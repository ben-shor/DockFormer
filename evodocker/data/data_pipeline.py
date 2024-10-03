# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import time
from typing import List

import numpy as np
import torch
import ml_collections as mlc
from rdkit import Chem

from evodocker.data import data_transforms
from evodocker.data.data_transforms import get_restype_atom37_mask, get_restypes
from evodocker.data.ligand_features import make_ligand_features
from evodocker.data.protein_features import make_protein_features
from evodocker.data.utils import FeatureTensorDict, FeatureDict
from evodocker.utils import protein


def _np_filter_and_to_tensor_dict(np_example: FeatureDict, features_to_keep: List[str]) -> FeatureTensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    # torch generates warnings if feature is already a torch Tensor
    to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
    tensor_dict = {
        k: to_tensor(v) for k, v in np_example.items() if k in features_to_keep
    }

    return tensor_dict


def _add_protein_probablistic_features(features: FeatureDict, cfg: mlc.ConfigDict, mode: str) -> FeatureDict:
    if mode == "train":
        p = torch.rand(1).item()
        use_clamped_fape_value = float(p < cfg.supervised.clamp_prob)
        features["use_clamped_fape"] = np.float32(use_clamped_fape_value)
    else:
        features["use_clamped_fape"] = np.float32(0.0)
    return features


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def _apply_protein_transforms(tensors: FeatureTensorDict) -> FeatureTensorDict:
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.squeeze_features,
        data_transforms.make_atom14_masks,
        data_transforms.make_atom14_positions,
        data_transforms.atom37_to_frames,
        data_transforms.atom37_to_torsion_angles(""),
        data_transforms.make_pseudo_beta(),
        data_transforms.get_backbone_frames,
        data_transforms.get_chi_angles,
    ]

    tensors = compose(transforms)(tensors)

    return tensors


def _apply_protein_probablistic_transforms(tensors: FeatureTensorDict, cfg: mlc.ConfigDict, mode: str) \
        -> FeatureTensorDict:
    transforms = [data_transforms.make_target_feat()]

    crop_feats = dict(cfg.common.feat)

    if cfg[mode].fixed_size:
        transforms.append(data_transforms.select_feat(list(crop_feats)))
        # TODO bshor: restore transforms for training on cropped proteins, need to handle pocket somehow
        # if so, look for random_crop_to_size and make_fixed_size in data_transforms.py

    compose(transforms)(tensors)

    return tensors


class DataPipeline:
    """Assembles input features."""
    def __init__(self, config: mlc.ConfigDict, mode: str):
        self.config = config
        self.mode = mode

        self.feature_names = config.common.unsupervised_features
        if config[mode].supervised:
            self.feature_names += config.supervised.supervised_features

    def process_pdb(self, pdb_path: str) -> FeatureTensorDict:
        """
            Assembles features for a protein in a PDB file.
        """
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()

        protein_object = protein.from_pdb_string(pdb_str)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_protein_features(protein_object, description)
        pdb_feats = _add_protein_probablistic_features(pdb_feats, self.config, self.mode)

        tensor_feats = _np_filter_and_to_tensor_dict(pdb_feats, self.feature_names)

        tensor_feats = _apply_protein_transforms(tensor_feats)
        tensor_feats = _apply_protein_probablistic_transforms(tensor_feats, self.config, self.mode)

        return tensor_feats

    def process_smiles(self, smiles: str) -> FeatureTensorDict:
        ligand = Chem.MolFromSmiles(smiles)
        return make_ligand_features(ligand)

    def process_mol2(self, mol2_path: str) -> FeatureTensorDict:
        """
            Assembles features for a ligand in a mol2 file.
        """
        ligand = Chem.MolFromMol2File(mol2_path)
        assert ligand is not None, f"Failed to parse ligand from {mol2_path}"

        conf = ligand.GetConformer()
        positions = torch.tensor(conf.GetPositions())

        return {
            **make_ligand_features(ligand),
            "gt_ligand_positions": positions.float()
        }

    def process_sdf(self, sdf_path: str) -> FeatureTensorDict:
        """
            Assembles features for a ligand in a mol2 file.
        """
        ligand = Chem.MolFromMolFile(sdf_path)
        assert ligand is not None, f"Failed to parse ligand from {sdf_path}"

        conf = ligand.GetConformer(0)
        positions = torch.tensor(conf.GetPositions())

        return {
            **make_ligand_features(ligand),
            "ligand_positions": positions.float()
        }

    def process_sdf_list(self, sdf_path_list: List[str]) -> FeatureTensorDict:
        all_sdf_feats = [self.process_sdf(sdf_path) for sdf_path in sdf_path_list]

        all_sizes = [sdf_feats["ligand_target_feat"].shape[0] for sdf_feats in all_sdf_feats]

        joined_ligand_feats = {}
        for k in all_sdf_feats[0].keys():
            if k == "ligand_positions":
                joined_positions = all_sdf_feats[0][k]
                prev_offset = joined_positions.max(dim=0).values + 100

                for i, sdf_feats in enumerate(all_sdf_feats[1:]):
                    offset = prev_offset - sdf_feats[k].min(dim=0).values
                    joined_positions = torch.cat([joined_positions, sdf_feats[k] + offset], dim=0)
                    prev_offset = joined_positions.max(dim=0).values + 100
                joined_ligand_feats[k] = joined_positions
            elif k in ["ligand_target_feat", "ligand_atype", "ligand_charge", "ligand_chirality", "ligand_bonds"]:
                joined_ligand_feats[k] = torch.cat([sdf_feats[k] for sdf_feats in all_sdf_feats], dim=0)
                if k == "ligand_target_feat":
                    joined_ligand_feats["ligand_idx"] = torch.cat([torch.full((sdf_feats[k].shape[0],), i)
                                                                   for i, sdf_feats in enumerate(all_sdf_feats)], dim=0)
                elif k == "ligand_bonds":
                    joined_ligand_feats["ligand_bonds_idx"] = torch.cat([torch.full((sdf_feats[k].shape[0],), i)
                                                                        for i, sdf_feats in enumerate(all_sdf_feats)],
                                                                        dim=0)
            elif k == "ligand_bonds_feat":
                joined_feature = torch.zeros((sum(all_sizes), sum(all_sizes), all_sdf_feats[0][k].shape[2]))
                for i, sdf_feats in enumerate(all_sdf_feats):
                    start_idx = sum(all_sizes[:i])
                    end_idx = sum(all_sizes[:i + 1])
                    joined_feature[start_idx:end_idx, start_idx:end_idx, :] = sdf_feats[k]
                joined_ligand_feats[k] = joined_feature
            else:
                raise ValueError(f"Unknown key in sdf list features {k}")
        return joined_ligand_feats

    def get_matching_positions_list(self, ref_path_list: List[str], gt_path_list: List[str]):
        joined_gt_positions = []

        for ref_ligand_path, gt_ligand_path in zip(ref_path_list, gt_path_list):
            ref_ligand = Chem.MolFromMolFile(ref_ligand_path)
            gt_ligand = Chem.MolFromMolFile(gt_ligand_path)

            gt_original_positions = gt_ligand.GetConformer(0).GetPositions()

            gt_positions = [gt_original_positions[idx] for idx in gt_ligand.GetSubstructMatch(ref_ligand)]

            joined_gt_positions.extend(gt_positions)

        return torch.tensor(np.array(joined_gt_positions)).float()

    def get_matching_positions(self, ref_ligand_path: str, gt_ligand_path: str):
        ref_ligand = Chem.MolFromMolFile(ref_ligand_path)
        gt_ligand = Chem.MolFromMolFile(gt_ligand_path)

        gt_original_positions = gt_ligand.GetConformer(0).GetPositions()

        gt_positions = [gt_original_positions[idx] for idx in gt_ligand.GetSubstructMatch(ref_ligand)]

        # ref_positions = ref_ligand.GetConformer(0).GetPositions()
        # for i in range(len(ref_positions)):
        #     for j in range(i + 1, len(ref_positions)):
        #         dist_ref = np.linalg.norm(ref_positions[i] - ref_positions[j])
        #         dist_gt = np.linalg.norm(gt_positions[i] - gt_positions[j])
        #         dist_gt = np.linalg.norm(gt_original_positions[i] - gt_original_positions[j])
        #         if abs(dist_ref - dist_gt) > 1.0:
        #             print(f"Distance mismatch {i} {j} {dist_ref} {dist_gt}")

        return torch.tensor(np.array(gt_positions)) .float()


def _prepare_recycles(feat: torch.Tensor, num_recycles: int) -> torch.Tensor:
    return feat.unsqueeze(-1).repeat(*([1] * len(feat.shape)), num_recycles)


def _fit_to_crop(target_tensor: torch.Tensor, crop_size: int, start_ind: int) -> torch.Tensor:
    if len(target_tensor.shape) == 1:
        ret = torch.zeros((crop_size, ), dtype=target_tensor.dtype)
        ret[start_ind:start_ind + target_tensor.shape[0]] = target_tensor
        return ret
    elif len(target_tensor.shape) == 2:
        ret = torch.zeros((crop_size, target_tensor.shape[-1]), dtype=target_tensor.dtype)
        ret[start_ind:start_ind + target_tensor.shape[0], :] = target_tensor
        return ret
    else:
        ret = torch.zeros((crop_size, *target_tensor.shape[1:]), dtype=target_tensor.dtype)
        ret[start_ind:start_ind + target_tensor.shape[0], ...] = target_tensor
        return ret


def parse_input_json(input_path: str, mode: str, config: mlc.ConfigDict, data_pipeline: DataPipeline,
                     data_dir: str, idx: int) -> FeatureTensorDict:
    start_load_time = time.time()
    input_data = json.load(open(input_path, "r"))
    if mode == "train" or mode == "eval":
        print("loading", input_data["pdb_id"], end=" ")

    num_recycles = config.common.max_recycling_iters + 1

    input_pdb_path = os.path.join(data_dir, input_data["input_structure"])
    input_protein_feats = data_pipeline.process_pdb(pdb_path=input_pdb_path)

    # load ref sdf
    if "ref_sdf" in input_data:
        ref_sdf_path = os.path.join(data_dir, input_data["ref_sdf"])
        ref_ligand_feats = data_pipeline.process_sdf(sdf_path=ref_sdf_path)
        ref_ligand_feats["ligand_idx"] = torch.zeros((ref_ligand_feats["ligand_target_feat"].shape[0],))
        ref_ligand_feats["ligand_bonds_idx"] = torch.zeros((ref_ligand_feats["ligand_bonds"].shape[0],))
    elif "ref_sdf_list" in input_data:
        sdf_path_list = [os.path.join(data_dir, i) for i in input_data["ref_sdf_list"]]
        ref_ligand_feats = data_pipeline.process_sdf_list(sdf_path_list=sdf_path_list)
    else:
        raise ValueError("ref_sdf or ref_sdf_list must be in input_data")

    n_res = input_protein_feats["protein_target_feat"].shape[0]
    n_lig = ref_ligand_feats["ligand_target_feat"].shape[0]
    n_affinity = 1

    # add 1 for affinity token
    crop_size = n_res + n_lig + n_affinity
    if (mode == "train" or mode == "eval") and config.train.fixed_size:
        crop_size = config.train.crop_size

    assert crop_size >= n_res + n_lig + n_affinity, f"crop_size: {crop_size}, n_res: {n_res}, n_lig: {n_lig}"

    token_mask = torch.zeros((crop_size,), dtype=torch.float32)
    token_mask[:n_res + n_lig + n_affinity] = 1

    protein_mask = torch.zeros((crop_size,), dtype=torch.float32)
    protein_mask[:n_res] = 1

    ligand_mask = torch.zeros((crop_size,), dtype=torch.float32)
    ligand_mask[n_res:n_res + n_lig] = 1

    affinity_mask = torch.zeros((crop_size,), dtype=torch.float32)
    affinity_mask[n_res + n_lig] = 1

    structural_mask = torch.zeros((crop_size,), dtype=torch.float32)
    structural_mask[:n_res + n_lig] = 1

    inter_pair_mask = torch.zeros((crop_size, crop_size), dtype=torch.float32)
    inter_pair_mask[:n_res, n_res:n_res + n_lig] = 1
    inter_pair_mask[n_res:n_res + n_lig, :n_res] = 1

    protein_tf_dim = input_protein_feats["protein_target_feat"].shape[-1]
    ligand_tf_dim = ref_ligand_feats["ligand_target_feat"].shape[-1]
    joined_tf_dim = protein_tf_dim + ligand_tf_dim

    target_feat = torch.zeros((crop_size, joined_tf_dim + 3), dtype=torch.float32)
    target_feat[:n_res, :protein_tf_dim] = input_protein_feats["protein_target_feat"]
    target_feat[n_res:n_res + n_lig, protein_tf_dim:joined_tf_dim] = ref_ligand_feats["ligand_target_feat"]

    target_feat[:n_res, joined_tf_dim] = 1  # Set "is_protein" flag for protein rows
    target_feat[n_res:n_res + n_lig, joined_tf_dim + 1] = 1  # Set "is_ligand" flag for ligand rows
    target_feat[n_res + n_lig, joined_tf_dim + 2] = 1  # Set "is_affinity" flag for affinity row

    ligand_bonds_feat = torch.zeros((crop_size, crop_size, ref_ligand_feats["ligand_bonds_feat"].shape[-1]),
                                    dtype=torch.float32)
    ligand_bonds_feat[n_res:n_res + n_lig, n_res:n_res + n_lig] = ref_ligand_feats["ligand_bonds_feat"]

    input_positions = torch.zeros((crop_size, 3), dtype=torch.float32)
    input_positions[:n_res] = input_protein_feats["pseudo_beta"]
    input_positions[n_res:n_res + n_lig] = ref_ligand_feats["ligand_positions"]

    protein_distogram_mask = torch.zeros(crop_size)
    if mode == "train":
        ones_indices = torch.randperm(n_res)[:int(n_res * config.train.protein_distogram_mask_prob)]
        # print(ones_indices)
        protein_distogram_mask[ones_indices] = 1
        input_positions = input_positions * (1 - protein_distogram_mask).unsqueeze(-1)

    # Implement ligand as amino acid type 20
    ligand_aatype = 20 * torch.ones((n_lig,), dtype=input_protein_feats["aatype"].dtype)
    aatype = torch.cat([input_protein_feats["aatype"], ligand_aatype], dim=0)

    restype_atom14_to_atom37, restype_atom37_to_atom14, restype_atom14_mask = get_restypes(target_feat.device)
    lig_residx_atom37_to_atom14 = restype_atom37_to_atom14[20].repeat(n_lig, 1)
    residx_atom37_to_atom14 = torch.cat([input_protein_feats["residx_atom37_to_atom14"], lig_residx_atom37_to_atom14],
                                        dim=0)

    restype_atom37_mask = get_restype_atom37_mask(target_feat.device)
    lig_atom37_atom_exists = restype_atom37_mask[20].repeat(n_lig, 1)
    atom37_atom_exists = torch.cat([input_protein_feats["atom37_atom_exists"], lig_atom37_atom_exists], dim=0)

    feats = {
        "token_mask": token_mask,
        "protein_mask": protein_mask,
        "ligand_mask": ligand_mask,
        "affinity_mask": affinity_mask,
        "structural_mask": structural_mask,
        "inter_pair_mask": inter_pair_mask,

        "target_feat": target_feat,
        "ligand_bonds_feat": ligand_bonds_feat,
        "input_positions": input_positions,
        "protein_distogram_mask": protein_distogram_mask,
        "protein_residue_index": _fit_to_crop(input_protein_feats["residue_index"], crop_size, 0),
        "aatype": _fit_to_crop(aatype, crop_size, 0),
        "residx_atom37_to_atom14": _fit_to_crop(residx_atom37_to_atom14, crop_size, 0),
        "atom37_atom_exists": _fit_to_crop(atom37_atom_exists, crop_size, 0),
    }

    if mode == "predict":
        feats.update({
            "in_chain_residue_index": input_protein_feats["in_chain_residue_index"],
            "chain_index": input_protein_feats["chain_index"],
            "ligand_atype": ref_ligand_feats["ligand_atype"],
            "ligand_chirality": ref_ligand_feats["ligand_chirality"],
            "ligand_charge": ref_ligand_feats["ligand_charge"],
            "ligand_bonds": ref_ligand_feats["ligand_bonds"],
            "ligand_idx": ref_ligand_feats["ligand_idx"],
            "ligand_bonds_idx": ref_ligand_feats["ligand_bonds_idx"],
        })

    if mode == 'train' or mode == 'eval':
        gt_pdb_path = os.path.join(data_dir, input_data["gt_structure"])
        gt_protein_feats = data_pipeline.process_pdb(pdb_path=gt_pdb_path)

        if "gt_sdf" in input_data:
            gt_ligand_positions = data_pipeline.get_matching_positions(
                os.path.join(data_dir, input_data["ref_sdf"]),
                os.path.join(data_dir, input_data["gt_sdf"]),
            )
        elif "gt_sdf_list" in input_data:
            gt_ligand_positions = data_pipeline.get_matching_positions_list(
                [os.path.join(data_dir, i) for i in input_data["ref_sdf_list"]],
                [os.path.join(data_dir, i) for i in input_data["gt_sdf_list"]],
            )
        else:
            raise ValueError("gt_sdf or gt_sdf_list must be in input_data")

        affinity_loss_factor = torch.tensor([1.0], dtype=torch.float32)
        if input_data["affinity"] is None:
            eps = 1e-6
            affinity_loss_factor = torch.tensor([eps], dtype=torch.float32)
            affinity = torch.tensor([0.0], dtype=torch.float32)
        else:
            affinity = torch.tensor([input_data["affinity"]], dtype=torch.float32)

        resolution = torch.tensor(input_data["resolution"], dtype=torch.float32)

        # prepare inter_contacts
        expanded_prot_pos = gt_protein_feats["pseudo_beta"].unsqueeze(1)  # Shape: (N_prot, 1, 3)
        expanded_lig_pos = gt_ligand_positions.unsqueeze(0)  # Shape: (1, N_lig, 3)
        distances = torch.sqrt(torch.sum((expanded_prot_pos - expanded_lig_pos) ** 2, dim=-1))
        inter_contact = (distances < 5.0).float()
        binding_site_mask = inter_contact.any(dim=1).float()

        inter_contact_reshaped_to_crop = torch.zeros((crop_size, crop_size), dtype=torch.float32)
        inter_contact_reshaped_to_crop[:n_res, n_res:n_res + n_lig] = inter_contact
        inter_contact_reshaped_to_crop[n_res:n_res + n_lig, :n_res] = inter_contact.T

        # Use CA positions only
        lig_single_res_atom37_mask = torch.zeros((37,), dtype=torch.float32)
        lig_single_res_atom37_mask[1] = 1
        lig_atom37_mask = lig_single_res_atom37_mask.unsqueeze(0).expand(n_lig, -1)
        lig_single_res_atom14_mask = torch.zeros((14,), dtype=torch.float32)
        lig_single_res_atom14_mask[1] = 1
        lig_atom14_mask = lig_single_res_atom14_mask.unsqueeze(0).expand(n_lig, -1)

        lig_atom37_positions = gt_ligand_positions.unsqueeze(1).expand(-1, 37, -1)
        lig_atom37_positions = lig_atom37_positions * lig_single_res_atom37_mask.view(1, 37, 1).expand(n_lig, -1, 3)

        lig_atom14_positions = gt_ligand_positions.unsqueeze(1).expand(-1, 14, -1)
        lig_atom14_positions = lig_atom14_positions * lig_single_res_atom14_mask.view(1, 14, 1).expand(n_lig, -1, 3)

        atom37_gt_positions = torch.cat([gt_protein_feats["all_atom_positions"], lig_atom37_positions], dim=0)
        atom37_atom_exists_in_res = torch.cat([gt_protein_feats["atom37_atom_exists"], lig_atom37_mask], dim=0)
        atom37_atom_exists_in_gt = torch.cat([gt_protein_feats["all_atom_mask"], lig_atom37_mask], dim=0)

        atom14_gt_positions = torch.cat([gt_protein_feats["atom14_gt_positions"], lig_atom14_positions], dim=0)
        atom14_atom_exists_in_res = torch.cat([gt_protein_feats["atom14_atom_exists"], lig_atom14_mask], dim=0)
        atom14_atom_exists_in_gt = torch.cat([gt_protein_feats["atom14_gt_exists"], lig_atom14_mask], dim=0)

        gt_pseudo_beta_with_lig = torch.cat([gt_protein_feats["pseudo_beta"], gt_ligand_positions], dim=0)
        gt_pseudo_beta_with_lig_mask = torch.cat(
            [gt_protein_feats["pseudo_beta_mask"],
             torch.ones((n_lig,), dtype=gt_protein_feats["pseudo_beta_mask"].dtype)],
            dim=0)

        # IGNORES: residx_atom14_to_atom37, rigidgroups_group_exists,
        # rigidgroups_group_is_ambiguous, pseudo_beta_mask, backbone_rigid_mask, protein_target_feat
        gt_protein_feats = {
            "atom37_gt_positions": atom37_gt_positions,  # torch.Size([n_struct, 37, 3])
            "atom37_atom_exists_in_res": atom37_atom_exists_in_res,  # torch.Size([n_struct, 37])
            "atom37_atom_exists_in_gt": atom37_atom_exists_in_gt,  # torch.Size([n_struct, 37])

            "atom14_gt_positions": atom14_gt_positions,  # torch.Size([n_struct, 14, 3])
            "atom14_atom_exists_in_res": atom14_atom_exists_in_res,  # torch.Size([n_struct, 14])
            "atom14_atom_exists_in_gt": atom14_atom_exists_in_gt,  # torch.Size([n_struct, 14])

            "gt_pseudo_beta_with_lig": gt_pseudo_beta_with_lig,  # torch.Size([n_struct, 3])
            "gt_pseudo_beta_with_lig_mask": gt_pseudo_beta_with_lig_mask,  # torch.Size([n_struct])

            # These we don't need to add the ligand to, because padding is sufficient (everything should be 0)
            "atom14_alt_gt_positions": gt_protein_feats["atom14_alt_gt_positions"],  # torch.Size([n_res, 14, 3])
            "atom14_alt_gt_exists": gt_protein_feats["atom14_alt_gt_exists"],  # torch.Size([n_res, 14])
            "atom14_atom_is_ambiguous": gt_protein_feats["atom14_atom_is_ambiguous"],  # torch.Size([n_res, 14])
            "rigidgroups_gt_frames": gt_protein_feats["rigidgroups_gt_frames"],  # torch.Size([n_res, 8, 4, 4])
            "rigidgroups_gt_exists": gt_protein_feats["rigidgroups_gt_exists"],  # torch.Size([n_res, 8])
            "rigidgroups_alt_gt_frames": gt_protein_feats["rigidgroups_alt_gt_frames"],  # torch.Size([n_res, 8, 4, 4])
            "backbone_rigid_tensor": gt_protein_feats["backbone_rigid_tensor"],  # torch.Size([n_res, 4, 4])
            "backbone_rigid_mask": gt_protein_feats["backbone_rigid_mask"],  # torch.Size([n_res])
            "chi_angles_sin_cos": gt_protein_feats["chi_angles_sin_cos"],
            "chi_mask": gt_protein_feats["chi_mask"],
        }

        for k, v in gt_protein_feats.items():
            gt_protein_feats[k] = _fit_to_crop(v, crop_size, 0)

        feats = {
            **feats,
            **gt_protein_feats,
            "gt_ligand_positions": _fit_to_crop(gt_ligand_positions, crop_size, n_res),
            "resolution": resolution,
            "affinity": affinity,
            "affinity_loss_factor": affinity_loss_factor,
            "seq_length": torch.tensor(n_res + n_lig),
            "binding_site_mask": _fit_to_crop(binding_site_mask, crop_size, 0),
            "gt_inter_contacts": inter_contact_reshaped_to_crop,
        }

    for k, v in feats.items():
        # print(k, v.shape)
        feats[k] = _prepare_recycles(v, num_recycles)

    feats["batch_idx"] = torch.tensor(
        [idx for _ in range(crop_size)], dtype=torch.int64, device=feats["aatype"].device
    )

    print("load time", round(time.time() - start_load_time, 4))

    return feats
