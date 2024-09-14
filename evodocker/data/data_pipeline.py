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

import os
from typing import List

import numpy as np
import torch
import ml_collections as mlc
from rdkit import Chem

from evodocker.data import data_transforms
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


