import copy
import itertools
import time
from collections import Counter
from functools import partial
import json
import os
import pickle
from typing import Optional, Sequence, Any

import ml_collections as mlc
import lightning as L
import numpy as np
import torch
from torch.utils.data import RandomSampler

from evodocker.config import NUM_RES
from evodocker.data.data_transforms import get_restypes, get_restype_atom37_mask
from evodocker.data.utils import FeatureTensorDict
from evodocker.utils.residue_constants import restypes
from evodocker.data import data_pipeline
from evodocker.utils.consts import POSSIBLE_ATOM_TYPES
from evodocker.utils.tensor_utils import dict_multimap
from evodocker.utils.tensor_utils import (
    tensor_tree_map,
)


class OpenFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir: str,
                 config: mlc.ConfigDict,
                 mode: str = "train",
                 ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                config:
                    A dataset config object. See openfold.config
                mode:
                    "train", "val", or "predict"
        """
        super(OpenFoldSingleDataset, self).__init__()
        self.data_dir = data_dir

        self.config = config
        self.mode = mode

        valid_modes = ["train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        self._all_input_files = [i for i in os.listdir(data_dir)
                                 if i.endswith(".json") and self._verify_json_input_file(i)]

        self.data_pipeline = data_pipeline.DataPipeline(config, mode)


    def _verify_json_input_file(self, file_name: str) -> bool:
        with open(os.path.join(self.data_dir, file_name), "r") as f:
            try:
                loaded = json.load(f)
                for i in ["input_structure"]:
                    if i not in loaded:
                        return False
                if self.mode != "predict":
                    for i in ["gt_structure", "resolution"]:
                        if i not in loaded:
                            return False
            except json.JSONDecodeError:
                return False
        return True

    def get_metadata_for_idx(self, idx: int) -> dict:
        input_path = os.path.join(self.data_dir, self._all_input_files[idx])
        input_data = json.load(open(input_path, "r"))
        parent_dir = os.path.dirname(self.data_dir)
        input_pdb_path = os.path.join(parent_dir, input_data["input_structure"])
        input_structure = self.data_pipeline.process_pdb(pdb_path=input_pdb_path)
        metadata = {
            "resolution": input_data.get("resolution", 99.0),
            "input_path": input_path,
            "input_name": os.path.basename(input_path).split(".")[0],
        }
        return metadata

    @staticmethod
    def _prepare_recycles(feat: torch.Tensor, num_recycles: int) -> torch.Tensor:
        return feat.unsqueeze(-1).repeat(*([1] * len(feat.shape)), num_recycles)

    # def _add_affinity_to_ligand(self, ref_ligand_feats: FeatureTensorDict) -> FeatureTensorDict:
    #     # TODO: delete
    #     # add affinity, add additional token
    #     affinity_target_feat = torch.zeros((1, ref_ligand_feats["ligand_target_feat"].shape[-1]), dtype=torch.float32)
    #     affinity_target_feat[0, POSSIBLE_ATOM_TYPES.index("[AFFINITY]")] = 1
    #     ligand_target_feat = torch.cat([ ref_ligand_feats["ligand_target_feat"], affinity_target_feat], dim=0)
    #
    #     n_lig = ref_ligand_feats["ligand_target_feat"].shape[0]
    #     bonds_feat_size = ref_ligand_feats["ligand_bonds_feat"].shape[-1]
    #     column_zeros = torch.zeros(n_lig, 1, bonds_feat_size)
    #     row_zeros = torch.zeros(1, n_lig + 1, bonds_feat_size)
    #     tensor_with_col = torch.cat([ref_ligand_feats["ligand_target_feat"], column_zeros], dim=1)
    #     ligand_bonds_feat = torch.cat([tensor_with_col, row_zeros], dim=0)
    #
    #     center = ref_ligand_feats["ref_ligand_positions"].mean(dim=0)
    #     ref_ligand_positions = torch.cat([ref_ligand_feats["ref_ligand_positions"], center.unsqueeze(0)], dim=0)
    #
    #     return {
    #         "ligand_target_feat": ligand_target_feat,
    #         "ligand_bonds_feat": ligand_bonds_feat,
    #         "ref_ligand_positions": ref_ligand_positions,
    #     }
    #
    # def _pad_feats(self, feats):
    #     # TODO: delte
    #     if (self.mode == "train" or self.mode == "eval") and self.config.train.fixed_size:
    #         shape_schema = self.config.common.feat
    #         filtered_feats = {}
    #         for k, v in feats.items():
    #             if k not in shape_schema:
    #                 continue
    #             pad_size_map = {NUM_RES: self.config.train.crop_size}
    #             shape = list(v.shape)
    #             schema = shape_schema[k]
    #             msg = "Rank mismatch between shape and shape schema for"
    #             assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
    #
    #             pad_size = [
    #                 pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
    #             ]
    #
    #             padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
    #             padding.reverse()
    #             padding = list(itertools.chain(*padding))
    #             if padding:
    #                 filtered_feats[k] = torch.nn.functional.pad(v, padding)
    #                 filtered_feats[k] = torch.reshape(filtered_feats[k], pad_size)
    #         feats = filtered_feats
    #     return feats

    def fit_to_crop(self, target_tensor: torch.Tensor, crop_size: int, start_ind: int) -> torch.Tensor:
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

    def __getitem__(self, idx):
        start_load_time = time.time()

        input_path = os.path.join(self.data_dir, self._all_input_files[idx])
        input_data = json.load(open(input_path, "r"))
        if self.mode == "train" or self.mode == "eval":
            print("loading", input_data["pdb_id"], end=" ")

        num_recycles = self.config.common.max_recycling_iters + 1

        parent_dir = os.path.dirname(self.data_dir)
        input_pdb_path = os.path.join(parent_dir, input_data["input_structure"])
        input_protein_feats = self.data_pipeline.process_pdb(pdb_path=input_pdb_path)

        # load ref sdf
        ref_sdf_path = os.path.join(parent_dir, input_data["ref_sdf"])
        ref_ligand_feats = self.data_pipeline.process_sdf(sdf_path=ref_sdf_path)

        n_res = input_protein_feats["protein_target_feat"].shape[0]
        n_lig = ref_ligand_feats["ligand_target_feat"].shape[0]
        n_affinity = 1

        # ref_ligand_feats = self._add_affinity_to_ligand(ref_ligand_feats)

        # add 1 for affinity token
        crop_size = n_res + n_lig + n_affinity
        if (self.mode == "train" or self.mode == "eval") and self.config.train.fixed_size:
            crop_size = self.config.train.crop_size

        assert crop_size >= n_res + n_lig + n_affinity, f"crop_size: {crop_size}, n_res: {n_res}, n_lig: {n_lig}"

        token_mask = torch.zeros((crop_size,), dtype=torch.float32)
        token_mask[:n_res + n_lig + n_affinity] = 1

        protein_mask = torch.zeros((crop_size,), dtype=torch.float32)
        protein_mask[:n_res] = 1

        ligand_mask = torch.zeros((crop_size,), dtype=torch.float32)
        ligand_mask[n_res:n_res + n_lig] = 1

        affinity_mask = torch.zeros((crop_size,), dtype=torch.float32)
        affinity_mask[n_res + n_lig] = 1
        affinity_mask[n_res + n_lig - 1] = 1 # TODO: should we do this? it worked better in previous versions, but not sure why

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

        # Implement ligand as amino acid type 20
        ligand_aatype = 20 * torch.ones((n_lig, ), dtype=input_protein_feats["aatype"].dtype)
        aatype = torch.cat([input_protein_feats["aatype"], ligand_aatype], dim=0)

        restype_atom14_to_atom37, restype_atom37_to_atom14, restype_atom14_mask = get_restypes(target_feat.device)
        lig_residx_atom37_to_atom14 = restype_atom37_to_atom14[20].repeat(n_lig, 1)
        residx_atom37_to_atom14 = torch.cat([input_protein_feats["residx_atom37_to_atom14"], lig_residx_atom37_to_atom14], dim=0)

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
            "protein_residue_index": self.fit_to_crop(input_protein_feats["residue_index"], crop_size, 0),
            "aatype": self.fit_to_crop(aatype, crop_size, 0),
            "residx_atom37_to_atom14": self.fit_to_crop(residx_atom37_to_atom14, crop_size, 0),
            "atom37_atom_exists": self.fit_to_crop(atom37_atom_exists, crop_size, 0),
        }

        if self.mode == "predict":
            feats.update({
                "in_chain_residue_index": self.fit_to_crop(input_protein_feats["in_chain_residue_index"], crop_size, 0),
                "chain_index": self.fit_to_crop(input_protein_feats["chain_index"], crop_size, 0),
                "ligand_atype": self.fit_to_crop(ref_ligand_feats["ligand_atype"], crop_size, n_res),
                "ligand_chirality": self.fit_to_crop(ref_ligand_feats["ligand_chirality"], crop_size, n_res),
                "ligand_charge": self.fit_to_crop(ref_ligand_feats["ligand_charge"], crop_size, n_res),
                "ligand_bonds": ref_ligand_feats["ligand_bonds"],
            })

        if self.mode == 'train' or self.mode == 'eval':
            gt_pdb_path = os.path.join(parent_dir, input_data["gt_structure"])
            gt_protein_feats = self.data_pipeline.process_pdb(pdb_path=gt_pdb_path)

            gt_ligand_positions = self.data_pipeline.get_matching_positions(
                os.path.join(parent_dir, input_data["ref_sdf"]),
                os.path.join(parent_dir, input_data["gt_sdf"]),
            )

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
            lig_single_res_atom37_mask = torch.zeros((37, ), dtype=torch.float32)
            lig_single_res_atom37_mask[1] = 1
            lig_atom37_mask = lig_single_res_atom37_mask.unsqueeze(0).expand(n_lig, -1)
            lig_single_res_atom14_mask = torch.zeros((14, ), dtype=torch.float32)
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
                 torch.ones((n_lig, ), dtype=gt_protein_feats["pseudo_beta_mask"].dtype)],
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
                "atom14_alt_gt_positions": gt_protein_feats["atom14_alt_gt_positions"],    # torch.Size([n_res, 14, 3])
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
                gt_protein_feats[k] = self.fit_to_crop(v, crop_size, 0)

            feats = {
                **feats,
                **gt_protein_feats,
                "gt_ligand_positions": self.fit_to_crop(gt_ligand_positions, crop_size, n_res),
                "resolution": resolution,
                "affinity": affinity,
                "seq_length": torch.tensor(n_res + n_lig),
                "binding_site_mask": self.fit_to_crop(binding_site_mask, crop_size, 0),
                "gt_inter_contacts": inter_contact_reshaped_to_crop,
            }

        for k, v in feats.items():
            # print(k, v.shape)
            feats[k] = self._prepare_recycles(v, num_recycles)

        feats["batch_idx"] = torch.tensor(
            [idx for _ in range(crop_size)], dtype=torch.int64, device=feats["aatype"].device
        )

        print("load time", round(time.time() - start_load_time, 4))

        return feats

    def __len__(self):
        return len(self._all_input_files)


def resolution_filter(resolution: int, max_resolution: float) -> bool:
    """Check that the resolution is <= max_resolution permitted"""
    return resolution is not None and resolution <= max_resolution


def all_seq_len_filter(seqs: list, minimum_number_of_residues: int) -> bool:
    """Check if the total combined sequence lengths are >= minimum_numer_of_residues"""
    total_len = sum([len(i) for i in seqs])
    return total_len >= minimum_number_of_residues


class OpenFoldDataset(torch.utils.data.Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an OpenFoldFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """

    def __init__(self,
                 datasets: Sequence[OpenFoldSingleDataset],
                 probabilities: Sequence[float],
                 epoch_len: int,
                 generator: torch.Generator = None,
                 _roll_at_init: bool = True,
                 ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator

        self._samples = [self.looped_samples(i) for i in range(len(self.datasets))]
        if _roll_at_init:
            self.reroll()

    @staticmethod
    def deterministic_train_filter(
        cache_entry: Any,
        max_resolution: float = 9.,
        max_single_aa_prop: float = 0.8,
        *args, **kwargs
    ) -> bool:
        # Hard filters
        resolution = cache_entry["resolution"]

        return all([
            resolution_filter(resolution=resolution,
                              max_resolution=max_resolution)
        ])

    @staticmethod
    def get_stochastic_train_filter_prob(
        cache_entry: Any,
        *args, **kwargs
    ) -> float:
        # Stochastic filters
        probabilities = []

        cluster_size = cache_entry.get("cluster_size", None)
        if cluster_size is not None and cluster_size > 0:
            probabilities.append(1 / cluster_size)

        # Risk of underflow here?
        out = 1
        for p in probabilities:
            out *= p

        return out

    def looped_shuffled_dataset_idx(self, dataset_len):
        while True:
            # Uniformly shuffle each dataset's indices
            weights = [1. for _ in range(dataset_len)]
            shuf = torch.multinomial(
                torch.tensor(weights),
                num_samples=dataset_len,
                replacement=False,
                generator=self.generator,
            )
            for idx in shuf:
                yield idx

    def looped_samples(self, dataset_idx):
        max_cache_len = int(self.epoch_len * self.probabilities[dataset_idx])
        dataset = self.datasets[dataset_idx]
        idx_iter = self.looped_shuffled_dataset_idx(len(dataset))
        while True:
            weights = []
            idx = []
            for _ in range(max_cache_len):
                candidate_idx = next(idx_iter)
                # chain_id = dataset.idx_to_chain_id(candidate_idx)
                # chain_data_cache_entry = chain_data_cache[chain_id]
                # data_entry = dataset[candidate_idx.item()]
                entry_metadata_for_filter = dataset.get_metadata_for_idx(candidate_idx.item())
                if not self.deterministic_train_filter(entry_metadata_for_filter):
                    continue

                p = self.get_stochastic_train_filter_prob(
                    entry_metadata_for_filter,
                )
                weights.append([1. - p, p])
                idx.append(candidate_idx)

            samples = torch.multinomial(
                torch.tensor(weights),
                num_samples=1,
                generator=self.generator,
            )
            samples = samples.squeeze()

            cache = [i for i, s in zip(idx, samples) if s]

            for datapoint_idx in cache:
                yield datapoint_idx

    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        # TODO bshor: I have removed support for filters (currently done in preprocess) and to weighting clusters
        # now it is much faster, because it doesn't call looped_samples
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )
        self.datapoints = []
        counter_datasets = Counter(dataset_choices.tolist())
        for dataset_idx, num_samples in counter_datasets.items():
            dataset = self.datasets[dataset_idx]
            sample_choices = torch.randint(0, len(dataset), (num_samples,), generator=self.generator)
            for datapoint_idx in sample_choices:
                self.datapoints.append((dataset_idx, datapoint_idx))


class OpenFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)


class OpenFoldDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters

        if stage_cfg.uniform_recycling:
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.

        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs]

        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        # gt_features = batch.pop('gt_features', None)
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1,  # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["aatype"]
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample,
                device=aatype.device,
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if key == "no_recycling_iters":
                no_recycling = sample

        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)
        # batch['gt_features'] = gt_features

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class OpenFoldDataModule(L.LightningDataModule):
    def __init__(self,
                 config: mlc.ConfigDict,
                 train_data_dir: Optional[str] = None,
                 val_data_dir: Optional[str] = None,
                 predict_data_dir: Optional[str] = None,
                 batch_seed: Optional[int] = None,
                 train_epoch_len: int = 50000,
                 **kwargs
                 ):
        super(OpenFoldDataModule, self).__init__()

        self.config = config
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.predict_data_dir = predict_data_dir
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        if self.train_data_dir is None and self.predict_data_dir is None:
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_data_dir is not None

        # if not self.training_mode and predict_alignment_dir is None:
        #     raise ValueError(
        #         'In inference mode, predict_alignment_dir must be specified'
        #     )
        # elif val_data_dir is not None and val_alignment_dir is None:
        #     raise ValueError(
        #         'If val_data_dir is specified, val_alignment_dir must '
        #         'be specified as well'
        #     )

    def setup(self, stage):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(OpenFoldSingleDataset,
                              config=self.config)

        if self.training_mode:
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                mode="train",
            )

            datasets = [train_dataset]
            probabilities = [1.]

            generator = None
            if self.batch_seed is not None:
                generator = torch.Generator()
                generator = generator.manual_seed(self.batch_seed + 1)

            self.train_dataset = OpenFoldDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=self.train_epoch_len,
                generator=generator,
                _roll_at_init=False,
            )

            if self.val_data_dir is not None:
                self.eval_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    mode="eval",
                )
            else:
                self.eval_dataset = None
        else:
            self.predict_dataset = dataset_gen(
                data_dir=self.predict_data_dir,
                mode="predict",
            )

    def _gen_dataloader(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        if stage == "train":
            dataset = self.train_dataset
            # Filter the dataset, if necessary
            dataset.reroll()
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "predict":
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = OpenFoldBatchCollator()

        dl = OpenFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            # num_workers=self.config.data_module.data_loaders.num_workers,
            num_workers=0, # TODO bshor: solve generator pickling issue and then bring back num_workers, or just remove generator
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train")

    def val_dataloader(self):
        if self.eval_dataset is not None:
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, batch_path):
        with open(batch_path, "rb") as f:
            self.batch = pickle.load(f)

    def __getitem__(self, idx):
        return copy.deepcopy(self.batch)

    def __len__(self):
        return 1000


class DummyDataLoader(L.LightningDataModule):
    def __init__(self, batch_path):
        super().__init__()
        self.dataset = DummyDataset(batch_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset)
