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
import sys

from env_consts import TEST_INPUT_DIR, TEST_OUTPUT_DIR, CKPT_PATH
import json
import logging
import numpy as np
import os
import pickle

from evodocker.data.data_modules import OpenFoldSingleDataset

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import torch
torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if (
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from evodocker.config import model_config
from evodocker.utils.script_utils import (load_models_from_command_line, run_model, save_output_structure,
                                          get_latest_checkpoint)
from evodocker.utils.tensor_utils import tensor_tree_map


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def override_config(base_config, overriding_config):
    for k, v in overriding_config.items():
        if isinstance(v, dict):
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def run_on_folder(input_dir: str, output_dir: str, run_config_path: str, skip_relaxation=True,
                  long_sequence_inference=False, skip_exists=False):
    config_preset = "initial_training"
    save_outputs = False
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    run_config = json.load(open(run_config_path))

    ckpt_path = CKPT_PATH
    if ckpt_path is None:
        ckpt_path = get_latest_checkpoint(os.path.join(run_config["train_output_dir"], "checkpoint"))
    print("Using checkpoint: ", ckpt_path)

    config = model_config(config_preset, long_sequence_inference=long_sequence_inference)
    config = override_config(config, run_config.get("override_conf", {}))

    model_generator = load_models_from_command_line(
        config,
        model_device=device_name,
        model_checkpoint_path=ckpt_path,
        output_dir=output_dir)
    print("Model loaded")
    model, output_directory = next(model_generator)

    dataset = OpenFoldSingleDataset(data_dir=input_dir, config=config.data, mode="predict")
    for i, processed_feature_dict in enumerate(dataset):
        tag = dataset.get_metadata_for_idx(i)["input_name"]
        output_name = f"{tag}_predicted"
        protein_output_path = os.path.join(output_directory, f'{output_name}_protein.pdb')
        if os.path.exists(protein_output_path) and skip_exists:
            print("skipping exists", output_name)
            continue

        # turn into a batch of size 1
        processed_feature_dict = {key: value.unsqueeze(0).to(device_name)
                                  for key, value in processed_feature_dict.items()}

        out = run_model(model, processed_feature_dict, tag, output_dir)

        # Toss out the recycling dimensions --- we don't need them anymore
        processed_feature_dict = tensor_tree_map(
            lambda x: np.array(x[..., -1].cpu()),
            processed_feature_dict
        )
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        affinity = torch.sum(torch.softmax(torch.tensor(out["affinity_2d_logits"]), -1) * torch.linspace(0, 15, 32),
                             dim=-1).item()
        print("Affinity: ", affinity)

        # binding_site = torch.sigmoid(torch.tensor(out["binding_site_logits"])) * 100
        # binding_site = binding_site[:processed_feature_dict["aatype"].shape[1]].flatten()

        predicted_contacts = torch.sigmoid(torch.tensor(out["inter_contact_logits"])) * 100
        binding_site = torch.max(predicted_contacts, dim=2).values.flatten()

        protein_binding_output_path = os.path.join(output_directory, f'{output_name}_protein_affinity.pdb')
        ligand_output_path = os.path.join(output_directory, f'{output_name}_ligand.sdf')

        protein_mask = processed_feature_dict["protein_mask"][0].astype(bool)
        ligand_mask = processed_feature_dict["ligand_mask"][0].astype(bool)


        save_output_structure(
            aatype=processed_feature_dict["aatype"][0][protein_mask],
            residue_index=processed_feature_dict["in_chain_residue_index"][0][protein_mask],
            chain_index=processed_feature_dict["chain_index"][0][protein_mask],
            plddt=out["plddt"][0][protein_mask],
            final_atom_protein_positions=out["final_atom_positions"][0][protein_mask],
            final_atom_mask=out["final_atom_mask"][0][protein_mask],
            ligand_atype=processed_feature_dict["ligand_atype"][0][ligand_mask].astype(int),
            ligand_chiralities=processed_feature_dict["ligand_chirality"][0][ligand_mask].astype(int),
            ligand_charges= processed_feature_dict["ligand_charge"][0][ligand_mask].astype(int),
            ligand_bonds=processed_feature_dict["ligand_bonds"][0][ligand_mask].astype(int),
            final_ligand_atom_positions=out["final_atom_positions"][0][ligand_mask][:, 1, :], # only ca index
            protein_output_path=protein_output_path,
            ligand_output_path=ligand_output_path,
            protein_affinity_output_path=protein_binding_output_path,
            affinity=affinity,
            binding_site_probs=binding_site
        )

        logger.info(f"Output written to {protein_output_path}...")

        if not skip_relaxation:
            # Relax the prediction.
            logger.info(f"Running relaxation on {protein_output_path}...")
            from evodocker.utils.relax import relax_complex
            try:
                relax_complex(protein_output_path,
                              ligand_output_path,
                              os.path.join(output_directory, f'{output_name}_protein_relaxed.pdb'),
                              os.path.join(output_directory, f'{output_name}_ligand_relaxed.sdf'))
            except Exception as e:
                logger.error(f"Failed to relax {protein_output_path} due to {e}...")

        if save_outputs:
            output_dict_path = os.path.join(
                output_directory, f'{output_name}_output_dict.pkl'
            )
            with open(output_dict_path, "wb") as fp:
                pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Model output written to {output_dict_path}...")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run_config.json")
    input_dir, output_dir = TEST_INPUT_DIR, TEST_OUTPUT_DIR
    options = {"skip_relaxation": True, "long_sequence_inference": False}
    if len(sys.argv) > 3:
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        if "--relax" in sys.argv:
            options["skip_relaxation"] = False
        if "--long" in sys.argv:
            options["long_sequence_inference"] = True
        if "--allow-skip" in sys.argv:
            options["skip_exists"] = True

    run_on_folder(input_dir, output_dir, config_path, **options)
