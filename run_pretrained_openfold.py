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
import logging
import numpy as np
import os
import pickle

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

from openfold.config import model_config
from openfold.data import feature_pipeline, data_pipeline
from openfold.np import protein
from openfold.utils.script_utils import (load_models_from_command_line, run_model, prep_output, relax_protein)
from openfold.utils.tensor_utils import tensor_tree_map


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def manual_main():
    input_dir = "/Users/benshor/Documents/Data/202401_pred_affinity/sample_test_input_folders/json"
    output_dir = "/Users/benshor/Documents/Data/202401_pred_affinity/sample_test_input_folders/output"
    ckpt_path = "/Users/benshor/Documents/Data/202401_pred_affinity/repos/EvoDocker/logs/evodocker_try1/version_7/checkpoints/epoch=0-step=100.ckpt"
    # ckpt_path = "/Users/benshor/Documents/Data/202401_pred_affinity/repos/EvoDocker/logs/evodocker_try1/version_22/checkpoints/epoch=0-step=100.ckpt"

    config_preset = "initial_training"
    skip_relaxation = True
    save_outputs = False
    device_name = "cpu"

    config = model_config(config_preset, long_sequence_inference=False)

    data_processor = data_pipeline.DataPipeline()

    random_seed = 43
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    feature_dicts = {}
    model_generator = load_models_from_command_line(
        config,
        model_device=device_name,
        openfold_checkpoint_path=ckpt_path,
        output_dir=output_dir)

    for model, output_directory in model_generator:
        for input_filename in list_files_with_extensions(input_dir, ".json"):
            tag = input_filename.split(".")[0]
            output_name = f"{tag}_predicted"

            input_path = os.path.join(input_dir, input_filename)
            input_data = json.load(open(input_path, "r"))

            input_structure = data_processor.process_pdb(pdb_path=input_data["input_structure"])

            input_feats = feature_processor.process_features(input_structure, "predict")
            processed_feature_dict = {**input_feats, "input_pseudo_beta": input_feats["pseudo_beta"]}
            processed_feature_dict = {
                k: torch.as_tensor(v, device=device_name)
                for k, v in processed_feature_dict.items()
            }
            # turn into a batch of size 1
            processed_feature_dict = {key: value.unsqueeze(0) for key, value in processed_feature_dict.items()}

            out = run_model(model, processed_feature_dict, tag, output_dir)

            # Toss out the recycling dimensions --- we don't need them anymore
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()),
                processed_feature_dict
            )
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            squeezed_feats = {k: v[0] for k, v in processed_feature_dict.items()}
            squeezed_out = {
                "final_atom_mask": out["final_atom_mask"][0],
                "final_atom_positions": out["final_atom_positions"][0],
                "plddt": out["plddt"][0],
            }
            # squeezed_out = {k: v[0] for k, v in out.items()}
            # for k, v in out.items():
            #     print("doing ", k)
            #     try:
            #         print(v.shape)
            #     except:
            #         print(f"failed for {k} with {v}")

            unrelaxed_protein = prep_output(squeezed_out, squeezed_feats)

            unrelaxed_file_suffix = "_unrelaxed.pdb"
            unrelaxed_output_path = os.path.join(
                output_directory, f'{output_name}{unrelaxed_file_suffix}'
            )

            with open(unrelaxed_output_path, 'w') as fp:
                fp.write(protein.to_pdb(unrelaxed_protein))

            logger.info(f"Output written to {unrelaxed_output_path}...")

            if not skip_relaxation:
                # Relax the prediction.
                logger.info(f"Running relaxation on {unrelaxed_output_path}...")
                relax_protein(config, device_name, unrelaxed_protein, output_directory, output_name)

            if save_outputs:
                output_dict_path = os.path.join(
                    output_directory, f'{output_name}_output_dict.pkl'
                )
                with open(output_dict_path, "wb") as fp:
                    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info(f"Model output written to {output_dict_path}...")


if __name__ == "__main__":
    manual_main()
