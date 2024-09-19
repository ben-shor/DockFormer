import sys

from env_consts import SCREEN_INPUT_DIR, SCREEN_OUTPUT_DIR, CKPT_PATH
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

from openfold.config import model_config
from openfold.utils.script_utils import (load_models_from_command_line, run_model, save_output_structure,
                                         get_latest_checkpoint)
from openfold.utils.tensor_utils import tensor_tree_map


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def override_config(base_config, overriding_config):
    for k, v in overriding_config.items():
        if isinstance(v, dict):
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def run_on_folder(input_dir: str, output_dir: str, run_config_path: str, long_sequence_inference=False):
    config_preset = "initial_training"
    save_outputs = False
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    run_config = json.load(open(run_config_path))

    ckpt_path = CKPT_PATH
    if ckpt_path is None:
        ckpt_path = get_latest_checkpoint(os.path.join(run_config["train_output_dir"], "checkpoint"))
    print("Using checkpoint: ", ckpt_path)

    config = model_config(config_preset, long_sequence_inference=long_sequence_inference)

    # if "globals" not in run_config["override_conf"]:
    #     run_config["override_conf"]["globals"] = {}
    # run_config["override_conf"]["globals"]["only_affinity"] = True
    if "data" not in run_config["override_conf"]:
        run_config["override_conf"]["data"] = {}
    if "common" not in run_config["override_conf"]["data"]:
        run_config["override_conf"]["data"]["common"] = {}
    run_config["override_conf"]["data"]["common"]["max_recycling_iters"] = 0

    config = override_config(config, run_config.get("override_conf", {}))

    model_generator = load_models_from_command_line(
        config,
        model_device=device_name,
        openfold_checkpoint_path=ckpt_path,
        output_dir=output_dir)
    print("Model loaded")
    model, output_directory = next(model_generator)

    output_file = open(os.path.join(output_dir, "screening_results.csv"), "w")
    output_file.write("protein_name,lig_ind,label,affinity_2d_sum,affinity_2d_max,affinity_cls_sum,affinity_cls_max\n")

    dataset = OpenFoldSingleDataset(data_dir=input_dir, config=config.data, mode="predict")
    for i, processed_feature_dict in enumerate(dataset):
        input_name = dataset.get_metadata_for_idx(i)["input_name"]
        gene_name, ind, label = input_name.split("_")

        # turn into a batch of size 1
        processed_feature_dict = {key: value.unsqueeze(0).to(device_name)
                                  for key, value in processed_feature_dict.items()}

        out = run_model(model, processed_feature_dict, input_name, output_dir)
        # print(out)
        # print(out.keys())
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        affinity_bins = torch.linspace(0, 15, 32)
        affinity_2d = torch.softmax(torch.tensor(out["affinity_2d_logits"]), -1)
        affinity_cls = torch.softmax(torch.tensor(out["affinity_cls_logits"]), -1)

        affinity_2d_sum = torch.sum(affinity_2d * affinity_bins, dim=-1).item()
        affinity_2d_max = affinity_bins[torch.argmax(affinity_2d)].item()
        affinity_cls_sum = torch.sum(affinity_cls * affinity_bins, dim=-1).item()
        affinity_cls_max = affinity_bins[torch.argmax(affinity_cls)].item()

        # round to 3 decimal places
        # affinity_2d_sum = round(affinity_2d_sum, 3)
        # affinity_2d_max = round(affinity_2d_max, 3)
        # affinity_cls_sum = round(affinity_cls_sum, 3)
        # affinity_cls_max = round(affinity_cls_max, 3)

        output_file.write(f"{gene_name},{ind},{label},"
                          f"{affinity_2d_sum},{affinity_2d_max},{affinity_cls_sum},{affinity_cls_max}\n")
        output_file.flush()
    output_file.close()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run_config.json")
    input_dir, output_dir = SCREEN_INPUT_DIR, SCREEN_OUTPUT_DIR
    options = {"long_sequence_inference": False}
    if len(sys.argv) > 3:
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        if "--long" in sys.argv:
            options["long_sequence_inference"] = True

    run_on_folder(input_dir, output_dir, config_path, **options)
