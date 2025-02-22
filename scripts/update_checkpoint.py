import json
import os
import sys

import torch
import pytorch_lightning as pl

from dockformer.config import model_config
from dockformer.model.model import AlphaFold
from dockformer.utils.script_utils import get_latest_checkpoint
from run_pretrained_model import override_config


def main(run_config_path: str):
    run_config = json.load(open(run_config_path))
    ckpt_path = get_latest_checkpoint(os.path.join(run_config["train_output_dir"], "checkpoint"))
    updated_ckpt_path = ckpt_path.replace(".ckpt", "_updated.ckpt")

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    state_dict = checkpoint['state_dict']
    # old_keys = list(state_dict.keys())
    # for old_name in old_keys:
    #     name = old_name
    #     if "msa_" in old_name:
    #         name = old_name.replace("msa_", "single_")
    #         print(f"Renaming {old_name} to {name}")
    #     state_dict[name] = state_dict.pop(old_name)
    #
    #     ema_old_name, ema_new_name = old_name[6:], name[6:]  # remove "model." prefix
    #     checkpoint["ema"]["params"][ema_new_name] = checkpoint["ema"]["params"].pop(ema_old_name)

    # # Load the modified state dictionary into the model
    config = model_config(run_config["stage"])
    config = override_config(config, run_config.get("override_conf", {}))
    model = AlphaFold(config)
    model.load_state_dict(state_dict, strict=False)

    # # If needed, re-save the checkpoint with the new names
    torch.save({'state_dict': model.state_dict()}, updated_ckpt_path)
    # torch.save(checkpoint, updated_ckpt_path)

    print("Updated checkpoint saved to", updated_ckpt_path)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "run_config.json")
    main(config_path)
