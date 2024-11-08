import json
import os
import sys
from typing import Optional

from dockformer.config import model_config
from dockformer.data.data_modules import OpenFoldSingleDataset


def override_config(base_config, overriding_config):
    for k, v in overriding_config.items():
        if isinstance(v, dict):
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def check_folder(jsons_folder_path: str, failed_jsons_folder: str, run_config_path: Optional[str] = None):
    config = model_config("initial_training", long_sequence_inference=False)
    os.makedirs(failed_jsons_folder, exist_ok=True)
    if run_config_path is not None:
        run_config = json.load(open(run_config_path))
        config = override_config(config, run_config.get("override_conf", {}))

    dataset = OpenFoldSingleDataset(data_dir=jsons_folder_path, config=config.data, mode="train")

    all_samples, good_samples = [], []

    for i in range(len(dataset)):
        input_path = dataset.get_metadata_for_idx(i)["input_path"]

        try:
            print("checking", input_path)
            all_samples.append(input_path)
            processed_feature_dict = dataset[i]
            good_samples.append(input_path)
        except Exception as e:
            print(f"ERROR in load: {input_path}, error: {e}")
            moved_json_path = os.path.join(failed_jsons_folder, os.path.basename(input_path))
            os.rename(input_path, moved_json_path)


    print(f"Total samples: {len(dataset)}", f"Good samples: {len(good_samples)}")


if __name__ == '__main__':
    # check_folder("/Users/benshor/Documents/Data/202401_pred_affinity/plinder/v2/processed_tests/plinder_jsons",
    #              "/Users/benshor/Documents/Data/202401_pred_affinity/plinder/v2/processed_tests/plinder_jsons_bad")
    check_folder(sys.argv[1], sys.argv[2], sys.argv[3])