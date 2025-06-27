"""
Before running this script, make sure you completed stage3
"""
import os
import shutil
from collections import defaultdict

from utils import get_all_descriptors, update_metadata, generate_af_input_gt_in_af

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

PDBBIND_PATH = os.path.join(DATA_DIR, "pdbbind", "raw")
DATA_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_data.2020")
NAME_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_name.2020")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "pdbbind", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")


def main():
    all_descriptors = get_all_descriptors(NAME_INDEX_PATH, DATA_INDEX_PATH)

    status_dict = defaultdict(list)
    for desc_ind, desc in enumerate(all_descriptors):
        output_folder = os.path.join(MODELS_FOLDER, desc.pdb_id)
        if not os.path.exists(output_folder):
            continue
        print("processing", desc.pdb_id, desc_ind + 1, "/", len(all_descriptors), flush=True)

        gt_pdb_path = os.path.join(output_folder, f"gt_protein.pdb")
        apo_pdb_path = os.path.join(output_folder, f"apo_protein.pdb")
        metadata_path = os.path.join(output_folder, "metadata.json")
        if os.path.exists(apo_pdb_path):
            print("Already processed apo", desc.pdb_id)
            status_dict["already_processed_apo"].append(desc.pdb_id)
            continue
        shutil.copy(gt_pdb_path, apo_pdb_path)
        update_metadata(metadata_path, {"apo_source": "holo"})
        status_dict["success_holo_as_apo"].append(desc.pdb_id)

    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
