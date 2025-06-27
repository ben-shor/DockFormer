"""
Before running this script, run AF2 on all sequences in the sequences folder and save all output pdbss filed in
folder named manual_af_results.
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
SEQUENCES_FOR_AF_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "sequences")
MANUAL_AF_RESULTS_PATH = os.path.join(BASE_OUTPUT_FOLDER, "manual_af_results")


def main():
    all_descriptors = get_all_descriptors(NAME_INDEX_PATH, DATA_INDEX_PATH)

    all_manual_af_files = [i for i in os.listdir(MANUAL_AF_RESULTS_PATH) if i.endswith(".pdb")]

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

        try:
            manual_af_files = [filename for filename in all_manual_af_files if filename.startswith(desc.pdb_id + "_")]
            assert manual_af_files, f"No manual AF files found"

            manual_af_path = os.path.join(MANUAL_AF_RESULTS_PATH, manual_af_files[0])

            success = generate_af_input_gt_in_af(gt_pdb_path, manual_af_path, apo_pdb_path)
            assert success, f"Failed to generate apo from AF"
            status_dict["success_manual_af"].append(desc.pdb_id)
            update_metadata(metadata_path, {"apo_source": "af_manual"})
            print("Success manual AF")
        except Exception as e:
            print("Error with descriptor", desc.pdb_id, e)
            status_dict["error_" + str(e)].append(desc.pdb_id)
            # if os.path.exists(output_folder):
            #     shutil.rmtree(output_folder)
            continue
    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
