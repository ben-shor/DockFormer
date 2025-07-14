"""
Before running this script, run AF2 on all sequences in the sequences folder and save all output pdbss filed in
folder named manual_af_results.
"""
import os
import shutil
from collections import defaultdict

from utils import generate_af_input_gt_in_af, merge_to_single_chain

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

RAW_PATH = os.path.join(DATA_DIR, "posebusters", "raw")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "posebusters", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
SEQUENCES_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "sequences")
MANUAL_AF_RESULTS_PATH = os.path.join(BASE_OUTPUT_FOLDER, "manual_af_results")


def main():
    os.makedirs(SEQUENCES_FOLDER, exist_ok=True)

    all_manual_af_files = [i for i in os.listdir(MANUAL_AF_RESULTS_PATH) if i.endswith(".pdb")]

    status_dict = defaultdict(list)
    for folder_name in os.listdir(RAW_PATH):
        output_folder = os.path.join(MODELS_FOLDER, folder_name)
        if not os.path.isdir(output_folder):
            continue

        # if folder_name not in ('8F8E_XJI', '7XRL_FWK', '7MS7_ZQ1', "7SUC_COM", ):
        #     continue

        print("processing", folder_name, flush=True)

        gt_pdb_full_sc_path = os.path.join(output_folder, f"gt_protein_full_mc.pdb")  # all chains, multi chain
        gt_pdb_pocket_sc_path = os.path.join(output_folder, f"gt_protein_pocket_mc.pdb")  # pocket chains, multi chain
        apo_pdb_full_sc_path = os.path.join(output_folder, f"apo_protein_full.pdb")  # all chains, multi chain
        apo_pdb_pocket_sc_path = os.path.join(output_folder, f"apo_protein_pocket.pdb")  # pocket chains, multi chain

        manual_af_files = [filename for filename in all_manual_af_files if filename.startswith(folder_name + "_")]
        if not manual_af_files:
            print("No manual AF files found for", folder_name)
            status_dict["no_manual_af_files"].append(folder_name)
            continue

        if len(manual_af_files) > 1:
            print("Multiple manual AF files found for", folder_name, manual_af_files)

        af_path = os.path.join(MANUAL_AF_RESULTS_PATH, manual_af_files[0])
        # tmp_af_path = os.path.join(output_folder, "tmp_af.pdb")
        # if "alphafold2_multimer" in manual_af_files[0]:
        #     merge_to_single_chain(af_path, tmp_af_path)
        #     af_path = tmp_af_path
        #     print("using merged AF file for", folder_name, "from", manual_af_files[0])

        success_full = generate_af_input_gt_in_af(gt_pdb_full_sc_path, af_path, apo_pdb_full_sc_path)
        success_pocket = generate_af_input_gt_in_af(gt_pdb_pocket_sc_path, af_path, apo_pdb_pocket_sc_path)

        if not success_full and not success_pocket:
            print("Failed to generate apo from AF for", folder_name)
            status_dict["failed_both"].append(folder_name)
        elif not success_full:
            print("Failed to generate apo full from AF for", folder_name)
            status_dict["failed_full"].append(folder_name)
        elif not success_pocket:
            print("Failed to generate apo pocket from AF for", folder_name)
            status_dict["failed_pocket"].append(folder_name)
        else:
            print("Successfully generated apo from AF for", folder_name)
            status_dict["success"].append(folder_name)

    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
