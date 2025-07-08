"""
Before running this script, make sure you completed stage1
"""
import os
import shutil
from collections import defaultdict
from typing import Optional
import requests

from utils import update_metadata, generate_af_input_gt_in_af, get_sequence_from_pdb, get_all_sequences_from_pdb

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

RAW_PATH = os.path.join(DATA_DIR, "posebusters", "raw")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "posebusters", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
SEQUENCES_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "sequences")


def main():
    os.makedirs(SEQUENCES_FOLDER, exist_ok=True)

    status_dict = defaultdict(list)
    for folder_name in os.listdir(RAW_PATH):
        output_folder = os.path.join(MODELS_FOLDER, folder_name)
        if not os.path.isdir(output_folder):
            continue

        gt_pdb_full_mc_path = os.path.join(output_folder, f"gt_protein_full_mc.pdb")  # all chains, multi chain
        gt_pdb_pocket_mc_path = os.path.join(output_folder, f"gt_protein_pocket_mc.pdb")  # pocket chains, single chain
        gt_pdb_full_sc_path = os.path.join(output_folder, f"gt_protein_full_sc.pdb")  # all chains, multi chain
        gt_pdb_pocket_sc_path = os.path.join(output_folder, f"gt_protein_pocket_sc.pdb")  # pocket chains, multi chain
        sequence_path = os.path.join(SEQUENCES_FOLDER, f"{folder_name}.fasta")

        metadata_path = os.path.join(output_folder, "metadata.json")
        update_metadata(metadata_path, {"resolution": -1,
                                        "affinity": -1,
                                        "release_year": -1,
                                        "uniprot": ""})

        # if os.path.exists(sequence_path):
        #     print("Already processed sequence", folder_name)
        #     status_dict["already_processed_sequence"].append(folder_name)
        #     continue

        try:
            sequences = get_all_sequences_from_pdb(gt_pdb_full_mc_path)
            all_seq_len = sum(len(seq) for seq in sequences.values())
            if all_seq_len > 1800:
                sequences = get_all_sequences_from_pdb(gt_pdb_pocket_mc_path)
                status_dict["seq_created_short"].append(folder_name)
            else:
                status_dict["seq_created_long"].append(folder_name)

            with open(sequence_path, "w") as f:
                seq = ":".join(sequences.values())
                f.write(f">{folder_name}\n{seq}\n")

        except Exception as e:
            print("Error with descriptor", folder_name, e)
            status_dict["error_" + str(e)].append(desc.pdb_id)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise
            continue
    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
