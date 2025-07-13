"""
Before running this script, make sure you finished stage3.
"""
import os
from collections import defaultdict
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

RAW_PATH = os.path.join(DATA_DIR, "posebusters", "raw")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "posebusters", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")

JSONS_FULL_PATH = os.path.join(BASE_OUTPUT_FOLDER, "jsons_full")
JSONS_POCKET_PATH = os.path.join(BASE_OUTPUT_FOLDER, "jsons_pocket")
JSONS_HOLO_POCKET_PATH = os.path.join(BASE_OUTPUT_FOLDER, "jsons_holo_pocket")


def main():
    os.makedirs(JSONS_FULL_PATH, exist_ok=True)
    os.makedirs(JSONS_POCKET_PATH, exist_ok=True)
    os.makedirs(JSONS_HOLO_POCKET_PATH, exist_ok=True)

    paths_to_save = {"full": [], "pocket": [], "holo_pocket": []}

    status_dict = defaultdict(list)
    for folder_name in os.listdir(RAW_PATH):
        output_folder = os.path.join(MODELS_FOLDER, folder_name)
        if not os.path.isdir(output_folder):
            continue

        print("processing", folder_name)

        models_folder_name = os.path.basename(MODELS_FOLDER)
        base_relative_path = os.path.join(models_folder_name, folder_name)
        base_json_data = {
            "input_structure": None,
            "gt_structure": None,
            "gt_sdf": os.path.join(base_relative_path, f"gt_ligand.sdf"),
            "ref_sdf": os.path.join(base_relative_path, f"ref_ligand.sdf"),
            "resolution": -1,
            "release_year": -1,
            "affinity": -1,
            "uniprot": -1,
            "pdb_id": folder_name,
        }

        # json holo pocket
        json_holo_pocket_path = os.path.join(JSONS_HOLO_POCKET_PATH, f"{folder_name}.json")
        json_data = {**base_json_data,
                     # "input_structure": os.path.join(base_relative_path, f"gt_protein_pocket_sc.pdb"),
                     # "gt_structure": os.path.join(base_relative_path, f"gt_protein_pocket_sc.pdb"),
                     "input_structure": os.path.join(base_relative_path, f"gt_protein_pocket_mc.pdb"),
                     "gt_structure": os.path.join(base_relative_path, f"gt_protein_pocket_mc.pdb"),
                     }
        open(json_holo_pocket_path, "w").write(json.dumps(json_data, indent=4))
        paths_to_save["holo_pocket"].append(f"{os.path.basename(JSONS_HOLO_POCKET_PATH)}/{folder_name}.json")

        # json pocket
        pocket_exists = os.path.exists(os.path.join(MODELS_FOLDER, folder_name, "apo_protein_pocket.pdb"))

        if pocket_exists:
            json_pocket_path = os.path.join(JSONS_POCKET_PATH, f"{folder_name}.json")
            json_data = {**base_json_data,
                         "input_structure": os.path.join(base_relative_path, f"apo_protein_pocket.pdb"),
                         # "gt_structure": os.path.join(base_relative_path, f"gt_protein_pocket_sc.pdb"),
                         "gt_structure": os.path.join(base_relative_path, f"gt_protein_pocket_mc.pdb"),
                         }
            open(json_pocket_path, "w").write(json.dumps(json_data, indent=4))
            paths_to_save["pocket"].append(f"{os.path.basename(JSONS_POCKET_PATH)}/{folder_name}.json")
        else:
            status_dict["no_pocket"].append(folder_name)
            continue

        # json full
        if os.path.exists(os.path.join(MODELS_FOLDER, folder_name, "apo_protein_full.pdb")):
            json_full_path = os.path.join(JSONS_FULL_PATH, f"{folder_name}.json")
            json_data = {**base_json_data,
                         "input_structure": os.path.join(base_relative_path, f"apo_protein_full.pdb"),
                         # "gt_structure": os.path.join(base_relative_path, f"gt_protein_full_sc.pdb"),
                         "gt_structure": os.path.join(base_relative_path, f"gt_protein_full_mc.pdb"),
                         }
            open(json_full_path, "w").write(json.dumps(json_data, indent=4))
            paths_to_save["full"].append(f"{os.path.basename(JSONS_FULL_PATH)}/{folder_name}.json")
            print("using full structure")
            status_dict["full"].append(folder_name)
        else:
            print("using pocket instead of full")
            paths_to_save["full"].append(f"{os.path.basename(JSONS_POCKET_PATH)}/{folder_name}.json")
            status_dict["only_pocket"].append(folder_name)

    # json.dump(paths_to_save["full"], open(os.path.join(BASE_OUTPUT_FOLDER, "full.json"), "w"), indent=4)
    # json.dump(paths_to_save["pocket"], open(os.path.join(BASE_OUTPUT_FOLDER, "pocket.json"), "w"), indent=4)
    # json.dump(paths_to_save["holo_pocket"], open(os.path.join(BASE_OUTPUT_FOLDER, "holo_pocket.json"), "w"), indent=4)

    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
