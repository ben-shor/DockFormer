"""
Before running this script, make sure you completed stage1
"""
import os
import shutil
from collections import defaultdict
from typing import Optional
import requests

from utils import get_all_descriptors, update_metadata, generate_af_input_gt_in_af, get_sequence_from_pdb

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

PDBBIND_PATH = os.path.join(DATA_DIR, "pdbbind", "raw")
DATA_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_data.2020")
NAME_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_name.2020")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "pdbbind", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
SEQUENCES_FOR_AF_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "sequences")
AF_DB_DOWNLOADED_PATH = os.path.join(BASE_OUTPUT_FOLDER, "af_db_downloaded")


def get_af_db_path(uniprot: str) -> Optional[str]:
    af_path = os.path.join(AF_DB_DOWNLOADED_PATH, f"{uniprot}.pdb")
    if os.path.exists(af_path):
        return af_path

    print("Downloading from AF-DB", uniprot)
    url_to_download = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb"
    result = requests.get(url_to_download)
    if result.ok:
        open(af_path, "wb").write(result.content)
    else:
        print("Could not to download", uniprot)
        return None
    return af_path


def get_esm_structure(sequence: str) -> Optional[str]:
    if len(sequence) > 400:
        print("Sequence too long for ESMFold", len(sequence))
        return None

    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    try:
        response = requests.post(url, data=sequence)
        response.raise_for_status()
        return response.text
    except:
        return None


def main():
    all_descriptors = get_all_descriptors(NAME_INDEX_PATH, DATA_INDEX_PATH)

    os.makedirs(AF_DB_DOWNLOADED_PATH, exist_ok=True)
    os.makedirs(SEQUENCES_FOR_AF_FOLDER, exist_ok=True)

    status_dict = defaultdict(list)
    for desc_ind, desc in enumerate(all_descriptors):
        output_folder = os.path.join(MODELS_FOLDER, desc.pdb_id)
        if not os.path.exists(output_folder):
            continue
        print("processing", desc.pdb_id, desc_ind + 1, "/", len(all_descriptors), flush=True)

        gt_pdb_path = os.path.join(output_folder, f"gt_protein.pdb")
        apo_pdb_path = os.path.join(output_folder, f"apo_protein.pdb")
        sequence_path = os.path.join(SEQUENCES_FOR_AF_FOLDER, f"{desc.pdb_id}.fasta")

        metadata_path = os.path.join(output_folder, "metadata.json")
        update_metadata(metadata_path, {"resolution": desc.resolution,
                                        "affinity": desc.affinity,
                                        "release_year": desc.release_year,
                                        "uniprot": desc.uniprot})

        if os.path.exists(apo_pdb_path):
            print("Already processed", desc.pdb_id)
            status_dict["already_processed_apo"].append(desc.pdb_id)
            continue
        if os.path.exists(sequence_path):
            print("Already processed sequence", desc.pdb_id)
            status_dict["already_processed_sequence"].append(desc.pdb_id)
            continue

        try:
            af_path = get_af_db_path(desc.uniprot)
            if af_path is not None:
                success = generate_af_input_gt_in_af(gt_pdb_path, af_path, apo_pdb_path)
                if success:
                    print("Success AF_DB", desc.pdb_id)
                    update_metadata(metadata_path, {"apo_source": "af_db"})
                    status_dict["success_af"].append(desc.pdb_id)
                    continue

            sequence = get_sequence_from_pdb(gt_pdb_path, "A")
            sequence = sequence.replace("X" * 50, "X" * 32)
            print("searching sequence", sequence)

            esm_pdb = get_esm_structure(sequence)
            if esm_pdb is not None:
                print("Success ESMFold", desc.pdb_id)
                open(apo_pdb_path, "w").write(esm_pdb)
                assert generate_af_input_gt_in_af(gt_pdb_path, apo_pdb_path, apo_pdb_path), \
                    "Weirdly Failed to generate apo from ESMFold structure"
                update_metadata(metadata_path, {"apo_source": "esmfold"})
                status_dict["success_esm"].append(desc.pdb_id)
                continue

            open(sequence_path, "w").write(f">{desc.pdb_id}\n{sequence}\n")
            status_dict["seq_created"].append(desc.pdb_id)
            print("Sequence saved", desc.pdb_id)

        except Exception as e:
            print("Error with descriptor", desc.pdb_id, e)
            status_dict["error_" + str(e)].append(desc.pdb_id)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            continue
    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
