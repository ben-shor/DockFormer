"""
Before running this script, make sure you completed stage1
"""
import os
import shutil
from collections import defaultdict
from typing import Optional

import Bio.PDB
import Bio.SeqIO
import Bio.SeqUtils
import requests

from utils import get_all_descriptors, get_pdb_model, get_pdb_model_readonly

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


def get_chain_object_to_seq(chain: Bio.PDB.Chain.Chain) -> str:
    res_id_to_res = {res.get_id()[1]: res for res in chain.get_residues() if "CA" in res}

    if len(res_id_to_res) == 0:
        print("skipping empty chain", chain.get_id())
        return ""
    seq = ""
    for i in range(1, max(res_id_to_res) + 1):
        if i in res_id_to_res:
            seq += Bio.SeqUtils.seq1(res_id_to_res[i].get_resname())
        else:
            seq += "X"
    return seq


def get_sequence_from_pdb(pdb_path: str, chain_id: str = "A") -> str:
    model = get_pdb_model_readonly(pdb_path)
    return get_chain_object_to_seq(model[chain_id])


def generate_af_input_gt_in_af(gt_path: str, af_path: str, output_path: str):
    gt_model = get_pdb_model(gt_path)
    af_model = get_pdb_model(af_path)

    gt_chain = gt_model["A"]
    af_chain = af_model["A"]

    gt_seq = get_chain_object_to_seq(gt_chain)
    af_seq = get_chain_object_to_seq(af_chain)

    largest_segment = max(gt_seq.split("X"), key=len)

    i = -1
    all_possible_af_offsets = []
    while True:
        try:
            i = af_seq.index(largest_segment, i + 1)
        except ValueError:
            break
        all_possible_af_offsets.append(i)

    if len(all_possible_af_offsets) == 0:
        print("Can't find offset")
        return False

    gt_offset = gt_seq.index(largest_segment)
    af_offset = None

    for possible_af_offset in all_possible_af_offsets:
        offset = possible_af_offset - gt_offset

        if len(gt_seq.strip("X")) + offset > len(af_seq):
            continue
        if offset < 0:
            continue

        error_in_seq = False
        for i in range(len(gt_seq)):
            if gt_seq[i] == "X":
                continue
            if af_seq[i + offset] == "X":
                error_in_seq = True
                break
            if gt_seq[i] != af_seq[i + offset]:
                error_in_seq = True
                break
        if error_in_seq:
            continue
        af_offset = possible_af_offset
        break

    if af_offset is None:
        print("Can't find offset")
        return False

    offset = af_offset - gt_offset

    af_res_by_res_id = {res.get_id()[1]: res for res in af_chain.get_residues() if "CA" in res}
    new_chain = Bio.PDB.Chain.Chain("A")
    for gt_res in gt_chain.get_residues():
        res = af_res_by_res_id[gt_res.id[1] + offset]
        af_chain.detach_child(res.id)
        res.id = (" ", gt_res.id[1], " ")
        # print(gt_res.get_resname(), res.get_resname(), res.id)
        assert res.get_resname() == gt_res.get_resname(), \
            f"Residue mismatch {gt_res.get_resname()} {res.get_resname()}"
        new_chain.add(res)

    io = Bio.PDB.PDBIO()
    io.set_structure(new_chain)
    io.save(output_path)

    return True



def main():
    all_descriptors = get_all_descriptors(NAME_INDEX_PATH, DATA_INDEX_PATH)

    os.makedirs(AF_DB_DOWNLOADED_PATH, exist_ok=True)
    os.makedirs(SEQUENCES_FOR_AF_FOLDER, exist_ok=True)

    status_dict = defaultdict(list)
    for desc_ind, desc in enumerate(all_descriptors):
        output_folder = os.path.join(MODELS_FOLDER, desc.pdb_id)
        if not os.path.exists(output_folder):
            continue
        print("processing", desc.pdb_id, desc_ind + 1, "/", len(all_descriptors))

        gt_pdb_path = os.path.join(output_folder, f"gt_protein.pdb")
        apo_pdb_path = os.path.join(output_folder, f"apo_protein.pdb")
        sequence_path = os.path.join(SEQUENCES_FOR_AF_FOLDER, f"{desc.pdb_id}.fasta")

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
