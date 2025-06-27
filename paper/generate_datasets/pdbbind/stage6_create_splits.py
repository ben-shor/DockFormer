"""
Before running this script, make sure you completed stage5 and prepare CASF dataset
downlaod the casf dataset from https://www.pdbbind-plus.org.cn/casf and crop the CASF-2106 folder
into DockFormer/data/casf2016/raw/
"""
import os
import shutil
from collections import defaultdict
import json
from Bio import pairwise2
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from utils import get_all_descriptors, get_all_sequences_from_pdb, get_sequence_from_pdb

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

PDBBIND_PATH = os.path.join(DATA_DIR, "pdbbind", "raw")
DATA_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_data.2020")
NAME_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_name.2020")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "pdbbind", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "jsons")
CROP_SIZE = 256

CASF2016_PATH = os.path.join(DATA_DIR, "casf2016", "raw", "CASF-2016")


def get_all_casf_seqs(casf_folder):
    input_data = os.path.join(casf_folder, "power_scoring", "CoreSet.dat")
    lines = open(input_data).readlines()[1:]
    all_chains = set()
    for i in range(len(lines)):
        pdb_id = lines[i].split()[0]
        pdb_path = os.path.join(casf_folder, "coreset", pdb_id, f"{pdb_id}_protein.pdb")
        seqs = get_all_sequences_from_pdb(pdb_path)
        # print(len(seqs), seqs)
        for chain, seq in seqs.items():
            if len(seq) > 20:
                all_chains.add(seq)
            else:
                print("Too short:", pdb_id, chain, seq)
    print("Total chains:", len(all_chains), len(set(all_chains)))
    return list(all_chains)


def get_all_casf_ligands(casf_folder):
    input_data = os.path.join(casf_folder, "power_scoring", "CoreSet.dat")
    lines = open(input_data).readlines()[1:]
    all_smiles = []
    for i in range(len(lines)):
        pdb_id = lines[i].split()[0]
        lig_path = os.path.join(casf_folder, "coreset", pdb_id, f"{pdb_id}_ligand.mol2")
        ligand = Chem.MolFromMol2File(lig_path)
        all_smiles.append(Chem.MolToSmiles(ligand))
    print("Total ligands:", len(all_smiles), len(set(all_smiles)))
    return list(set(all_smiles))


def get_ident_percent(seq1: str, seq2: str):
    if min(len(seq1), len(seq2)) / max(len(seq1), len(seq2)) < 0.3:
        return 0
    a = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    return a.score / (a.end - a.start)


def tanimoto_ecfp4(smiles1, smiles2):
    """
    Calculate the Tanimoto coefficient between two molecules using ECFP4 fingerprints.

    Parameters:
        smiles1 (str): SMILES representation of the first molecule.
        smiles2 (str): SMILES representation of the second molecule.

    Returns:
        float: Tanimoto similarity between the two fingerprints.
    """
    # Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES input")

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp1 = mfpgen.GetFingerprint(mol1)
    fp2 = mfpgen.GetFingerprint(mol2)

    # Compute Tanimoto similarity
    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

    return tanimoto


def is_similar_to_casf(protein_pdb_path: str, ligand_sdf_path: str, all_seqs: list, all_ligands: list):
    ligand = Chem.MolFromMolFile(ligand_sdf_path)

    similar_ligand = False
    for casf_ligand in all_ligands:
        if tanimoto_ecfp4(Chem.MolToSmiles(ligand), casf_ligand) > 0.9:
            similar_ligand = True
            break
    if not similar_ligand:
        print(f"No ligand match")
        return False

    joined_sequence = get_sequence_from_pdb(protein_pdb_path, chain_id="A")
    pdb_seqs = [i for i in joined_sequence.split("X * 32") if len(i) > 20]

    for casf_chain in all_seqs:
        for seq in pdb_seqs:
            if get_ident_percent(casf_chain, seq) > 0.9:
                print("Found chain and ligand similar to casf")
                return True
    print("No chain match")
    return False


def save_list_as_clusters(paths_to_write: list, output_path: str):
    data_to_write = {f"c{i:05}": [p] for i, p in enumerate(paths_to_write)}
    json.dump(data_to_write, open(output_path, "w"))


def main():
    all_descriptors = get_all_descriptors(NAME_INDEX_PATH, DATA_INDEX_PATH)
    paths_to_save = {"train": [], "validation": []}

    all_casf_seqs = get_all_casf_seqs(CASF2016_PATH)
    all_casf_ligands = get_all_casf_ligands(CASF2016_PATH)

    print("loaded CASF sequences and ligands, total CASF sequences:", len(all_casf_seqs),
          "total CASF ligands:", len(all_casf_ligands))

    os.makedirs(JSONS_FOLDER, exist_ok=True)

    status_dict = defaultdict(list)
    for desc_ind, desc in enumerate(all_descriptors):
        output_folder = os.path.join(MODELS_FOLDER, desc.pdb_id)
        if not os.path.exists(output_folder):
            continue
        print("processing", desc.pdb_id, desc_ind + 1, "/", len(all_descriptors), flush=True)

        gt_ligand_path = os.path.join(output_folder, f"gt_ligand.sdf")
        gt_pdb_path = os.path.join(output_folder, f"gt_protein.pdb")

        try:
            similar_to_casf = is_similar_to_casf(gt_pdb_path, gt_ligand_path, all_casf_seqs, all_casf_ligands)
            split = "train" if not similar_to_casf else "validation"

            json_path = os.path.join(JSONS_FOLDER, f"{desc.pdb_id}.json")
            base_relative_path = os.path.join("models", desc.pdb_id)
            json_data = {
                "input_structure": os.path.join(base_relative_path, f"apo_protein_cropped_{CROP_SIZE}.pdb"),
                "gt_structure": os.path.join(base_relative_path, f"gt_protein_cropped_{CROP_SIZE}.pdb"),
                "gt_sdf": os.path.join(base_relative_path, f"gt_ligand.sdf"),
                "ref_sdf": os.path.join(base_relative_path, f"ref_ligand.sdf"),
                # "input_smiles": input_smiles,
                # "pocket_res_ids": pocket_residues,
                "resolution": desc.resolution,
                "release_year": desc.release_year,
                "affinity": desc.affinity,
                # "protein_seq": seq,
                # "protein_seq_len": seq_len,
                "uniprot": desc.uniprot,
                # "ligand_num_atoms": ligand_num_atoms,
            }
            open(json_path, "w").write(json.dumps(json_data, indent=4))

            paths_to_save[split].append(f"jsons/{desc.pdb_id}.json")
            status_dict[split].append(desc.pdb_id)
        except Exception as e:
            print("Error with descriptor", desc.pdb_id, e)
            status_dict["error_" + str(e)].append(desc.pdb_id)
            continue

    save_list_as_clusters(paths_to_save["train"], os.path.join(BASE_OUTPUT_FOLDER, "train.json"))
    save_list_as_clusters(paths_to_save["validation"], os.path.join(BASE_OUTPUT_FOLDER, "validation.json"))
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
