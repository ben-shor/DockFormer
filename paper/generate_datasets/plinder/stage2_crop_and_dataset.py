"""
Before running this script, make sure you completed stage4, and set CROP_SIZE to the desired value.
"""
import json
import os
from collections import defaultdict
from typing import List

import numpy as np
import Bio.PDB

from rdkit import Chem

from utils import get_pdb_model

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "plinder", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
TRAIN_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "jsons_train")
VALIDATION_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "jsons_validation")
TEST_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "jsons_test")
CROP_SIZE = 256

OUTPUT_TRAIN_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, f"jsons_train_cs{CROP_SIZE}")
OUTPUT_VALIDATION_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, f"jsons_validation_cs{CROP_SIZE}")
OUTPUT_TEST_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, f"jsons_test_cs{CROP_SIZE}")

TRAIN_CLUSTERS_DATASET_FILE = os.path.join(BASE_OUTPUT_FOLDER, f"clusters_train_cs{CROP_SIZE}.json")
TRAIN_NO_CLUSTERS_DATASET_FILE = os.path.join(BASE_OUTPUT_FOLDER, f"no_clusters_train_cs{CROP_SIZE}.json")
VALIDATION_CLUSTERS_DATASET_FILE = os.path.join(BASE_OUTPUT_FOLDER, f"clusters_validation_cs{CROP_SIZE}.json")


def get_protein_res_ids_to_keep(pdb_path: str, sdf_paths: List[str], crop_size: int, neighborhood_size: int = 20) -> set:
    protein = Chem.MolFromPDBFile(pdb_path, sanitize=False)

    ligand_pos = None
    total_ligand_atoms = 0
    for sdf_path in sdf_paths:
        ligand = Chem.MolFromMolFile(sdf_path)
        ligand_conf = ligand.GetConformer()
        if ligand_pos is None:
            ligand_pos = ligand_conf.GetPositions()
        else:
            ligand_pos = np.vstack((ligand_pos, ligand_conf.GetPositions()))
        total_ligand_atoms += ligand.GetNumAtoms()

    protein_conf = protein.GetConformer()
    protein_pos = protein_conf.GetPositions()
    protein_atoms = list(protein.GetAtoms())
    assert len(protein_pos) == len(protein_atoms), f"Positions and atoms mismatch in {pdb_path}"

    inter_dists = ligand_pos[:, np.newaxis, :] - protein_pos[np.newaxis, :, :]
    inter_dists = np.sqrt((inter_dists ** 2).sum(-1))
    min_inter_dist_per_protein_atom = inter_dists.min(axis=0)

    # sort protein atom idx by distance to ligand
    protein_idx_by_dist = np.argsort(min_inter_dist_per_protein_atom)

    all_protein_residues = set()
    for atom in protein_atoms:
        res = atom.GetPDBResidueInfo()
        if res.GetIsHeteroAtom():
            continue
        all_protein_residues.add((res.GetChainId(), res.GetResidueNumber()))

    res_to_save_count = crop_size - total_ligand_atoms
    if res_to_save_count >= len(all_protein_residues):
        print("All residues will be kept, no need to crop")
        return all_protein_residues

    res_ids_to_keep = set()
    for idx in protein_idx_by_dist:
        if len(res_ids_to_keep) >= res_to_save_count:
            break
        res = protein_atoms[idx].GetPDBResidueInfo()
        if res.GetIsHeteroAtom():
            continue
        base_res_id = (res.GetChainId(), res.GetResidueNumber())
        for i in range(0, int(neighborhood_size / 2)):
            res_plus_1 = (base_res_id[0], base_res_id[1] + i)
            res_minus_1 = (base_res_id[0], base_res_id[1] - i)
            if res_minus_1 in all_protein_residues:
                res_ids_to_keep.add(res_minus_1)
            if len(res_ids_to_keep) >= res_to_save_count:
                break
            if res_plus_1 in all_protein_residues:
                res_ids_to_keep.add(res_plus_1)
    print("cropped", len(res_ids_to_keep), "residues from", len(all_protein_residues), "total residues",
          total_ligand_atoms)

    return res_ids_to_keep


def crop_residue_ids(pdb_path: str, res_ids_to_keep: set, output_path: str):
    pdb_model = get_pdb_model(pdb_path)

    for chain in pdb_model.get_chains():
        res_to_remove = []
        for res in chain.get_residues():
            if (chain.id, res.id[1]) not in res_ids_to_keep:
                res_to_remove.append(res)
        for res in res_to_remove:
            chain.detach_child(res.id)

    io = Bio.PDB.PDBIO()
    io.set_structure(pdb_model)
    io.save(output_path)


def save_list_as_clusters(paths_to_write: list, output_path: str):
    data_to_write = {f"c{i:05}": [p] for i, p in enumerate(paths_to_write)}
    json.dump(data_to_write, open(output_path, "w"))


def main():
    status_dict = defaultdict(list)

    for jsons_folder, output_jsons_folder, output_file_path, output_no_clusters_path in [
        (TRAIN_JSONS_FOLDER, OUTPUT_TRAIN_JSONS_FOLDER, TRAIN_CLUSTERS_DATASET_FILE, TRAIN_NO_CLUSTERS_DATASET_FILE),
        (VALIDATION_JSONS_FOLDER, OUTPUT_VALIDATION_JSONS_FOLDER, VALIDATION_CLUSTERS_DATASET_FILE, None)
    ]:
        cluster_to_jsons = defaultdict(list)
        os.makedirs(output_jsons_folder, exist_ok=True)
        all_files = os.listdir(jsons_folder)
        for i, filename in enumerate(all_files):
            system_name = filename[:-5]  # remove .json extension
            json_path = os.path.join(jsons_folder, filename)
            models_folder = os.path.join(MODELS_FOLDER, system_name)  # remove .json extension

            if not os.path.exists(json_path) or not os.path.exists(models_folder):
                continue
            print(f"processing {filename} ({i}/{len(all_files)})", flush=True)

            gt_pdb_path = os.path.join(models_folder, f"gt_protein.pdb")
            apo_pdb_path = os.path.join(models_folder, f"input_protein.pdb")
            ligand_paths = [os.path.join(models_folder, f"gt_ligand_{j}.sdf") for j in range(10)
                            if os.path.exists(os.path.join(models_folder, f"gt_ligand_{j}.sdf"))]

            cropped_gt_pdb_path = os.path.join(models_folder, f"gt_protein_cropped_{CROP_SIZE}.pdb")
            cropped_apo_pdb_path = os.path.join(models_folder, f"apo_protein_cropped_{CROP_SIZE}.pdb")

            if not os.path.exists(gt_pdb_path) or not os.path.exists(apo_pdb_path):
                print("Missing gt or apo pdb, skipping", filename)
                status_dict["missing"].append(system_name)
                continue

            # res_ids_to_keep = get_protein_res_ids_to_keep(gt_pdb_path, ligand_paths, CROP_SIZE)
            #
            # crop_residue_ids(gt_pdb_path, res_ids_to_keep, cropped_gt_pdb_path)
            # crop_residue_ids(apo_pdb_path, res_ids_to_keep, cropped_apo_pdb_path)

            loaded_json = json.load(open(json_path, "r"))
            models_folder_name = os.path.basename(MODELS_FOLDER)
            loaded_json["input_structure"] = os.path.join(models_folder_name, system_name, f"apo_protein_cropped_{CROP_SIZE}.pdb")
            loaded_json["gt_structure"] = os.path.join(models_folder_name, system_name, f"gt_protein_cropped_{CROP_SIZE}.pdb")

            json.dump(loaded_json, open(os.path.join(output_jsons_folder, filename), "w"), indent=4)

            jsons_folder_name = os.path.basename(output_jsons_folder)
            cluster_to_jsons[loaded_json["cluster"]].append(os.path.join(jsons_folder_name, filename))

            status_dict["cropped"].append(system_name)
        # Save clusters dataset
        json.dump(cluster_to_jsons, open(output_file_path, "w"), indent=4)

        if output_no_clusters_path:
            data_to_write = {f"c{i:05}": [p] for i, p in enumerate(sum(cluster_to_jsons.values(), []))}
            json.dump(data_to_write, open(output_no_clusters_path, "w"), indent=4)

    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
