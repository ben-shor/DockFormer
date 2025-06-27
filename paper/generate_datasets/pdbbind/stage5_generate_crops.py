"""
Before running this script, make sure you completed stage4, and set CROP_SIZE to the desired value.
"""
import os
from collections import defaultdict
import numpy as np
import Bio.PDB

from rdkit import Chem

from utils import get_all_descriptors, update_metadata, generate_af_input_gt_in_af, get_pdb_model

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

PDBBIND_PATH = os.path.join(DATA_DIR, "pdbbind", "raw")
DATA_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_data.2020")
NAME_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_name.2020")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "pdbbind", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
CROP_SIZE = 256


def get_protein_res_ids_to_keep(pdb_path: str, sdf_path: str, crop_size: int, neighborhood_size: int = 20) -> set:
    protein = Chem.MolFromPDBFile(pdb_path, sanitize=False)
    ligand = Chem.MolFromMolFile(sdf_path)

    protein_conf = protein.GetConformer()
    protein_pos = protein_conf.GetPositions()
    protein_atoms = list(protein.GetAtoms())
    assert len(protein_pos) == len(protein_atoms), f"Positions and atoms mismatch in {pdb_path}"

    ligand_conf = ligand.GetConformer()
    ligand_pos = ligand_conf.GetPositions()

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
        all_protein_residues.add(res.GetResidueNumber())

    res_to_save_count = crop_size - ligand.GetNumAtoms()
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
        base_res_id = res.GetResidueNumber()
        for i in range(0, int(neighborhood_size / 2)):
            if base_res_id - i in all_protein_residues:
                res_ids_to_keep.add(base_res_id - i)
            if len(res_ids_to_keep) >= res_to_save_count:
                break
            if base_res_id + i in all_protein_residues:
                res_ids_to_keep.add(base_res_id + i)
    print("cropped", len(res_ids_to_keep), "residues from", len(all_protein_residues), "total residues",
          ligand.GetNumAtoms())

    return res_ids_to_keep


def crop_residue_ids(pdb_path: str, res_ids_to_keep: set, output_path: str):
    pdb_model = get_pdb_model(pdb_path)

    chain = next(iter(pdb_model.get_chains()))
    res_to_remove = []
    for res in chain.get_residues():
        if res.id[1] not in res_ids_to_keep:
            res_to_remove.append(res)
    for res in res_to_remove:
        chain.detach_child(res.id)

    io = Bio.PDB.PDBIO()
    io.set_structure(pdb_model)
    io.save(output_path)


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
        ligand_path = os.path.join(output_folder, f"gt_ligand.sdf")

        cropped_gt_pdb_path = os.path.join(output_folder, f"gt_protein_cropped_{CROP_SIZE}.pdb")
        cropped_apo_pdb_path = os.path.join(output_folder, f"apo_protein_cropped_{CROP_SIZE}.pdb")

        assert os.path.exists(apo_pdb_path), f"Apo pdb does not exist"

        res_ids_to_keep = get_protein_res_ids_to_keep(gt_pdb_path, ligand_path, CROP_SIZE)

        crop_residue_ids(gt_pdb_path, res_ids_to_keep, cropped_gt_pdb_path)
        crop_residue_ids(apo_pdb_path, res_ids_to_keep, cropped_apo_pdb_path)

        status_dict["cropped"].append(desc.pdb_id)

    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()
