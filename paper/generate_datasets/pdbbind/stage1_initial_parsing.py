"""
Before running this script, download the PDBBind v2020 dataset
https://www.pdbbind-plus.org.cn/download

Extract PDBbind_v2020_refined.tar.gz and PDBbind_v2020_other_PL.tar.gz contents
into on DockFormer/data/pdbbind/raw

stage 1 - prepare folder + ref ligand + gt_protein renumber
stage 2 - prepare apo structure: download AF_DB/ESM + align to gt_protein + create dir of sequences for af
stage 3 - either load structures created on stage 2.5, or use manual AF structures
stage 4 - cropping + filtering + create jsons + train/validation jsons
"""


import dataclasses
import os
import shutil
import rdkit
from collections import defaultdict
from typing import Optional

import Bio.PDB
import Bio.SeqIO
import Bio.SeqUtils
from rdkit import Chem
from rdkit.Chem import AllChem

from utils import get_all_descriptors, get_pdb_model, get_pdb_model_readonly


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

PDBBIND_PATH = os.path.join(DATA_DIR, "pdbbind", "raw")
DATA_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_data.2020")
NAME_INDEX_PATH = os.path.join(PDBBIND_PATH, "index/INDEX_general_PL_name.2020")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "pdbbind", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")


def reconstruct_mol(mol):
    new_mol = Chem.RWMol()

    # Add atoms
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        new_atom = Chem.Atom(symbol)
        new_atom.SetChiralTag(atom.GetChiralTag())
        new_atom.SetFormalCharge(atom.GetFormalCharge())

        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        new_mol.AddBond(atom1, atom2, bond_type)

    return new_mol


def is_same(mol1, mol2):
    return mol1.GetNumAtoms() == mol2.GetNumAtoms() \
           and mol1.GetSubstructMatch(mol2) == tuple(range(mol1.GetNumAtoms())) \
           and mol2.GetSubstructMatch(mol1) == tuple(range(mol2.GetNumAtoms()))


def validate_mol(mol2_obj):
    # First validate that the smiles can create some molecule
    mol2_obj = Chem.MolFromSmiles(Chem.MolToSmiles(mol2_obj))
    if mol2_obj is None:
        print("Failed to create ligand from smiles")
        return False

    # Then validate that only symbol+chirality+charge are needed to represent the molecule
    reconstructed = reconstruct_mol(mol2_obj)
    return is_same(mol2_obj, reconstructed)


def create_conformers(smiles, output_path, num_conformers=1, multiplier_samples=1):
    target_mol = Chem.MolFromSmiles(smiles)
    target_mol = Chem.AddHs(target_mol)

    params = AllChem.ETKDGv3()
    params.numThreads = 0  # Use all available threads
    params.pruneRmsThresh = 0.1  # Pruning threshold for RMSD
    conformer_ids = AllChem.EmbedMultipleConfs(target_mol, numConfs=num_conformers * multiplier_samples, params=params)

    # Optional: Optimize each conformer using MMFF94 force field
    # for conf_id in conformer_ids:
    #     AllChem.UFFOptimizeMolecule(target_mol, confId=conf_id)

    # remove hydrogen atoms
    target_mol = Chem.RemoveHs(target_mol)

    # Save aligned conformers to a file (optional)
    w = Chem.SDWriter(output_path)
    for i, conf_id in enumerate(conformer_ids):
        if i >= num_conformers:
            break
        w.write(target_mol, confId=conf_id)
    w.close()


def do_robust_chain_object_renumber(chain: Bio.PDB.Chain.Chain, new_chain_id: str) -> Optional[Bio.PDB.Chain.Chain]:
    all_residues = [res for res in chain.get_residues()
                    if "CA" in res and Bio.SeqUtils.seq1(res.get_resname()) not in ("X", "", " ")]
    if not all_residues:
        return None

    res_and_res_id = [(res, res.get_id()[1]) for res in all_residues]

    min_res_id = min([i[1] for i in res_and_res_id])
    if min_res_id < 1:
        print("Negative res id", chain, min_res_id)
        factor = -1 * min_res_id + 1
        res_and_res_id = [(res, res_id + factor) for res, res_id in res_and_res_id]

    res_and_res_id_no_collisions = []
    for res, res_id in res_and_res_id[::-1]:
        if res_and_res_id_no_collisions and res_and_res_id_no_collisions[-1][1] == res_id:
            # there is a collision, usually an insertion residue
            res_and_res_id_no_collisions = [(i, j + 1) for i, j in res_and_res_id_no_collisions]
        res_and_res_id_no_collisions.append((res, res_id))

    first_res_id = min([i[1] for i in res_and_res_id_no_collisions])
    factor = 1 - first_res_id  # start from 1
    new_chain = Bio.PDB.Chain.Chain(new_chain_id)

    res_and_res_id_no_collisions.sort(key=lambda x: x[1])

    for res, res_id in res_and_res_id_no_collisions:
        chain.detach_child(res.id)
        res.id = (" ", res_id + factor, " ")
        new_chain.add(res)

    return new_chain


def generate_gt_protein(pdbbind_gt_path: str, pdbbind_pocket_path: str, output_path: str, pdb_id: str):
    """
    (1) get chains in pocket
    (2) renumber all chains
    (3) if multiple chains - join them into 1, with gaps of 50 residues
    """
    gt_model = get_pdb_model(pdbbind_gt_path)
    pocket_model = get_pdb_model_readonly(pdbbind_pocket_path)
    pocket_chain_ids = [c.id for c in pocket_model if c.id not in ("", " ")]
    if len(pocket_chain_ids) != 1:
        print(f"multiple chains in pocket ({len(pocket_chain_ids)})", pdb_id)
        renum_chains = [do_robust_chain_object_renumber(gt_model[cid], "A") for cid in pocket_chain_ids]
        gt_chain = Bio.PDB.Chain.Chain("A")
        factor_res_id = 0
        for chain in renum_chains:
            all_res = list(chain.get_residues())
            for res in all_res:
                chain.detach_child(res.id)
            max_new_id = 0
            for res in all_res:
                res.id = (" ", res.id[1] + factor_res_id, " ")
                max_new_id = max(max_new_id, res.id[1])
                gt_chain.add(res)
            factor_res_id = max_new_id + 32
    else:
        pdb_chain_id = pocket_chain_ids[0]
        gt_chain = do_robust_chain_object_renumber(gt_model[pdb_chain_id], "A")
    io = Bio.PDB.PDBIO()
    io.set_structure(gt_chain)
    io.save(output_path)


def main():
    all_descriptors = get_all_descriptors(NAME_INDEX_PATH, DATA_INDEX_PATH)

    os.makedirs(MODELS_FOLDER, exist_ok=True)

    status_dict = defaultdict(list)
    for desc_ind, desc in enumerate(all_descriptors):
        output_folder = os.path.join(MODELS_FOLDER, desc.pdb_id)
        print("processing", desc.pdb_id, desc_ind + 1, "/", len(all_descriptors))
        try:
            pdbbind_gt_path = os.path.join(PDBBIND_PATH, f"{desc.pdb_id}/{desc.pdb_id}_protein.pdb")
            pdb_pocket_path = os.path.join(PDBBIND_PATH, f"{desc.pdb_id}/{desc.pdb_id}_pocket.pdb")
            ligand_mol2_path = os.path.join(PDBBIND_PATH, f"{desc.pdb_id}/{desc.pdb_id}_ligand.mol2")

            ref_ligand_path = os.path.join(output_folder, f"ref_ligand.sdf")
            gt_pdb_path = os.path.join(output_folder, f"gt_protein.pdb")
            gt_ligand_path = os.path.join(output_folder, f"gt_ligand.sdf")

            if os.path.exists(output_folder):
                print("Already processed", desc.pdb_id)
                status_dict["already_processed"].append(desc.pdb_id)
                continue
            os.makedirs(output_folder, exist_ok=True)

            ligand = Chem.MolFromMol2File(ligand_mol2_path)
            assert ligand is not None, "Failed to load ligand"
            assert ligand.GetNumAtoms() < 50, "Too many atoms in ligand"
            assert validate_mol(ligand), "Failed to reconstruct ligand"
            # save gt ligand
            w = Chem.SDWriter(gt_ligand_path)
            w.write(ligand)
            w.close()

            # prepare ref ligand
            smiles = Chem.MolToSmiles(ligand)
            create_conformers(smiles, ref_ligand_path, num_conformers=1, multiplier_samples=1)
            ref_ligand = Chem.MolFromMolFile(ref_ligand_path)
            assert ref_ligand is not None, "Failed to create ref ligand"
            try:
                rdkit.Chem.rdMolAlign.CalcRMS(ref_ligand, ligand, prbId=0)
            except:
                assert False, f"Ref ligand is not the same as original"

            # prepare gt protein structure
            try:
                generate_gt_protein(pdbbind_gt_path, pdb_pocket_path, gt_pdb_path, desc.pdb_id)
            except:
                assert False, f"Failed to generate gt protein"
            pdb_model = get_pdb_model(gt_pdb_path)
            all_res_ids = [res.get_id()[1] for res in pdb_model.get_residues()]
            assert max(all_res_ids) - min(all_res_ids) < 1800, "Too large gap in residues"

            status_dict["success"].append(desc.pdb_id)

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
