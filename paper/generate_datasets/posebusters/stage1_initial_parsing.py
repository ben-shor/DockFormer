"""
download posebusters_paper_data.zip from https://zenodo.org/records/8278563
extract posebusters_benchmark_set folder to data/posebusters/raw
"""

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
import numpy as np

from utils import get_pdb_model, merge_to_single_chain


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

RAW_PATH = os.path.join(DATA_DIR, "posebusters", "raw")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "posebusters", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
SEQUENCES_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "sequences")


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


def remove_non_canonical_chains_and_residues(pdb_path: str, output_path: str):
    gt_model = get_pdb_model(pdb_path)
    new_chain_ids = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"]

    new_model = Bio.PDB.Model.Model(0)

    for chain in gt_model.get_chains():
        new_chain = do_robust_chain_object_renumber(chain, new_chain_ids.pop(0))
        if new_chain is not None:
            new_model.add(new_chain)

    io = Bio.PDB.PDBIO()
    io.set_structure(new_model)
    io.save(output_path)


def get_only_pocket_chains(pdb_path: str, sdf_path: str, output_path: str):
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
    protein_idx_by_dist = np.argsort(min_inter_dist_per_protein_atom)

    chains_to_save = set()
    for idx in protein_idx_by_dist:
        res = protein_atoms[idx].GetPDBResidueInfo()
        if min_inter_dist_per_protein_atom[idx] > 5.0:
            break
        if res.GetIsHeteroAtom():
            continue
        chains_to_save.add(res.GetChainId())

    pdb_model = get_pdb_model(pdb_path)
    all_chain_ids = [c.id for c in pdb_model.get_chains()]
    if len(chains_to_save) != len(all_chain_ids):
        print(f"Removing some chains ({len(chains_to_save)} vs {len(all_chain_ids)}) in {pdb_path}")
    else:
        print(f"All chains are saved ({len(chains_to_save)}) in {pdb_path}")

    chains_to_remove = set(all_chain_ids) - chains_to_save

    for chain_id in chains_to_remove:
        pdb_model.detach_child(chain_id)

    io = Bio.PDB.PDBIO()
    io.set_structure(pdb_model)
    io.save(output_path)



def main():
    status_dict = defaultdict(list)
    for folder_name in os.listdir(RAW_PATH):
        folder_path = os.path.join(RAW_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue
        print("Processing folder:", folder_name)

        output_folder = os.path.join(MODELS_FOLDER, folder_name)

        try:
            raw_protein_gt_path = os.path.join(folder_path, f"{folder_name}_protein.pdb")
            raw_ligand_gt_path = os.path.join(folder_path, f"{folder_name}_ligand.sdf")

            ref_ligand_path = os.path.join(output_folder, f"ref_ligand.sdf")
            gt_pdb_full_mc_path = os.path.join(output_folder, f"gt_protein_full_mc.pdb")  # all chains, multi chain
            gt_pdb_pocket_mc_path = os.path.join(output_folder, f"gt_protein_pocket_mc.pdb")  # pocket chains, single chain
            gt_pdb_full_sc_path = os.path.join(output_folder, f"gt_protein_full_sc.pdb")  # all chains, multi chain
            gt_pdb_pocket_sc_path = os.path.join(output_folder, f"gt_protein_pocket_sc.pdb")  # pocket chains, multi chain
            gt_ligand_path = os.path.join(output_folder, f"gt_ligand.sdf")

            if os.path.exists(output_folder):
                print("Already processed", folder_name)
                status_dict["already_processed"].append(folder_name)
                continue
            os.makedirs(output_folder, exist_ok=True)

            ligand = Chem.MolFromMolFile(raw_ligand_gt_path)
            assert ligand is not None, "Failed to load ligand"
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
            remove_non_canonical_chains_and_residues(raw_protein_gt_path, gt_pdb_full_mc_path)
            get_only_pocket_chains(gt_pdb_full_mc_path, raw_ligand_gt_path, gt_pdb_pocket_mc_path)

            merge_to_single_chain(gt_pdb_full_mc_path, gt_pdb_full_sc_path)
            merge_to_single_chain(gt_pdb_pocket_mc_path, gt_pdb_pocket_sc_path)

            status_dict["success"].append(folder_name)

        except Exception as e:
            print("############## Error with descriptor", folder_name, e)
            status_dict["error_" + str(e)].append(folder_name)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            raise

            continue

    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()




