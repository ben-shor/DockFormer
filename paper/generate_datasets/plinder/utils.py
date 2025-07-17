from typing import Optional, List

import Bio.PDB
import Bio.SeqIO
import Bio.SeqUtils
import requests
from Bio import pairwise2, PDB
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def get_pdb_model(pdb_path: str):
    if pdb_path.endswith(".pdb"):
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        pdb_struct = pdb_parser.get_structure("original_pdb", pdb_path)
        assert len(list(pdb_struct)) == 1, f"Not one model ({len(list(pdb_struct))})! {pdb_path}"
        return next(iter(pdb_struct))
    elif pdb_path.endswith(".cif"):
        cif_parser = Bio.PDB.MMCIFParser(QUIET=True)
        cif_struct = cif_parser.get_structure("original_cif", pdb_path)
        assert len(list(cif_struct)) == 1, f"Not one model ({len(list(cif_struct))})! {pdb_path}"
        return next(iter(cif_struct))
    raise ValueError(f"Unsupported file format for {pdb_path}. Expected .pdb or .cif.")


def cif_to_pdb(cif_path: str, pdb_path: str):
    protein = Bio.PDB.MMCIFParser().get_structure("s_cif", cif_path)
    io = Bio.PDB.PDBIO()
    io.set_structure(protein)
    io.save(pdb_path)


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


def validate_mol(ligand):
    # First validate that the smiles can create some molecule
    mol2_obj = Chem.MolFromSmiles(Chem.MolToSmiles(ligand))
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


def renumber_and_remove_non_canonical_chains_and_residues(pdb_path: str, output_path: str):
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


def get_only_pocket_chains(pdb_path: str, sdf_paths: List[str], output_path: str):
    protein = Chem.MolFromPDBFile(pdb_path, sanitize=False)

    protein_conf = protein.GetConformer()
    protein_pos = protein_conf.GetPositions()
    protein_atoms = list(protein.GetAtoms())
    assert len(protein_pos) == len(protein_atoms), f"Positions and atoms mismatch in {pdb_path}"

    ligand_pos = None
    for sdf_path in sdf_paths:
        ligand = Chem.MolFromMolFile(sdf_path)
        ligand_conf = ligand.GetConformer()
        if ligand_pos is None:
            ligand_pos = ligand_conf.GetPositions()
        else:
            ligand_pos = np.vstack((ligand_pos, ligand_conf.GetPositions()))

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


def generate_shared_gt_and_apo(gt_path: str, apo_path: str, output_gt_path: str, output_apo_path: str,
                               max_missing: float = 0.1) -> bool:
    gt_model = get_pdb_model(gt_path)
    apo_model = get_pdb_model(apo_path)

    assert len(gt_model) == 1, f"GT model has multiple chains: {len(gt_model)}"
    assert len(apo_model) == 1, f"APO model has multiple chains: {len(apo_model)}"

    gt_chain = next(iter(gt_model.get_chains()))
    apo_chain = next(iter(apo_model.get_chains()))

    gt_seq = get_chain_object_to_seq(gt_chain).replace("X", "")
    apo_seq = get_chain_object_to_seq(apo_chain).replace("X", "")

    alignments = pairwise2.align.globalms(gt_seq, apo_seq, 2, -10000, -1, 0)

    if not alignments:
        print("No alignment found.")
        return False

    best_alignment = alignments[0]
    aligned_gt, aligned_apo, score, start, end = best_alignment

    missing_residues_percentage = aligned_apo.count("-") / len(gt_seq)

    if missing_residues_percentage > max_missing:
        print(f"Too many missing residues in APO sequence: {aligned_apo.count('-')} / {len(gt_seq)}")
        return False

    # For debugging, optional
    # print(format_alignment(*best_alignment))

    gt_residues = list(gt_chain.get_residues())
    apo_residues = list(apo_chain.get_residues())

    new_gt_chain = PDB.Chain.Chain("A")
    new_apo_chain = PDB.Chain.Chain("A")

    gt_index, apo_index = 0, 0
    for a_gt, a_apo in zip(aligned_gt, aligned_apo):
        if a_gt == "-":
            apo_index += 1
            continue
        if a_apo == "-":
            gt_index += 1
            continue
        gt_res = gt_residues[gt_index]
        apo_res = apo_residues[apo_index]

        if gt_res.get_resname() != apo_res.get_resname():
            print(f"ERROR: Residue mismatch at {gt_res.id[1]}: GT {gt_res.get_resname()} != AF {apo_res.get_resname()}")
            return False

        apo_chain.detach_child(apo_res.id)  # remove from original AF chain
        apo_res.id = (" ", gt_res.id[1], " ")  # reset to GT's numbering
        new_apo_chain.add(apo_res)

        gt_chain.detach_child(gt_res.id)  # remove from original GT chain
        new_gt_chain.add(gt_res)

        gt_index += 1
        apo_index += 1

    # Write to output
    structure = Bio.PDB.Structure.Structure("X_gt")
    model = Bio.PDB.Model.Model(0)
    structure.add(model)
    model.add(new_gt_chain)
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_gt_path)

    structure = Bio.PDB.Structure.Structure("X_apo")
    model = Bio.PDB.Model.Model(0)
    structure.add(model)
    model.add(new_apo_chain)
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_apo_path)

    return True


def get_sequence_from_pdb(pdb_path: str) -> str:
    model = get_pdb_model(pdb_path)
    assert len(model) == 1, f"Model has multiple chains: {len(model)}"
    return get_chain_object_to_seq(next(iter(model.get_chains())))


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