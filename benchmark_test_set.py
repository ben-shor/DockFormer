import json
import os
import sys
from typing import Optional

import Bio.PDB
import Bio.SeqUtils
import numpy as np
import rdkit.Chem.rdMolAlign
import torch
from Bio import pairwise2
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS
from rdkit.Geometry import Point3D

from dockformer.data.data_pipeline import get_psuedo_beta
from run_pretrained_model import run_on_folder


def get_pdb_model(pdb_path: str):
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    pdb_struct = pdb_parser.get_structure("original_pdb", pdb_path)
    assert len(list(pdb_struct)) == 1, f"Too many models! {pdb_path}"
    return next(iter(pdb_struct))


def create_embeded_molecule(ref_mol: Chem.Mol, smiles: str):
    # Convert SMILES to a molecule
    target_mol = Chem.MolFromSmiles(smiles)

    assert target_mol is not None, f"Failed to parse molecule from SMILES {smiles}"

    # Set up parameters for conformer generation
    params = AllChem.ETKDGv3()
    params.numThreads = 0  # Use all available threads
    params.pruneRmsThresh = 0.1  # Pruning threshold for RMSD

    # Generate multiple conformers
    num_conformers = 1000  # Define the number of conformers to generate
    conformer_ids = AllChem.EmbedMultipleConfs(target_mol, numConfs=num_conformers, params=params)

    # Optional: Optimize each conformer using MMFF94 force field
    # for conf_id in conformer_ids:
    #     AllChem.UFFOptimizeMolecule(target_mol, confId=conf_id)

    # Align each generated conformer to the initial aligned conformer of the target molecule
    rmsd_list = []
    for conf_id in conformer_ids:
        rmsd = rdMolAlign.AlignMol(target_mol, ref_mol, prbCid=conf_id)
        rmsd_list.append(rmsd)
        # print(f"Conformer ID: {conf_id}, RMSD: {rmsd:.4f}")

    # w = Chem.SDWriter(os.path.join('aligned_conformers.sdf'))
    # for conf_id in conformer_ids:
    #     w.write(target_mol, confId=conf_id)
    # w.close()

    best_rmsd_index = int(np.argmin(rmsd_list))
    # print(f"Best RMSD: {rmsd_list[best_rmsd_index]:.4f} (ind={best_rmsd_index})")
    # return target_mol.GetConformers()[best_rmsd_index]
    return target_mol, conformer_ids[best_rmsd_index]


def get_matching_residues(gt_chain, pred_chain):
    """
    Returns lists of matching residues between two chains using sequence alignment.
    """
    # Convert residues to sequences
    gt_seq = ''.join([Bio.PDB.Polypeptide.three_to_one(res.get_resname())
                      for res in gt_chain if "CA" in res])
    pred_seq = ''.join([Bio.PDB.Polypeptide.three_to_one(res.get_resname())
                        for res in pred_chain if "CA" in res])

    # Perform global alignment
    alignments = pairwise2.align.globalxx(gt_seq, pred_seq)
    best_alignment = alignments[0]
    aligned_gt, aligned_pred = best_alignment[0], best_alignment[1]

    # Calculate sequence identity
    matches = sum(1 for a, b in zip(aligned_gt, aligned_pred) if a == b)
    seq_identity = (matches / len(aligned_gt)) * 100

    if seq_identity < 90:
        raise ValueError(f"Sequence identity {seq_identity:.1f}% is below 90% threshold")

    # Get residues with CA atoms
    gt_res = [res for res in gt_chain if "CA" in res]
    pred_res = [res for res in pred_chain if "CA" in res]

    # Map aligned sequences back to residues
    matching_pairs = []
    gt_idx = pred_idx = 0

    for a, b in zip(aligned_gt, aligned_pred):
        if a != '-' and b != '-' and a == b:  # Matching residues
            if gt_idx < len(gt_res) and pred_idx < len(pred_res):
                gt_r = gt_res[gt_idx]
                pred_r = pred_res[pred_idx]
                if Bio.PDB.Polypeptide.three_to_one(gt_r.get_resname()) == a:
                    matching_pairs.append((gt_r, pred_r))

        if a != '-':
            gt_idx += 1
        if b != '-':
            pred_idx += 1

    return matching_pairs


def calculate_ligand_rmsd_with_fallback(pred_ligand, gt_ligand, conf_id=0):
    # return the ligand_rmsd, and the percentage of matching atoms
    try:
        ligand_rmsd = rdkit.Chem.rdMolAlign.CalcRMS(pred_ligand, gt_ligand, prbId=conf_id)
        return ligand_rmsd, None
    except Exception as e:
        print(f"Standard RMSD calculation failed: {str(e)}")
        print("Falling back to MCS-based RMSD calculation")

        # Use MCS to find matching atoms between the molecules
        mcs_result = rdFMCS.FindMCS([pred_ligand, gt_ligand])
        assert not mcs_result.canceled, "MCS search canceled, Error!!!! Can't map ref ligand to gt ligand"
        assert mcs_result.numAtoms > 0, "MCS search found no common substructure"

        mcs_mol = rdkit.Chem.MolFromSmarts(mcs_result.smartsString)
        pred_match = pred_ligand.GetSubstructMatch(mcs_mol)
        gt_match = gt_ligand.GetSubstructMatch(mcs_mol)

        assert pred_match and gt_match, "Failed to find MCS matches in molecules"

        pred_to_gt_atom = {pred_idx: gt_idx for pred_idx, gt_idx in zip(pred_match, gt_match)}

        pred_ligand_conformer = pred_ligand.GetConformer(conf_id)
        pred_original_positions = [pred_ligand_conformer.GetAtomPosition(i) for i in range(pred_ligand.GetNumAtoms())]
        gt_ligand_conformer = gt_ligand.GetConformer()
        gt_original_positions = [gt_ligand_conformer.GetAtomPosition(i) for i in range(gt_ligand.GetNumAtoms())]

        pred_coords = []
        gt_coords = []

        for pred_idx, gt_idx in pred_to_gt_atom.items():
            pred_pos = pred_original_positions[pred_idx]
            gt_pos = gt_original_positions[gt_idx]

            pred_coords.append((pred_pos.x, pred_pos.y, pred_pos.z))
            gt_coords.append((gt_pos.x, gt_pos.y, gt_pos.z))

        # Convert to numpy arrays for RMSD calculation
        pred_coords = np.array(pred_coords)
        gt_coords = np.array(gt_coords)

        # Calculate RMSD manually
        squared_diff = np.sum((pred_coords - gt_coords) ** 2, axis=1)
        mcs_rmsd = np.sqrt(np.mean(squared_diff))

        print(f"MCS-based RMSD calculated using {len(pred_match)}/{len(pred_ligand.GetAtoms())} matching atoms")
        return mcs_rmsd, len(pred_match) / len(pred_ligand.GetAtoms())


def get_rmsd(gt_protein_path: str, gt_ligand_path: str, pred_protein_path: str, pred_ligand_path:str,
             reembed_smiles: Optional[str] = None, save_aligned: bool = False):

    gt_protein = get_pdb_model(gt_protein_path)
    pred_protein = get_pdb_model(pred_protein_path)

    assert len(list(gt_protein.get_chains())) == len(list(pred_protein.get_chains())), \
        "Different number of chains in proteins"
    chain_name_map = {a: b for a, b in zip(sorted([c.id for c in gt_protein.get_chains()]),
                                           sorted([c.id for c in pred_protein.get_chains()]))}

    gt_res_id_to_res, pred_res_id_to_res = {}, {}
    res_name_map = {}
    for (gt_chain_id, pred_chain_id) in chain_name_map.items():
        matching_pairs = get_matching_residues(gt_protein[gt_chain_id], pred_protein[pred_chain_id])

        for gt_res, pred_res in matching_pairs:
            assert gt_res.get_resname() == pred_res.get_resname(), "Different residue names"
            gt_res_id = (gt_chain_id, str(gt_res.get_id()[1]) + str(gt_res.get_id()[2]))
            pred_res_id = (pred_chain_id, str(pred_res.get_id()[1]) + str(pred_res.get_id()[2]))
            gt_res_id_to_res[gt_res_id] = gt_res
            pred_res_id_to_res[pred_res_id] = pred_res
            res_name_map[gt_res_id] = pred_res_id

    super_imposer = Bio.PDB.Superimposer()
    ref_atoms = []
    sample_atoms = []
    for gt_res_id, pred_res_id in res_name_map.items():
        gt_res = gt_res_id_to_res[gt_res_id]
        pred_res = pred_res_id_to_res[pred_res_id]
        ref_atoms.append(gt_res["CA"])
        sample_atoms.append(pred_res["CA"])
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    protein_rmsd = super_imposer.rms

    # calculate pocket rmsd
    gt_protein_ca = [res for chain in gt_protein for res in chain]
    gt_ligand = Chem.SDMolSupplier(gt_ligand_path)[0]
    gt_ligand_atoms = [gt_ligand.GetConformer().GetAtomPosition(i) for i in range(gt_ligand.GetNumAtoms())]

    protein_ca_coords = np.array([i["CA"].get_coord() for i in gt_protein_ca if "CA" in i])  # Shape: (n_protein, 3)
    ligand_atom_coords = np.array(gt_ligand_atoms)  # Shape: (n_ligand, 3)

    assert len(ref_atoms) == len(sample_atoms), f"Different number of CA atoms in proteins {len(ref_atoms)} {len(sample_atoms)}"
    # don't sanitize as when the positions are bad you have issues of "Can't kekulize mol."
    original_pred_ligand = Chem.MolFromMolFile(pred_ligand_path, sanitize=False)

    try:
        original_pred_ligand = Chem.RemoveHs(original_pred_ligand)
    except Exception as e:
        print("Failed to remove hydrogens", e)

    assert gt_ligand is not None, f"Failed to parse ligand from {gt_ligand_path}"
    assert original_pred_ligand is not None, f"Failed to parse ligand from {pred_ligand_path}"
    assert gt_ligand.GetNumAtoms() == original_pred_ligand.GetNumAtoms(), f"Different number of atoms in ligands " \
                                                                          f"{gt_ligand.GetNumAtoms()} {original_pred_ligand.GetNumAtoms()}"

    original_pred_ligand, conf_id = original_pred_ligand, 0
    if reembed_smiles:
        original_pred_ligand, conf_id = create_embeded_molecule(original_pred_ligand, reembed_smiles)
    pred_ligand = Chem.Mol(original_pred_ligand)

    # Compute squared distances between each protein CA and each ligand atom
    diff = protein_ca_coords[:, np.newaxis, :] - ligand_atom_coords[np.newaxis, :, :]  # Shape: (n_protein, n_ligand, 3)
    dist_sq = np.sum(diff ** 2, axis=2)  # Shape: (n_protein, n_ligand)
    is_interface = dist_sq <= 8 ** 2  # Boolean array of shape (n_protein, n_ligand)
    is_interface_res = np.any(is_interface, axis=1)  # Boolean array of shape (n_protein,)
    indices = np.where(is_interface_res)[0]

    gt_pocket_res_ids = []
    for i in indices:
        res = gt_protein_ca[i]
        gt_pocket_res_ids.append((res.parent.id, str(res.get_id()[1]) + str(res.get_id()[2])))

    pocket_most_common_chain = max(set([res_id[0] for res_id in gt_pocket_res_ids]),
                                   key=[res_id[0] for res_id in gt_pocket_res_ids].count)
    print(f"Number of interface residues: {len(indices)} {pocket_most_common_chain} {gt_pocket_res_ids}")

    super_imposer = Bio.PDB.Superimposer()
    ref_atoms = []
    sample_atoms = []
    for gt_res_id in gt_pocket_res_ids:
        if gt_res_id[0] != pocket_most_common_chain:
            continue
        pred_res_id = res_name_map[gt_res_id]
        gt_res = gt_res_id_to_res[gt_res_id]
        pred_res = pred_res_id_to_res[pred_res_id]
        ref_atoms.append(gt_res["CA"])
        sample_atoms.append(pred_res["CA"])
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    pocket_rmsd = super_imposer.rms

    # alt_chains = []
    # for possible_sample_chain_id in sorted([c.id for c in pred_protein.get_chains()]):
    #     if sample_chain_id == possible_sample_chain_id:
    #         continue


    # debug issues
    # for res1, res2 in zip(gt_protein[gt_chain_id].get_residues(), pred_protein.get_residues()):
    #     print(res1.get_id()[1], res1.get_resname(), res2.get_id()[1], res2.get_resname())

    # for a1, a2 in zip(ref_atoms, sample_atoms):
    #     res1, res2 = a1.get_parent(), a2.get_parent()
    #     print(res1.get_id()[1], res1.get_resname(), res2.get_id()[1], res2.get_resname())
    #
    # print([res.id[1] for res in gt_protein[gt_chain_id].get_residues() if "CA" in res])
    # print([res.id[1] for res in pred_protein.get_residues() if "CA" in res])

    pred_ligand = Chem.Mol(pred_ligand)
    pred_ligand_conf = pred_ligand.GetConformer(id=conf_id)
    # apply transformation to pred_ligand
    for i in range(pred_ligand.GetNumAtoms()):
        pos = pred_ligand_conf.GetAtomPosition(i)
        new_pos = np.dot(pos, super_imposer.rotran[0]) + super_imposer.rotran[1]
        pred_ligand_conf.SetAtomPosition(i, Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))

    # get the RMSD
    ligand_rmsd, matching_atoms_ratio = calculate_ligand_rmsd_with_fallback(pred_ligand, gt_ligand, conf_id)

    # check if there are alternative pockets
    original_pocket_res_ids = [k for k in gt_pocket_res_ids if k[0] == pocket_most_common_chain]
    for alt_gt_chain_id in chain_name_map.keys():
        if alt_gt_chain_id == pocket_most_common_chain:
            continue
        alt_pocket_res_ids = []
        for gt_res_id in gt_pocket_res_ids:
            alt_res_id = (alt_gt_chain_id, gt_res_id[1])
            if alt_res_id in gt_res_id_to_res:
                if gt_res_id_to_res[alt_res_id].get_resname() == gt_res_id_to_res[gt_res_id].get_resname():
                    alt_pocket_res_ids.append(alt_res_id)
        if len(alt_pocket_res_ids)/len(original_pocket_res_ids) > 0.7:
            alt_super_imposer = Bio.PDB.Superimposer()
            ref_atoms = []
            sample_atoms = []
            for gt_res_id in gt_pocket_res_ids:
                if gt_res_id[0] != pocket_most_common_chain:
                    continue
                alt_res_id = (alt_gt_chain_id, gt_res_id[1])
                if alt_res_id not in gt_res_id_to_res:
                    continue
                pred_res_id = res_name_map[gt_res_id]
                gt_res = gt_res_id_to_res[alt_res_id]
                pred_res = pred_res_id_to_res[pred_res_id]
                ref_atoms.append(gt_res["CA"])
                sample_atoms.append(pred_res["CA"])
            if not ref_atoms:
                continue
            alt_super_imposer.set_atoms(ref_atoms, sample_atoms)

            alt_pred_ligand = Chem.Mol(original_pred_ligand)
            alt_pred_ligand_conf = alt_pred_ligand.GetConformer(id=conf_id)

            # apply transformation to pred_ligand
            for i in range(pred_ligand.GetNumAtoms()):
                pos = alt_pred_ligand_conf.GetAtomPosition(i)
                new_pos = np.dot(pos, alt_super_imposer.rotran[0]) + alt_super_imposer.rotran[1]
                alt_pred_ligand_conf.SetAtomPosition(i, Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
            # get the RMSD
            alt_ligand_rmsd = rdkit.Chem.rdMolAlign.CalcRMS(alt_pred_ligand, gt_ligand, prbId=conf_id)
            print(f"Alternative pocket found {alt_gt_chain_id} {alt_super_imposer.rms} {alt_ligand_rmsd} ")

            if alt_ligand_rmsd < ligand_rmsd:
                print("Alternative is better", alt_ligand_rmsd, ligand_rmsd)
                ligand_rmsd = alt_ligand_rmsd
                pocket_rmsd = alt_super_imposer.rms
                super_imposer = alt_super_imposer
                pred_ligand = alt_pred_ligand

    # output aligned
    if save_aligned:
        aligned_pdb_path = pred_protein_path + "_aligned.pdb"
        io = Bio.PDB.PDBIO()
        super_imposer.apply(pred_protein.get_atoms())
        io.set_structure(pred_protein)
        io.save(aligned_pdb_path)
        aligned_ligand_path = pred_ligand_path + "_aligned.sdf"
        Chem.MolToMolFile(pred_ligand, aligned_ligand_path, confId=conf_id)

    return ligand_rmsd, pocket_rmsd, protein_rmsd, matching_atoms_ratio


def simple_get_rmsd(protein_path1: str, protein_path2: str):
    gt_protein = get_pdb_model(protein_path1)
    pred_protein = get_pdb_model(protein_path2)

    super_imposer = Bio.PDB.Superimposer()
    ref_atoms = []
    sample_atoms = []
    for chain in gt_protein:
        for res in chain:
            if "CA" in res:
                ref_atoms.append(res["CA"])
    for chain in pred_protein:
        for res in chain:
            if "CA" in res:
                sample_atoms.append(res["CA"])
    assert len(ref_atoms) == len(sample_atoms), f"Different number of CA atoms in proteins {len(ref_atoms)} {len(sample_atoms)}"
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    return super_imposer.rms


def get_inter_contacts(pdb_path: str, sdf_path: str, threshold: float = 5.0):
    ligand = Chem.MolFromMolFile(sdf_path)
    ligand_positions = ligand.GetConformer(0).GetPositions()
    ligand_positions = torch.tensor(np.array(ligand_positions)).float()

    # prepare inter_contacts
    expanded_prot_pos = get_psuedo_beta(pdb_path).unsqueeze(1)
    expanded_lig_pos = ligand_positions.unsqueeze(0)  # Shape: (1, N_lig, 3)
    distances = torch.sqrt(torch.sum((expanded_prot_pos - expanded_lig_pos) ** 2, dim=-1))
    mask = distances < threshold
    indices = mask.nonzero(as_tuple=False)  # shape (N, 2), where N is the number of True elements
    return [(int(row), int(col)) for row, col in indices]


def main(config_path, jsons_path, base_output_folder, ckpt_path=None, should_skip_structures=False):
    use_relaxed = False
    use_reembed = False
    save_aligned = False
    assert not (use_relaxed and use_reembed), "Can't use both relaxed and reembed"

    i = 0
    while os.path.exists(os.path.join(base_output_folder, f"output_{i}")):
        i += 1
    output_dir = os.path.join(base_output_folder, f"output_{i}")
    os.makedirs(output_dir, exist_ok=True)

    run_on_folder(jsons_path, output_dir, config_path, long_sequence_inference=True, ckpt_path=ckpt_path)
    # output_dir = os.path.join(base_output_folder, f"output__manual")
    # output_dir = os.path.join(base_output_folder, f"output_0")

    if not use_relaxed:
        jobnames = ["_".join(filename.split("_")[:-2])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein.pdb") and "relaxed" not in filename]
    else:
        jobnames = ["_".join(filename.split("_")[:-3])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein_relaxed.pdb")]
        # jobnames = ["_".join(filename.split("_")[1:3])
        #             for filename in os.listdir(os.path.join(output_dir, "predictions"))
        #             if filename.endswith("_protein.pdb") and "relaxed" in filename]

    print(f"Analyzing {len(jobnames)} jobs...")

    all_rmsds = {}
    for jobname in sorted(jobnames):
        input_json = os.path.join(jsons_path, f"{jobname}.json")
        input_data = json.load(open(input_json, "r"))
        parent_dir = os.path.dirname(jsons_path)

        gt_affinity = input_data["affinity"]

        affinity_output_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_affinity.json")
        pred_affinities = json.load(open(affinity_output_path, "r"))
        pred_inter_contacts = [(i[0], i[1]) for i in pred_affinities.pop("inter_contacts", [])]
        all_rmsds[jobname] = {"gt_affinity": gt_affinity, **pred_affinities}

        if not should_skip_structures:
            gt_protein_path = os.path.join(parent_dir, input_data["gt_structure"])
            # assert len(input_data["ref_sdf_list"]) == 1, "Multiple ligands not supported"
            if len(input_data["ref_sdf_list"]) > 1:
                print("**** Multiple ligands not supported, taking first", jobname)

            gt_ligand_path = os.path.join(parent_dir, input_data["gt_sdf_list"][0])
            pred_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_protein.pdb")
            pred_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_ligand_0.sdf")

            relaxed_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_protein_relaxed.pdb")
            relaxed_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_ligand_relaxed.sdf")

            input_pdb_path = os.path.join(parent_dir, input_data["input_structure"])

            if use_relaxed and not os.path.exists(relaxed_ligand_path):
                print("skipping, no relaxed", jobname)
                continue

            print(jobname)
            gt_inter_contacts = get_inter_contacts(gt_protein_path, gt_ligand_path)

            min_len = min(len(gt_inter_contacts), len(pred_inter_contacts))
            if min_len == 0:
                precision = recall = auc_score = 0
            else:
                precision = len([i for i in pred_inter_contacts[:len(gt_inter_contacts)]
                                 if i in gt_inter_contacts]) / min_len
                recall = len([i for i in gt_inter_contacts
                              if i in pred_inter_contacts[:len(gt_inter_contacts)]]) / min_len
                try:
                    from sklearn.metrics import roc_auc_score
                    y_label = [1 if i in gt_inter_contacts else 0 for i in pred_inter_contacts]
                    y_score = list(range(len(pred_inter_contacts), 0, -1))
                    auc_score = roc_auc_score(y_label, y_score)
                    auc_score = auc_score if not np.isnan(auc_score) else 0
                except ImportError:
                    print("Failed to import sklearn, skipping auc")
                    auc_score = None

            try:
                if use_relaxed:
                    rmsds = get_rmsd(gt_protein_path, gt_ligand_path, relaxed_protein_path, relaxed_ligand_path,
                                     save_aligned=save_aligned)
                else:
                    smiles = None
                    if use_reembed:
                        smiles = input_data["input_smiles_list"][0]
                    rmsds = get_rmsd(gt_protein_path, gt_ligand_path, pred_protein_path, pred_ligand_path,
                                     reembed_smiles=smiles, save_aligned=save_aligned)
            except Exception as e:
                print(f"Failed to compute RMSD for {jobname}", e)
                raise e
                continue

            ligand_rmsd, pocket_rmsd, protein_rmsd, matching_ligand_atoms = rmsds
            try:
                input_protein_to_pred_rmsd = simple_get_rmsd(input_pdb_path, pred_protein_path)
                input_protein_to_gt_rmsd = simple_get_rmsd(input_pdb_path, gt_protein_path)
            except:
                input_protein_to_pred_rmsd = input_protein_to_gt_rmsd = None
            print(ligand_rmsd, pocket_rmsd, protein_rmsd, input_protein_to_pred_rmsd, input_protein_to_gt_rmsd)

            all_rmsds[jobname] = {"ligand_rmsd": ligand_rmsd, "pocket_rmsd": pocket_rmsd, "protein_rmsd": protein_rmsd,
                                  "input_protein_to_pred_rmsd": input_protein_to_pred_rmsd,
                                  "input_protein_to_gt_rmsd": input_protein_to_gt_rmsd,
                                  "inter_contacts_precision": precision, "inter_contacts_recall": recall,
                                  "inter_contacts_auc": auc_score, "matching_ligand_atoms": matching_ligand_atoms,
                                  **all_rmsds[jobname]
                                  }
    print(all_rmsds)
    print("real total", len(all_rmsds))
    ligand_rmsds = {k: v["ligand_rmsd"] for k, v in all_rmsds.items() if "ligand_rmsd" in v}
    missing_keys = [k for k in all_rmsds.keys() if k not in ligand_rmsds]
    print("Missing ligand_rmsd for:", missing_keys)
    print("saved on", output_dir)
    print("Total: ", len(ligand_rmsds), "Under 2: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 2),
          "Under 5: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 5))

    json.dump(all_rmsds, open(os.path.join(output_dir, "rmsds.json"), "w"), indent=4)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run_config.json")
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "basic_local"
    arg_ckpt_path = sys.argv[3] if len(sys.argv) > 3 else None

    should_skip_structures = False
    if dataset_name == "basic_local":
        jsons_path = "/Users/benshor/Documents/Data/202401_pred_affinity/local_tests/input_manual"
        output_folder = "/Users/benshor/Documents/Data/202401_pred_affinity/local_tests/test_outputs"
    elif dataset_name == "casf2016":
        jsons_path = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/CASF2016/CASF-2016/dockformer_jsons_dockformer_pocket"
        output_folder = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/CASF2016/dockformer_predictions_20250121"
    elif dataset_name == "casp_test":
        jsons_path = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/jsons"
        output_folder = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/output"
    elif dataset_name == "casp_test_diverse":
        jsons_path = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/diverse_jsons"
        output_folder = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/output"
    elif dataset_name == "casp_test_only_aff":
        jsons_path = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/jsons_only_aff"
        output_folder = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/output_only_aff"
        should_skip_structures = True
    elif dataset_name == "casp_test_aff":
        jsons_path = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/jsons_aff_dataset"
        output_folder = "/cs/usr/bshor/sci/projects/pred_affinity/202405_evodocker/casp_test_set/output_aff"
        should_skip_structures = True
    elif dataset_name == "plinder_test":
        jsons_path = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/202409_plinder/processed/plinder_jsons_test"
        output_folder = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/test_set_plinder/output"
    elif dataset_name == "posebusters2":
        jsons_path = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/posebusters_dataset2/input_json_with_gt"
        output_folder = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/posebusters_dataset2/output"
    elif dataset_name == "posebusters3":
        jsons_path = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/posebusters_dataset3/jsons_no_big"
        output_folder = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/posebusters_dataset3/output"
    elif dataset_name == "posebusters3_big":
        jsons_path = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/posebusters_dataset3/jsons_no_big"
        output_folder = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/posebusters_dataset3/output"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if dataset_name != "basic_local":
        name = f"{dataset_name}_{os.path.basename(config_path).split('.')[0]}"
        output_folder = f"/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/benchmark_outputs/{name}"

    main(config_path, jsons_path, output_folder, ckpt_path=arg_ckpt_path, should_skip_structures=should_skip_structures)
