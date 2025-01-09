import json
import os
import sys
from typing import Optional

import Bio.PDB
import Bio.SeqUtils
import numpy as np
import rdkit.Chem.rdMolAlign
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D

from run_pretrained_model import run_on_folder


TEST_SET_PATH = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/processed/plinder_jsons_test"
OUTPUT_PATH = "/sci/labs/dina/bshor/projects/pred_affinity/202405_evodocker/test_set_plinder/output"


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
        gt_res_in_chain = [res for res in gt_protein[gt_chain_id] if "CA" in res and
                           not (Bio.SeqUtils.seq1(res.get_resname()) in ("X", "", " "))]
        pred_res_in_chain = [res for res in pred_protein[pred_chain_id] if "CA" in res]
        assert len(gt_res_in_chain) == len(pred_res_in_chain), "Different number of residues in proteins"

        for gt_res, pred_res in zip(gt_res_in_chain, pred_res_in_chain):
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

    protein_ca_coords = np.array([i["CA"].get_coord() for i in gt_protein_ca])  # Shape: (n_protein, 3)
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
    ligand_rmsd = rdkit.Chem.rdMolAlign.CalcRMS(pred_ligand, gt_ligand, prbId=conf_id)


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

    return ligand_rmsd, pocket_rmsd, protein_rmsd


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



def main(config_path):
    use_relaxed = False
    use_reembed = False
    save_aligned = False
    assert not (use_relaxed and use_reembed), "Can't use both relaxed and reembed"

    i = 0
    while os.path.exists(os.path.join(OUTPUT_PATH, f"output_{i}")):
        i += 1
    output_dir = os.path.join(OUTPUT_PATH, f"output_{i}")
    os.makedirs(output_dir, exist_ok=True)

    run_on_folder(TEST_SET_PATH, output_dir, config_path, long_sequence_inference=True)
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_10_run61")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_18_run85_67K")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_18_run85_67K")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_22_run87_93K")
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_23_run86_260K")

    if not use_relaxed:
        jobnames = ["_".join(filename.split("_")[:-1])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein.pdb") and "relaxed" not in filename]
    else:
        jobnames = ["_".join(filename.split("_")[:-2])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein_relaxed.pdb")]
        # jobnames = ["_".join(filename.split("_")[1:3])
        #             for filename in os.listdir(os.path.join(output_dir, "predictions"))
        #             if filename.endswith("_protein.pdb") and "relaxed" in filename]

    print(f"Analyzing {len(jobnames)} jobs...")

    all_rmsds = {}
    for jobname in sorted(jobnames):
        input_json = os.path.join(TEST_SET_PATH, f"{jobname}.json")
        input_data = json.load(open(input_json, "r"))
        parent_dir = os.path.dirname(TEST_SET_PATH)

        smiles = input_data["input_smiles"][0]
        gt_affinity = input_data["affinity"]

        gt_protein_path = os.path.join(parent_dir, input_data["gt_structure"])
        assert len(input_data["ref_sdf_list"]) == 1, "Multiple ligands not supported"

        gt_ligand_path = os.path.join(parent_dir, input_data["ref_sdf_list"][0])
        pred_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_protein.pdb")
        pred_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_ligand_0.sdf")

        relaxed_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_protein_relaxed.pdb")
        relaxed_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_ligand_relaxed.sdf")

        affinity_output_path = os.path.join(output_dir, "predictions", f"{jobname}_affinity.json")
        pred_affinities = json.load(open(affinity_output_path, "r"))

        input_pdb_path = os.path.join(parent_dir, input_data["input_structure"])

        if use_relaxed and not os.path.exists(relaxed_ligand_path):
            print("skipping, no relaxed", jobname)
            continue

        print(jobname)
        try:
            if use_relaxed:
                rmsds = get_rmsd(gt_protein_path, gt_ligand_path, relaxed_protein_path, relaxed_ligand_path,
                                 save_aligned=save_aligned)
            else:
                if not use_reembed:
                    smiles = None
                rmsds = get_rmsd(gt_protein_path, gt_ligand_path, pred_protein_path, pred_ligand_path,
                                 reembed_smiles=smiles, save_aligned=save_aligned)
        except Exception as e:
            print(f"Failed to compute RMSD for {jobname}", e)
            # raise e
            continue

        ligand_rmsd, pocket_rmsd, protein_rmsd = rmsds
        input_protein_to_pred_rmsd = simple_get_rmsd(input_pdb_path, pred_protein_path)
        input_protein_to_gt_rmsd = simple_get_rmsd(input_pdb_path, gt_protein_path)

        print(ligand_rmsd, pocket_rmsd, protein_rmsd, input_protein_to_pred_rmsd, input_protein_to_gt_rmsd)
        all_rmsds[jobname] = {"ligand_rmsd": ligand_rmsd, "pocket_rmsd": pocket_rmsd, "protein_rmsd": protein_rmsd,
                              "input_protein_to_pred_rmsd": input_protein_to_pred_rmsd,
                              "input_protein_to_gt_rmsd": input_protein_to_gt_rmsd,
                              "gt_affinity": gt_affinity, **pred_affinities
                              }

    print(all_rmsds)
    json.dump(all_rmsds, open(os.path.join(output_dir, "rmsds.json"), "w"), indent=4)

    ligand_rmsds = {k: v["ligand_rmsd"] for k, v in all_rmsds.items()}
    print("Total: ", len(ligand_rmsds), "Under 2: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 2),
          "Under 5: ", sum(1 for rmsd in ligand_rmsds.values() if rmsd < 5))


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run_config.json")
    main(config_path)
