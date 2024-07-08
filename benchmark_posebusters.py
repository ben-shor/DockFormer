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
from env_consts import POSEBUSTERS_JSONS, POSEBUSTERS_OUTPUT, POSEBUSTERS_GT, CKPT_PATH


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
    num_conformers = 5  # Define the number of conformers to generate
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


def get_rmsd(gt_protein_path: str, gt_ligand_path: str, pred_protein_path: str, pred_ligand_path:str, gt_chain_id: str,
             reembed_smiles: Optional[str] = None, save_aligned: bool = False):

    gt_protein = get_pdb_model(gt_protein_path)
    pred_protein = get_pdb_model(pred_protein_path)

    # impose the proteins
    super_imposer = Bio.PDB.Superimposer()

    ref_atoms = [res["CA"] for res in gt_protein[gt_chain_id].get_residues() if "CA" in res
                 and Bio.SeqUtils.seq1(res.get_resname()) not in ("X", "")]
    sample_atoms = [res["CA"] for res in pred_protein.get_residues() if "CA" in res]

    # debug issues
    # for res1, res2 in zip(gt_protein[gt_chain_id].get_residues(), pred_protein.get_residues()):
    #     print(res1.get_id()[1], res1.get_resname(), res2.get_id()[1], res2.get_resname())

    # for a1, a2 in zip(ref_atoms, sample_atoms):
    #     res1, res2 = a1.get_parent(), a2.get_parent()
    #     print(res1.get_id()[1], res1.get_resname(), res2.get_id()[1], res2.get_resname())
    #
    # print([res.id[1] for res in gt_protein[gt_chain_id].get_residues() if "CA" in res])
    # print([res.id[1] for res in pred_protein.get_residues() if "CA" in res])

    assert len(ref_atoms) == len(sample_atoms), f"Different number of CA atoms in proteins {len(ref_atoms)} {len(sample_atoms)}"

    super_imposer.set_atoms(ref_atoms, sample_atoms)

    gt_ligand = Chem.SDMolSupplier(gt_ligand_path)[0]
    # don't sanitize as when the positions are bad you have issues of "Can't kekulize mol."
    pred_ligand = Chem.MolFromMolFile(pred_ligand_path, sanitize=False)

    # remove the hydrogens
    try:
        pred_ligand = Chem.RemoveHs(pred_ligand)
    except Exception as e:
        print("Failed to remove hydrogens", e)

    assert gt_ligand is not None, f"Failed to parse ligand from {gt_ligand_path}"
    assert pred_ligand is not None, f"Failed to parse ligand from {pred_ligand_path}"
    assert gt_ligand.GetNumAtoms() == pred_ligand.GetNumAtoms(), f"Different number of atoms in ligands " \
                                                                 f"{gt_ligand.GetNumAtoms()} {pred_ligand.GetNumAtoms()}"

    pred_ligand, conf_id = pred_ligand, 0
    if reembed_smiles:
        pred_ligand, conf_id = create_embeded_molecule(pred_ligand, reembed_smiles)

    pred_ligand_conf = pred_ligand.GetConformer(id=conf_id)

    # apply transformation to pred_ligand
    for i in range(pred_ligand.GetNumAtoms()):
        pos = pred_ligand_conf.GetAtomPosition(i)
        new_pos = np.dot(pos, super_imposer.rotran[0]) + super_imposer.rotran[1]
        pred_ligand_conf.SetAtomPosition(i, Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))

    # get the RMSD
    rmsd = rdkit.Chem.rdMolAlign.CalcRMS(pred_ligand, gt_ligand, prbId=conf_id)

    # output aligned
    if save_aligned:
        aligned_pdb_path = pred_protein_path + "_aligned.pdb"
        io = Bio.PDB.PDBIO()
        super_imposer.apply(pred_protein.get_atoms())
        io.set_structure(pred_protein)
        io.save(aligned_pdb_path)
        aligned_ligand_path = pred_ligand_path + "_aligned.sdf"
        Chem.MolToMolFile(pred_ligand, aligned_ligand_path)

    return rmsd


def main(config_path):
    use_relaxed = False
    use_reembed = False
    assert not (use_relaxed and use_reembed), "Can't use both relaxed and reembed"

    i = 0
    while os.path.exists(os.path.join(POSEBUSTERS_OUTPUT, f"output_{i}")):
        i += 1
    output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_{i}")
    os.makedirs(output_dir, exist_ok=True)
    run_on_folder(POSEBUSTERS_JSONS, output_dir, config_path)
    # output_dir = os.path.join(POSEBUSTERS_OUTPUT, f"output_32_relaxed")

    if not use_relaxed:
        jobnames = ["_".join(filename.split("_")[:2])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein.pdb") and "relaxed" not in filename]
    else:
        jobnames = ["_".join(filename.split("_")[:2])
                    for filename in os.listdir(os.path.join(output_dir, "predictions"))
                    if filename.endswith("_protein_relaxed.pdb")]
        # jobnames = ["_".join(filename.split("_")[1:3])
        #             for filename in os.listdir(os.path.join(output_dir, "predictions"))
        #             if filename.endswith("_protein.pdb") and "relaxed" in filename]

    print(f"Analyzing {len(jobnames)} jobs...")

    all_rmsds = []
    for jobname in sorted(jobnames):
        gt_protein_path = os.path.join(POSEBUSTERS_GT, jobname, f"{jobname}_protein.pdb")
        gt_ligand_path = os.path.join(POSEBUSTERS_GT, jobname, f"{jobname}_ligand.sdf")
        pred_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_protein.pdb")
        pred_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_ligand.sdf")
        # relaxed_protein_path = os.path.join(output_dir, "predictions", f"relaxed_{jobname}_protein.pdb")
        # relaxed_ligand_path = os.path.join(output_dir, "predictions", f"relaxed_{jobname}_ligand.sdf")
        relaxed_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_protein_relaxed.pdb")
        relaxed_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_ligand_relaxed.sdf")
        input_json = os.path.join(POSEBUSTERS_JSONS, f"{jobname}.json")
        smiles = json.load(open(input_json, "r"))["input_smiles"]

        gt_chain_id = open(gt_ligand_path, "r").readline().strip().split()[0].split("_")[2][0]

        print(jobname)
        try:
            if use_relaxed:
                rmsd = get_rmsd(gt_protein_path, gt_ligand_path, relaxed_protein_path, relaxed_ligand_path, gt_chain_id)
            else:
                if not use_reembed:
                    smiles = None
                rmsd = get_rmsd(gt_protein_path, gt_ligand_path, pred_protein_path, pred_ligand_path, gt_chain_id,
                                reembed_smiles=smiles)
        except Exception as e:
            print(f"Failed to compute RMSD for {jobname}", e)
            # raise e
            continue
        print(rmsd)
        all_rmsds.append(rmsd)

    print(all_rmsds)

    print("Total: ", len(all_rmsds), "Under 2: ", sum(1 for rmsd in all_rmsds if rmsd < 2),
          "Under 5: ", sum(1 for rmsd in all_rmsds if rmsd < 5))


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run_config.json")
    main(config_path)
