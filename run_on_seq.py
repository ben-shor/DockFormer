import json
import os
import tempfile

import Bio.PDB
import Bio.SeqUtils
import numpy as np
from Bio import pairwise2
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from run_pretrained_model import run_on_folder


def get_seq_based_on_template(seq: str, template_path: str, output_path: str):
    # get a list of all residues in template
    parser = Bio.PDB.PDBParser()
    template_structure = parser.get_structure("template", template_path)
    chain = template_structure[0].get_chains().__next__()
    template_residues = [i for i in chain.get_residues() if "CA" in i
                         and Bio.SeqUtils.seq1(i.get_resname()) not in ("X", "", " ")]
    template_seq = "".join([Bio.SeqUtils.seq1(i.get_resname()) for i in template_residues])

    # align the sequence to the template
    alignment = pairwise2.align.globalxx(seq, template_seq, one_alignment_only=True)[0]
    aligned_seq, aligned_template_seq = alignment.seqA, alignment.seqB

    # create a new pdb file with the aligned residues
    new_structure = Bio.PDB.Structure.Structure("new_structure")
    new_model = Bio.PDB.Model.Model(0)
    new_structure.add(new_model)
    new_chain = Bio.PDB.Chain.Chain("A")  # Using chain ID 'A' for the output
    new_model.add(new_chain)

    template_ind = -1
    seq_ind = 0
    print(aligned_seq, aligned_template_seq, len(template_residues))
    for seq_res, template_res in zip(aligned_seq, aligned_template_seq):
        if template_res != "-":
            template_ind += 1

        if seq_res != "-":
            seq_ind += 1

        if seq_res == "-":
            continue

        if template_res == "-":
            seq_res_3_letter = Bio.SeqUtils.seq3(seq_res).upper()
            residue = Bio.PDB.Residue.Residue((' ', seq_ind, ' '), seq_res_3_letter, '')
            atom = Bio.PDB.Atom.Atom("C", (0.0, 0.0, 0.0), 1.0, 1.0, ' ', "CA", 0, element="C")
            residue.add(atom)
            new_chain.add(residue)
        else:
            residue = template_residues[template_ind].copy()
            residue.detach_parent()
            residue.id = (' ', seq_ind, ' ')
            new_chain.add(residue)
    io = Bio.PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)


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

    best_rmsd_index = int(np.argmin(rmsd_list))
    return target_mol, conformer_ids[best_rmsd_index], rmsd_list[best_rmsd_index]


def run_on_sample_seqs(seq_protein: str, template_protein_path: str, smiles: str, output_prot_path: str,
                       output_lig_path: str, run_config_path: str):
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    metrics = {}

    get_seq_based_on_template(seq_protein, template_protein_path, f"{temp_dir_path}/prot.pdb")
    create_conformers(smiles, f"{temp_dir_path}/lig.sdf")

    json_data = {
        "input_structure": f"prot.pdb",
        "ref_sdf": f"lig.sdf",
    }
    tmp_json_folder = f"{temp_dir_path}/jsons"
    os.makedirs(tmp_json_folder, exist_ok=True)
    json.dump(json_data, open(f"{tmp_json_folder}/input.json", "w"))
    tmp_output_folder = f"{temp_dir_path}/output"

    run_on_folder(tmp_json_folder, tmp_output_folder, run_config_path, skip_relaxation=True,
                  long_sequence_inference=False, skip_exists=False)
    predicted_protein_path = tmp_output_folder + "/predictions/input_predicted_protein.pdb"
    predicted_ligand_path = tmp_output_folder + "/predictions/input_predicted_ligand_0.sdf"
    predicted_affinity = json.load(open(tmp_output_folder + "/predictions/input_predicted_affinity.json"))
    metrics = {**metrics, **predicted_affinity}

    try:
        original_pred_ligand = Chem.MolFromMolFile(predicted_ligand_path, sanitize=False)
        try:
            original_pred_ligand = Chem.RemoveHs(original_pred_ligand)
        except Exception as e:
            print("Failed to remove hydrogens", e)

        assert original_pred_ligand is not None, f"Failed to parse ligand from {predicted_ligand_path}"
        rembed_pred_ligand, conf_id, rmsd = create_embeded_molecule(original_pred_ligand, smiles)
        metrics["ligand_reembed_rmsd"] = rmsd
        print("reembed with rmsd", rmsd)

        # save conformation to predicted_ligand_path
        w = Chem.SDWriter(predicted_ligand_path)
        w.write(rembed_pred_ligand, confId=conf_id)
        w.close()
    except Exception as e:
        print("Failed to reembed the ligand", e)

    os.rename(predicted_protein_path, output_prot_path)
    os.rename(predicted_ligand_path , output_lig_path)
    print("moved output to ", output_prot_path, output_lig_path)

    temp_dir.cleanup()

    return metrics
