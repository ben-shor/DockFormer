import json
import logging
import os
import re
import time
from typing import List, Tuple

import numpy
import torch
from rdkit import Chem

from evodocker.model.model import AlphaFold
from evodocker.utils import residue_constants, protein
from evodocker.utils.consts import POSSIBLE_ATOM_TYPES, POSSIBLE_BOND_TYPES, POSSIBLE_CHARGES, POSSIBLE_CHIRALITIES

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def count_models_to_evaluate(model_checkpoint_path):
    model_count = 0
    if model_checkpoint_path:
        model_count += len(model_checkpoint_path.split(","))
    return model_count


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


# Function to get the latest checkpoint
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest_checkpoint)


def load_models_from_command_line(config, model_device, model_checkpoint_path, output_dir):
    # Create the output directory

    multiple_model_mode = count_models_to_evaluate(model_checkpoint_path) > 1
    if multiple_model_mode:
        logger.info(f"evaluating multiple models")

    if model_checkpoint_path:
        for path in model_checkpoint_path.split(","):
            model = AlphaFold(config)
            model = model.eval()
            checkpoint_basename = get_model_basename(path)
            assert os.path.isfile(path), f"Model checkpoint not found at {path}"
            ckpt_path = path
            d = torch.load(ckpt_path)

            if "ema" in d:
                # The public weights have had this done to them already
                d = d["ema"]["params"]
            model.load_state_dict(d)


            model = model.to(model_device)
            logger.info(
                f"Loaded Model parameters at {path}..."
            )
            output_directory = make_output_directory(output_dir, checkpoint_basename, multiple_model_mode)
            yield model, output_directory

    if not model_checkpoint_path:
        raise ValueError("model_checkpoint_path must be specified.")


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs


def update_timings(timing_dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return output_file


def run_model(model, batch, tag, output_dir):
    with torch.no_grad():
        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
        update_timings({tag: {"inference": inference_time}}, os.path.join(output_dir, "timings.json"))

    return out


def get_molecule_from_output(atoms_atype: List[int], atom_chiralities: List[int], atom_charges: List[int],
                             bonds: List[Tuple[int, int, int]], atom_positions: List[Tuple[float, float, float]]):
    mol = Chem.RWMol()

    assert len(atoms_atype) == len(atom_chiralities) == len(atom_charges) == len(atom_positions)
    for atype_idx, chirality_idx, charge_idx in zip(atoms_atype, atom_chiralities, atom_charges):
        new_atom = Chem.Atom(POSSIBLE_ATOM_TYPES[atype_idx])
        new_atom.SetChiralTag(POSSIBLE_CHIRALITIES[chirality_idx])
        new_atom.SetFormalCharge(POSSIBLE_CHARGES[charge_idx])

        mol.AddAtom(new_atom)

    # Add bonds
    for bond in bonds:
        atom1, atom2, bond_type_idx = bond
        bond_type = POSSIBLE_BOND_TYPES[bond_type_idx]
        mol.AddBond(int(atom1), int(atom2), bond_type)

    # Set atom positions
    conf = Chem.Conformer(len(atoms_atype))
    for i, pos in enumerate(atom_positions.astype(float)):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    return mol


def save_output_structure(aatype, residue_index, plddt, final_atom_protein_positions, final_atom_mask, ligand_atype,
                          ligand_chiralities, ligand_charges, ligand_bonds, final_ligand_atom_positions,
                          protein_output_path, ligand_output_path, protein_affinity_output_path, affinity,
                          binding_site_probs):
    plddt_b_factors = numpy.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    unrelaxed_protein = protein.from_prediction(
        aatype=aatype,
        residue_index=residue_index,
        atom_mask=final_atom_mask,
        atom_positions=final_atom_protein_positions,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False,
    )

    with open(protein_output_path, 'w') as fp:
        fp.write(protein.to_pdb(unrelaxed_protein))

    binding_site_b_factors = numpy.repeat(
        binding_site_probs[..., None], residue_constants.atom_type_num, axis=-1
    )

    protein_binding_site = protein.from_prediction(
        aatype=aatype,
        residue_index=residue_index,
        atom_mask=final_atom_mask,
        atom_positions=final_atom_protein_positions,
        b_factors=binding_site_b_factors,
        remove_leading_feature_dimension=False,
        remark=f"affinity: {affinity:.3f}",
    )

    with open(protein_affinity_output_path, 'w') as fp:
        fp.write(protein.to_pdb(protein_binding_site))

    ligand = get_molecule_from_output(ligand_atype, ligand_chiralities, ligand_charges, ligand_bonds,
                                      final_ligand_atom_positions)
    # protein_obj = Chem.MolFromPDBFile(output_path, sanitize=False)
    # assert protein_obj is not None, "Failed to unrelaxed read protein from PDB file"
    # combined = Chem.CombineMols(protein_obj, ligand)

    with open(ligand_output_path, 'w') as f:
        f.write(Chem.MolToMolBlock(ligand, kekulize=False))
    print("Output written to", protein_output_path, ligand_output_path)
