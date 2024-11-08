import os
import numpy as np
import torch
from torch import nn
from rdkit import Chem

from dockformer.data.utils import FeatureTensorDict
from dockformer.utils.consts import POSSIBLE_BOND_TYPES, POSSIBLE_ATOM_TYPES, POSSIBLE_CHARGES, POSSIBLE_CHIRALITIES


def get_atom_features(atom: Chem.Atom):
    # TODO: this is temporary, we need to add more features, for example for Zn
    if atom.GetSymbol() not in POSSIBLE_ATOM_TYPES:
        print(f"********Unknown atom type {atom.GetSymbol()}")
        atom_type = POSSIBLE_ATOM_TYPES.index("Ni")
    else:
        atom_type = POSSIBLE_ATOM_TYPES.index(atom.GetSymbol())
    atom_charge = POSSIBLE_CHARGES.index(max(min(atom.GetFormalCharge(), 1), -1))
    atom_chirality = POSSIBLE_CHIRALITIES.index(atom.GetChiralTag())

    return {"atom_type": atom_type, "atom_charge": atom_charge, "atom_chirality": atom_chirality}


def get_bond_features(bond: Chem.Bond):
    bond_type = POSSIBLE_BOND_TYPES.index(bond.GetBondType())
    return {"bond_type": bond_type}


def make_ligand_features(ligand: Chem.Mol) -> FeatureTensorDict:
    atoms_features = []
    atom_idx_to_atom_pos_idx = {}
    for atom in ligand.GetAtoms():
        atom_idx_to_atom_pos_idx[atom.GetIdx()] = len(atoms_features)
        atoms_features.append(get_atom_features(atom))

    atom_types = torch.tensor(np.array([atom["atom_type"] for atom in atoms_features], dtype=np.int64))
    atom_types_one_hot = nn.functional.one_hot(atom_types, num_classes=len(POSSIBLE_ATOM_TYPES), )
    atom_charges = torch.tensor(np.array([atom["atom_charge"] for atom in atoms_features], dtype=np.int64))
    atom_charges_one_hot = nn.functional.one_hot(atom_charges, num_classes=len(POSSIBLE_CHARGES))
    atom_chiralities = torch.tensor(np.array([atom["atom_chirality"] for atom in atoms_features], dtype=np.int64))
    atom_chiralities_one_hot = nn.functional.one_hot(atom_chiralities, num_classes=len(POSSIBLE_CHIRALITIES))

    ligand_target_feat = torch.cat([atom_types_one_hot.float(), atom_charges_one_hot.float(),
                                    atom_chiralities_one_hot.float()], dim=1)

    # create one-hot matrix encoding for bonds
    ligand_bonds_feat = torch.zeros((len(atoms_features), len(atoms_features), len(POSSIBLE_BOND_TYPES)))
    ligand_bonds = []
    for bond in ligand.GetBonds():
        atom1_idx = atom_idx_to_atom_pos_idx[bond.GetBeginAtomIdx()]
        atom2_idx = atom_idx_to_atom_pos_idx[bond.GetEndAtomIdx()]
        bond_features = get_bond_features(bond)
        ligand_bonds.append((atom1_idx, atom2_idx, bond_features["bond_type"]))
        ligand_bonds_feat[atom1_idx, atom2_idx, bond_features["bond_type"]] = 1

    return {
        # These are used for reconstruction at the end of the pipeline
        "ligand_atype": atom_types,
        "ligand_charge": atom_charges,
        "ligand_chirality": atom_chiralities,
        "ligand_bonds": torch.tensor(ligand_bonds, dtype=torch.int64),
        # these are the actual features
        "ligand_target_feat": ligand_target_feat.float(),
        "ligand_bonds_feat": ligand_bonds_feat.float(),
    }

