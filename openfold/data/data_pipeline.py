# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, MutableMapping
import numpy as np
import torch
from torch import nn

from openfold.np import residue_constants, protein
from rdkit import Chem

FeatureDict = MutableMapping[str, np.ndarray]


def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=object
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=object
    )
    return features


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]]
        for i in range(len(aatype))
    ])


def make_protein_features(
    protein_object: protein.Protein,
    description: str,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)

    return pdb_feats


def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description
    )

    return pdb_feats


POSSIBLE_ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "other"]
def get_atom_features(atom: Chem.Atom):
    if atom.GetSymbol() in POSSIBLE_ATOM_TYPES:
        atom_type = POSSIBLE_ATOM_TYPES.index(atom.GetSymbol())
    else:
        atom_type = len(POSSIBLE_ATOM_TYPES) - 1  # Other

    return {"atom_type": atom_type}


POSSIBLE_BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                       Chem.rdchem.BondType.AROMATIC, "other"]
def get_bond_features(bond: Chem.Bond):
    chem_bond_type = bond.GetBondType()
    if chem_bond_type in POSSIBLE_BOND_TYPES:
        bond_type = POSSIBLE_BOND_TYPES.index(chem_bond_type)
    else:
        print("Unknown bond type", chem_bond_type)
        bond_type = len(POSSIBLE_BOND_TYPES) - 1  # Other

    return {"bond_type": bond_type}


class DataPipeline:
    """Assembles input features."""
    def __init__(
        self,
    ):
        pass

    def process_pdb(
        self,
        pdb_path: str,
        chain_id: Optional[str] = None,
        _structure_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file.
        """
        if(_structure_index is not None):
            db_dir = os.path.dirname(pdb_path)
            db = _structure_index["db"]
            db_path = os.path.join(db_dir, db)
            fp = open(db_path, "rb")
            _, offset, length = _structure_index["files"][0]
            fp.seek(offset)
            pdb_str = fp.read(length).decode("utf-8")
            fp.close()
        else:
            with open(pdb_path, 'r') as f:
                pdb_str = f.read()

        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features(
            protein_object,
            description,
        )

        return {**pdb_feats}

    def process_smiles(self, smiles: str) -> FeatureDict:
        ligand = Chem.MolFromSmiles(smiles)

        # Add ligand atoms
        atoms_features = []
        atom_idx_to_atom_pos_idx = {}
        for atom in ligand.GetAtoms():
            atom_idx_to_atom_pos_idx[atom.GetIdx()] = len(atoms_features)
            atoms_features.append(get_atom_features(atom))

        atom_types = torch.tensor(np.array([atom["atom_type"] for atom in atoms_features], dtype=np.int64))
        atom_types_one_hot = nn.functional.one_hot(atom_types, num_classes=len(POSSIBLE_ATOM_TYPES), )

        ligand_bonds_feat = torch.zeros((len(atoms_features), len(atoms_features), len(POSSIBLE_BOND_TYPES)))
        for bond in ligand.GetBonds():
            atom1_idx = atom_idx_to_atom_pos_idx[bond.GetBeginAtomIdx()]
            atom2_idx = atom_idx_to_atom_pos_idx[bond.GetEndAtomIdx()]
            bond_features = get_bond_features(bond)
            ligand_bonds_feat[atom1_idx, atom2_idx, bond_features["bond_type"]] = 1

        return {
            "ligand_target_feat": atom_types_one_hot.float(),
            "ligand_bonds_feat": ligand_bonds_feat.float(),
        }

    def process_mol2(self, mol2_path: str) -> FeatureDict:
        """
            Assembles features for a ligand in a mol2 file.
        """
        ligand = Chem.MolFromMol2File(mol2_path)
        assert ligand is not None, f"Failed to parse ligand from {mol2_path}"

        conf = ligand.GetConformer()
        positions = torch.tensor(conf.GetPositions())

        return {
            **self.process_smiles(Chem.MolToSmiles(ligand)),
            "gt_ligand_positions": positions.float()
        }


