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
from openfold.np import residue_constants, protein

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
