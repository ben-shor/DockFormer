import numpy as np

from dockformer.data.utils import FeatureDict
from dockformer.utils import residue_constants, protein


def _make_sequence_features(sequence: str, description: str, num_res: int) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=object
    )
    # features["residue_index"] = np.array(range(num_res), dtype=np.int32)
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


def _make_protein_structure_features(protein_object: protein.Protein) -> FeatureDict:
    pdb_feats = {}

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)
    pdb_feats["in_chain_residue_index"] = protein_object.residue_index.astype(np.int32)

    gapped_res_indexes = []
    prev_chain_index = protein_object.chain_index[0]
    chain_start_res_ind = 0
    for relative_res_ind, chain_index in zip(protein_object.residue_index, protein_object.chain_index):
        if chain_index != prev_chain_index:
            chain_start_res_ind = gapped_res_indexes[-1] + 50
            prev_chain_index = chain_index
        gapped_res_indexes.append(relative_res_ind + chain_start_res_ind)

    pdb_feats["residue_index"] = np.array(gapped_res_indexes).astype(np.int32)
    pdb_feats["chain_index"] = np.array(protein_object.chain_index).astype(np.int32)
    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)

    return pdb_feats


def make_protein_features(protein_object: protein.Protein, description: str) -> FeatureDict:
    feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    feats.update(
        _make_sequence_features(sequence=sequence, description=description, num_res=len(protein_object.aatype))
    )

    feats.update(
        _make_protein_structure_features(protein_object=protein_object)
    )

    return feats
