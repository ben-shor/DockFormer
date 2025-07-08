import dataclasses
import json
from functools import lru_cache
from typing import List, Dict
import os

import Bio.PDB
import Bio.SeqIO
import Bio.SeqUtils
from Bio import pairwise2, PDB


@lru_cache(5)
def get_pdb_model_readonly(pdb_path: str):
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    pdb_struct = pdb_parser.get_structure("original_pdb", pdb_path)
    assert len(list(pdb_struct)) == 1, f"Not one model ({len(list(pdb_struct))})! {pdb_path}"
    return next(iter(pdb_struct))


def get_pdb_model(pdb_path: str):
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    pdb_struct = pdb_parser.get_structure("original_pdb", pdb_path)
    assert len(list(pdb_struct)) == 1, f"Not one model ({len(list(pdb_struct))})! {pdb_path}"
    return next(iter(pdb_struct))


def update_metadata(metadata_path: str, data: dict):
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as f:
            f.write("{}\n")
    current_data = json.load(open(metadata_path, "r"))
    current_data.update(data)
    with open(metadata_path, "w") as f:
        json.dump(current_data, f, indent=4)


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


def get_sequence_from_pdb(pdb_path: str, chain_id: str = "A") -> str:
    model = get_pdb_model_readonly(pdb_path)
    return get_chain_object_to_seq(model[chain_id])


def get_all_sequences_from_pdb(pdb_path: str) -> Dict[str, str]:
    model = get_pdb_model_readonly(pdb_path)
    chain_to_seq = {}
    for chain in model.get_chains():
        seq = get_chain_object_to_seq(chain)
        if seq:
            chain_to_seq[chain.id] = seq
    return chain_to_seq


def generate_af_input_gt_in_af(gt_path: str, af_path: str, output_path: str):
    gt_model = get_pdb_model(gt_path)
    af_model = get_pdb_model(af_path)

    gt_chain = gt_model["A"]
    af_chain = af_model["A"]

    # Convert chains to sequences
    gt_seq = get_chain_object_to_seq(gt_chain).replace("X", "")
    af_seq = get_chain_object_to_seq(af_chain).replace("X", "")

    # Perform global alignment (match/mismatch/gap scores can be tuned)
    alignments = pairwise2.align.globalms(gt_seq, af_seq, 2, -10000, -1, 0)

    if not alignments:
        print("No alignment found.")
        return False

    best_alignment = alignments[0]
    aligned_gt, aligned_af, score, start, end = best_alignment

    if "-" in aligned_af:
        print("Alignment contains gaps in AF sequence, which is not allowed.")
        return False

    # For debugging, optional
    # print(format_alignment(*best_alignment))

    gt_residues = list(gt_chain.get_residues())
    af_residues = list(af_chain.get_residues())

    gt_index, af_index = 0, 0
    new_chain = PDB.Chain.Chain("A")

    for a_gt, a_af in zip(aligned_gt, aligned_af):
        if a_gt == "-":
            af_index += 1
            continue
        gt_res = gt_residues[gt_index]
        af_res = af_residues[af_index]

        if gt_res.get_resname() != af_res.get_resname():
            print(f"Residue mismatch at {gt_res.id[1]}: GT {gt_res.get_resname()} != AF {af_res.get_resname()}")
            return False

        af_chain.detach_child(af_res.id)  # remove from original AF chain
        af_res.id = (" ", gt_res.id[1], " ")  # reset to GT's numbering
        new_chain.add(af_res)

        gt_index += 1
        af_index += 1

    # Write to output
    structure = Bio.PDB.Structure.Structure("X")
    model = Bio.PDB.Model.Model(0)
    structure.add(model)
    model.add(new_chain)

    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_path)

    return True


def merge_to_single_chain(pdb_path: str, output_path: str):
    pdb_model = get_pdb_model(pdb_path)
    chains = [c for c in pdb_model if c.id not in ("", " ")]
    if len(pdb_model) != 1:
        print(f"multiple chains in model ({len(pdb_model)})")
    gt_chain = Bio.PDB.Chain.Chain("A")
    factor_res_id = 0
    for chain in chains:
        all_res = list(chain.get_residues())
        for res in all_res:
            chain.detach_child(res.id)
        max_new_id = 0
        for res in all_res:
            res.id = (" ", res.id[1] + factor_res_id, " ")
            max_new_id = max(max_new_id, res.id[1])
            gt_chain.add(res)
        factor_res_id = max_new_id + 32

    io = Bio.PDB.PDBIO()
    io.set_structure(gt_chain)
    io.save(output_path)