import dataclasses
from functools import lru_cache
from typing import List

import Bio.PDB


@dataclasses.dataclass
class PdbBindDescriptor:
    pdb_id: str
    uniprot: str
    resolution: float
    release_year: int
    affinity: float


def get_all_descriptors(name_index_path: str, data_index_path: str) -> List[PdbBindDescriptor]:
    pdb_id_to_uniprot = {}
    for line in open(name_index_path, "r").read().split("\n"):
        if len(line) < 2 or line[0] == "#":
            continue
        split_line = line.split()
        pdb_id, uniprot = split_line[0], split_line[2]
        if "-" in uniprot:
            continue
        pdb_id_to_uniprot[pdb_id] = uniprot

    all_descriptors = []
    for line in open(data_index_path, "r").read().split("\n"):
        if len(line) < 2 or line[0] == "#":
            continue
        split_line = line.split()
        pdb_id = split_line[0]
        try:
            resolution = float(split_line[1])
        except ValueError:
            resolution = 50.0
            print("Could not parse resolution", pdb_id, split_line[1])
            # continue
        year = int(split_line[2])
        affinity = float(split_line[3])

        all_descriptors.append(PdbBindDescriptor(pdb_id, pdb_id_to_uniprot.get(pdb_id), resolution, year, affinity))
    return all_descriptors


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
