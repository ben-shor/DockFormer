"""
Before running this script, go to https://predictioncenter.org/download_area/CASP16/targets/pharma_ligands/
download all files LX000_exper_affinity.csv,LX000_exper_struct.tar.gz and extract the tar.gz files
and save them in `data/casp16/raw`
also run AF2 prediction on the sequence of L1000/L3000 and save the files in "data/casp16/raw/LX000_apo.pdb"
"""
import os
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import Bio.PDB, Bio.SeqUtils
from typing import Optional
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

CASP16_PATH = os.path.join(DATA_DIR, "casp16", "raw")
BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "casp16", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, f"jsons")


def reconstruct_mol(mol):
    new_mol = Chem.RWMol()

    # Add atoms
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        new_atom = Chem.Atom(symbol)
        new_atom.SetChiralTag(atom.GetChiralTag())
        new_atom.SetFormalCharge(atom.GetFormalCharge())

        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        new_mol.AddBond(atom1, atom2, bond_type)

    return new_mol


def is_same(mol1, mol2):
    return mol1.GetNumAtoms() == mol2.GetNumAtoms() \
           and mol1.GetSubstructMatch(mol2) == tuple(range(mol1.GetNumAtoms())) \
           and mol2.GetSubstructMatch(mol1) == tuple(range(mol2.GetNumAtoms()))


def validate_mol(mol2_obj):
    # First validate that the smiles can create some molecule
    mol2_obj = Chem.MolFromSmiles(Chem.MolToSmiles(mol2_obj))
    if mol2_obj is None:
        print("Failed to create ligand from smiles")
        return False

    # Then validate that only symbol+chirality+charge are needed to represent the molecule
    reconstructed = reconstruct_mol(mol2_obj)
    return is_same(mol2_obj, reconstructed)


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


def do_robust_chain_object_renumber(chain: Bio.PDB.Chain.Chain, new_chain_id: str) -> Optional[Bio.PDB.Chain.Chain]:
    all_residues = [res for res in chain.get_residues()
                    if "CA" in res and Bio.SeqUtils.seq1(res.get_resname()) not in ("X", "", " ")]
    if not all_residues:
        return None

    res_and_res_id = [(res, res.get_id()[1]) for res in all_residues]

    min_res_id = min([i[1] for i in res_and_res_id])
    if min_res_id < 1:
        print("Negative res id", chain, min_res_id)
        factor = -1 * min_res_id + 1
        res_and_res_id = [(res, res_id + factor) for res, res_id in res_and_res_id]

    res_and_res_id_no_collisions = []
    for res, res_id in res_and_res_id[::-1]:
        if res_and_res_id_no_collisions and res_and_res_id_no_collisions[-1][1] == res_id:
            # there is a collision, usually an insertion residue
            res_and_res_id_no_collisions = [(i, j + 1) for i, j in res_and_res_id_no_collisions]
        res_and_res_id_no_collisions.append((res, res_id))

    first_res_id = min([i[1] for i in res_and_res_id_no_collisions])
    factor = 1 - first_res_id  # start from 1
    new_chain = Bio.PDB.Chain.Chain(new_chain_id)

    res_and_res_id_no_collisions.sort(key=lambda x: x[1])

    for res, res_id in res_and_res_id_no_collisions:
        if res.id[2] == "P":
            print(f"Residue {res.id} in chain {chain.id} has weird insertion code, skipping")
            continue
        chain.detach_child(res.id)
        res.id = (" ", res_id + factor, " ")
        new_chain.add(res)

    return new_chain


def get_pdb_model(pdb_path: str):
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    pdb_struct = pdb_parser.get_structure("original_pdb", pdb_path)
    assert len(list(pdb_struct)) == 1, f"Not one model ({len(list(pdb_struct))})! {pdb_path}"
    return next(iter(pdb_struct))


def remove_non_canonical_chains_and_residues(pdb_path: str, output_path: str):
    gt_model = get_pdb_model(pdb_path)
    new_chain_ids = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"]

    new_model = Bio.PDB.Model.Model(0)

    for chain in gt_model.get_chains():
        new_chain = do_robust_chain_object_renumber(chain, new_chain_ids.pop(0))
        if new_chain is not None:
            new_model.add(new_chain)

    io = Bio.PDB.PDBIO()
    io.set_structure(new_model)
    io.save(output_path)


def main():
    # for lig_target in ["L1000", "L2000", "L3000", "L4000"]:
    for lig_target in ["L1000", "L3000"]:
        raw_input_main_folder = os.path.join(CASP16_PATH, f"{lig_target}_exper_struct")
        affinity_data_path = os.path.join(CASP16_PATH, f"{lig_target}_exper_affinity.csv")
        base_output_folder = os.path.join(BASE_OUTPUT_FOLDER, lig_target)
        models_output_folder = os.path.join(base_output_folder, "models")
        jsons_output_folder = os.path.join(base_output_folder, "jsons")
        os.makedirs(models_output_folder, exist_ok=True)
        os.makedirs(jsons_output_folder, exist_ok=True)

        apo_path = os.path.join(CASP16_PATH, f"{lig_target}_apo.pdb")
        shutil.copyfile(apo_path, os.path.join(models_output_folder, f"apo.pdb"))

        affininty_by_id = {}
        if os.path.exists(affinity_data_path):
            lines = open(affinity_data_path).read().split("\n")
            header = lines[0].split(",")
            name_ind = 0
            affinity_ind = header.index("binding_affinity")
            for line in lines[1:]:
                if not line:
                    continue
                parts = line.split(",")
                affininty_by_id[parts[name_ind]] = float(parts[affinity_ind]) * -1

        for model_tgz in sorted(os.listdir(raw_input_main_folder)):
            if not model_tgz.endswith(".tgz"):
                continue
            model_id = model_tgz.split(".")[0]
            # target_models_folder = os.path.join(raw_input_main_folder, model_id)
            # os.makedirs(target_models_folder)
            target_output_models_folder = os.path.join(models_output_folder, model_id)
            os.makedirs(target_output_models_folder, exist_ok=True)

            # extract tgz to temp dir
            temp_extract_dir = os.path.join(raw_input_main_folder, "temp_extract")
            os.makedirs(temp_extract_dir, exist_ok=True)
            print("running ", f"tar -xzf {os.path.join(raw_input_main_folder, model_tgz)} -C {temp_extract_dir}")
            os.system(f"tar -xzf {os.path.join(raw_input_main_folder, model_tgz)} -C {temp_extract_dir}")

            raw_models_folder = os.path.join(temp_extract_dir, f"{lig_target}_prepared", model_id)
            assert os.path.exists(raw_models_folder)
            files = os.listdir(raw_models_folder)
            pdb_files = [f for f in files if f.startswith("protein")]
            assert len(pdb_files) == 1
            ligand_files = sorted([f for f in files if f.startswith("ligand")])
            # if len(ligand_files) != 1:
            #     print("Many ligand files", model_id, ligand_files)
            #     ligand_files = ligand_files[:1]

            shutil.copyfile(os.path.join(raw_models_folder, pdb_files[0]),
                            os.path.join(target_output_models_folder, f"gt_protein.pdb"))
            smiles_list = []
            for i, ligand_file in enumerate(ligand_files):
                ligand_path = os.path.join(raw_models_folder, ligand_file)
                ligand_output_path = os.path.join(target_output_models_folder, f"gt_ligand_{i}.sdf")

                mol = Chem.MolFromPDBFile(ligand_path, removeHs=False)
                assert mol is not None, f"Could not load the molecule from {ligand_path}."
                writer = Chem.SDWriter(ligand_output_path)
                writer.write(mol)
                writer.close()

                smiles = Chem.MolToSmiles(mol)
                create_conformers(smiles, os.path.join(target_output_models_folder, f"ref_ligand_{i}.sdf"),
                                  num_conformers=1, multiplier_samples=1)
                smiles_list.append(smiles)

            shutil.rmtree(temp_extract_dir)

            models_folder_name = os.path.basename(MODELS_FOLDER)
            base_relative_path = os.path.join(models_folder_name, model_id)
            sample_data = {
                "input_structure": os.path.join(base_relative_path, f"apo.pdb"),
                "gt_structure": os.path.join(base_relative_path, f"gt_protein.pdb"),
                "gt_sdf_list": [os.path.join(base_relative_path, f"gt_ligand_{i}.sdf") for i in range(len(ligand_files))],
                "ref_sdf_list": [os.path.join(base_relative_path, f"ref_ligand_{i}.sdf") for i in range(len(ligand_files))],
                "input_smiles_list": smiles_list,
                "resolution": -1,
                "affinity": affininty_by_id.get(model_id),
                "pdb_id": model_id
            }

            json_path = os.path.join(jsons_output_folder, f"{model_id}.json")
            json.dump(sample_data, open(json_path, "w"), indent=4)

    print("done")


if __name__ == '__main__':
    main()
