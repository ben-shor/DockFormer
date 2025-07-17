"""
before running this script
(1) download the following files and save them in data/plinder/raw:
https://storage.googleapis.com/plinder/2024-06/v2/index/annotation_table.parquet
https://storage.googleapis.com/plinder/2024-06/v2/splits/split.parquet
https://console.cloud.google.com/storage/browser/_details/plinder/2024-06/v2/links/kind%3Dapo/links.parquet
https://console.cloud.google.com/storage/browser/_details/plinder/2024-06/v2/links/kind%3Dpred/links.parquet

(2) download the following zip and extract them to data/plinder/raw
https://storage.googleapis.com/plinder/2024-06/v2/linked_structures/apo.zip
https://storage.googleapis.com/plinder/2024-06/v2/linked_structures/pred.zip

(3) install GSUtil and put the folder google-cloud-sdk in data/plinder/raw
https://cloud.google.com/sdk/docs/install#linux

(4) install pandas and pyarrow on your python environment
pip install pandas pyarrow
"""
import json
import os
import shutil
import rdkit
from collections import defaultdict
from typing import Optional

import Bio.PDB
import Bio.SeqIO
import Bio.SeqUtils
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

from utils import renumber_and_remove_non_canonical_chains_and_residues, validate_mol, get_pdb_model, \
    generate_shared_gt_and_apo, get_sequence_from_pdb, get_esm_structure, create_conformers

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                        "data")

RAW_PATH = os.path.join(DATA_DIR, "plinder", "raw")
PLINDER_ANNOTATIONS = os.path.join(RAW_PATH, "2024-06_v2_index_annotation_table.parquet")
PLINDER_SPLITS = os.path.join(RAW_PATH, "2024-06_v2_splits_split.parquet")
PLINDER_LINKED_PRED_MAP = os.path.join(RAW_PATH, "2024-06_v2_links_kind=pred_links.parquet")
PLINDER_LINKED_APO_MAP = os.path.join(RAW_PATH, "2024-06_v2_links_kind=apo_links.parquet")

PLINDER_LINKED_PRED_PATH = os.path.join(RAW_PATH, "pred")
PLINDER_LINKED_APO_PATH = os.path.join(RAW_PATH, "apo")

GSUTIL_PATH = os.path.join(RAW_PATH, "google-cloud-sdk", "bin", "gsutil")

BASE_OUTPUT_FOLDER = os.path.join(DATA_DIR, "plinder", "processed")
MODELS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "models")
TRAIN_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "jsons_train")
VALIDATION_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "jsons_validation")
TEST_JSONS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "jsons_test")

SYSTEMS_PICKLE = os.path.join(BASE_OUTPUT_FOLDER, "systems.pkl")


def print_systems(name, systems):
    print(f"{name:<40}",
          f"train systems size: {len(systems):<6}",
          f"unique: {systems['system_id'].nunique():<6}",
          f"unique clusters: {systems['cluster'].nunique():<6}",
          f"unique proteins: {systems['system_pocket_UniProt'].nunique():<6}",
          f"with apo or pred: {(systems['linked_pred_id'].notna() | systems['linked_apo_id'].notna()).sum():<6}",
          f"with affinity: {systems['ligand_binding_affinity'].notna().sum():<6}",
          )


def get_systems():
    systems = pd.read_parquet(PLINDER_ANNOTATIONS,
                              columns=['system_id', 'entry_pdb_id', 'ligand_binding_affinity', 'entry_release_date',
                                       'system_pocket_UniProt', 'entry_resolution', 'system_num_protein_chains',
                                       'system_num_ligand_chains', 'system_proper_unique_ccd_codes',
                                       'ligand_unique_ccd_code', 'ligand_smiles', 'ligand_ccd_code',
                                       'ligand_num_heavy_atoms',  # 'fake_name_for_error_with_all_columns',
                                       ])
    print("Initial total systems:", len(systems))

    splits = pd.read_parquet(PLINDER_SPLITS, columns=['system_id', 'split', 'cluster',
                                                      'system_pass_validation_criteria', 'system_has_binding_affinity',
                                                      'system_has_apo_or_pred'])
    systems = pd.merge(systems, splits, on='system_id', how='inner')
    print(f"Total with splits: {len(systems)}")
    # print("\n------", systems["split"].value_counts(), "\n-------")

    systems = systems[systems['split'].isin(['train', 'val', 'test'])]
    print("Has valid splits (train/test/val):", len(systems))

    linked_pred = pd.read_parquet(PLINDER_LINKED_PRED_MAP)
    linked_pred_agg = linked_pred.groupby('reference_system_id')['id'].apply(list).reset_index()
    systems = pd.merge(systems, linked_pred_agg[['reference_system_id', 'id']],
                       left_on='system_id', right_on='reference_system_id',
                       how='left')
    systems.rename(columns={'id': 'linked_pred_id'}, inplace=True)

    # Merge the result with linked_apo on the same condition
    linked_apo = pd.read_parquet(PLINDER_LINKED_APO_MAP)
    linked_apo_agg = linked_apo.groupby('reference_system_id')['id'].apply(list).reset_index()
    systems = pd.merge(systems, linked_apo_agg[['reference_system_id', 'id']],
                       left_on='system_id', right_on='reference_system_id',
                       how='left')
    systems.rename(columns={'id': 'linked_apo_id'}, inplace=True)

    # Drop the reference_system_id columns that were added during the two merges
    systems.drop(columns=['reference_system_id_x', 'reference_system_id_y'], inplace=True)

    print("---- Filters:")
    # only filter train systems
    non_train_systems = systems[systems['split'] != 'train']
    systems = systems[systems['split'] == 'train']

    filters = systems['system_id'].notna()

    # filters = systems["system_pass_validation_criteria"]
    # print("pass validation", filters.sum(), "/", len(systems))

    filters &= (systems['linked_pred_id'].notna() | systems['linked_apo_id'].notna())
    print("only with linked structures filter", filters.sum())

    filters |= systems['ligand_binding_affinity'].notna()
    print("add all with affinity", filters.sum())

    filters &= (systems["ligand_num_heavy_atoms"] <= 50) & (systems["ligand_num_heavy_atoms"] >= 5)
    print("ligand size filter", filters.sum())
    filters &= systems['system_pocket_UniProt'].notna()
    print("has uniprot id filter", filters.sum())
    systems = systems[filters]

    aff_agg = systems[systems['ligand_binding_affinity'].notna()].groupby(
        ['system_pocket_UniProt', 'ligand_ccd_code'])[
        "ligand_binding_affinity"].apply(list).reset_index()
    print("unique affinities per uniprot+ligand:", aff_agg.shape[0])
    # Debug option - make sure all affinities are the same if same uniprot and ligand by seeing std 0 or very small
    # aff_agg['affinity_std'] = aff_agg['ligand_binding_affinity'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    # print("Affinity std value counts:", aff_agg['affinity_std'].value_counts())

    print_systems("Before train filters", systems)

    # -- Filter out multiple positions for same ligand on same protein, as this prevents the model from learning
    # How: If there are two samples with same protein & ligand in different clusters, remove the one from the smaller
    # cluster, as it is likely that this is the less common position for the ligand
    cluster_sizes = systems['cluster'].value_counts().to_dict()
    systems = systems.copy()  # prevent SettingWithCopyWarning
    systems['cluster_size'] = systems['cluster'].map(cluster_sizes)
    systems_sorted = systems.sort_values(by='cluster_size', ascending=False)
    systems = systems_sorted.drop_duplicates(subset=['system_pocket_UniProt', 'ligand_ccd_code'])

    print_systems("After deduplication of uniprot+ligand", systems)

    systems_sorted = systems.sort_values(by=['ligand_ccd_code', 'cluster_size'])
    systems = systems_sorted.groupby('ligand_ccd_code').head(20).reset_index(drop=True)

    print_systems("After limiting ligand count to 20", systems)

    # -- keep only one row per system_id, because in later steps we will add all ligands  in the system anyway
    systems = systems.drop_duplicates(subset=['system_id'])

    systems = pd.concat([systems, non_train_systems], ignore_index=True)
    print_systems("After deduplication of system_id", systems)

    # optional debug prints
    # print(systems["cluster"].value_counts())
    # print(systems["ligand_ccd_code"].value_counts())
    # print(systems["system_pocket_UniProt"].value_counts())

    systems['_bucket_id'] = systems['entry_pdb_id'].str[1:3]

    return systems


def generate_gt_and_apo_structure(row, current_gt_path: str, tmp_apo_path: str, output_gt_path: str,
                                  output_apo_path: str) -> str:
    if row["system_num_protein_chains"] == 1:
        if isinstance(row["linked_apo_id"], list):
            for template_name in row['linked_apo_id']:
                apo_pdb_path = os.path.join(PLINDER_LINKED_APO_PATH, f"{template_name}.cif")
                if os.path.exists(apo_pdb_path):
                    renumber_and_remove_non_canonical_chains_and_residues(apo_pdb_path, tmp_apo_path)
                    if generate_shared_gt_and_apo(current_gt_path, tmp_apo_path, output_gt_path, output_apo_path):
                        return "apo"
                    else:
                        print("Failed to generate shared apo structure from apo", template_name)
                else:
                    print("Linked apo structure not found", template_name)

        if isinstance(row["linked_pred_id"], list):
            for template_name in row['linked_pred_id']:
                apo_pdb_path = os.path.join(PLINDER_LINKED_PRED_PATH, f"{template_name}.cif")
                if os.path.exists(apo_pdb_path):
                    renumber_and_remove_non_canonical_chains_and_residues(apo_pdb_path, tmp_apo_path)
                    if generate_shared_gt_and_apo(current_gt_path, tmp_apo_path, output_gt_path, output_apo_path):
                        return "pred"
                    else:
                        print("**** ERROR to generate shared apo structure from pred", template_name)
                else:
                    print("**** ERROR Linked pred structure not found", template_name)

        sequence = get_sequence_from_pdb(current_gt_path)
        if len(sequence) <= 400:
            esm_model_text = get_esm_structure(sequence)
            if esm_model_text is not None:
                open(tmp_apo_path, "w").write(esm_model_text)
                if generate_shared_gt_and_apo(current_gt_path, tmp_apo_path, output_gt_path, output_apo_path):
                    return "esm"

    shutil.copyfile(current_gt_path, output_gt_path)
    shutil.copyfile(current_gt_path, output_apo_path)
    return "gt"


def prepare_system(row, system_folder, output_models_folder, output_jsons_folder, should_overwrite=False):
    output_json_path = os.path.join(output_jsons_folder, f"{row['system_id']}.json")
    os.makedirs(output_models_folder, exist_ok=True)

    if os.path.exists(output_json_path) and not should_overwrite:
        return "Already exists"

    plinder_gt_pdb_path = os.path.join(system_folder, f"receptor.pdb")
    plinder_gt_ligands_folder = os.path.join(system_folder, "ligand_files")
    plinder_gt_ligand_paths = [os.path.join(plinder_gt_ligands_folder, i) for i in os.listdir(plinder_gt_ligands_folder)
                               if i.endswith(".sdf")]

    gt_output_path = os.path.join(output_models_folder, f"gt_protein.pdb")
    tmp_gt_pdb_path = os.path.join(output_models_folder, f"tmp_gt_protein.pdb")
    gt_pocket_output_path = os.path.join(output_models_folder, f"gt_pocket.pdb")
    protein_input_path = os.path.join(output_models_folder, f"input_protein.pdb")
    tmp_input_path = os.path.join(output_models_folder, f"tmp_input_protein.pdb")
    pocket_input_path = os.path.join(output_models_folder, f"input_pocket.pdb")
    gt_ligand_paths = []
    ref_ligand_paths = []

    for plinder_ligand in plinder_gt_ligand_paths:
        ligand = Chem.MolFromMolFile(plinder_ligand)
        if ligand is None or not validate_mol(ligand):
            print("Failed to load ligand", plinder_ligand)
            continue
        gt_ligand_path = os.path.join(output_models_folder, f"gt_ligand_{len(gt_ligand_paths)}.sdf")
        ref_ligand_path = os.path.join(output_models_folder, f"ref_ligand_{len(gt_ligand_paths)}.sdf")
        writer = Chem.SDWriter(gt_ligand_path)
        writer.write(ligand)
        writer.close()

        smiles = Chem.MolToSmiles(ligand)
        create_conformers(smiles, ref_ligand_path, num_conformers=1, multiplier_samples=5)
        ref_ligand = Chem.MolFromMolFile(ref_ligand_path)
        if ref_ligand is None:
            print("Failed to create ref ligand", ref_ligand_path)
            os.remove(gt_ligand_path)
            continue

        gt_ligand_paths.append(gt_ligand_path)
        ref_ligand_paths.append(ref_ligand_path)

    if not gt_ligand_paths:
        return "No valid ligands"

    if not os.path.exists(plinder_gt_pdb_path):
        return "No receptor"
    renumber_and_remove_non_canonical_chains_and_residues(plinder_gt_pdb_path, tmp_gt_pdb_path)

    input_source = generate_gt_and_apo_structure(row, tmp_gt_pdb_path, tmp_input_path, gt_output_path, protein_input_path)
    if os.path.exists(tmp_gt_pdb_path):
        os.remove(tmp_gt_pdb_path)
    if os.path.exists(tmp_input_path):
        os.remove(tmp_input_path)
    print("created input structure", input_source)

    affinity = row["ligand_binding_affinity"]
    if not pd.notna(affinity):
        affinity = None

    relative_models_folder_name = os.path.basename(output_models_folder)
    json_data = {
        "input_structure": os.path.join(relative_models_folder_name, "input_protein.pdb"),
        "gt_structure": os.path.join(relative_models_folder_name, "gt_protein.pdb"),
        "gt_sdf_list": gt_ligand_paths,
        "resolution": row.fillna(99)["entry_resolution"],
        "release_year": row["entry_release_date"],
        "affinity": affinity,
        "uniprot": row["system_pocket_UniProt"],
        "cluster": row["cluster"],
        "cluster_size": row["cluster_size"],
        "pdb_id": row["system_id"],
        "input_source": input_source,
    }
    open(output_json_path, "w").write(json.dumps(json_data, indent=4))

    return "success"


def is_system_done(system_id):
    return os.path.exists(os.path.join(TRAIN_JSONS_FOLDER, f"{system_id}.json")) \
           or os.path.exists(os.path.join(VALIDATION_JSONS_FOLDER, f"{system_id}.json")) \
           or os.path.exists(os.path.join(TEST_JSONS_FOLDER, f"{system_id}.json"))


def main():
    status_dict = defaultdict(list)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_JSONS_FOLDER, exist_ok=True)
    os.makedirs(VALIDATION_JSONS_FOLDER, exist_ok=True)
    os.makedirs(TEST_JSONS_FOLDER, exist_ok=True)

    if not os.path.exists(SYSTEMS_PICKLE):
        systems = get_systems()
        systems.to_pickle(SYSTEMS_PICKLE)
    systems = pd.read_pickle(SYSTEMS_PICKLE)
    split_to_folder = {"train": TRAIN_JSONS_FOLDER, "val": VALIDATION_JSONS_FOLDER, "test": TEST_JSONS_FOLDER}


    # s = systems[systems["linked_apo_id"].notna() | systems["linked_pred_id"].notna()]
    # print(s["system_num_protein_chains"].value_counts())
    # print(s[s["system_num_protein_chains"] == 2][["system_id", "linked_apo_id", "ligand_binding_affinity", "split"]])
    # print(s[s["system_num_protein_chains"] == 2][["system_id", "ligand_binding_affinity", "split"]])
    # print(s[s["system_num_protein_chains"] == 2]["split"].value_counts())

    for bucket_id, bucket_systems in systems.groupby('_bucket_id', sort=True):
        systems_done = [is_system_done(row['system_id']) for _, row in bucket_systems.iterrows()]
        if all(systems_done):
            print("All systems in bucket", bucket_id, "are already done, skipping")
            continue
        print("Starting bucket", bucket_id, len(bucket_systems))


        tmp_output_models_folder = os.path.join(RAW_PATH, f"tmp_{bucket_id}")
        os.makedirs(tmp_output_models_folder, exist_ok=True)
        os.system(
            f'{GSUTIL_PATH} -q -m cp -r "gs://plinder/2024-06/v2/systems/{bucket_id}.zip" {tmp_output_models_folder}')
        systems_folder = os.path.join(tmp_output_models_folder, "systems")
        bucket_zip_path = os.path.join(tmp_output_models_folder, f"{bucket_id}.zip")
        # os.system(f'unzip -o {os.path.join(tmp_output_models_folder, f"{bucket_id}.zip")} -d {systems_folder}')

        for i, row in bucket_systems.iterrows():
            # if not str(row['system_id']).startswith("4z22__1__1.A__1.C"):
            #     continue
            system_id = row['system_id']
            print("doing", system_id, row["system_num_protein_chains"], row["system_num_ligand_chains"])

            os.system(f'unzip -q -o {bucket_zip_path} "{system_id}/*" -d {systems_folder}')
            raw_system_folder = os.path.join(systems_folder, system_id)
            assert os.path.exists(raw_system_folder), f"System folder {raw_system_folder} does not exist"

            try:
                success = prepare_system(row, raw_system_folder,
                                         output_models_folder=os.path.join(MODELS_FOLDER, system_id),
                                         output_jsons_folder=split_to_folder[row["split"]],
                                         should_overwrite=False)
                print("done", row['system_id'], success, flush=True)
                status_dict[success].append(row['system_id'])
            except Exception as e:
                raise
                print("Failed", row['system_id'], e, flush=True)
                status_dict["error_" + str(e)].append(row['system_id'])

        shutil.rmtree(tmp_output_models_folder)

    print(status_dict)
    print("counts", {k: len(v) for k, v in status_dict.items()})


if __name__ == "__main__":
    main()




