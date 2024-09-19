import json
import os
import sys
from typing import Optional

import Bio.PDB
import Bio.SeqUtils
import numpy as np
import rdkit.Chem.rdMolAlign
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D

from run_pretrained_model import run_on_folder
from env_consts import AFFINITY_TEST_JSONS, AFFINITY_TEST_OUTPUT


def main(config_path):
    i = 0
    while os.path.exists(os.path.join(AFFINITY_TEST_OUTPUT, f"output_{i}")):
        i += 1
    output_dir = os.path.join(AFFINITY_TEST_OUTPUT, f"output_{i}")
    os.makedirs(output_dir, exist_ok=True)
    run_on_folder(AFFINITY_TEST_JSONS, output_dir, config_path, long_sequence_inference=True)
    # output_dir = os.path.join(AFFINITY_TEST_OUTPUT, f"output_13")

    jobnames = [filename.split("_predicted_protein.pdb")[0]
                for filename in os.listdir(os.path.join(output_dir, "predictions"))
                if filename.endswith("_predicted_protein.pdb") and "relaxed" not in filename]

    print(f"Analyzing {len(jobnames)} jobs...")

    all_affinity_data = {}
    for jobname in sorted(jobnames):
        # gt_protein_path = os.path.join(POSEBUSTERS_GT, jobname, f"{jobname}_protein.pdb")
        # gt_ligand_path = os.path.join(POSEBUSTERS_GT, jobname, f"{jobname}_ligand.sdf")
        pred_protein_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_protein.pdb")
        pred_affinity_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_affinity.json")
        pred_ligand_path = os.path.join(output_dir, "predictions", f"{jobname}_predicted_ligand.sdf")
        input_json = os.path.join(AFFINITY_TEST_JSONS, f"{jobname}.json")
        gt_affinity = json.load(open(input_json, "r"))["affinity"]

        pred_affinity_data = json.load(open(pred_affinity_path, "r"))
        affinity_2d, affinity_cls, affinity_1d = pred_affinity_data["affinity_2d"], \
                                                 pred_affinity_data["affinity_cls"], pred_affinity_data["affinity_1d"]
        affinity_2d_max, affinity_cls_max, affinity_1d_max = pred_affinity_data["affinity_2d_max"], \
                                                             pred_affinity_data["affinity_cls_max"], \
                                                             pred_affinity_data["affinity_1d_max"]


        all_affinity_data[jobname] = {"gt_affinity": gt_affinity, "pred_affinity_2d": affinity_2d,
                                      "pred_affinity_cls": affinity_cls, "pred_affinity_1d": affinity_1d,
                                      "pred_affinity_2d_max": affinity_2d_max, "pred_affinity_cls_max": affinity_cls_max,
                                      "pred_affinity_1d_max": affinity_1d_max}

    open(os.path.join(output_dir, "all_affinity_data.json"), "w").write(json.dumps(all_affinity_data, indent=4))

    # 2d affinity dists and rmse
    gt_affinities = np.array([data["gt_affinity"] for data in all_affinity_data.values()])
    pred_affinities_2d = np.array([data["pred_affinity_2d"] for data in all_affinity_data.values()])
    pred_affinities_cls = np.array([data["pred_affinity_cls"] for data in all_affinity_data.values()])
    pred_affinities_1d = np.array([data["pred_affinity_1d"] for data in all_affinity_data.values()])

    pred_affinities_avg = (pred_affinities_2d + pred_affinities_cls + pred_affinities_1d) / 3
    pred_affinities_avg_2d_cls = (pred_affinities_2d + pred_affinities_cls) / 2

    affinity_2d_rmse = np.sqrt(np.mean((gt_affinities - pred_affinities_2d) ** 2))
    affinity_cls_rmse = np.sqrt(np.mean((gt_affinities - pred_affinities_cls) ** 2))
    affinity_1d_rmse = np.sqrt(np.mean((gt_affinities - pred_affinities_1d) ** 2))

    affinity_avg_rmse = np.sqrt(np.mean((gt_affinities - pred_affinities_avg) ** 2))
    affinity_avg_2d_cls_rmse = np.sqrt(np.mean((gt_affinities - pred_affinities_avg_2d_cls) ** 2))

    print(all_affinity_data)
    print("2d average distance", np.mean(np.abs(gt_affinities - pred_affinities_2d)))
    print("2d pearson", np.corrcoef(gt_affinities, pred_affinities_2d)[0, 1])
    print(f"2d affinity RMSE: {affinity_2d_rmse}")

    print("cls average distance", np.mean(np.abs(gt_affinities - pred_affinities_cls)))
    print("cls pearson", np.corrcoef(gt_affinities, pred_affinities_cls)[0, 1])
    print(f"cls affinity RMSE: {affinity_cls_rmse}")

    print("1d average distance", np.mean(np.abs(gt_affinities - pred_affinities_1d)))
    print("1d pearson", np.corrcoef(gt_affinities, pred_affinities_1d)[0, 1])
    print(f"1d affinity RMSE: {affinity_1d_rmse}")
    print(f"avg affinity RMSE: {affinity_avg_rmse}")
    print(f"avg 2d cls affinity RMSE: {affinity_avg_2d_cls_rmse}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run_config.json")
    main(config_path)
