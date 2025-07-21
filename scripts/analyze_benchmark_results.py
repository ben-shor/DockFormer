import json
import os
import sys
from typing import List, Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def pretty_print_result_files(results_files: Dict[str, str]):
    all_data = {}
    for name, file_path in results_files.items():
        data = json.load(open(file_path, "r"))
        ligand_rmsd = [i["ligand_rmsd"] for i in data.values()]
        all_data[name] = {
            "ligand_rmsd_count": len(ligand_rmsd),
            "ligand_rmsd_lt2": sum(1 for i in ligand_rmsd if i < 2.0),
            "ligand_rmsd_lt5": sum(1 for i in ligand_rmsd if i < 5.0),
            "ligand_rmsd_lt10": sum(1 for i in ligand_rmsd if i < 10.0),
        }
        gt_affinity = np.array([i["gt_affinity"] for i in data.values()])
        all_data[name]["gt_affinity_count"] = len([i for i in gt_affinity if i])
        if all_data[name]["gt_affinity_count"] > 0:
            for predicted_aff_name in ["affinity_2d", "affinity_cls"]:
                predicted_aff = [i[predicted_aff_name] for i, gt_aff in zip(data.values(), gt_affinity)
                                 if gt_aff is not None]
                predicted_aff = np.array(predicted_aff)

                all_data[name][f"{predicted_aff_name}_pearson"], _ = pearsonr(gt_affinity, predicted_aff)
                all_data[name][f"{predicted_aff_name}_spearman"], _ = spearmanr(gt_affinity, predicted_aff)
                all_data[name][f"{predicted_aff_name}_kendall"], _ = kendalltau(gt_affinity, predicted_aff)

                all_data[name][f"{predicted_aff_name}_rmse"] = np.sqrt(((gt_affinity - predicted_aff) ** 2).mean())

    for val_name in next(iter(all_data.values())).keys():
        print(f"------ {val_name:30s} -----")
        for name, data in all_data.items():
            name = name.replace("output_dockformer_", "")
            print(f"{name:30s}: {data[val_name]:>10.2f}")


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: python <script> <path_to_results_file_or_dir>"
    results_path = os.path.abspath(sys.argv[1])
    results_files = {}
    if os.path.isfile(results_path):
        results_files[os.path.basename(os.path.dirname(results_path))] = results_path
    elif os.path.isdir(results_path):
        for folder_name in os.listdir(results_path):
            folder_path = os.path.join(results_path, folder_name)
            file_path = os.path.join(folder_path, "rmsds.json")
            if os.path.exists(file_path):
                results_files[folder_name] = file_path
    else:
        raise ValueError(f"Invalid path: {results_path}")
    pretty_print_result_files(results_files)
