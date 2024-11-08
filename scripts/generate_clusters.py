import json
import os
import sys


def generate_clusters_file(jsons_folder, output_file):
    clusters = {}
    jsons_folder_name = os.path.basename(jsons_folder)
    for json_filename in os.listdir(jsons_folder):
        if not json_filename.endswith(".json"):
            continue
        with open(os.path.join(jsons_folder, json_filename), "r") as json_file:
            json_data = json.load(json_file)
            rel_path = os.path.join(jsons_folder_name, json_filename)
            cluster_id = json_data["cluster"]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(rel_path)
    with open(output_file, "w") as output_file:
        json.dump(clusters, output_file)
    print("Number of clusters:", len(clusters))
    print("Number of jsons:", sum(len(cluster) for cluster in clusters.values()))


if __name__ == '__main__':
    # generate_clusters_file("/Users/benshor/Documents/Data/202401_pred_affinity/plinder/v2/processed/plinder_jsons_train",
    #                        "/Users/benshor/Documents/Data/202401_pred_affinity/plinder/v2/processed/cluster_train.json")
    # generate_clusters_file("/Users/benshor/Documents/Data/202401_pred_affinity/plinder/v2/processed/plinder_jsons_train_small",
    #                        "/Users/benshor/Documents/Data/202401_pred_affinity/plinder/v2/processed/cluster_train_small.json")
    if len(sys.argv) != 3:
        print("Usage: <script> <jsons_folder> <output_file_path>")
    else:
        generate_clusters_file(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
