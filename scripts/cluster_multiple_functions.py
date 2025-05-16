import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import csv
import ast
import glob
csv.field_size_limit(sys.maxsize)

from typing import Dict
from tqdm import tqdm
from dataset import dataset_mapping, Dataset

def get_clusters(
    idx: int, 
    dataset: Dataset,
    functions: Dict,
):
    clusters = {}
    total_funcs = sum(len(c) for c in functions.values())
    count = 0

    for model in functions.keys():
        for func in functions[model]:
            res, out = dataset.test_absolute(idx, func)
            id = (res, out)
            if id not in clusters:
                clusters[id] = []
            clusters[id].append({
                'model': model,
                'function': func
            })

            print(f"{count} / {total_funcs}: {res}")
            count += 1

    return clusters


def main(config):
    dataset = dataset_mapping(config["dataset"])
    fieldnames = ["idx", "clusters"]

    # Skip processed
    processed = set()
    if "reset" not in config or not config["reset"]:
        if os.path.isfile(config["cluster_file"]):
            with open(config["cluster_file"], mode="r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    processed.add(int(row["idx"]))
    else:
        open(config['cluster_file'], 'w').close()

    # Get functions
    function_files = glob.glob(config['function_files']+'*')

    functions = {}
    for file in function_files:
        if os.path.isfile(file):
            with open(file, mode="r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    idx = int(row["idx"])
                    completions = ast.literal_eval(row["completions"])

                    if idx not in functions.keys():
                        functions[idx] = completions
                    else:
                        for key, value in completions.items():
                            if key in functions[idx]:
                                functions[idx][key] += value
                            else:
                                functions[idx][key] = value
        else:
            raise Exception(f'Function file {config["function_file"]} not found.')

    with tqdm(total=len(dataset), desc=f"Processing {config['dataset']}") as pbar:
        for idx in range(len(dataset)):
            if dataset.skip(idx):
                pbar.update(1)
                continue
            if idx in processed:
                pbar.update(1)
                continue
            if idx not in functions.keys():
                pbar.update(1)
                continue

            clusters = get_clusters(idx, dataset, functions[idx])
            result = {
                "idx": idx,
                "clusters": clusters
            }

            with open(config["cluster_file"], mode="a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(result)

            pbar.update(1)
    
    print("\n\nFINISHED PROCESSING")
    print(f"RESULTS SAVED TO {config['cluster_file']}")
    print(f"DATASET: {config['dataset']}")
    print(f"N: {config['n']}")
    print(f"MODELS:")
    for model in config['models']:
        print(f"   {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/test.json")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    main(config)
