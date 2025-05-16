import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import csv
csv.field_size_limit(sys.maxsize)

from typing import List
from tqdm import tqdm
from dataset import dataset_mapping, Dataset
from model import model_mapping, Model

def get_functions(
    idx: int, 
    dataset: Dataset, 
    models: List[Model],
    n: int,
    n_retries: int = 5
):
    prompt = dataset.get_function_prompt(idx)
    completions = {}
    for model in models:
        model_completions = []
        retries = 0
        while len(model_completions) < n // len(models) and retries < n_retries:
            new_completions = model.generate_function_completions(prompt, n // len(models))
            model_completions += new_completions
            print(f"Generated {len(new_completions)} functions")
            retries += 1
        completions[model.model] = model_completions
    
    return completions


def main(config):
    dataset = dataset_mapping(config["dataset"])
    models = [model_mapping(model["name"], model["temperature"]) for model in config["models"]]
    if "updated_prompts" in config:
        dataset.update_prompts(config["updated_prompts"])
    if "skip_unupdated" in config:
        dataset.skip_unupdated = True

    fieldnames = ["idx", "completions"]

    processed = set()
    if "reset" not in config or not config["reset"]:
        if os.path.isfile(config["function_file"]):
            with open(config["function_file"], mode="r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    processed.add(int(row["idx"]))
    else:
        open(config['function_file'], 'w').close()

    with tqdm(total=len(dataset), desc=f"Processing {config['dataset']}") as pbar:
        for idx in range(len(dataset)):
            if dataset.skip(idx):
                pbar.update(1)
                continue

            if idx in processed:
                pbar.update(1)
                continue

            completions = get_functions(idx, dataset, models, config['n'])
            result = {
                "idx": idx,
                "completions": completions
            }
            
            with open(config["function_file"], mode="a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(result)

            pbar.update(1)
    
    print("\n\nFINISHED PROCESSING")
    print(f"RESULTS SAVED TO {config['function_file']}")
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
