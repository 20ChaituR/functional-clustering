import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import csv
import ast
csv.field_size_limit(sys.maxsize)

from typing import List, Dict, Tuple
from itertools import product
from dataset import dataset_mapping
from model.chatgpt import ChatGPTModel

problems = {
    "X": [],
    "O": [207, 56, 206, 125, 150, 176, 86],
    "D": [212, 60, 47, 105, 95, 70, 154, 107, 12, 77, 158, 68],
    "C": [185, 137, 217, 71, 0, 67, 32, 89, 214, 213, 51, 131, 104, 145],
    "B": [64, 40, 209, 81, 8, 48, 99, 62, 163, 34],
    "A": [139, 215, 120, 135, 37, 162, 156, 190, 78, 216, 13, 218, 210, 184, 63, 29, 123],
    "T": [3, 5, 6, 18, 23, 26, 27, 33, 41, 44, 52, 57, 58, 61, 73, 75, 79, 80, 84, 87, 93, 98, 103, 109, 110, 112, 113, 116, 127, 140, 144, 146, 151, 161, 164, 167, 172, 174, 177, 179, 182, 183, 187, 189, 192, 194, 196, 198, 200, 203, 204, 205]
}

id_to_name = {
    "X": "Baseline",
    "O": "Out of Scope",
    "D": "Dumb Mistake",
    "C": "Hard Mistake (Missing Condition)",
    "B": "Hard Mistake (One Line Bug)",
    "A": "Hard Mistake (Wrong Algorithm)",
    "T": "Time Limit Exceeded"
}

def toPercent(count, total):
    return 100 * count / total

def toPercentStr(prob):
    return f"{100 * prob:.2f}%"

# Calculates the average probability of getting the right solution
# Essentially, the expected value for the accuracy of the model
def calculate_expected_accuracy(cluster_sizes: Dict[int, List[Tuple[str, int]]], dropped: List[int] = []):
    n_problems = 0
    total_prob = 0

    for idx, clusters in cluster_sizes.items():
        if idx in dropped:
            continue

        n_generated = sum(x[1] for x in clusters)
        n_ac = sum(x[1] for x in clusters if x[0] == 'AC')

        total_prob += n_ac / n_generated
        n_problems += 1
    
    return total_prob / n_problems

# Calculates the percentage of problems that the model is able to solve
# How many of the problems can the model generate even one correct solution for?
def calculate_maximum_accuracy(cluster_sizes: Dict[int, List[Tuple[str, int]]], dropped: List[int] = []):
    n_problems = 0
    n_solved = 0

    for idx, clusters in cluster_sizes.items():
        if idx in dropped:
            continue

        solved = any(x[0] == 'AC' for x in clusters)
        if solved:
            n_solved += 1
        n_problems += 1
    
    return n_solved / n_problems

# Calculates the percentage of problems where the largest cluster contains the correct solution
# Essentially, if I do clustering without any thresholding, what is the accuracy of the model?
def calculate_clustered_accuracy(cluster_sizes: Dict[int, List[Tuple[str, int]]], dropped: List[int] = []):
    n_problems = 0
    n_solved = 0

    for idx, clusters in cluster_sizes.items():
        if idx in dropped:
            continue

        if clusters[0][0] == 'AC':
            n_solved += 1
        n_problems += 1
    
    return n_solved / n_problems

# Calculates the accuracy after thresholding while trying to achieve a certain error rate
# This is absolute thresholding, so I will be looking at the largest cluster only to determine the threshold
# `max_error` should be a probability between 0 and 1. The thresholding will achieve an error rate of at most `max_error`.
def calculate_threshold_accuracy(cluster_sizes: Dict[int, List[Tuple[str, int]]], dropped: List[int] = [], max_error: float = 0.0):
    total_problems = len(cluster_sizes) - len(dropped)
    total_errors = int(max_error * total_problems)

    considered_cluster_sizes = [clusters for idx, clusters in cluster_sizes.items() if idx not in dropped]
    sorted_cluster_sizes = sorted(considered_cluster_sizes, key=lambda clusters: -clusters[0][1])

    n_ac = 0
    n_wa = 0
    for clusters in sorted_cluster_sizes:
        if clusters[0][0] == 'AC':
            n_ac += 1
        elif clusters[0][0] == 'WA':
            n_wa += 1
        
        if n_wa > total_errors:
            break
    
    return n_ac / total_problems

# Calculate all accuracies (avg, max, clustered, threshold at different errors) for a given list of dropped problems
def calculate_dropped_accuracies(cluster_sizes, dropped):
    accuracies = {
        "Threshold Accuracy (<= 0% Error)": calculate_threshold_accuracy(cluster_sizes, dropped, 0),
        "Threshold Accuracy (<= 1% Error)": calculate_threshold_accuracy(cluster_sizes, dropped, 0.01),
        "Threshold Accuracy (<= 2% Error)": calculate_threshold_accuracy(cluster_sizes, dropped, 0.02),
        "Expected Accuracy": calculate_expected_accuracy(cluster_sizes, dropped),
        "Clustered Accuracy": calculate_clustered_accuracy(cluster_sizes, dropped),
        "Maximum Accuracy": calculate_maximum_accuracy(cluster_sizes, dropped)
    }
    
    return accuracies

# Calculate all accuracies for all considered sets of dropped problems
def calculate_accuracies(config):
    raw_clusters = {}
    if os.path.isfile(config["cluster_file"]):
        with open(config["cluster_file"], mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raw_clusters[int(row["idx"])] = ast.literal_eval(row["clusters"])
    else:
        raise Exception(f'Cluster file {config["cluster_file"]} not found.')
    
    cluster_sizes = {}
    for idx, raw_cluster in raw_clusters.items():
        clusters = []
        for result in raw_cluster.keys():
            cluster_size = len(raw_cluster[result])
            clusters.append((result[0], cluster_size))
        
        clusters.sort(key=lambda x: -x[1])
        cluster_sizes[idx] = clusters

    dropped_sets = [
        "X", "O", 
        "OD", "OC", "OB", "OA",
        "ODC", "ODB", "ODA", "OCB", "OCA", "OBA", 
        "ODCB", "ODCA", "ODBA", "OCBA",
        "ODCBA", "ODCBAT"
    ]
    dropped_problems = []
    for s in dropped_sets:
        p = []
        for l in s:
            p += problems[l]
        dropped_problems.append(p)

    data = {
        f'{s} ({len(cluster_sizes) - len(p)})': 
        calculate_dropped_accuracies(cluster_sizes, p)
        for s, p in zip(dropped_sets, dropped_problems)
    }

    # Step 1: Collect all keys from the inner dictionaries.
    inner_keys = set()
    for inner_dict in data.values():
        inner_keys = inner_dict.keys()
        break
    inner_keys = list(inner_keys)

    # Step 2: Write data to CSV file.
    with open(config["accuracy_file"], mode="w", newline="") as csvfile:
        # Fieldnames includes a column for the outer key plus all the inner keys.
        fieldnames = ["outer_key"] + inner_keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for outer_key, inner_dict in data.items():
            # Start with the outer key
            row = {"outer_key": outer_key}
            # Update with the inner dict values; missing values will be left blank
            row.update({k: toPercentStr(v) for k, v in inner_dict.items()})
            writer.writerow(row)

def main(config, relative=False, rescale=True, model=''):
    clusters = {}
    if os.path.isfile(config["cluster_file"]):
        with open(config["cluster_file"], mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                clusters[int(row["idx"])] = ast.literal_eval(row["clusters"])
    else:
        raise Exception(f'Cluster file {config["cluster_file"]} not found.')

    if model != '':
        num_models = len(config['models'])
        config['n'] //= num_models

    threshold_rates = [{
        'failed': 0,
        'unknown_tl': 0,
        'unknown_er': 0,
        'unknown_lo': 0,
        'passed': 0
    } for _ in range(config['n'] + 1)]

    all_cluster_sizes = []
    avg_prob = 0
    max_prob = 0
    count = 0
    ac_largest = 0

    tl_largest = []
    
    for idx in clusters.keys():
        if config['dataset'] == 'Live Code Bench':
            if idx in problems["O"]:
                continue

        cluster = clusters[idx]
        cluster_sizes = []
        total_size = 0
        for result in cluster.keys():
            if model != '':
                cluster_size = 0
                for func in cluster[result]:
                    if func['model'] == model:
                        cluster_size += 1
            else:
                cluster_size = len(cluster[result])

            total_size += cluster_size
            cluster_sizes.append((result[0], cluster_size))
        
        cluster_sizes.sort(key=lambda x: -x[1])
        if len(cluster_sizes) == 0:
            continue

        if cluster_sizes[0][0] == 'TL':
            tl_largest.append(idx)

        if rescale:
            for i in range(len(cluster_sizes)):
                cluster_sizes[i] = (cluster_sizes[i][0], cluster_sizes[i][1] * config['n'] / total_size)

        first_non_err_size = 0
        for v, c in cluster_sizes:
            if v != 'TL' and v != 'ER':
                first_non_err_size = c
                break

        contains_ac = False
        ac_count = 0
        total_count = 0
        ac_biggest = cluster_sizes[0][0] == 'AC'
        for v, c in cluster_sizes:
            total_count += c
            if v == 'AC':
                ac_count += c
                contains_ac = True
        
        if total_count == 0:
            probability = 0
        else:
            probability = ac_count / total_count
        avg_prob += probability
        if contains_ac:
            max_prob += 1
        count += 1
        if cluster_sizes[0][0] == 'AC':
            ac_largest += 1
                
        # if not ac_biggest and cluster_sizes[0][0] == 'WA':
        all_cluster_sizes.append((idx, cluster_sizes, first_non_err_size))

        for threshold in range(config['n'] + 1):
            if relative:
                val = cluster_sizes[0][1] - cluster_sizes[1][1]
            else:
                val = cluster_sizes[0][1]

            if val >= threshold:
                if cluster_sizes[0][0] == 'AC':
                    threshold_rates[threshold]['passed'] += 1
                elif cluster_sizes[0][0] == 'WA':
                    threshold_rates[threshold]['failed'] += 1
                elif cluster_sizes[0][0] == 'TL':
                    threshold_rates[threshold]['unknown_tl'] += 1
                elif cluster_sizes[0][0] == 'ER':
                    threshold_rates[threshold]['unknown_er'] += 1
            else:
                threshold_rates[threshold]['unknown_lo'] += 1

    print('Thres\tFailed\tPassed\tTLE\tERR\tLOW')
    for threshold in range(len(threshold_rates)):
        failed = threshold_rates[threshold]['failed']
        unknown_tl = threshold_rates[threshold]['unknown_tl']
        unknown_er = threshold_rates[threshold]['unknown_er']
        unknown_lo = threshold_rates[threshold]['unknown_lo']
        passed = threshold_rates[threshold]['passed']
        total = failed + unknown_tl + unknown_er + unknown_lo + passed

        print(f'{threshold}\t{toPercent(failed, total):.2f}%\t{toPercent(passed, total):.2f}%\t{toPercent(unknown_tl, total):.2f}%\t{toPercent(unknown_er, total):.2f}%\t{toPercent(unknown_lo, total):.2f}%')

    all_cluster_sizes.sort(key=lambda x: x[2])
    for val in all_cluster_sizes:
        print(val)
    
    print(f'Expected Acc: {toPercent(avg_prob, count)}')
    print(f'Cluster Acc: {toPercent(ac_largest, count)}')
    print(f'Max Acc: {toPercent(max_prob, count)}')
    print(f'TL Largest:', tl_largest)


def multiple_thresholds(config):
    clusters = {}
    if os.path.isfile(config["cluster_file"]):
        with open(config["cluster_file"], mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                clusters[int(row["idx"])] = ast.literal_eval(row["clusters"])
    else:
        raise Exception(f'Cluster file {config["cluster_file"]} not found.')
    
    to_remove = 'O'
    to_remove_list = []
    for l in to_remove:
        to_remove_list += problems[l]

    for key in to_remove_list:
        clusters.pop(key, None)
    
    model_clusters = {}
    for m in config['models']:
        model_clusters[m['name']] = {}
    
    only_dominating = []
    for idx in clusters.keys():
        cluster = clusters[idx]
        dominating_cluster = {}
        for model_name in model_clusters.keys():
            cluster_sizes = []
            cluster_idx = 0
            for result in cluster.keys():
                cluster_size = 0
                for func in cluster[result]:
                    if func['model'] == model_name:
                        cluster_size += 1
                cluster_sizes.append((result[0], cluster_size, cluster_idx))
                cluster_idx += 1
            cluster_sizes.sort(key=lambda x: -x[1])

            model_clusters[model_name][idx] = cluster_sizes
            dominating_cluster[model_name] = cluster_sizes[0][2]
        
        if len(set(dominating_cluster.values())) == 1:
            size = 0
            for model_name in model_clusters.keys():
                res = model_clusters[model_name][idx][0][0]
                size += model_clusters[model_name][idx][0][1]
            
            only_dominating.append((idx, res, size))
        
    only_dominating.sort(key=lambda x: -x[2])
    count_ac = 0
    for val in only_dominating:
        print(val)
        if val[1] == "AC":
            count_ac += 1
        if val[1] == 'WA':
            break
    
    print(count_ac / len(clusters))
    
    all_clusters = {}
    for idx in clusters.keys():
        cluster = clusters[idx]
        cluster_sizes = []
        for result in cluster.keys():
            cluster_size = len(cluster[result])
            cluster_sizes.append((result[0], cluster_size))
        cluster_sizes.sort(key=lambda x: -x[1])
        all_clusters[idx] = cluster_sizes

    model_max_t = {}
    for model_name in model_clusters.keys():
        model_max_t[model_name] = 0
        for idx in model_clusters[model_name].keys():
            model_max_t[model_name] = max(model_max_t[model_name], model_clusters[model_name][idx][0][1])
    
    keys = list(model_max_t.keys())
    ranges = [range(model_max_t[k] + 1) for k in keys]

    error_0 = []
    error_1 = []
    error_2 = []
    for combo in product(*ranges):
        thresholds = dict(zip(keys, combo))

        accuracy = 0
        error = 0
        for idx in clusters.keys():
            pass_threshold = True
            for model_name in model_clusters.keys():
                if model_clusters[model_name][idx][0][1] <= thresholds[model_name]:
                    pass_threshold = False
                    break
            
            if pass_threshold:
                largest_cluster = all_clusters[idx][0]
                if largest_cluster[0] == 'AC':
                    accuracy += 1
                elif largest_cluster[0] == 'WA':
                    error += 1
        
        accuracy /= len(clusters)
        error /= len(clusters)

        if error <= 0:
            error_0.append((accuracy, thresholds))
        if error <= 0.01:
            error_1.append((accuracy, thresholds))
        if error <= 0.02:
            error_2.append((accuracy, thresholds))
    
    error_0.sort(key=lambda x: -x[0])
    error_1.sort(key=lambda x: -x[0])
    error_2.sort(key=lambda x: -x[0])
    print(error_0[0])
    print(error_1[0])
    print(error_2[0])


def show_item(config, ind, print_folder, cluster_ind = 0, use_model = False):
    dataset = dataset_mapping(config["dataset"])
    if os.path.isfile(config["cluster_file"]):
        with open(config["cluster_file"], mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if int(row['idx']) == ind:
                    cluster = ast.literal_eval(row["clusters"])
    else:
        raise Exception(f'Cluster file {config["cluster_file"]} not found.')
    
    model = ChatGPTModel("gpt-4o", os.environ.get("OPENAI_API_KEY"))
    
    cluster_sizes = []
    total_size = 0
    for result in cluster.keys():
        cluster_size = len(cluster[result])
        total_size += cluster_size
        cluster_sizes.append((result, cluster_size))
    
    cluster_sizes.sort(key=lambda x: -x[1])
    if not use_model:
        message = f"{cluster_sizes[cluster_ind]}\n## Problem\n{dataset.get_question_content(ind)}\n## Solution\n```py\n{cluster[cluster_sizes[cluster_ind][0]][0]['function']}\n```\n"
        with open(f"{print_folder}/{ind}-{cluster_ind}.md", "w") as f:
            f.write(message)
        print(message)

    def print_last_message(m):
        message_str = f"# {m[-1]["role"].upper()}\n{m[-1]["content"]}"
        print(message_str)
        return message_str
    
    if use_model:
        full_response = ""
        messages = [{
            "role": "user", 
            "content": (
                "I will give you a competitive programming problem and a solution written in Python. I am certain the solution is incorrect. Can you explain what this solution is doing, then simulate running it on some sample test cases? "
                "Then, based on your reasoning, can you categorize the solution into one of these three categories?\n"
                "   1) Dumb mistake that most humans will not make\n"
                "   2) Hard mistake that an intelligent programmers can make\n"
                "   3) The problem itself is truly ambiguous\n"
                f"## Problem\n{dataset.get_question_content(ind)}\n## Solution\n```py\n{cluster[cluster_sizes[cluster_ind][0]][0]['function']}\n```\n"
            )
        }]
        full_response += print_last_message(messages)

        messages = model.custom_prompt(messages)
        full_response += print_last_message(messages)

        with open(f"{print_folder}/{ind}-{cluster_ind}.md", "w") as f:
            f.write(full_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/test.json")
    parser.add_argument("-i", "--show-item", type=int, required=False)
    parser.add_argument("-x", "--cluster_ind", type=int, default=0)
    parser.add_argument('-m', "--use_model", action='store_true')
    parser.add_argument('-f', "--print_folder", type=str, default='results/solution_explanations')
    parser.add_argument('-a', "--get_accuracies", action='store_true')
    parser.add_argument('-t', "--multiple_thresholds", action='store_true')
    parser.add_argument('-n', "--model_name", type=str, default='')
    parser.add_argument("--relative", type=bool, default=False)
    parser.add_argument("--rescale", type=bool, default=False)
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    if args.get_accuracies:
        calculate_accuracies(config)
    elif args.multiple_thresholds:
        multiple_thresholds(config)
    elif args.show_item is not None:
        show_item(config, args.show_item, args.print_folder, args.cluster_ind, args.use_model)
    else:
        main(config, relative=args.relative, rescale=args.rescale, model=args.model_name)
