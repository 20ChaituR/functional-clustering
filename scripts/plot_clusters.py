import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
csv.field_size_limit(sys.maxsize)


def scatter_plot(config, output_file):
    # ---------- load clusters -------------------------------------------------
    with open(config["cluster_file"], newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        raw_clusters = {int(r["idx"]): ast.literal_eval(r["clusters"])
                        for r in reader}
        
    if config['dataset'] == 'Live Code Bench':
        OUT_OF_DIST = [207, 56, 206, 125, 150, 176, 86]
    elif config['dataset'] == 'Human Eval':
        OUT_OF_DIST = []

    cluster_sizes = {}
    for idx, raw in raw_clusters.items():
        if idx in OUT_OF_DIST:
            continue
        # list of (outcome, cluster_size)
        cluster_sizes[idx] = sorted(
            [(res[0], len(members)) for res, members in raw.items()
             if res[0] in {"AC", "WA"}],
            key=lambda t: -t[1])

    # ---------- flatten to points --------------------------------------------
    xs, ys, cs = [], [], []
    palette = {"AC": "green",   # blue
               "WA": "red"}   # orange
    for clusters in cluster_sizes.values():
        for outcome, size in clusters:
            xs.append(size / config["n"])
            ys.append(random.random())           # jitter only for visibility
            cs.append(palette[outcome])

    # ---------- figure --------------------------------------------------------
    plt.figure(figsize=(10, 4.5))
    plt.scatter(xs, ys, c=cs, alpha=0.55, edgecolors='k', linewidths=0.4)

    plt.xlabel('Estimated confidence $\\hat C_n$')
    plt.yticks([])            # no semantic meaning
    plt.title(f'Correctness vs. Estimated confidence for {config["dataset"]}')

    # ---------- thresholds + inline labels ---------------------------------
    if config['dataset'] == 'Live Code Bench':
        thresholds = [
            (0.57, 'τ$_{0\\%}$', 0.156),
            (0.45, 'τ$_{1\\%}$', 0.255),
            (0.34, 'τ$_{2\\%}$', 0.368)
        ]
    elif config['dataset'] == 'Human Eval':
        thresholds = [
            (0.84, 'τ$_{0\\%}$', 0.784),
            (0.76, 'τ$_{1\\%}$', 0.827),
            (0.70, 'τ$_{2\\%}$', 0.852)
        ]

    handles = []
    for tau, name, cov in thresholds:
        h = plt.axvline(tau, color='blue', ls='--', lw=1.5)
        # place label 5 % below the top of the axes
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        ax.text(tau, 0.97 * y_max, f'{name}        ',
                ha='center', va='top', fontsize=9, color='blue')
        handles.append(h)

    # ---------- legend ------------------------------------------------------
    point_legend = [
        Line2D([0], [0], marker='o', color='w', label='Correct',
            markerfacecolor='green', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Wrong',
            markerfacecolor='red', markersize=8, markeredgecolor='k')
    ]
    thresh_legend = [
        Line2D([0], [0], color='blue', lw=1.5, ls='--',
            label=f'{name} (accuracy {cov*100:.1f}%)')
        for (_, name, cov) in thresholds
    ]

    plt.legend(handles=point_legend + thresh_legend,
            bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)


    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def dual_plot(config, output_file):
    # ---------- load clusters -------------------------------------------------
    with open(config["cluster_file"], newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        raw_clusters = {int(r["idx"]): ast.literal_eval(r["clusters"])
                        for r in reader}
        
    if config['dataset'] == 'Live Code Bench':
        OUT_OF_DIST = [207, 56, 206, 125, 150, 176, 86]
    elif config['dataset'] == 'Human Eval':
        OUT_OF_DIST = []
    
    models = []
    for m in config['models']:
        models.append(m['name'])

    points = []
    for idx, raw in raw_clusters.items():
        if idx in OUT_OF_DIST:
            continue
        # list of (outcome, cluster_size)
        for result in raw.keys():
            cluster = raw[result]
            point = [0] * len(models)
            for i in range(len(models)):
                model = models[i]
                for func in cluster:
                    if func['model'] == model:
                        point[i] += 1
            
            if result[0] == 'AC' or result[0] == 'WA':
                points.append((result[0], point))

    # ---------- flatten to points --------------------------------------------
    xs, ys, cs = [], [], []
    palette = {"AC": "green",   # blue
               "WA": "red"}   # orange
    for result, point in points:
        xs.append(len(models) * point[0] / config["n"])
        ys.append(len(models) * point[1] / config["n"])
        cs.append(palette[result])

    # ---------- figure --------------------------------------------------------
    plt.figure(figsize=(10, 4.5))
    plt.scatter(xs, ys, c=cs, alpha=0.55, edgecolors='k', linewidths=0.4)

    plt.xlabel(f'Estimated confidence $\\hat C_n$ for {models[0]}')
    plt.ylabel(f'Estimated confidence $\\hat C_n$ for {models[1]}')
    plt.title(f'Scatter plot of confidences across models for {config["dataset"]}')

    # ---------- legend ------------------------------------------------------
    point_legend = [
        Line2D([0], [0], marker='o', color='w', label='Correct',
            markerfacecolor='green', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Wrong',
            markerfacecolor='red', markersize=8, markeredgecolor='k')
    ]

    plt.legend(handles=point_legend,
            bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def cumulative_plot(config, output_file):
    # ---------- load clusters -------------------------------------------------
    with open(config["cluster_file"], newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        raw_clusters = {int(r["idx"]): ast.literal_eval(r["clusters"])
                        for r in reader}
        
    if config['dataset'] == 'Live Code Bench':
        OUT_OF_DIST = [207, 56, 206, 125, 150, 176, 86]
    elif config['dataset'] == 'Human Eval':
        OUT_OF_DIST = []

    cluster_sizes = {}
    for idx, raw in raw_clusters.items():
        if idx in OUT_OF_DIST:
            continue
        # list of (outcome, cluster_size)
        cluster_sizes[idx] = sorted(
            [(res[0], len(members)) for res, members in raw.items()
             if res[0] in {"AC", "WA"}],
            key=lambda t: -t[1])

    # ---------- flatten to points --------------------------------------------
    xs, ys, cs = [], [], []
    palette = {"AC": "green",
               "WA": "red"}
    for clusters in cluster_sizes.values():
        for outcome, size in clusters:
            xs.append(size / config["n"])
            ys.append(random.random())           # jitter only for visibility
            cs.append(palette[outcome])
            # break

    # ---------- pre-compute cumulative curves --------------------------------
    xs_arr = np.asarray(xs)
    cs_arr = np.asarray(cs)

    red_x   = np.sort(xs_arr[cs_arr == 'red'])[::-1]
    green_x = np.sort(xs_arr[cs_arr == 'green'])[::-1]     # descending

    # ---------- figure --------------------------------------------------------
    plt.figure(figsize=(10, 4.5))
    plt.scatter(xs, ys, c=cs, alpha=0.3, edgecolors='k', linewidths=0.4)

    # ---- cumulative wrong (left → right) ------------------------------------
    if len(red_x):
        eps = 1e-6
        red_x_p = np.concatenate(([red_x[0] + eps], red_x))
        red_y_p = np.concatenate(([0.0],
                                  np.arange(1, len(red_x) + 1) / len(red_x)))
        plt.step(red_x_p, red_y_p, where='pre',
                 color='red', lw=2, zorder=3)

    # ---- cumulative correct (right → left) ----------------------------------
    if len(green_x):
        eps = 1e-6
        green_x_p = np.concatenate(([green_x[0] + eps], green_x))
        green_y_p = np.concatenate(([0.0],
                                    np.arange(1, len(green_x) + 1) / len(green_x)))
        plt.step(green_x_p, green_y_p, where='pre',
                 color='green', lw=2, zorder=3)

    plt.ylim(0, 1)                     # make curves span full height
    plt.xlabel('Estimated confidence $\\hat C_n$')
    plt.ylabel('Cumulative fraction of answers ≥ confidence')
    plt.yticks(np.linspace(0, 1, 5))
    # plt.yticks([])                     # no semantic meaning
    plt.title(f'Correct/incorrect responses vs. confidence ({config['dataset']})')

    # plt.title(f'Correctness vs. Estimated confidence for {config["dataset"]}')

    # ---------- thresholds + inline labels -----------------------------------
    if config['dataset'] == 'Live Code Bench':
        thresholds = [
            (0.57, 'τ$_{0\\%}$', 0.156),
            (0.45, 'τ$_{1\\%}$', 0.255),
            (0.34, 'τ$_{2\\%}$', 0.368)
        ]
    elif config['dataset'] == 'Human Eval':
        thresholds = [
            (0.84, 'τ$_{0\\%}$', 0.784),
            (0.76, 'τ$_{1\\%}$', 0.827),
            (0.70, 'τ$_{2\\%}$', 0.852)
        ]

    thresh_legend = []
    ax = plt.gca()
    y_max = ax.get_ylim()[1]
    for tau, name, cov in thresholds:
        plt.axvline(tau, color='blue', ls='--', lw=1.5)
        ax.text(tau, 0.97 * y_max, f'{name}        ',
                ha='center', va='top', fontsize=9, color='blue')
        thresh_legend.append(
            Line2D([0], [0], color='blue', ls='--', lw=1.5,
                   label=f'{name} (accuracy {cov*100:.1f}%)')
        )

    # ---------- legend --------------------------------------------------------
    point_legend = [
        Line2D([0], [0], marker='o', color='w', label='Correct',
               markerfacecolor='green', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Wrong',
               markerfacecolor='red', markersize=8, markeredgecolor='k')
    ]
    cum_legend = [
        Line2D([0], [0], color='green', lw=2, label='Cum. correct'),
        Line2D([0], [0], color='red',   lw=2, label='Cum. wrong')
    ]

    plt.legend(handles=point_legend + cum_legend + thresh_legend,
               bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/test.json")
    parser.add_argument("-o", "--output_file", type=str)
    parser.add_argument("-s", "--scatter", action='store_true')
    parser.add_argument("-d", "--dual", action='store_true')
    parser.add_argument("-l", "--cumulative", action='store_true')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    if args.scatter:
        scatter_plot(config, args.output_file)
    elif args.dual:
        dual_plot(config, args.output_file)
    elif args.cumulative:
        cumulative_plot(config, args.output_file)
