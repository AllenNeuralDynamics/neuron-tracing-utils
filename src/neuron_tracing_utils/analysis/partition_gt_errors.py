import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scyjava

from neuron_tracing_utils.util import sntutil, swcutil
from neuron_tracing_utils.util.ioutil import open_ts
from neuron_tracing_utils.util.java import snt


def partition_graph(g, label_mask):
    omit_graph = snt.DirectedWeightedGraph()
    split_graph = snt.DirectedWeightedGraph()
    correct_graph = snt.DirectedWeightedGraph()
    non_duplicate_edges = []
    z_coords = []
    y_coords = []
    x_coords = []
    for e in g.edgeSet():
        s = e.getSource()
        t = e.getTarget()
        sx, sy, sz = int(s.getX()), int(s.getY()), int(s.getZ())
        tx, ty, tz = int(t.getX()), int(t.getY()), int(t.getZ())
        if (tx, ty, tz) != (sx, sy, sz):
            non_duplicate_edges.append(e)
            z_coords.extend([sz, tz])
            y_coords.extend([sy, ty])
            x_coords.extend([sx, tx])
    values = label_mask[z_coords, y_coords, x_coords].read().result()
    for i, e in enumerate(non_duplicate_edges):
        s = e.getSource()
        t = e.getTarget()
        v1 = values[2 * i]
        v2 = values[2 * i + 1]
        if v1 == 0 or v2 == 0:
            graph = omit_graph
        elif v1 != v2:
            graph = split_graph
        else:
            graph = correct_graph
        graph.addVertex(s)
        graph.addVertex(t)
        graph.addEdge(s, t, e)
    return omit_graph.getComponents(), split_graph.getComponents(), correct_graph.getComponents()


def _mkdir(path):
    os.makedirs(path, exist_ok=True)


def _rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def _save_swc(tree, output_path, voxel_size=[0.748, 0.748, 1.0], node_type=2, color=(1, 1, 1), radius=1.0):
    arr = sntutil.tree_to_ndarray(tree)
    arr[:, 1] = node_type
    arr[:, 2:5] *= np.array(voxel_size)
    arr[:, 5] = radius
    swcutil.ndarray_to_swc(arr, output_path, color)


def process_swc_file(swc_path, label_mask, output_base_dir):
    g = snt.Tree(swc_path).getGraph()
    omit, split, correct = partition_graph(g, label_mask)
    partitioned_graphs = {
        "omit": omit,
        "split": split,
        "correct": correct
    }
    colors = {
        "omit": (1, 0, 0),
        "split": (0, 0, 1),
        "correct": (0, 1, 0)
    }
    for part_name, components in partitioned_graphs.items():
        part_dir = os.path.join(output_base_dir, part_name)
        _mkdir(part_dir)
        for i, c in enumerate(components):
            output_path = os.path.join(part_dir, os.path.basename(swc_path).replace(".swc", f"_{part_name}_{i}.swc"))
            _save_swc(c.getTree(), output_path, color=colors[part_name])


def process_swc_file_wrapper(swc, args, store):
    print(f"Processing {swc}")
    neuron_dir = os.path.join(args.out_dir, swc.replace(".swc", ""))
    _mkdir(neuron_dir)
    process_swc_file(os.path.join(args.swc_dir, swc), store, neuron_dir)


def main():
    parser = argparse.ArgumentParser(description="Process SWC files and partition the graph based on a label mask.")

    parser.add_argument('--label-mask',
                        type=str,
                        help='Path to the TensorStore on Google Cloud Storage.')

    parser.add_argument('--out-dir',
                        type=str,
                        help='Output directory path.')

    parser.add_argument('--swc-dir',
                        type=str,
                        help='Directory containing SWC files.')

    args = parser.parse_args()

    scyjava.start_jvm()

    label_mask = open_ts(args.label_mask)
    label_mask = label_mask[0]
    print(label_mask)

    _rmtree(args.out_dir)
    _mkdir(args.out_dir)

    with ThreadPoolExecutor(max_workers=4) as executor:
        swc_files = os.listdir(args.swc_dir)
        executor.map(process_swc_file_wrapper, swc_files, [args] * len(swc_files), [label_mask] * len(swc_files))


if __name__ == "__main__":
    main()
