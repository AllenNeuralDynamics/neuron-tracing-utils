import argparse
import os

import scyjava

from neuron_tracing_utils.util.java import snt
from neuron_tracing_utils.util.swcutil import *


def print_stats(swc_dir):
    total_length = 0
    total_bps = 0
    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith(".swc")]
        if not swcs:
            continue
        for f in swcs:
            swc = os.path.join(root, f)
            t = snt.Tree(swc)
            graph = t.getGraph()
            total_length += graph.sumEdgeWeights()
            total_bps += len(graph.getBPs())
    print(f"Total length: {total_length}")
    print(f"Total branch points: {total_bps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to render"
    )

    args = parser.parse_args()

    scyjava.start_jvm()

    print_stats(args.input)


if __name__ == "__main__":
    main()
