import os
import logging
import argparse
from pathlib import Path

from . import snt

import zarr
import scyjava
import tifffile
import numpy as np


def get_tiff_shape(tiffpath):
    with tifffile.TiffFile(tiffpath) as tif:
        # Open as zarr-store in case we're dealing with
        # ImageJ hyperstacks.
        # tifffile.TiffFile is unable to parse Z-slices with ImageJ
        # Tiffs larger than 4GB.
        z = zarr.open(tif.aszarr(), 'r')
        shape = np.array([s for s in z.shape])
        # rearrange to XYZ
        axes = [2, 1, 0]
        return shape[axes]


def prune_graph(g, mini, maxi):
    to_remove = []
    for v in g.vertexSet():
        point = np.array([v.getX(), v.getY(), v.getZ()])
        if np.any(point < mini) or np.any(point > maxi):
            to_remove.append(v)
    g.removeAllVertices(to_remove)


def prune_swcs(in_swc_dir, out_swc_dir, imdir):
    for root, dirs, files in os.walk(in_swc_dir):
        for f in files:
            if not f.endswith(".swc"):
                continue

            img_name = os.path.basename(root) + ".tif"
            tiff = os.path.join(imdir, img_name)
            img_shape = get_tiff_shape(tiff)

            swc = os.path.join(root, f)
            print(f"pruning {swc}")
            outswc = os.path.join(out_swc_dir, os.path.relpath(swc, in_swc_dir))
            Path(outswc).parent.mkdir(exist_ok=True, parents=True)

            graph = snt.Tree(swc).getGraph()

            mini = np.array([0, 0, 0])
            maxi = img_shape - 1
            num_points_before = graph.vertexSet().size()
            prune_graph(graph, mini, maxi)
            print(f"{num_points_before - graph.vertexSet().size()} points pruned")

            if graph.vertexSet().isEmpty():
                continue

            trees = [c.getTree() for c in graph.getComponents()]
            for i, t in enumerate(trees):
                t.saveAsSWC(outswc.replace(".swc", f"-{i}.swc"))


def main():
    parser = argparse.ArgumentParser(description="Prune vertices that lay outside the bounds of the image. "
                                                 "If a vertex of degree > 1 is pruned, this will break connectivity"
                                                 "and result in additional .swc outputs.")
    parser.add_argument('--input', type=str, help='directory of .swc files to prune')
    parser.add_argument('--output', type=str,  help='directory to output pruned .swc files')
    parser.add_argument('--images', type=str, help='directory of images associated with the .swc files')
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    logging.info("Starting prune...")
    prune_swcs(args.input, args.output, args.images)
    logging.info("Finished prune.")


if __name__ == "__main__":
    main()

