import os
import logging
import argparse
from pathlib import Path

from ..javahelpers import snt
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


def clip_graph(g, mini, maxi):
    for v in g.vertexSet():
        point = np.array([v.getX(), v.getY(), v.getZ()])
        clipped = np.clip(point, mini, maxi)
        if not np.array_equal(point, clipped):
            logging.info(f"input coords: {point}")
            logging.info(f"adjusted coords: {clipped}")
            v.x = clipped[0]
            v.y = clipped[1]
            v.z = clipped[2]


def prune_swcs(in_swc_dir, out_swc_dir, imdir, mode="clip"):
    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith('.swc')]
        for f in swcs:
            tiff = os.path.join(imdir, os.path.basename(root) + ".tif")
            img_shape = get_tiff_shape(tiff)

            swc = os.path.join(root, f)
            logging.info(f"fixing {swc}")
            out_swc = os.path.join(out_swc_dir, os.path.relpath(swc, in_swc_dir))
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)

            graph = snt.Tree(swc).getGraph()

            mini = np.array([0, 0, 0])
            maxi = img_shape - 1
            num_points_before = graph.vertexSet().size()
            if mode == "prune":
                prune_graph(graph, mini, maxi)
                logging.info(f"{num_points_before - graph.vertexSet().size()} points pruned")
            elif mode == "clip":
                clip_graph(graph, mini, maxi)
            else:
                raise ValueError(f"Invalid mode {mode}")

            if graph.vertexSet().size() <= 1:
                continue

            components = get_components_iterative(graph)
            for i, c in enumerate(components):
                if c.vertexSet().size() <= 1:
                    continue
                c.getTree().saveAsSWC(out_swc.replace(".swc", f"-{i}.swc"))


def get_components_iterative(graph):
    roots = [v for v in graph.vertexSet() if graph.inDegreeOf(v) == 0]
    components = []
    for root in roots:
        # create an empty graph
        comp = snt.DirectedWeightedGraph()
        # iterative depth-first search
        stack = [root]
        while stack:
            v = stack.pop()
            comp.addVertex(v)
            out_edges = graph.outgoingEdgesOf(v)
            for edge in out_edges:
                child = edge.getTarget()
                comp.addVertex(child)
                comp.addEdge(v, child)
                stack.append(child)
        components.append(comp)

    return components


def main():
    parser = argparse.ArgumentParser(description="Prune vertices that lay outside the bounds of the image. "
                                                 "If a vertex of degree > 1 is pruned, this will break connectivity"
                                                 "and result in additional .swc outputs.")
    parser.add_argument('--input', type=str, help='directory of .swc files to prune')
    parser.add_argument('--output', type=str, help='directory to output pruned .swc files')
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
