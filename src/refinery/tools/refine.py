import argparse
import logging
import os
from pathlib import Path

from ..javahelpers import snt, imglib2
from ..util import sntutil, imgutil

import scyjava
from jpype import JArray, JLong


def refine_point(swc_point, img, sphere_radius=1, region_shape="block", region_pad=None):
    """Creates a shape neighborhood centered at an SWCPoint in img,
    iterates through each pixel in the neighborhood, creates a local hypersphere around each pixel
    and measures mean intensity, keeping track of the maximum encountered mean,
    then use the position of the final maximum to snap the SWCPoint to. Because this is a brute-force solution,
    it is very slow."""
    if region_pad is None:
        region_pad = [1, 1, 1]
    chunk = None
    if region_shape == "block":
        chunk = sntutil.swcpoint_to_block(img, swc_point, region_pad)
    elif region_shape == "sphere":
        chunk = sntutil.swcpoint_to_sphere(img, swc_point, sum(region_pad) / len(region_pad))
    cursor = chunk.localizingCursor()
    maximum = float("-inf")
    bestPosition = JArray(JLong, 1)(3)
    # right now this is just the maximum of the mean intensities
    while cursor.hasNext():
        cursor.fwd()
        local_distribution = list(imgutil.local_intensities(imglib2.HyperSphere(img, cursor, sphere_radius)))
        if len(local_distribution) == 0:
            continue
        mean = sum(local_distribution) / len(local_distribution)
        if mean > maximum:
            maximum = mean
            cursor.localize(bestPosition)
    swc_point.x = bestPosition[0]
    swc_point.y = bestPosition[1]
    swc_point.z = bestPosition[2]


def refine_graph(graph, img, radius=1):
    for v in graph.vertexSet():
        refine_point(v, img, radius)


def refine_swcs(in_swc_dir, out_swc_dir, imdir):
    loader = imglib2.IJLoader()
    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith('.swc')]
        for f in swcs:
            swc_path = os.path.join(root, f)
            logging.info(f"Refining {swc_path}")

            graph = snt.Tree(swc_path).getGraph()

            img = loader.get(os.path.join(imdir, os.path.basename(root) + ".tif"))

            refine_graph(graph, img, radius=1)

            out_swc = os.path.join(out_swc_dir, os.path.relpath(swc_path, in_swc_dir))
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            tree = graph.getTree()
            tree.saveAsSWC(out_swc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='directory of .swc files to refine')
    parser.add_argument('--output', type=str,  help='directory to output refined .swc files')
    parser.add_argument('--images', type=str, help='directory of images associated with the .swc files')
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    logging.info("Starting refinement...")
    refine_swcs(args.input, args.output, args.images)
    logging.info("Finished refinement.")


if __name__ == "__main__":
    main()
