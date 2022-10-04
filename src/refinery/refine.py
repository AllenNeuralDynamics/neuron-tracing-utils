import argparse
import json
import logging
import os
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import itertools
from time import time

from refinery.util.java import snt
from refinery.util.java import imglib2, imagej1
from refinery.util import sntutil, imgutil

import scyjava
from jpype import JArray, JLong
import zarr
import imglyb


class RefineMode(Enum):
    naive = "naive"
    fit = "fit"


def refine_point(
    swc_point, img, sphere_radius=1, region_shape="block", region_pad=None
):
    """Creates a shape neighborhood centered at an SWCPoint in img,
    iterates through each pixel in the neighborhood, creates a local hypersphere around each pixel
    and measures mean intensity, keeping track of the maximum encountered mean,
    then use the position of the final maximum to snap the SWCPoint to. Because this is a brute-force solution,
    it is very slow."""
    print(f"refining point {str(swc_point)}")
    if region_pad is None:
        region_pad = [1, 1, 1]
    chunk = None
    if region_shape == "block":
        chunk = sntutil.swcpoint_to_block(img, swc_point, region_pad)
    elif region_shape == "sphere":
        chunk = sntutil.swcpoint_to_sphere(
            img, swc_point, sum(region_pad) / len(region_pad)
        )
    cursor = chunk.localizingCursor()
    maximum = float("-inf")
    bestPosition = JArray(JLong, 1)(3)
    # right now this is just the maximum of the mean intensities
    while cursor.hasNext():
        cursor.fwd()
        local_distribution = list(
            imgutil.local_intensities(
                imglib2.HyperSphere(img, cursor, sphere_radius)
            )
        )
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
    vertices = (v for v in graph.vertexSet())
    times = graph.vertexSet().size()
    with ThreadPoolExecutor(16) as executor:
        executor.map(
            refine_point, 
            vertices, 
            itertools.repeat(img, times), 
            itertools.repeat(radius, times)
            )


def fit_tree(tree, img, radius=1):
    PathFitter = snt.PathFitter
    for path in tree.list():
        fitter = PathFitter(img, path)
        fitter.setScope(PathFitter.RADII_AND_MIDPOINTS)
        fitter.setReplaceNodes(True)
        fitter.setMaxRadius(radius)
        fitter.call()


def refine_swcs(in_swc_dir, out_swc_dir, imdir, radius=1, mode=RefineMode.naive.value):
    loader = imglib2.IJLoader()
    IJ = imagej1.IJ
    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        for f in swcs:
            swc_path = os.path.join(root, f)
            logging.info(f"Refining {swc_path}")

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(swc_path, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)

            tree = snt.Tree(swc_path)

            if mode == RefineMode.naive.value:
                n5 = "s3://janelia-mouselight-imagery/carveouts/2018-08-01/fluorescence-near-consensus.n5/"
                store = zarr.N5FSStore(n5)
                z = zarr.open(store, 'r')
                ds = z['volume-rechunked']
                print(ds.shape)
                print(ds.chunks)
                img, ref_store = imglyb.as_cell_img(
                    ds, 
                    chunk_shape=tuple(reversed(ds.chunks)), 
                    cache=100
                )
                view = imglib2.Views.hyperSlice(img, 3, 0)
                graph = tree.getGraph()
                refine_graph(graph, view, radius)
                graph.getTree().saveAsSWC(out_swc)
            elif mode == RefineMode.fit.value:
                # Versions prior to 4.04 only accept ImagePlus inputs
                # this will be orders of magnitude slower than it should be
                # until SNT versions >= 4.0.4 are available on the scijava maven repository
                imp = IJ.openImage(
                    os.path.join(imdir, os.path.basename(root) + ".tif")
                )
                fit_tree(tree, imp)
                tree.saveAsSWC(out_swc)
            else:
                raise ValueError(f"Invalid mode {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to refine"
    )
    parser.add_argument(
        "--output", type=str, help="directory to output refined .swc files"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RefineMode],
        default=RefineMode.naive.value,
        help="algorithm type",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="search radius for point refinement",
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'args.json'), 'w') as f:
        args.__dict__['script'] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    logging.info("Starting refinement...")
    refine_swcs(args.input, args.output, args.images, args.radius, args.mode)
    logging.info("Finished refinement.")


if __name__ == "__main__":
    main()
