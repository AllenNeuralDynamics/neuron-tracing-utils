import argparse
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import numpy as np
import scyjava
from scipy.interpolate import RegularGridInterpolator
from tifffile import tifffile

from refinery.util.java import imglib2, imagej1
from refinery.util.java import snt


class RefineMode(Enum):
    mean_shift = "mean_shift"
    fit = "fit"


def interpolate(img, ival=None):
    if ival is None:
        z = np.arange(0, img.shape[0], dtype=float)
        y = np.arange(0, img.shape[1], dtype=float)
        x = np.arange(0, img.shape[2], dtype=float)
    else:
        # interpolate on the interval that defines the image
        z = np.arange(ival.min(0), ival.max(0), dtype=float)
        y = np.arange(ival.min(1), ival.max(1), dtype=float)
        x = np.arange(ival.min(2), ival.max(2), dtype=float)
    return RegularGridInterpolator((z, y, x), img)


def sphere_coords(x, y, z, r):
    rr = r ** 2
    coords = []
    for dz in np.linspace(z - r, z + r, 2 * r + 1, dtype=float):
        for dy in np.linspace(y - r, y + r, 2 * r + 1, dtype=float):
            for dx in np.linspace(x - r, x + r, 2 * r + 1, dtype=float):
                dd = (dx - x) ** 2 + (dy - y) ** 2 + (dz - z) ** 2
                if dd <= rr:
                    coords.append([dz, dy, dx])
    return np.array(coords)


def mean_shift_point(swc_point, img, radius, n_iter=10, interval=None, return_disp=False):
    cx = swc_point.x
    cy = swc_point.y
    cz = swc_point.z

    logging.debug(f"initial point: {[cz, cy, cx]}")

    interpolant = interpolate(img, interval)

    count = 0
    displacements = []
    disp = float("inf")
    while disp >= 0.5 and count < n_iter:
        count += 1

        # get locations corresponding to the sphere centered at z,y,x
        sc = sphere_coords(cx, cy, cz, radius)

        # get the interpolated intensity values
        vals = interpolant(sc)

        # compute weights
        w = vals - vals.mean()

        # get positive weights and their indices
        w_idx = np.nonzero(w > 0)
        if w_idx[0].size == 0:
            return
        w_pos = w[w_idx]

        # get corresponding sphere coordinates
        c_pos = sc[w_idx]

        # add dummy dim for broadcasting purposes
        w_pos = w_pos[..., np.newaxis]
        # compute shifted coordinate
        c = (c_pos * w_pos).sum(axis=0) / w_pos.sum()

        disp = np.linalg.norm(c - np.array([cz, cy, cx]))
        displacements.append(disp)

        cz = c[0]
        cy = c[1]
        cx = c[2]

    swc_point.x = cx
    swc_point.y = cy
    swc_point.z = cz

    logging.debug(f"adjusted: {cz} {cy} {cx}")

    if return_disp:
        return displacements


def refine_graph(graph, img, radius, n_iter, n_threads=8):
    vertices = (v for v in graph.vertexSet())
    times = graph.vertexSet().size()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(
            mean_shift_point,
            vertices,
            itertools.repeat(img, times),
            itertools.repeat(radius, times),
            itertools.repeat(n_iter, times)
        )


def fit_tree(tree, img, radius=1):
    PathFitter = snt.PathFitter
    for path in tree.list():
        fitter = PathFitter(img, path)
        fitter.setScope(PathFitter.RADII_AND_MIDPOINTS)
        fitter.setReplaceNodes(True)
        fitter.setMaxRadius(radius)
        fitter.call()


def refine_swcs(
        in_swc_dir, out_swc_dir, imdir, radius=1, mode=RefineMode.mean_shift.value, threads=1
):
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

            if mode == RefineMode.mean_shift.value:
                tif = os.path.join(imdir, os.path.basename(root) + ".tif")
                img = tifffile.imread(tif)
                graph = tree.getGraph()
                refine_graph(graph, img, radius, n_iter=5, n_threads=threads)
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
        "--input",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\aligned\swcs\block_3",
        help="directory of .swc files to refine",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\aligned\swcs\block_3\refined",
        help="directory to output refined .swc files",
    )
    parser.add_argument(
        "--images",
        default=r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\aligned\blocks\all",
        type=str,
        help="directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RefineMode],
        default=RefineMode.mean_shift.value,
        help="algorithm type",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="search radius for point refinement",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    logging.info("Starting refinement...")
    refine_swcs(args.input, args.output, args.images, args.radius, args.mode, args.threads)
    logging.info("Finished refinement.")


if __name__ == "__main__":
    main()
