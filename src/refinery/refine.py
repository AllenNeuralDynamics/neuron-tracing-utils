import argparse
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import dask.array
import numpy as np
import scyjava
import tifffile
from scipy.interpolate import RegularGridInterpolator
from tensorstore import TensorStore
from tqdm import tqdm

from refinery.util import ioutil
from refinery.util.chunkutil import chunk_center
from refinery.util.imgutil import get_hyperslice
from refinery.util.ioutil import ImgReaderFactory, open_ts, open_n5_zarr_as_ndarray
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
    rr = r**2
    coords = []
    for dz in np.linspace(z - r, z + r, 2 * r + 1, dtype=float):
        for dy in np.linspace(y - r, y + r, 2 * r + 1, dtype=float):
            for dx in np.linspace(x - r, x + r, 2 * r + 1, dtype=float):
                dd = (dx - x) ** 2 + (dy - y) ** 2 + (dz - z) ** 2
                if dd <= rr:
                    coords.append([dz, dy, dx])
    return np.array(coords)


def mean_shift_point(
    swc_point,
    im,
    radius,
    n_iter=10,
    interval=None,
    return_disp=False,
    crop_interval=True,
):
    if crop_interval:
        if isinstance(im, dask.array.Array):
            print(type(im))
            try:
                im = im[
                    interval.min(0) : interval.max(0),
                    interval.min(1) : interval.max(1),
                    interval.min(2) : interval.max(2),
                ].compute()
            except Exception as e:
                print(e)


        elif isinstance(im, TensorStore):
            im = im[
                interval.min(0) : interval.max(0),
                interval.min(1) : interval.max(1),
                interval.min(2) : interval.max(2),
            ].read().result()
        else:
            im = im[
                interval.min(0) : interval.max(0),
                interval.min(1) : interval.max(1),
                interval.min(2) : interval.max(2),
            ]

    cx = swc_point.x
    cy = swc_point.y
    cz = swc_point.z

    logging.info(f"initial point: {[cz, cy, cx]}")

    interpolant = interpolate(im, interval)

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

    logging.info(f"adjusted: {cz} {cy} {cx}")

    if return_disp:
        return displacements


def refine_graph(
    graph, img, radius, n_iter, n_threads=8, crop_intervals=False
):
    vertices = [v for v in graph.vertexSet()]
    times = graph.vertexSet().size()
    if crop_intervals:
        ivals = []
        for v in vertices:
            ivals.append(chunk_center(np.array([v.z, v.y, v.x]), [128, 128, 128]))
    else:
        ivals = [None] * times
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(len(vertices)):
            futures.append(
                executor.submit(mean_shift_point, vertices[i], img, radius, n_iter, ivals[i], False, crop_intervals)
            )
        for fut in tqdm(futures):
            try:
                fut.result()
            except Exception as e:
                print(e)


def fit_tree(tree, img, radius=1):
    PathFitter = snt.PathFitter
    for path in tree.list():
        fitter = PathFitter(img, path)
        fitter.setScope(PathFitter.RADII_AND_MIDPOINTS)
        fitter.setReplaceNodes(True)
        fitter.setMaxRadius(radius)
        fitter.call()


def refine_swcs_batch(
    in_swc_dir,
    out_swc_dir,
    im_dir,
    radius=1,
    mode=RefineMode.mean_shift.value,
    threads=1,
    key=None,
    mean_shift_iter=1,
):
    im_fmt = ioutil.get_file_format(im_dir)
    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        im_path = os.path.join(im_dir, os.path.basename(root) + im_fmt)

        for f in swcs:
            swc_path = os.path.join(root, f)
            logging.info(f"Refining {swc_path}")

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(swc_path, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)

            tree = snt.Tree(swc_path)

            if mode == RefineMode.mean_shift.value:
                # FIXME support more formats for Python
                im = tifffile.imread(im_path)
                graph = tree.getGraph()
                refine_graph(
                    graph,
                    im,
                    radius,
                    n_iter=mean_shift_iter,
                    n_threads=threads,
                    crop_intervals=False,
                )
                graph.getTree().saveAsSWC(out_swc)
            elif mode == RefineMode.fit.value:
                img = get_hyperslice(
                    ImgReaderFactory.create(im_path).load(im_path, key=key),
                    ndim=3,
                )
                fit_tree(tree, img, radius=radius)
                tree.saveAsSWC(out_swc)
            else:
                raise ValueError(f"Invalid mode {mode}")


def refine_swcs(
    in_swc_dir,
    out_swc_dir,
    im_path,
    radius=1,
    mode=RefineMode.mean_shift.value,
    threads=1,
    key=None,
    mean_shift_iter=1,
):
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
                im = open_ts(im_path, dataset=key, total_bytes_limit=500_000_000)
                while im.ndim > 3:
                    im = im[0, ...]
                graph = tree.getGraph()
                refine_graph(
                    graph,
                    im,
                    radius,
                    n_iter=mean_shift_iter,
                    n_threads=threads,
                    crop_intervals=True,
                )
                graph.getTree().saveAsSWC(out_swc)
            elif mode == RefineMode.fit.value:
                reader = ImgReaderFactory.create(im_path)
                img = get_hyperslice(reader.load(im_path, key=key), ndim=3)
                fit_tree(tree, img, radius=radius)
                tree.saveAsSWC(out_swc)
            else:
                raise ValueError(f"Invalid mode {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\repos\exaSpim-training-data\exaSPIM_609281_2022-11-03_13-49-18"
        r"\whole-brain\swcs-transformed",
        help="directory of .swc files to refine",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\repos\exaSpim-training-data\exaSPIM_609281_2022-11-03_13-49-18"
        r"\whole-brain\swcs-refined",
        help="directory to output refined .swc files",
    )
    parser.add_argument(
        "--image",
        default=r"https://aind-open-data.s3.amazonaws.com/exaSPIM_609281_2022-11-03_13-49-18_stitched_2022-11-22_12"
        r"-07-00/fused.zarr/fused.zarr",
        type=str,
        help="image or directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--dataset", type=str, default="0", help="key for the N5/Zarr dataset"
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
    parser.add_argument("--mean-shift-iter", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
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
    if os.path.isdir(args.image):
        refine_swcs_batch(
            args.input,
            args.output,
            args.image,
            args.radius,
            args.mode,
            args.threads,
            args.dataset,
            args.mean_shift_iter,
        )
    else:
        refine_swcs(
            args.input,
            args.output,
            args.image,
            args.radius,
            args.mode,
            args.threads,
            args.dataset,
            args.mean_shift_iter,
        )
    logging.info("Finished refinement.")


if __name__ == "__main__":
    main()
