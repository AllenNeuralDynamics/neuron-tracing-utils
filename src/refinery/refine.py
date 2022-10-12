import argparse
import itertools
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, wait
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

from refinery.util.java import snt
from refinery.util.java import imglib2, imagej1
from refinery.util import sntutil, imgutil

import scyjava
from jpype import JArray, JLong, JDouble


class RefineMode(Enum):
    naive = "naive"
    fit = "fit"


def refine_point_old(
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


def minmax(cursor):
    import java

    mn = float("inf")
    mx = float("-inf")
    while cursor.hasNext():
        cursor.fwd()
        try:
            val = cursor.get().get()
        except java.lang.ArrayIndexOutOfBoundsException:
            continue
        mn = min(val, mn)
        mx = max(val, mx)
    return mn, mx


def mean(cursor):
    import java

    s = 0
    c = 0
    while cursor.hasNext():
        cursor.fwd()
        try:
            val = cursor.get().get()
        except java.lang.ArrayIndexOutOfBoundsException:
            continue
        s += val
        c += 1
    return s / c


def mean_interp(ra, x, y, z, radius):
    rr = radius ** 2
    pos = JArray(JDouble, 1)(3)
    s = 0
    n = 0
    for dx in np.linspace(x - radius, x + radius, 2 * radius + 1, dtype=float):
        for dy in np.linspace(y - radius, y + radius, 2 * radius + 1, dtype=float):
            for dz in np.linspace(z - radius, z + radius, 2 * radius + 1, dtype=float):
                dd = (dx - x) ** 2 + (dy - y) ** 2 + (dz - z) ** 2
                if dd > rr:
                    continue
                pos[0] = dx
                pos[1] = dy
                pos[2] = dz
                s += ra.setPositionAndGet(pos).get()
                n += 1
    return s / n


def mean_shift_point(swc_point, img, radius, n_iter=10):
    eps = 1e-6
    disp = 1.0
    x = swc_point.x
    y = swc_point.y
    z = swc_point.z
    cx = x
    cy = y
    cz = z
    rr = radius ** 2

    ra = img.realRandomAccess()
    pos = JArray(JDouble, 1)(3)

    print("initial point: ", [swc_point.x, swc_point.y, swc_point.z])

    count = 0
    displacements = []
    max_disp = 2.0
    while disp >= 0.25 and count < n_iter:
        count += 1
        sx = 0
        sy = 0
        sz = 0
        sw = 0
        n = 0
        avg = mean_interp(ra, x, y, z, radius)
        for dx in np.linspace(x - radius, x + radius, 2 * radius + 1, dtype=float):
            for dy in np.linspace(y - radius, y + radius, 2 * radius + 1, dtype=float):
                for dz in np.linspace(z - radius, z + radius, 2 * radius + 1, dtype=float):
                    dd = (dx - x) ** 2 + (dy - y) ** 2 + (dz - z) ** 2
                    if dd > rr:
                        continue
                    pos[0] = dx
                    pos[1] = dy
                    pos[2] = dz
                    w = ra.setPositionAndGet(pos).get() - avg
                    if w > 0:
                        sx += dx * w
                        sy += dy * w
                        sz += dz * w
                        sw += w
                        n += 1

        if n == 0:
            return [0]

        cx = sx / sw
        cy = sy / sw
        cz = sz / sw

        disp = (cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2
        displacements.append(disp)

        x = cx
        y = cy
        z = cz

    swc_point.x = cx
    swc_point.y = cy
    swc_point.z = cz

    print(f"adjusted: {cx} {cy} {cz}")

    return displacements


def refine_graph(graph, img, radius, n_iter):
    Converters = imglib2.Converters
    RealDoubleConverter = imglib2.RealDoubleConverter
    DoubleType = imglib2.DoubleType
    Views = imglib2.Views
    NLinearInterpolatorFactory = imglib2.NLinearInterpolatorFactory
    RandomAccessibleInterval = imglib2.RandomAccessibleInterval

    floatImg = Converters.convert(RandomAccessibleInterval @ img, RealDoubleConverter(), DoubleType())
    interpolant = Views.interpolate(Views.extendZero(floatImg), NLinearInterpolatorFactory())

    vertices = (v for v in graph.vertexSet())
    displacements = []
    for v in vertices:
        d = mean_shift_point(v, interpolant, radius, n_iter)
        while len(d) < n_iter:
            d.append(0)
        displacements.append(d)
    return np.array(displacements, dtype=float)

    # times = graph.vertexSet().size()
    # with ThreadPoolExecutor(8) as executor:
    #     ret = executor.map(
    #         mean_shift_point,
    #         vertices,
    #         itertools.repeat(interpolant, times),
    #         itertools.repeat(radius, times),
    #         itertools.repeat(10,  times)
    #     )


def fit_tree(tree, img, radius=1):
    PathFitter = snt.PathFitter
    for path in tree.list():
        fitter = PathFitter(img, path)
        fitter.setScope(PathFitter.RADII_AND_MIDPOINTS)
        fitter.setReplaceNodes(True)
        fitter.setMaxRadius(radius)
        fitter.call()


def refine_swcs(
        in_swc_dir, out_swc_dir, imdir, radius=1, mode=RefineMode.naive.value
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

            if mode == RefineMode.naive.value:
                tif = os.path.join(imdir, os.path.basename(root) + ".tif")
                img = loader.get(tif)
                graph = tree.getGraph()
                refine_graph(graph, img, radius, n_iter=20)
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
        "--input", type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\aligned\swcs\block_3",
        help="directory of .swc files to refine"
    )
    parser.add_argument(
        "--output", type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\aligned\swcs\block_3\refined",
        help="directory to output refined .swc files"
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
        default=RefineMode.naive.value,
        help="algorithm type",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="search radius for point refinement",
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
    refine_swcs(args.input, args.output, args.images, args.radius, args.mode)
    logging.info("Finished refinement.")


if __name__ == "__main__":
    main()
