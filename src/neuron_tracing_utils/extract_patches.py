import argparse
import ast
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scyjava
import tifffile
from scipy.ndimage import gaussian_laplace
from skimage.exposure import rescale_intensity

from neuron_tracing_utils.refine import mean_shift_point
from neuron_tracing_utils.transform import WorldToVoxel
from neuron_tracing_utils.util.chunkutil import chunk_center
from neuron_tracing_utils.util.ioutil import open_ts
from neuron_tracing_utils.util.java import snt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=r"https://s3.amazonaws.com/janelia-mouselight-imagery/carveouts/2018-08-01/fluorescence-near-consensus.n5"
    )
    parser.add_argument("--dataset", type=str, default="volume-rechunked")
    parser.add_argument(
        "--swcs",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\swcs-batch1",
    )
    parser.add_argument(
        "--structure",
        type=str,
        choices=["endpoints", "branches", "soma"],
        default="endpoints",
    )
    parser.add_argument("--block-size", type=str, default="128,128,64")
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-all",
    )
    parser.add_argument(
        "--do-mean-shift",
        default=True,
        action="store_true"
    )
    parser.add_argument(
        "--mean-shift-iter",
        type=int,
        default=5
    )
    parser.add_argument(
        "--mean-shift-radius",
        type=float,
        default=5
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\transform.txt"
    )
    parser.add_argument(
        "--laplacian",
        default=True,
        action="store_true",
        help="use Laplacian of Gaussian filter for point refinement"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3,
        help="sigma for LoG filter"
    )
    parser.add_argument(
        "--add-shift",
        default=True,
        action="store_true",
        help="add a random shift to the center location of each patch"
    )
    parser.add_argument(
        "--shift-border",
        type=str,
        default="32,32,16",
        help="restrict the random shift to pixels within this border of block size"
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=logging.INFO
    )
    args = parser.parse_args()
    return args


# def objective(current_answer, r, img, side_lengths, max_value, min_value):
#     cx, cy, cz = current_answer
#     cx = round(cx)
#     cy = round(cy)
#     cz = round(cz)
#     interval = chunk_center([cx, cy, cz], side_lengths)
#     block = imglib2.Views.interval(img, interval)
#     pos = JArray(JLong, 1)(3)
#     ra = block.randomAccess()
#     badness = 0
#     for x in range(block.min(0), block.max(0) + 1):
#         for y in range(block.min(1), block.max(1) + 1):
#             for z in range(block.min(2), block.max(2) + 1):
#                 pos[0] = x
#                 pos[1] = y
#                 pos[2] = z
#                 val = ra.setPositionAndGet(pos).get()
#                 if r ** 2 > ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2):
#                     badness += (max_value - val) ** 2
#                 else:
#                     badness += (val - min_value) ** 2
#     # normalize by number of voxels
#     badness /= np.product(side_lengths, dtype=int)
#     return badness
#
#
# def optimize_point(point, radius, img, side_lengths, max_value, min_value):
#     x0 = np.array([point.x, point.y, point.z], dtype=int)
#     print(f"Initial coordinate: {x0}")
#     bounds = [
#         (point.x - side_lengths[0] // 2, point.x + side_lengths[0] // 2),
#         (point.y - side_lengths[1] // 2, point.y + side_lengths[1] // 2),
#         (point.z - side_lengths[2] // 2, point.z + side_lengths[2] // 2),
#     ]
#     args = (radius, img, side_lengths, max_value, min_value)
#     ret = optimize.minimize(objective, x0, args=args, bounds=bounds, method="Nelder-Mead")
#     print(ret.message)
#     solution = ret.x
#     point.x = solution[0]
#     point.y = solution[1]
#     point.z = solution[2]
#     print(f"Optimized coordinate: {solution}")
#
#
# def optimize_points(points, radius, img, side_lengths, max_value, min_value):
#     N = len(points)
#     for i, p in enumerate(points):
#         logging.info(f"optimizing point {i + 1}/{N}")
#         optimize_point(p, radius, img, side_lengths, max_value, min_value)


def constrained_random_shift(imin, imax, border):
    sz = imin[0] + border[0]
    sy = imin[1] + border[1]
    sx = imin[2] + border[2]
    ez = imax[0] - border[0]
    ey = imax[1] - border[1]
    ex = imax[2] - border[2]
    return np.array([random.randint(sz, ez), random.randint(sy, ey), random.randint(sx, ex)])


def patches_from_points(darray, points, block_size, add_shift=True, shift_border=None):
    if shift_border is None:
        shift_border = [10, 10, 10]
    logging.debug(f"Using window size: {block_size}")
    patches = []
    translated_points = []
    for p in points:
        arr = np.array([p.getZ(), p.getY(), p.getX()])
        mn, mx = chunk_center(arr, block_size)
        if add_shift:
            s = constrained_random_shift(mn, mx, shift_border)
            mn, mx = chunk_center(s, block_size)
        offset = arr - mn
        translated_points.append(snt.SWCPoint(0, 1, offset[2], offset[1], offset[0], 1.0, -1))
        patch = darray[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
        patches.append(patch)
    return patches, translated_points


def save_patch(darray, path):
    logging.debug(f"Saving patch: {path}")
    tifffile.imwrite(path, darray.read().result())


def save_patches(patches, out_dir, n_threads=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / (f"patch-{i:05}" + ".tif") for i in range(len(patches))]
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(save_patch, patches, paths)


def save_points(points, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for i, p in enumerate(points):
        patch_name = f"patch-{i:05}"
        g = snt.DirectedWeightedGraph()
        g.addVertex(p)
        g.getTree().saveAsSWC(str(out_dir / (patch_name + ".swc")))


def mean_shift_helper(point, img, radius=4, n_iter=1, do_log=False, log_sigma=1, voxel_size=None):
    mn, mx = chunk_center([point.z, point.y, point.x], [128, 128, 128])
    block = img[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]].read().result()
    if do_log:
        sigmas = log_sigma / np.array(voxel_size)
        block = gaussian_laplace(block.astype(np.float64), sigmas, output=np.float64)
        # Only keep negative part of response (bright changes)
        block = rescale_intensity(np.abs(np.clip(block, a_min=None, a_max=0)), out_range=np.uint16)
        # print(block.min(), block.max())
    mean_shift_point(point, block, radius, n_iter, interval=(mn, mx))


def mean_shift_points(
        points, img, radius=4, n_iter=1, n_threads=1, do_log=False, sigma=0.6, voxel_size=None
):
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for p in points:
            fut = executor.submit(mean_shift_helper, p, img, radius, n_iter, do_log, sigma, voxel_size)
            futures.append(fut)
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                logging.error(f"Exception during mean-shift: {e}")


def main():
    scyjava.start_jvm()

    args = parse_args()

    logging.getLogger().setLevel(args.log_level)

    transform = WorldToVoxel(args.transform)
    voxel_size = np.flip(transform.scale)
    logging.info(f"Voxel size: {voxel_size}")

    output = Path(args.output)

    block_size = list(reversed(ast.literal_eval(args.block_size)))
    logging.info(f"Using block size: {block_size}")

    # TensorStore opens n5 with axis order X,Y,Z, so get
    # a transposed view to be compatible with util code
    ds = open_ts(args.image, dataset=args.dataset)

    # FIXME deal with arbitrary dimensional images
    while ds.ndim > 3:
        ds = ds[0, ...]
    print(ds.shape)

    swc_dir = Path(args.swcs)

    swcs = [f for f in swc_dir.iterdir() if f.name.endswith(".swc")]

    for i, swc in enumerate(swcs):
        logging.info(f"Processing swc {i+1}/{len(swcs)}: {swc}")

        patch_dir = output / swc.stem
        patch_dir.mkdir(parents=True, exist_ok=True)

        image_dir = patch_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        point_dir = patch_dir / "swcs"
        point_dir.mkdir(parents=True, exist_ok=True)

        g = snt.Tree(str(swc)).getGraph()

        struct = args.structure

        if struct == "endpoints":
            points = list(g.getTips())
            logging.info(f"{len(points)} endpoints will be processed")
        elif struct == "branches":
            points = list(g.getBPs())
            logging.info(f"{len(points)} branches will be processed")
        elif struct == "soma":
            points = [g.getRoot()]
            logging.info("soma will be processed")
        else:
            raise ValueError(f"Invalid structure: {args.structure}")

        if args.do_mean_shift:
            logging.info(f"Doing mean shift for {struct}...")
            mean_shift_points(
                points=points,
                img=ds,
                radius=args.mean_shift_radius,
                n_iter=args.mean_shift_iter,
                n_threads=args.threads,
                do_log=args.laplacian,
                sigma=args.sigma,
                voxel_size=voxel_size
            )
            logging.info("Mean shift done.")

        logging.info("Computing patches")
        border = list(reversed(ast.literal_eval(args.shift_border)))
        patches, offset_points = patches_from_points(
            ds, points, block_size=block_size, add_shift=args.add_shift, shift_border=border
        )
        logging.info("Saving point coordinates as SWC")
        save_points(points, point_dir / struct / "locations")
        logging.info("Saving patch-aligned coordinates as SWC")
        save_points(offset_points, point_dir / struct / "patch-aligned")
        logging.info("Saving patches")
        save_patches(patches, image_dir / struct, n_threads=args.threads)


if __name__ == "__main__":
    main()
