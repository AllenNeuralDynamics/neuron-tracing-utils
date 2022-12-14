import argparse
import ast
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import scyjava

from refinery.refine import mean_shift_point
from refinery.util.chunkutil import chunk_center
from refinery.util.imgutil import get_hyperslice
from refinery.util.ioutil import ImgReaderFactory
from refinery.util.java import imglib2, imagej1
from refinery.util.java import snt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="s3://janelia-mouselight-imagery/carveouts/2018-08-01/fluorescence-near-consensus.n5",
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
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\patches-meanshift",
    )
    parser.add_argument(
        "--do-mean-shift",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--mean-shift-iter",
        type=int,
        default=1
    )
    parser.add_argument(
        "--mean-shift-radius",
        type=float,
        default=5
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1
    )
    args = parser.parse_args()
    return args


def patches_from_points(img, points, block_size):
    logging.debug(f"Using window size: {block_size}")
    patches = []
    for p in points:
        interval = chunk_center([p.getX(), p.getY(), p.getZ()], block_size)
        patch = imglib2.Views.interval(img, interval)
        patches.append(patch)
    return patches


def save_patch(patch, path):
    imp = imglib2.ImageJFunctions.wrap(patch, "")
    imagej1.IJ.saveAsTiff(imp, str(path))


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


def mean_shift_points(points, img, radius=4, n_iter=1, n_threads=1):
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        executor.map(
            mean_shift_point,
            points,
            repeat(img),
            repeat(radius),
            repeat(n_iter),
        )


def interploate_img(img):
    floatImg = imglib2.Converters.convert(
        imglib2.RandomAccessibleInterval @ img,
        imglib2.RealDoubleConverter(),
        imglib2.DoubleType(),
    )
    interpolant = imglib2.Views.interpolate(
        imglib2.Views.extendZero(floatImg),
        imglib2.NLinearInterpolatorFactory(),
    )

    return interpolant


def main():
    scyjava.start_jvm()

    logging.getLogger().setLevel(logging.DEBUG)

    args = parse_args()

    output = Path(args.output)

    block_size = list(ast.literal_eval(args.block_size))
    logging.info(f"Using block size: {block_size}")

    reader = ImgReaderFactory.create(args.image)
    img = get_hyperslice(reader.load(args.image, key=args.dataset), ndim=3)

    swc_dir = Path(args.swcs)

    for swc in swc_dir.iterdir():
        if not swc.name.endswith(".swc"):
            continue

        logging.info(f"Processing {swc}")

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
            interpolant = interploate_img(img)
            logging.info(f"Doing mean shift for {struct}...")
            mean_shift_points(
                points=points,
                img=interpolant,
                radius=args.mean_shift_radius,
                n_iter=args.mean_shift_iter,
                n_threads=args.threads,
            )
            logging.info("Mean shift done.")

        logging.info("Computing patches")
        patches = patches_from_points(
            img, points, block_size=block_size
        )
        logging.info("Saving point coordinates as SWC")
        save_points(points, point_dir / struct)
        logging.info("Saving patches")
        save_patches(patches, image_dir / struct, n_threads=args.threads)


if __name__ == "__main__":
    main()
