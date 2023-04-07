import argparse
import json
import os
from pathlib import Path

import numpy as np
from tifffile import tifffile

from neuron_tracing_utils.util import swcutil
from neuron_tracing_utils.util.ioutil import open_n5_zarr_as_ndarray


def rand_block(ds, block_shape):
    ds_shape = np.array([s for s in ds.shape], dtype=int)
    block_shape = np.array([s for s in block_shape], dtype=int)
    mn = np.array([0, 0, 0])
    mx = ds_shape - block_shape
    origin = np.zeros(3, dtype=int)
    origin[0] = np.random.randint(mn[0], mx[0], size=1)
    origin[1] = np.random.randint(mn[1], mx[1], size=1)
    origin[2] = np.random.randint(mn[2], mx[2], size=1)
    block = ds[
            origin[0]:origin[0] + block_shape[0],
            origin[1]:origin[1] + block_shape[1],
            origin[2]:origin[2] + block_shape[2]
            ]
    return block, origin


def from_points_bbox(ds, points, pad_px=10):
    mn, mx = np.min(points, axis=0), np.max(points, axis=0)
    origin = np.array(
        [
            max(mn[0] - pad_px, 0),
            max(mn[1] - pad_px, 0),
            max(mn[2] - pad_px, 0)
        ]
    )
    corner = np.array(
        [
            min(mx[0] + pad_px + 1, ds.shape[0]),
            min(mx[1] + pad_px + 1, ds.shape[1]),
            min(mx[2] + pad_px + 1, ds.shape[2])
        ]
    )
    block = ds[origin[0]:corner[0], origin[1]:corner[1], origin[2]:corner[2]]
    return block, origin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="path to the zarr/n5 image"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="s0",
        help="the n5/zarr key to the dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output directory"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        help="size of the subvolume in XYZ order (ignored when mode=swc)"
    )
    parser.add_argument(
        "--swc",
        type=str,
        help="path to swc (only used when mode=swc)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "swc"],
        default="swc"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    f = open_n5_zarr_as_ndarray(args.input)
    # load full res
    ds = f[args.dataset]
    print(f"dataset shape {ds.shape}")

    if args.mode == "random":
        block_shape = np.flip(args.block_size).astype(int)
        print(f"block shape {block_shape}")
        block, origin_vx = rand_block(ds, block_shape)
    elif args.mode == "swc":
        arr = swcutil.swc_to_ndarray(args.swc)
        points = np.flip(arr[:, 2:5]).astype(int)
        block, origin_vx = from_points_bbox(ds, points)
        block_shape = np.array(block.shape)
        print(f"block shape {block_shape}")
        arr[:, 2:5] -= np.flip(origin_vx)
        swcutil.ndarray_to_swc(
            arr,
            os.path.join(args.output, Path(args.swc).stem + "_aligned.swc")
        )
    else:
        raise ValueError(f"Invalid mode {args.mode}")

    with open(os.path.join(args.output, "metadata.json"), 'w') as locfile:
        # These are in XYZ order
        meta = {
            "tile_name": Path(args.input).name,
            "chunk_origin": origin_vx[[2, 1, 0]].tolist(),
            "chunk_shape": block_shape[[2, 1, 0]].tolist(),
            "image_url": args.input
        }
        print(meta)
        json.dump(meta, locfile)

    stackdir = os.path.join(args.output, "stack")
    os.makedirs(stackdir, exist_ok=True)
    tifffile.imwrite(os.path.join(stackdir, f"{Path(args.output).name}.tif"), block)


main()
