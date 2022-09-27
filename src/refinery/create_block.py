import argparse
import json
import os
from pathlib import Path

import hdf5plugin
import h5py as h5py
import numpy as np
from tifffile import tifffile


def get_rand_block(ds, block_shape):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=r"Z:\mnt\vast\aind\exaSPIM\exaSPIM_125L_20220805_172536\micr\tile_x_0001_y_0002_z_0000_ch_0000.ims",
        help="path to the image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=rf"C:\Users\cameron.arshadi\Desktop\repos\exaSpim-training-data\20220805_172536\blocks\block_53",
        help="output directory"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="+",
        default=[330,330,110],
        help="size of the subvolume in XYZ order"
    )
    args = parser.parse_args()

    block_id = Path(args.output).name

    block_shape = np.flip(args.block_size).astype(int)
    print(f"block shape {block_shape}")

    os.makedirs(args.output, exist_ok=True)

    with h5py.File(args.input, 'r') as f:
        # load full res
        ds = f["/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
        print(f"dataset shape {ds.shape}")
        block, origin_vx = get_rand_block(ds, block_shape)

        with open(os.path.join(args.output, "metadata.json"), 'w') as locfile:
            # These are in XYZ order
            meta = {
                "tile_name": Path(args.input).name,
                "chunk_origin": origin_vx[[2, 1, 0]].tolist(),
                "chunk_shape": block_shape[[2, 1, 0]].tolist()
            }
            print(meta)
            json.dump(meta, locfile)

        slicedir = os.path.join(args.output, "slices")
        os.makedirs(slicedir, exist_ok=True)
        for i in range(block.shape[0]):
            tifffile.imwrite(os.path.join(slicedir, f"{i:04d}.tif"), block[i])

        stackdir = os.path.join(args.output, "stack")
        os.makedirs(stackdir, exist_ok=True)
        tifffile.imwrite(os.path.join(stackdir, f"{block_id}.tif"), block)


main()