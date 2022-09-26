import argparse
import logging
import os
from pathlib import Path

from refinery.util import swcutil

import numpy as np


def _load_jws_transform(filepath):
    """parse transform.txt file in sample dir and return
    corrected origin and scale ndarrays"""
    transform_dict = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            pair = tuple(p.strip() for p in line.split(':'))
            transform_dict[pair[0]] = float(pair[1])

    origin = np.zeros(3, dtype=float)
    scale = np.zeros(3, dtype=float)
    # offset (nm)
    origin[0] = transform_dict['ox']
    origin[1] = transform_dict['oy']
    origin[2] = transform_dict['oz']
    # voxel spacing (nm)
    scale[0] = transform_dict['sx']
    scale[1] = transform_dict['sy']
    scale[2] = transform_dict['sz']
    # num imagery levels (int)
    nl = transform_dict['nl']

    # scale by number of imagery levels and convert nm to um
    divisor = 2.0 ** (nl - 1)
    scale /= divisor
    scale /= 1000.0
    origin /= 1000.0

    # correct origin for jaws misalignment
    origin_jw = scale * np.floor(origin / scale)

    return origin_jw, scale


class WorldToVoxel:

    def __init__(self, transform_path):
        self.origin, self.scale = _load_jws_transform(transform_path)

    def forward(self, world_coords):
        """m x d world array -> m x d voxel array"""
        # m x d array of world coordinates
        return np.round((world_coords - self.origin) / self.scale)

    def back(self, vox_coords):
        """m x d voxel array -> m x d world array"""
        return vox_coords * self.scale + self.origin


def transform_swcs(indir, outdir, transform: WorldToVoxel, forward, swap_xy):
    for root, dirs, files in os.walk(indir):
        swcs = [f for f in files if f.endswith('.swc')]
        for f in swcs:
            swc_path = os.path.join(root, f)
            outswc = os.path.join(outdir, os.path.relpath(swc_path, indir))
            Path(outswc).parent.mkdir(exist_ok=True, parents=True)
            arr = swcutil.swc_to_ndarray(swc_path, True)
            if forward:
                arr[:, 2:5] = transform.forward(arr[:, 2:5])
            else:
                arr[:, 2:5] = transform.back(arr[:, 2:5])

            if swap_xy:
                arr[:, [2, 3]] = arr[:, [3, 2]]

            swcutil.ndarray_to_swc(arr, outswc)


def main():
    parser = argparse.ArgumentParser(description="Transform .swc files created in the Janelia Workstation from"
                                                 "world coordinates to voxel coordinates.")
    parser.add_argument('--input', type=str, help='directory of .swc files to transform')
    parser.add_argument('--output', type=str,  help='directory to output transformed .swc files')
    parser.add_argument('--transform', type=str, help='path to the \"transform.txt\" file')
    parser.add_argument('--to-world', default=False, action='store_true')
    parser.add_argument('--log-level', type=int, default=logging.INFO)
    parser.add_argument("--swap-xy", default=False, action="store_true")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    forward = not args.to_world

    um2vx = WorldToVoxel(args.transform)

    logging.info("Starting transform...")
    transform_swcs(args.input, args.output, um2vx, forward, args.swap_xy)
    logging.info("Finished transform.")


if __name__ == "__main__":
    main()
