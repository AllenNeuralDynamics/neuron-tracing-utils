import argparse
import logging
import os
from pathlib import Path

from ..util import swcutil


def _align_to_cube(in_swc_dir, out_swc_dir):
    """One-off helper function to align jws-created swcs to the origin of
    the training ROI, where the origin is given by the first point
    in the "cube.swc" file"""

    for root, dirs, files in os.walk(in_swc_dir):
        if not files:
            continue
        cube_swc = [f for f in files if f.startswith("cube")][0]
        cube_arr = swcutil.swc_to_ndarray(os.path.join(root, cube_swc))
        origin = cube_arr[0, 2:5]
        for f in files:
            if not f.endswith(".swc"):
                continue
            if "cube" in f:
                continue
            swc = os.path.join(root, f)
            outswc = os.path.join(out_swc_dir, os.path.relpath(swc, in_swc_dir))
            Path(outswc).parent.mkdir(exist_ok=True, parents=True)
            arr = swcutil.swc_to_ndarray(swc, True)
            arr[:, 2:5] -= origin
            swcutil.ndarray_to_swc(arr, outswc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='directory of .swc files to align')
    parser.add_argument('--output', type=str,  help='directory to output aligned .swc files')
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    logging.info("Starting alignment...")
    _align_to_cube(args.input, args.output)
    logging.info("Finished alignment...")


if __name__ == "__main__":
    main()
