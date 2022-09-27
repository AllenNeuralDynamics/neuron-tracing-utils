import argparse
import logging
import os
from pathlib import Path

from refinery.util import swcutil


def _align_to_cube(in_swc_dir, out_swc_dir):
    """One-off helper function to align jws-created swcs to the origin of
    the training ROI, where the origin is given by the first point
    in the "cube.swc" file"""

    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        cube_swc = next(f for f in swcs if f.startswith("cube"))
        swcs.remove(cube_swc)
        cube_arr = swcutil.swc_to_ndarray(os.path.join(root, cube_swc))
        origin = cube_arr[0, 2:5]

        for f in swcs:
            swc_path = os.path.join(root, f)
            arr = swcutil.swc_to_ndarray(swc_path, True)
            arr[:, 2:5] -= origin

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(swc_path, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            swcutil.ndarray_to_swc(arr, out_swc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to align"
    )
    parser.add_argument(
        "--output", type=str, help="directory to output aligned .swc files"
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    logging.info("Starting alignment...")
    _align_to_cube(args.input, args.output)
    logging.info("Finished alignment...")


if __name__ == "__main__":
    main()
