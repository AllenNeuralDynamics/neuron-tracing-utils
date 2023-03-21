import argparse
import logging
from pathlib import Path

import numpy as np
import zarr

from refinery.util import swcutil


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


def map_filename_to_label(swc_dir: Path, label_mask):
    m = {}
    for f in swc_dir.iterdir():
        if f.name.endswith(".swc"):
            arr = swcutil.swc_to_ndarray(f)
            # Get the label of the first point in the swc, assuming
            # all others will have the same label (this may not be true where different neurons intersect).
            # FIXME: this should probably be most commonly occurring label for an swc.
            # round to voxel coordinates
            # convert XYZ -> ZYX
            first_point = np.flip(np.round(arr[0][2:5]).astype(int))
            label = label_mask[tuple(first_point)]
            m[f.stem] = label
            _LOGGER.info(f"{f.name} has label {label}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-mask",
        type=str,
        help="path to the N5 label mask"
    )
    parser.add_argument(
        "--swcs",
        type=str,
        help="directory of swcs associated with the text files"
    )
    parser.add_argument(
        "--text-files",
        type=str,
        help="directory of text files containing voxel indices to update"
    )
    args = parser.parse_args()

    label_mask = zarr.open(zarr.N5Store(args.label_mask), 'r+')['volume']

    m = map_filename_to_label(Path(args.swcs), label_mask)

    label_dir = Path(args.text_files)

    for f in label_dir.iterdir():
        _LOGGER.info(f"adding points from {f}")
        name = f.stem.replace("upd-", "")
        label = m[name]
        # XYZ -> ZYX
        add_points = np.flip(np.loadtxt(f, dtype=int))
        label_mask[tuple(add_points.T)] = label


if __name__ == "__main__":
    main()
