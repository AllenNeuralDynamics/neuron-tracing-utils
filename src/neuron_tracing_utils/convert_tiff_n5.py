import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Union, Any

import tifffile
import zarr
from numcodecs import GZip


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


PathLike = Union[os.PathLike, str]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="directory of Tiffs")
    parser.add_argument("-o", "--output", type=str, help="directory to output N5 images")
    parser.add_argument(
        "--copy-metadata",
        default=False,
        action="store_true",
        help="copy the metadata.json file to the output"
    )
    args = parser.parse_args()

    block_dir = Path(args.input)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for block in block_dir.iterdir():
        with open(block / "metadata.json", 'r') as f:
            metadata = json.load(f)
        voxel_size = metadata['voxel_spacing']
        out_block = out_dir / block.relative_to(block_dir)
        convert_tiff_to_n5(block / "input.tif", out_block / "input.n5", voxel_size=voxel_size)
        convert_tiff_to_n5(block / "Fill_Gray_Mask.tif", out_block / "Fill_Gray_Mask.n5", voxel_size=voxel_size)
        convert_tiff_to_n5(block / "Fill_Label_Mask.tif", out_block / "Fill_Label_Mask.n5", voxel_size=voxel_size)

        if args.copy_metadata:
            LOGGER.info(f"Copying metadata.json to {out_block}")
            shutil.copyfile(block / "metadata.json", out_block / "metadata.json")


def convert_tiff_to_n5(
        in_path: PathLike,
        out_path: PathLike,
        array_key: str = "volume",
        chunks: tuple = (64, 64, 64),
        compressor: Any = GZip(5),
        voxel_size: list = None
) -> None:
    """
    Convert a Tiff stack to an N5 dataset.

    Args:
         in_path: the path to the Tiff
         out_path: the path to write the N5
         array_key: the key for the array data
         chunks: the chunk shape of the N5 dataset
         compressor: the numcodecs compressor instance for the N5 dataset
         voxel_size: the voxel spacing of the image, in nanometers
    """
    LOGGER.info(f"converting {in_path}")
    im = tifffile.imread(in_path).squeeze()
    z = zarr.open(store=zarr.N5Store(out_path), mode='w')
    ds = z.create_dataset(
        array_key,
        shape=im.shape,
        chunks=chunks,
        dtype=im.dtype,
        compressor=compressor
    )
    ds[:] = im
    ds.attrs['resolution'] = voxel_size
    ds.attrs['units'] = ["nm", "nm", "nm"]


if __name__ == "__main__":
    main()
