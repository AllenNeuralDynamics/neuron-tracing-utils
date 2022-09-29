import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile
import zarr

from util.java.config import get_snt_version, get_fiji_version


def get_tiff_shape(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        # Open as zarr-store in case we're dealing with
        # ImageJ hyperstacks.
        # tifffile.TiffFile is unable to parse Z-slices with ImageJ
        # Tiffs larger than 4GB.
        z = zarr.open(tif.aszarr(), "r")
        return z.shape


def copy_blocks(src: Path, dst: Path, voxel_spacing: list):
    block_dir = src / "blocks"
    for block in block_dir.iterdir():
        if not block.name.startswith("block"):
            continue

        stack_dir = block / "stack"
        stack = stack_dir / f"{block.name}.tif"

        out_block = dst / block.relative_to(src)
        out_block.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(stack, out_block / "input.tif")

        block_metadata = {}

        # TODO: refactor for block creation script change
        origin_file = block / "origin_vx.txt"
        with open(origin_file, 'r') as f:
            s = f.readline()
            s = s.replace("[", "")
            s = s.replace("]", "")
            o = np.fromstring(s, dtype=int, sep=' ')
            origin = o.tolist()
            block_metadata['chunk_origin'] = origin

        block_shape = list(reversed(get_tiff_shape(stack)))
        block_metadata['chunk_shape'] = block_shape

        block_metadata['voxel_spacing'] = voxel_spacing

        block_metadata['provenance'] = ""

        with open(out_block / "metadata.json", 'w') as f:
            json.dump(block_metadata, f, indent=4)


def find_args(d: Path):
    for f in d.iterdir():
        if f.name == "args.json":
            with open(f, 'r') as jsonf:
                return json.load(jsonf)
    return None


def copy_masks(src: Path, dst: Path):
    block_pattern = re.compile(r"block_\d+")
    mask_pattern = re.compile(r"Fill_(Gray|Label)_Mask.tif$")

    blocks_dir = dst / "blocks"
    assert blocks_dir.is_dir()

    for p in src.iterdir():
        if not p.is_dir():
            continue
        if "masks" not in p.name:
            continue
        args = find_args(p)
        if args is None:
            raise ValueError(f"Could not find args.json in {p}")

        fill_threshold = float(args['threshold'])
        fill_cost_function = str(args['cost'])

        for f in p.iterdir():
            m = block_pattern.search(f.name)
            if not m:
                continue
            block_id = m.group(0)

            m = mask_pattern.search(f.name)
            mask_type = m.group(0)

            block_dir = blocks_dir / block_id
            shutil.copyfile(f, block_dir / mask_type)

            metadata_file = block_dir / "metadata.json"
            with open(metadata_file, 'r') as mf:
                metadata = json.load(mf)

            metadata['fill_threshold'] = fill_threshold
            metadata['fill_cost_function'] = fill_cost_function
            metadata['fill_method'] = "dijkstra"

            with open(metadata_file, 'w') as mf:
                json.dump(metadata, mf, indent=4)


def copy_swcs(src: Path, dst: Path):
    swc_dir = src / "swcs" / "final-trees"
    assert swc_dir.is_dir()

    out_swc_dir = dst / "swcs"
    out_swc_dir.mkdir(parents=True, exist_ok=True)

    for block_dir in swc_dir.iterdir():
        if not block_dir.is_dir():
            continue
        block_name = block_dir.name

        out_block_dir = out_swc_dir / block_name / "final-trees"
        out_block_dir.mkdir(parents=True, exist_ok=True)

        swcs = [f for f in block_dir.iterdir() if f.name.endswith(".swc")]
        for i, swc in enumerate(swcs):
            new_name = f"SNT_Data-{i:04}.swc"
            shutil.copyfile(swc, out_block_dir / new_name)


def remove_blocks_without_swcs(dst: Path):
    blocks_dir = dst / "blocks"
    swcs_dir = dst / "swcs"

    valid_blocks = [b.name for b in swcs_dir.iterdir() if b.name.startswith("block")]

    for b in blocks_dir.iterdir():
        if b.name not in valid_blocks:
            shutil.rmtree(b)


def write_component_versions(dst: Path):
    blocks_dir = dst / "blocks"

    for b in blocks_dir.iterdir():
        with open(b / "metadata.json", "r") as f:
            metadata = json.load(f)

        metadata['snt_version'] = get_snt_version()
        metadata['fiji_version'] = get_fiji_version()
        metadata['refinery_commit_sha'] = get_git_revision_hash()
        metadata['refinery_url'] = get_source_url()

        with open(b / "metadata.json", "w") as f:
            json.dump(metadata, f)


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_source_url() -> str:
    import toml
    data = toml.loads(find_pyproject().read_text())
    return data['project']['urls']['source']


def find_pyproject():
    curdir = Path(os.path.dirname(os.path.abspath(__file__)))
    root = curdir.parent.parent
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"Could not find pyproject.toml in {root}")
    return pyproject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--voxel-size", type=float, nargs='+', default=[1.0, 1.0, 1.0])
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    os.makedirs(out_dir, exist_ok=True)

    copy_blocks(in_dir, out_dir, args.voxel_size)
    copy_masks(in_dir, out_dir)
    copy_swcs(in_dir, out_dir)

    remove_blocks_without_swcs(out_dir)

    write_component_versions(out_dir)


if __name__ == "__main__":
    main()
