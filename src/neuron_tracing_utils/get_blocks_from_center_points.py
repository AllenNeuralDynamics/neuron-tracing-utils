import json
import logging
import os
import sys
from pathlib import Path

import argschema
import dask.array as da
import numpy as np
import scyjava
import tifffile
import zarr

from neuron_tracing_utils.util import chunkutil
from neuron_tracing_utils.util.java import snt

SWC_EXTENSION = ".swc"

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
_LOGGER = logging.getLogger(__name__)


class InputParameters(argschema.ArgSchema):
    points_file = argschema.fields.Str()
    image_path = argschema.fields.Str()
    out_dir = argschema.fields.Str()
    swc_dir = argschema.fields.Str()
    voxel_spacing = argschema.fields.List(argschema.fields.Float, cli_as_single_argument=True)
    block_shape = argschema.fields.List(argschema.fields.Int, cli_as_single_argument=True)


def crop_swc(swc, min, max):
    """
    Crop the given SWC file between the given min and max coordinates.

    Parameters
    ----------
    swc : str
        Path to the SWC file.
    min : list
        Minimum x, y, z coordinates.
    max : list
        Maximum x, y, z coordinates.

    Returns
    -------
    list
        Cropped trees.
    """
    g = snt.Tree(swc).getGraph()
    # Remove vertices outside the bounding box
    to_remove = [
        v for v in g.vertexSet() if
        not (
            min[0] <= v.z < max[0] and
            min[1] <= v.y < max[1] and
            min[2] <= v.x < max[2]
        )
    ]

    for v in to_remove:
        g.removeVertex(v)

    if g.vertexSet().isEmpty():
        return None

    # Align tracing to block origin (0,0,0)
    trees = []
    for c in g.getComponents():
        t = c.getTree()
        t.translate(-min[2], -min[1], -min[0])
        trees.append(t)

    return trees


def crop_swc_dir(swc_dir, out_dir, min, max):
    """
    Crop all SWC files in a directory.

    Parameters
    ----------
    swc_dir : str
        Directory containing SWC files.
    out_dir : str
        Output directory.
    min : list
        Minimum coordinates.
    max : list
        Maximum coordinates.
    """
    swcs = [os.path.join(swc_dir, f) for f in os.listdir(swc_dir) if f.endswith(SWC_EXTENSION)]

    for swc in swcs:
        _LOGGER.info(f"Cropping {swc}")
        trees = crop_swc(swc, min, max)

        if trees is None:
            _LOGGER.info(f"No trees left after cropping {swc}")
            continue

        for ti, t in enumerate(trees):
            t.saveAsSWC(os.path.join(out_dir, Path(swc).name.replace(SWC_EXTENSION, f"-{ti}-cropped.swc")))


def save_block(out_dir, i, block, image_path, block_origin, block_shape):
    """
    Save block data to a file.

    Parameters
    ----------
    out_dir : str
        Output directory.
    i : int
        Index of the block.
    block : ndarray
        Data to be saved.
    image_path : str
        Path to input image.
    block_origin : list
        Origin of the chunk.
    block_shape : list
        Shape of the chunk.
    """
    block_dir = os.path.join(out_dir, f"block_{i:03d}")
    os.makedirs(block_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(block_dir, f"block_{i:03d}.tiff"), block)
    meta = {
        "chunk_origin": list(reversed(block_origin.tolist())),
        "chunk_shape": list(reversed(block_shape.tolist())),
        "image_url": image_path
    }
    with open(os.path.join(block_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f)


def read_points_from_file(filepath):
    """
    Read points from the given file.

    Parameters
    ----------
    filepath : str
        Path to the file containing points.

    Returns
    -------
    list
        List of points.
    """
    if not os.path.exists(filepath):
        _LOGGER.error(f"Points file does not exist: {filepath}")
        sys.exit(1)

    points = []
    with open(filepath, 'r') as f:
        for line in f:
            point = json.loads(line.strip())
            points.append(point)
    return points


def main():
    parser = argschema.ArgSchemaParser(schema_type=InputParameters)
    args = parser.args
    _LOGGER.info(args)

    scyjava.start_jvm()

    points = read_points_from_file(args['points_file'])

    path = args['image_path']
    z = zarr.open(zarr.N5FSStore(path), "r")
    d = da.from_array(z['s0'])

    out_dir = args['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    swc_dir = args['swc_dir']
    voxel_spacing = np.array(args['voxel_spacing'])
    shape = np.array(args['block_shape'])

    for i, point in enumerate(points):
        point = np.array(list(reversed(point)))
        point_vx = point / voxel_spacing

        origin, corner = chunkutil.chunk_center(point_vx, shape)

        data = d[
               origin[0]: corner[0],
               origin[1]: corner[1],
               origin[2]: corner[2],
               ].compute()

        save_block(out_dir, i, data, path, origin, shape)

        if swc_dir is not None:
            crop_swc_dir(swc_dir, os.path.join(out_dir, f"block_{i:03d}"), origin, corner)


if __name__ == "__main__":
    main()
