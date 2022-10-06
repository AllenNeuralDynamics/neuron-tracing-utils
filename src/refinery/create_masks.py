import argparse
import gzip
import logging
import multiprocessing
import time
from enum import Enum
from functools import wraps
from pathlib import Path

import numpy as np
import zarr
from distributed import Client, LocalCluster
from numcodecs import blosc
from zarr.errors import PathNotFoundError

blosc.use_threads = False


class MaskType(Enum):
    binary = "binary"
    grayscale = "grayscale"
    labels = "labels"


def fill_to_indices(fill_path):
    coords = []
    with gzip.open(fill_path, "rb") as f:
        for line in f.readlines():
            parts = line.strip().split()
            zyx = list(reversed([int(p) for p in parts]))
            coords.append(zyx)
    return np.array(coords, dtype=int)


def bbox(points):
    return points.min(axis=0), points.max(axis=0)


def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time()
        logging.info(f"{func.__name__} took {t1 - t0}s")
        return ret

    return _time_it


@measure
def set_voxels(za, ds, indices, mask_type):
    if mask_type == MaskType.labels.value:
        raise NotImplementedError  # TODO
    elif mask_type == MaskType.grayscale.value:
        set_gray(za, ds, indices)
    elif mask_type == MaskType.binary.value:
        set_const(za, indices, value=255)
    else:
        raise ValueError(f"Invalid value for mask type: {mask_type}")


def set_const(za, indices, value=255):
    za.vindex[indices[:, 0], indices[:, 1], indices[:, 2]] = value


def set_gray(za, ds, indices):
    zs = indices[:, 0]
    ys = indices[:, 1]
    xs = indices[:, 2]
    # FIXME: assume the signal is always in channel/timepoint 0
    za.vindex[zs, ys, xs] = ds.vindex[
        tuple([0] * (len(ds.shape) - 3)) + (zs, ys, xs)
    ]


def get_dtype(mask_type, fills, ds):
    if mask_type == MaskType.labels.value:
        return get_labels_dtype(len(fills))
    elif mask_type == MaskType.grayscale.value:
        return ds.dtype
    elif mask_type == MaskType.binary.value:
        return np.uint8
    else:
        raise ValueError(f"Invalid value for mask type: {mask_type}")


def get_labels_dtype(n_labels):
    """
    Calculate the mask datatype given the cardinality of the label set.
    Labels can take any value in [1, 2 ** 64 - 1].
    0 is reserved for background.
    """
    if n_labels == 0:
        raise ValueError("n_labels must be > 0")
    elif 1 <= n_labels < 256:
        return np.uint8
    elif 256 <= n_labels < 2**16:
        return np.uint16
    elif 2**16 <= n_labels < 2**32:
        return np.uint32
    else:
        return np.uint64


def create_client(n_workers):
    client = Client(
        LocalCluster(
            processes=True,
            threads_per_worker=1,
            n_workers=n_workers,
            # dashboard_address=":1234"
        )
    )
    return client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--fills", type=str, help="directory of stored Fills"
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="path or url of the input image, either zarr or n5",
    )
    parser.add_argument("-o", "--output", type=str, help="the output zarr")
    parser.add_argument(
        "--type",
        type=str,
        choices=[mask_type.value for mask_type in MaskType],
        default=MaskType.grayscale.value,
        help="type of mask to output",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="chunk size for output mask zarr array",
    )
    parser.add_argument("-n", "--n-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    logging.getLogger().setLevel(logging.INFO)

    fill_dir = Path(args.fills)
    fills = [p for p in fill_dir.iterdir()]
    if not fills:
        logging.warning("Fill directory is empty, exiting.")
        return

    try:
        z = zarr.open(args.image, "r")
    except PathNotFoundError:
        try:
            z = zarr.open(store=zarr.N5FSStore(args.image), mode="r")
        except Exception as e:
            logging.error(
                f"Could not open {args.image} as either n5 or zarr. Error:\n{e}"
            )
            return

    # TODO: better way to specify array path
    array_key = next(iter(z.keys()))
    logging.info(array_key)
    ds = z[array_key]

    # get the spatial dimensions
    shape = ds.shape[-3:]
    logging.info(f"Full shape: {ds.shape}")
    logging.info(f"Spatial dims: {shape}")

    n_workers = max(1, min(args.n_workers, multiprocessing.cpu_count()))
    client = create_client(n_workers)

    compressor = blosc.Blosc(
        cname="zstd", clevel=1, shuffle=blosc.Blosc.SHUFFLE
    )

    chunks = tuple(args.chunks)

    za = zarr.create(
        store=args.output,
        shape=shape,
        chunks=chunks,
        dtype=get_dtype(args.type, fills, ds),
        compressor=compressor,
        fill_value=0,
        write_empty_chunks=False,
        overwrite=True,
    )

    t0 = time.time()
    for fill_path in fills:
        logging.info(f"Processing {fill_path}")
        indices = fill_to_indices(fill_path)
        logging.info(f"Num filled voxels: {indices.shape[0]}")

        bmin, bmax = bbox(indices)
        bbox_shape = bmax - bmin + 1
        logging.info(f"Bounding box of Fill: {bmin} {bmax}")
        logging.info(f"Bounding box shape: {bbox_shape}")
        logging.info(
            f"Bounding box size: {(np.product(bbox_shape) * ds.itemsize) / (1024 ** 2)}"
        )

        set_voxels(za, ds, indices, args.type)

        # block = za[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]]
        # tifffile.imwrite(r"C:\Users\cameron.arshadi\Desktop\block.tif", block)

    t1 = time.time()
    logging.info(f"Done. Took {t1 - t0}s")


if __name__ == "__main__":
    main()
