import gzip
from pathlib import Path

import numpy as np
import zarr
from distributed import Client, LocalCluster
from numcodecs import blosc

blosc.use_threads = False


def fill_to_indices(fill_path):
    coords = []
    with gzip.open(fill_path, 'rb') as f:
        for line in f.readlines():
            parts = line.strip().split()
            zyx = list(reversed([int(p) for p in parts]))
            coords.append(zyx)
    return np.array(coords, dtype=int)


def bbox(points):
    return points.min(axis=0), points.max(axis=0)


def setOne(za, indices):
    za.vindex[indices[:, 0], indices[:, 1], indices[:, 2]] = 1


def setGray(za, ds, indices):
    zs = indices[:, 0]
    ys = indices[:, 1]
    xs = indices[:, 2]
    za[zs, ys, xs] = ds[0, zs, ys, xs]


if __name__ == "__main__":

    client = Client(
        LocalCluster(
            processes=True,
            threads_per_worker=1,
            n_workers=4,
            dashboard_address=":1234"
        )
    )

    fill_dir = Path(r"C:\Users\cameron.arshadi\Desktop\2018-10-01\fills-axons")

    out_mask_dir = Path(r"C:\Users\cameron.arshadi\Desktop\2018-10-01\masks-axons")
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    n5_url = "s3://janelia-mouselight-imagery/carveouts/2018-08-01/fluorescence-near-consensus.n5"
    store = zarr.N5FSStore(n5_url)
    ds = zarr.open(store, 'r')['volume-rechunked']
    full_shape = ds.shape[1:]
    print(full_shape)

    compressor = blosc.Blosc(cname="zstd", clevel=1, shuffle=blosc.Blosc.SHUFFLE)

    store = zarr.DirectoryStore(r"C:\Users\cameron.arshadi\Desktop\2018-10-01\consensus-fill.zarr")
    za = zarr.create(
        store=store,
        shape=full_shape,
        chunks=(64, 64, 64),
        dtype=np.uint16,
        compressor=compressor,
        fill_value=0,
        write_empty_chunks=False,
        overwrite=True
    )

    for p in fill_dir.iterdir():
        if "Fill" not in p.name:
            continue
        indices = fill_to_indices(p)
        print(indices.shape)

        bmin, bmax = bbox(indices)
        print(bmin, bmax)
        print((np.product(bmax - bmin + 1) * 2) / (1024 ** 2))

        setGray(za, ds, indices)

    # d = da.from_array(za, chunks=(256, 256, 256))
    # print(d.chunks)
    #
    # block = d[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]]
    # print(block.shape)
    #
    # out_mask = out_mask_dir / (p.stem.replace(".txt", "") + ".zarr")
    # print(out_mask)
    #
    # del ds
    # del store
    #
    # print("Writing mask")
    # block = block.rechunk(256, 256, 256)
    # da.to_zarr(block, out_mask, compressor=compressor, overwrite=True)

    # tifffile.imwrite(out_mask, block, compression="ZLIB")

    print("done.")
