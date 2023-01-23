import logging
import os.path
from urllib.parse import urlparse

import tensorstore as ts
import zarr
from tensorstore import TensorStore
from zarr.errors import PathNotFoundError

from refinery.util.java import imglib2, n5, aws

_LOGGER = logging.getLogger(__name__)


class TiffReader:

    def __init__(self):
        self.loader = imglib2.IJLoader()

    def load(self, filepath, **kwargs):
        if "key" in kwargs:
            _LOGGER.warning("Ignoring keyword argument 'key' for TiffLoader")
        return self.loader.get(filepath)


class N5Reader:

    @staticmethod
    def _get_reader(path):
        path = str(path)

        parsed = urlparse(path)
        bucket = parsed.netloc
        prefix = parsed.path

        if path.startswith("s3://"):
            s3 = n5.AmazonS3ClientBuilder.defaultClient()
            reader = n5.N5AmazonS3Reader(s3, bucket, prefix)
            return reader
        elif path.startswith("gs://"):
            # TODO
            raise NotImplementedError("GCS is not currently supported")
        else:
            reader = n5.N5FSReader(path)
            return reader

    def load(self, path, **kwargs):
        key = kwargs.get("key", "volume")
        reader = self._get_reader(path)
        return n5.N5Utils.open(reader, key)


class OmeZarrReader:

    @staticmethod
    def _get_reader(path):
        path = str(path)

        parsed = urlparse(path)
        bucket = parsed.netloc
        prefix = parsed.path

        if path.startswith("s3://"):
            config = aws.ClientConfiguration().withMaxErrorRetry(10).withMaxConnections(32)
            s3 = aws.AmazonS3ClientBuilder.standard().withRegion(aws.Regions.US_WEST_2).withClientConfiguration(
                config
            ).build()
            reader = n5.N5S3OmeZarrReader(s3, None, bucket, prefix.strip('/'), "/")
            return reader
        elif path.startswith("gs://"):
            # TODO
            raise NotImplementedError("GCS is not currently supported")
        else:
            reader = n5.N5OmeZarrReader(path)
            return reader

    def load(self, path, **kwargs):
        key = kwargs.get("key", "volume")
        reader = self._get_reader(path)
        if "cache" in kwargs:
            cache = kwargs["cache"]
            if isinstance(cache, int):
                dataset = n5.N5Utils.openWithBoundedSoftRefCache(reader, key, cache)
            elif isinstance(cache, bool) and cache:
                dataset = n5.N5Utils.openWithDiskCache(reader, key)
            else:
                raise Exception("Expected type int or bool for cache")
        else:
            dataset = n5.N5Utils.open(reader, key)
        return dataset


class ImgReaderFactory:
    LOADERS = {
        ".tif": TiffReader,
        ".tiff": TiffReader,
        ".n5": N5Reader,
        ".zarr": OmeZarrReader
    }

    @staticmethod
    def create(path):
        _, ext = os.path.splitext(path)
        return ImgReaderFactory.LOADERS[ext]()


def get_file_format(imdir):
    f = next(iter(os.listdir(imdir)))
    _, ext = os.path.splitext(f)
    return ext


def open_n5_zarr_as_ndarray(path: str):
    try:
        z = zarr.open(path, "r")
    except PathNotFoundError:
        z = zarr.open(store=zarr.N5FSStore(path), mode="r")
    return z


def open_ts(
        path: str,
        dataset: str = None,
        total_bytes_limit: int = 200_000_000
) -> TensorStore:
    # TensorStore opens n5 with axis order X,Y,Z, so get
    # a transposed view to be compatible with util code
    spec = {
        "driver": _get_driver_string(path),
        "kvstore": path,
        "context": {
            "cache_pool": {
                "total_bytes_limit": total_bytes_limit
            }
        },
        "open": True,
        "recheck_cached_data": "open"
    }
    if dataset is not None:
        spec['path'] = dataset
    ds = ts.open(spec).result()
    if spec['driver'] == "n5":
        return ds.T
    return ds


def _get_driver_string(image_path: str):
    drivers = {
        ".zarr": "zarr",
        ".n5": "n5"
    }
    _, ext = os.path.splitext(image_path)
    return drivers[ext]
