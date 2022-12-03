import logging
import os.path
from urllib.parse import urlparse

from refinery.util.java import imglib2, n5

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


class ImgReaderFactory:
    LOADERS = {
        ".tif": TiffReader,
        ".tiff": TiffReader,
        ".n5": N5Reader
    }

    @staticmethod
    def create(path):
        _, ext = os.path.splitext(path)
        return ImgReaderFactory.LOADERS[ext]()


def get_file_format(imdir):
    f = next(iter(os.listdir(imdir)))
    _, ext = os.path.splitext(f)
    return ext
