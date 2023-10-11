import os

_default_snt_version = "4.2.1"
_snt_version = os.getenv("SNT_VERSION", _default_snt_version)

_default_fiji_version = "2.14.0"
_fiji_version = os.getenv("FIJI_VERSION", _default_fiji_version)

_default_mobie_io_version = "2.1.2"
_mobie_io_version = os.getenv("MOBIE_IO_VERSION", _default_mobie_io_version)

_default_java_opts = None
_java_opts = os.getenv("JAVA_OPTS", _default_java_opts)

_default_local_fiji = None
_local_fiji = os.getenv("FIJI_PATH", _default_local_fiji)


def set_snt_version(version):
    global _snt_version
    _snt_version = version


def get_snt_version():
    global _snt_version
    return _snt_version


def set_fiji_version(version):
    global _fiji_version
    _fiji_version = version


def get_fiji_version():
    global _fiji_version
    return _fiji_version


def set_mobie_io_version(version):
    global _mobie_io_version
    _mobie_io_version = version


def get_mobie_io_version():
    global _mobie_io_version
    return _mobie_io_version


def get_java_opts():
    global _java_opts
    return _java_opts


def get_local_fiji_path():
    global _local_fiji
    return _local_fiji


def set_local_fiji_path(path):
    global _local_fiji
    _local_fiji = path
