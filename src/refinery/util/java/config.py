import os

_default_snt_version = "4.0.3"
_snt_version = os.getenv(
    "SNT_VERSION", _default_snt_version
)

_default_fiji_version = "2.9.0"
_fiji_version = os.getenv(
    "FIJI_VERSION", _default_fiji_version
)

_default_java_opts = None
_java_opts = os.getenv(
    "JAVA_OPTS", _default_java_opts
)

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


def get_java_opts():
    global _java_opts
    return _java_opts