import logging

from .config import *

import scyjava.config as sjconf


_logger = logging.getLogger(__name__)

sjconf.add_option("-Djava.awt.headless=true")

_java_opts = get_java_opts()
if _java_opts is not None:
    for opt in _java_opts.split(" "):
        _logger.info(f"Adding Java option: {opt}")
        sjconf.add_option(opt)

_fiji_version = get_fiji_version()
_FIJI_ENDPOINT = f"sc.fiji:fiji:{_fiji_version}"
sjconf.endpoints.append(_FIJI_ENDPOINT)

_snt_version = get_snt_version()
_SNT_ENDPOINT = f"org.morphonets:SNT:{_snt_version}"
sjconf.endpoints.append(_SNT_ENDPOINT)

sjconf.endpoints.append("net.imagej:imagej-legacy:MANAGED")
