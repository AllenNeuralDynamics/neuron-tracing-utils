import logging

from .config import get_fiji_version, get_snt_version

import scyjava.config as sjconf


_logger = logging.getLogger(__name__)

_snt_version = get_snt_version()
_SNT_ENDPOINT = f"org.morphonets:SNT:{_snt_version}"
sjconf.endpoints.append(_SNT_ENDPOINT)

_fiji_version = get_fiji_version()
_FIJI_ENDPOINT = f"sc.fiji:fiji:{_fiji_version}"
sjconf.endpoints.append(_FIJI_ENDPOINT)
