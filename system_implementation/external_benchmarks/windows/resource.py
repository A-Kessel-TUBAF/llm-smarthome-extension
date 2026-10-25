"""Windows stand-in for the POSIX-only resource module (Home Assistant import-time only).

homeassistant.util.resource uses it only to raise the open-file soft limit.
getrlimit() reports a limit high enough that Home Assistant skips setrlimit().
"""
RLIMIT_NOFILE = 7
RLIM_INFINITY = -1
_LIMIT = (8192, 8192)


class error(OSError):
    pass


def getrlimit(resource):
    return _LIMIT


def setrlimit(resource, limits):
    return None
