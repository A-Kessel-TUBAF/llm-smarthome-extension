"""Windows stand-in for the POSIX-only fcntl module (Home Assistant import-time only).

homeassistant.runner imports fcntl at module level; it is used only by
ensure_single_execution() (single-instance lock file), which the pytest harness
never calls. flock() is a no-op here; fcntl()/ioctl() raise OSError.
Delete ha_windows_shims.pth (see install_shims.py) to restore stock behaviour.
"""
import errno as _errno

LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8
F_GETFL, F_SETFL, F_GETFD, F_SETFD = 3, 4, 1, 2
FD_CLOEXEC = 1


def flock(fd, operation):
    return None


def lockf(fd, cmd, len=0, start=0, whence=0):
    return None


def fcntl(fd, cmd, arg=0):
    raise OSError(_errno.ENOSYS, "fcntl.fcntl is not available on Windows (shim)")


def ioctl(fd, request, arg=0, mutate_flag=True):
    raise OSError(_errno.ENOSYS, "fcntl.ioctl is not available on Windows (shim)")
