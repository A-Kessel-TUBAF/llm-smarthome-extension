"""Windows compatibility for running Home Assistant's pytest harness (imported via .pth).

On Windows, socket.AF_UNIX does not exist, so socket.socketpair() falls back to a
127.0.0.1 TCP pair. asyncio's Proactor/Selector loops call socketpair() for their
internal self-pipe when every event loop is created. pytest_homeassistant_custom_component
enables pytest-socket (allow_unix_socket=True works on Linux only), which blocks that
fallback and makes every test fail before Home Assistant starts. This wrapper lets
socketpair() alone use the real socket class captured at interpreter start-up; all
other socket creation stays guarded by pytest-socket.
"""
import socket as _socket
import sys as _sys

if _sys.platform == "win32" and not hasattr(_socket, "AF_UNIX"):
    _real_socket = _socket.socket
    _orig_socketpair = _socket.socketpair

    def socketpair(*args, **kwargs):
        current = _socket.socket
        if current is _real_socket:
            return _orig_socketpair(*args, **kwargs)
        _socket.socket = _real_socket
        try:
            return _orig_socketpair(*args, **kwargs)
        finally:
            _socket.socket = current

    socketpair.__doc__ = _orig_socketpair.__doc__
    _socket.socketpair = socketpair
