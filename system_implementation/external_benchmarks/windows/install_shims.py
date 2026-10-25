"""Enable the Home Assistant Windows shims in the Python environment that runs this script.

    ../external/venvs/homeassistant/Scripts/python.exe external_benchmarks/windows/install_shims.py

Writes ha_windows_shims.pth into that environment's site-packages; delete it to undo.
fcntl.py and resource.py stand in for POSIX-only modules that Home Assistant imports but the
pytest harness never uses; ha_windows_compat.py lets asyncio's socketpair() pass pytest-socket.
"""
import sys
import sysconfig
from pathlib import Path

if sys.platform != "win32":
    raise SystemExit("The shims are only needed on Windows.")
pth = Path(sysconfig.get_paths()["purelib"]) / "ha_windows_shims.pth"
pth.write_text(Path(__file__).resolve().parent.as_posix() + "\nimport ha_windows_compat\n", encoding="utf-8")
print(pth)
