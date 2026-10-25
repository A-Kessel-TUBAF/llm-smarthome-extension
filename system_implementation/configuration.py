"""Shared configuration; no agent imports."""
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "experiment_config.toml"


def load_config(path=None):
    with Path(path or os.getenv("SMARTHOME_CONFIG") or CONFIG_PATH).open("rb") as file:
        return tomllib.load(file)


def activate_config(path):
    if path:
        data = load_config(path)
        if "mas" in data or isinstance(data.get("model"), dict):
            os.environ["SMARTHOME_CONFIG"] = str(Path(path).resolve())


def system_settings(name, path=None):
    defaults = load_config(CONFIG_PATH)
    selected = load_config(path)
    settings = {} if name == "mas" else dict(defaults["model"])
    settings.update(defaults[name])
    if "mas" in selected or isinstance(selected.get("model"), dict):
        if name != "mas":
            settings.update(selected.get("model", {}))
        settings.update(selected.get(name, {}))
    else:
        settings.update(selected)  # Older flat participant configs.
    return settings


def config_path(value, executable=False):
    """Normalize project paths without resolving venv symlinks."""
    if not value or (executable and "/" not in value and "\\" not in value):
        return value or None
    return os.path.abspath(ROOT / Path(value).expanduser())
