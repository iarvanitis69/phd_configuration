import json
import os
import yaml

# Resolve the shared configuration beside this loader, independent of its caller.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# Singleton load (μία φορά)
_CONFIG = None
_CONFIG_SOURCE_PATH = None


def _read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        # Keep explicitly selected legacy run snapshots readable.
        data = json.load(f) if os.fspath(path).lower().endswith('.json') else yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def _load_base():
    return _read_config(CONFIG_PATH)


def _load():
    global _CONFIG, _CONFIG_SOURCE_PATH
    if _CONFIG is None:
        _CONFIG = _load_base()
        _CONFIG_SOURCE_PATH = CONFIG_PATH
    return _CONFIG


# Public access
def get(key, default=None):
    return _load().get(key, default)


def all():
    return _load()


def base_all():
    return _load_base()


def source_path():
    _load()
    return _CONFIG_SOURCE_PATH


def activate(path):
    global _CONFIG, _CONFIG_SOURCE_PATH
    _CONFIG = _read_config(path)
    _CONFIG_SOURCE_PATH = path
    return _CONFIG
