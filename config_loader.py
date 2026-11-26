import json
import os
import yaml
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

# Resolve the shared configuration beside this loader, independent of its caller.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# Singleton load (μία φορά)
_CONFIG = None
_CONFIG_SOURCE_PATH = None
_FILE_OPENER = open
_RUN_CONFIG = ContextVar('preprocessing_run_config', default=None)
_PINNED_RUN = None


def _read_config(path):
    with _FILE_OPENER(path, "r", encoding="utf-8") as f:
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
        selected = os.environ.get('PHD_PREPROCESSING_CONFIG') or CONFIG_PATH
        _CONFIG = _read_config(selected)
        _CONFIG_SOURCE_PATH = selected
    return _CONFIG


# Public access
def get(key, default=None):
    return _load().get(key, default)


def all():
    return _load()


def base_all():
    pinned = _PINNED_RUN
    if pinned is not None:
        return deepcopy(pinned[1])
    if os.environ.get('PHD_PREPROCESSING_CONFIG'):
        return deepcopy(_load())
    return _load_base()


def source_path():
    _load()
    return _CONFIG_SOURCE_PATH


def activate(path):
    global _CONFIG, _CONFIG_SOURCE_PATH
    if run_config_path() is not None:
        return _load()
    _CONFIG = _read_config(path)
    _CONFIG_SOURCE_PATH = path
    return _CONFIG


def run_config_path():
    pinned = _PINNED_RUN
    return pinned[0] if pinned is not None else os.environ.get('PHD_PREPROCESSING_CONFIG')


@contextmanager
def run_configuration(path, settings):
    """Keep one physical config authoritative throughout a preprocessing run."""
    global _CONFIG, _CONFIG_SOURCE_PATH, _PINNED_RUN
    previous = _CONFIG, _CONFIG_SOURCE_PATH
    snapshot = deepcopy(settings)
    token = _RUN_CONFIG.set((str(path), snapshot))
    previous_pin = _PINNED_RUN
    _PINNED_RUN = (str(path), snapshot)
    _CONFIG, _CONFIG_SOURCE_PATH = deepcopy(snapshot), str(path)
    try:
        yield _CONFIG
    finally:
        _RUN_CONFIG.reset(token)
        _PINNED_RUN = previous_pin
        _CONFIG, _CONFIG_SOURCE_PATH = previous


def set_file_opener(opener):
    """Inject optional storage IO without coupling configuration to a backend."""
    global _FILE_OPENER
    _FILE_OPENER = opener
