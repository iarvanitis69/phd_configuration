import json
import os

# Absolute path (safe για όλα τα περιβάλλοντα)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_INFO_FILE = "session_info.txt"

# Singleton load (μία φορά)
_CONFIG = None
_CONFIG_SOURCE_PATH = None


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_base():
    return _read_json(CONFIG_PATH)


def _get_active_config_path(base_config):
    logs_dir = base_config.get("LOGS_DIR")
    if not logs_dir:
        return None

    state_path = os.path.join(REPO_ROOT, SESSION_INFO_FILE)
    if not os.path.exists(state_path):
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            folder_name = f.read().strip()
    except Exception:
        return None

    if not folder_name:
        return None

    run_config_path = os.path.join(logs_dir, folder_name, "config.json")
    if os.path.exists(run_config_path):
        return run_config_path
    return None


def _load():
    global _CONFIG, _CONFIG_SOURCE_PATH
    if _CONFIG is None:
        base_config = _load_base()
        active_config_path = _get_active_config_path(base_config)
        if active_config_path is None:
            _CONFIG = base_config
            _CONFIG_SOURCE_PATH = CONFIG_PATH
        else:
            _CONFIG = _read_json(active_config_path)
            _CONFIG_SOURCE_PATH = active_config_path
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
    _CONFIG = _read_json(path)
    _CONFIG_SOURCE_PATH = path
    return _CONFIG
