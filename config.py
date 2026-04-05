import json
import os

# Absolute path (safe για όλα τα περιβάλλοντα)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# Singleton load (μία φορά)
_CONFIG = None


def _load():
    global _CONFIG
    if _CONFIG is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
    return _CONFIG


# Public access
def get(key, default=None):
    return _load().get(key, default)


def all():
    return _load()