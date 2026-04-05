import json
import os

# Path προς το config.json (μέσα στο submodule σου)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Φόρτωση μία φορά (όταν γίνει import)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# Helper function (προαιρετική)
def get(key, default=None):
    return CONFIG.get(key, default)