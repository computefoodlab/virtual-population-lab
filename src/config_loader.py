"""
Shared config resolution for any pipeline entry point (main.py,
check_generalization.py, ...). Kept free of side-effecting top-level code
so it's safe to import from multiple scripts without triggering config
loading based on the wrong script's argv.
"""
import importlib
import os

DEFAULT_CONFIG = "apple_quality"
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
NON_DATASET_CONFIG_FILES = {"__init__.py", "base.py"}


def available_configs():
    return sorted(
        f[:-3] for f in os.listdir(CONFIGS_DIR)
        if f.endswith(".py") and f not in NON_DATASET_CONFIG_FILES
    )


def load_config(name):
    try:
        return importlib.import_module(f"src.configs.{name}")
    except ModuleNotFoundError:
        raise SystemExit(
            f"No config named '{name}' in src/configs/. Available: {', '.join(available_configs())}"
        )
