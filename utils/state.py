import os
from pathlib import Path
from distutils.util import strtobool

state = {
    "debug": bool(strtobool(os.environ.get('DEBUG', 'True'))),
    "debug_dir": Path('./debug')
}


def set_debug_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    state["debug_dir"] = dir_path


def get_debug_dir() -> Path:
    return state["debug_dir"]


def is_debug_mode_activated() -> bool:
    return state["debug"]
