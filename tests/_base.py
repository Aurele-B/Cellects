# tests/_base.py
from __future__ import annotations
from pathlib import Path
import unittest

def repo_root() -> Path:
    # .../repo/tests/_base.py -> go up one level to repo root
    return Path(__file__).resolve().parents[1]

def tests_dir() -> Path:
    return repo_root() / "tests"

def data_dir() -> Path:
    """
    Supports both layouts:
      - tests/data/{input,output,experiment}
      - tests/{input,output,experiment}
    """
    t = tests_dir()
    if (t / "data").exists():
        return t / "data"
    return t

class CellectsUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        d = data_dir()
        cls.path_input = d / "input"
        cls.path_output = d / "output"
        cls.path_experiment = d / "experiment"