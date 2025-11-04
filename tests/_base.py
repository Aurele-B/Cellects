# tests/_base.py
from __future__ import annotations

import os
from pathlib import Path
import unittest

class CellectsUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        d = Path(__file__).resolve().parents[1] / "data" # set up data path for the tests
        cls.path_input = d / "input"
        cls.path_output = d / "output"
        cls.path_experiment = d / "experiment"

        if not os.path.isdir(cls.path_output):
            os.mkdir(cls.path_output)