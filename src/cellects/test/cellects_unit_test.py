#!/usr/bin/env python3
"""
This script contains the class inherited in all cellects unit tests
"""
from unittest import TestCase
from cellects.core.cellects_paths import TEST_DIR


class CellectsUnitTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
            1/ setUpClass: Give the most basic parameters of the function
            2/ define one test_method per potential result/error/exception
        :return:
        """
        cls.path_input = TEST_DIR / "input"
        cls.path_output = TEST_DIR / "output"
        cls.path_experiment = TEST_DIR / "experiment"

