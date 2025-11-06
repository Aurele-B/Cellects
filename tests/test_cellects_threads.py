#!/usr/bin/env python3
"""
This script contains all unit tests of the cellects_threads script
"""
import unittest

from cellects.core.program_organizer import ProgramOrganizer
from tests._base import CellectsUnitTest
from cellects.core.cellects_threads import *


class TestLoadDataToRunCellectsQuickly(CellectsUnitTest):
    """Test suite for ProgressivelyAddDistantShapes class."""
    def setUp(self):
        parent = object
        parent.po = ProgramOrganizer()
        thread = {}
        thread["LoadDataToRunCellectsQuickly"] = LoadDataToRunCellectsQuicklyThread(parent)
        thread["LoadDataToRunCellectsQuickly"].start()
        thread["LoadDataToRunCellectsQuickly"].message_from_thread.connect(self.load_data_quickly_finished)

    def load_data_quickly_finished(self, message):
        pass

    def test_if_first_exp_ready_to_run(self):
        self.assertIsInstance(self.parent().po.first_exp_ready_to_run, bool)

if __name__ == '__main__':
    unittest.main()