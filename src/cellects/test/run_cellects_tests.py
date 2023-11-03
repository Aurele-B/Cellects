import unittest


import os
import sys
from pathlib import Path

#
# def add_import_path():
#     CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
#     split_dir = os.path.split(CURRENT_DIR)
#     while split_dir[1] != "Cellects":
#         split_dir = os.path.split(split_dir[0])
#     SRC_DIR = Path(split_dir[0]) / "Cellects" / "src"
#     sys.path.append(str(SRC_DIR))
#
#
# add_import_path()


from cellects.test import test_utilitarian
from cellects.test import test_formulas
from cellects.test import test_load_display_save
from cellects.test import test_image_segmentation
from cellects.test import test_morphological_operations
from cellects.test import test_integration


# Create a TestSuite combining all the test cases
def create_test_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_utilitarian))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_formulas))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_load_display_save))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_image_segmentation))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_morphological_operations))

    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_integration))

    return test_suite


if __name__ == '__main__':
    # Create the test suite
    suite = create_test_suite()

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the test suite
    result = runner.run(suite)
