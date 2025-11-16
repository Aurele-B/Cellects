#!/usr/bin/env python3
"""
This script contains all integration tests of Cellects
"""
import os.path
import unittest
from tests._base import CellectsUnitTest
from tests.test_based_run import *
from cellects.config.all_vars_dict import DefaultDicts
import numpy as np
from numba.typed import Dict as TDict


#     def test_run_image_analysis(self):
#         self.assertTrue(self.po.first_image.binary_image.sum() < self.po.last_image.binary_image.sum())
#
#     def test_video_writing(self):
#         self.po = run_write_videos_for_testing(self.po)
#         for video_i in np.arange(len(self.po.vars['analyzed_individuals'])) + 1:
#             self.assertTrue(os.path.isfile(f"ind_{video_i}.npy"))
#
#     def run_all_arenas_for_testing(self):
#         self.po.instantiate_tables()
#         self.po.vars['do_slope_segmentation'] = False
#         self.po.vars['do_value_segmentation'] = False
#         self.po.vars['frame_by_frame_segmentation'] = True
#         self.po.vars['keep_unaltered_videos'] = True
#         self.po = run_write_videos_for_testing(self.po)
#         run_all_arenas_for_testing(self.po)
#         for i, arena in enumerate(self.po.vars['analyzed_individuals']):
#             self.assertTrue(os.path.isfile(f"ind_{arena}.mp4"))
#         self.assertTrue(os.path.isfile(f"one_row_per_arena.csv"))
#         self.assertTrue(os.path.isfile(f"one_row_per_frame.csv"))
#         self.assertTrue(os.path.isfile(f"software_settings.csv"))
#         self.assertTrue(os.path.isfile(f"Analysis efficiency, 3th image.jpg"))
#         self.assertTrue(os.path.isfile(f"Analysis efficiency, last image.jpg"))
#
#     def test_detection(self):
#         self.po = run_write_videos_for_testing(self.po)
#         MA = run_one_video_analysis_for_testing(self.po)
#         self.assertTrue(MA.one_row_per_frame.shape[0] == 25)
#         self.assertTrue(MA.one_row_per_frame.shape[1] == self.descriptor_nb + 2)
#         self.assertTrue(np.any(MA.one_row_per_frame.iloc[:, 2:]))
#
#     def tearDown(self):
#
#         if os.path.isfile(self.path_experiment / f"Data to run Cellects quickly.pkl"):
#             os.remove(self.path_experiment / f"Data to run Cellects quickly.pkl")
#         if os.path.isfile(self.path_experiment / f"one_row_per_arena.csv"):
#             os.remove(self.path_experiment / f"one_row_per_arena.csv")
#         if os.path.isfile(self.path_experiment / f"one_row_per_frame.csv"):
#             os.remove(self.path_experiment / f"one_row_per_frame.csv")
#         if os.path.isfile(self.path_experiment / f"one_row_per_oscillating_cluster.csv"):
#             os.remove(self.path_experiment / f"one_row_per_oscillating_cluster.csv")
#         if os.path.isfile(self.path_experiment / f"software_settings.csv"):
#             os.remove(self.path_experiment / f"software_settings.csv")
#         if os.path.isfile(self.path_experiment / f"Analysis efficiency, 3th image.jpg"):
#             os.remove(self.path_experiment / f"Analysis efficiency, 3th image.jpg")
#         if os.path.isfile(self.path_experiment / f"Analysis efficiency, last image.jpg"):
#             os.remove(self.path_experiment / f"Analysis efficiency, last image.jpg")
#         for arena in self.po.vars['analyzed_individuals']:
#             if os.path.isfile(self.path_experiment / f"ind_{arena}.npy"):
#                 os.remove(self.path_experiment / f"ind_{arena}.npy")
#             if os.path.isfile(self.path_experiment / f"ind_{arena}.mp4"):
#                 os.remove(self.path_experiment / f"ind_{arena}.mp4")


if __name__ == '__main__':
    unittest.main()
