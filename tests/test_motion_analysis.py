#!/usr/bin/env python3
"""
This script contains all unit tests of the one_image_analysis script
"""
import unittest

from cellects.config.all_vars_dict import DefaultDicts
from cellects.core.motion_analysis import *
from tests.test_based_run import load_test_folder, run_image_analysis_for_testing, run_write_videos_for_testing
from tests._base import CellectsUnitTest, several_arenas_vid, several_arenas_bin_vid
import numpy as np

# color_space_combination = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
# visu = several_arenas_vid
# videos_already_in_ram = [several_arenas_vid, several_arenas_vid[:, :, :, 0]]
# origin_list = [several_arenas_bin_vid[0]]
# i = 0
# dd = DefaultDicts()
# vars = dd.vars
# for k in vars['descriptors'].keys():
#     vars['descriptors'][k] = True
# vars['origin_list'] = origin_list
# vars['first_move_threshold'] = 1
# vars['lighter_background'] = False
# vars['several_blob_per_arena'] = True
# l = [i, i + 1, vars, False, False, False, videos_already_in_ram]
# ma = MotionAnalysis(l)
# ma.get_origin_shape()
# ma.get_covering_duration(1)
# ma.origin
# ma.substantial_growth
# ma.substantial_image
# self = ma
# ma.detection(compute_all_possibilities=True)

class TestMotionAnalysis(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.color_space_combination = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
        cls.videos_already_in_ram = [several_arenas_vid, several_arenas_vid[:, :, :, 0]]
        cls.i = 0
        cls.dd = DefaultDicts()
        cls.vars = cls.dd.vars
        for k in cls.vars['descriptors'].keys():
            cls.vars['descriptors'][k] = True
        cls.origin_list = [several_arenas_bin_vid[0]]
        cls.vars['origin_list'] = cls.origin_list
        cls.vars['first_move_threshold'] = 1
        cls.vars['lighter_background'] = False
        cls.vars['several_blob_per_arena'] = True
        cls.l = [cls.i, cls.i + 1, cls.vars, False, False, False, cls.videos_already_in_ram]
        cls.ma = MotionAnalysis(cls.l)

    def test_motion_analysis(self):
        self.ma.get_origin_shape()
        self.assertTrue(self.ma.start is not None)
        self.ma.get_covering_duration(1)
        self.ma.detection(True)

    # def test_load_images_with_existing_video(self):
    #     """Test loading images from video file when not preloaded."""
    #     with patch('cellects.core.motion_analysis.video2numpy') as mock_video:
    #         mock_video.return_value = np.random.rand(10, 100, 100).astype(np.float32)
    #
    #         instance = MotionAnalysis(self.test_input)
    #         instance.load_images_and_videos(None, 0)
    #
    #         self.assertTrue(hasattr(instance, 'converted_video'))
    #         self.assertEqual(len(instance.converted_video.shape), 3)
    #
    # def test_load_preloaded_data(self):
    #     """Test loading with preloaded video data."""
    #     videos_already_in_ram = np.random.rand(10, 100, 100).astype(np.float32)
    #
    #     instance = MotionAnalysis([*self.test_input])
    #     instance.load_images_and_videos(videos_already_in_ram, 0)
    #
    #     self.assertTrue(hasattr(instance, 'converted_video'))
    #     self.assertIs(instance.converted_video, videos_already_in_ram)
    #
    #
    # def test_conversion_with_filtering(self):
    #     """Test conversion with filter application."""
    #     # Setup mock filter function
    #     with patch('cellects.core.motion_analysis.apply_filter') as mock_apply:
    #         mock_apply.return_value = np.random.rand(100, 100).astype(np.float32)
    #
    #         self.instance.get_converted_video()
    #
    #         self.assertTrue(hasattr(self.instance, 'converted_video'))
    #         # Verify that apply_filter was called with correct parameters
    #         self.assertEqual(mock_apply.call_count, 1)
    #
    # def test_conversion_without_background_subtraction(self):
    #     """Test conversion without background subtraction."""
    #     self.instance.background = None
    #
    #     self.instance.get_converted_video()
    #
    #     self.assertTrue(np.allclose(
    #         self.instance.converted_video.shape,
    #         (self.instance.visu.shape[0],) + self.instance.visu.shape[1:3]
    #     ))
    #
    # def test_frame_by_frame_segmentation(self):
    #     """Test frame-by-frame segmentation method."""
    #     # Create mock instance and data
    #     pass
    #
    # def test_lum_value_segmentation(self):
    #     """Test luminosity value-based segmentation."""
    #     pass
    #
    # def test_network_detection_basic_case(self):
    #     """Test network detection on valid input data."""
    #     pass


if __name__ == '__main__':
    unittest.main()