#!/usr/bin/env python3
"""
This script contains all unit tests of the one_video_per_blob script
"""
import unittest
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.core.one_video_per_blob import *
from cellects.core.program_organizer import ProgramOrganizer
from cellects.image_analysis.image_segmentation import generate_color_space_combination, extract_first_pc
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from tests.test_based_run import load_test_folder, run_image_analysis_for_testing
from tests._base import CellectsUnitTest, several_arenas_img, several_arenas_bin_img, several_arenas_vid, several_arenas_bin_vid
from pathlib import Path
import numpy as np
import cv2
import os
# po = load_test_folder("/Users/Directory/Scripts/python/Cellects/data/experiment", 1)
# po.get_first_image()
# greyscale, vap, csc = extract_first_pc(po.first_image.image)
# po.first_image.image = bracket_to_uint8_image_contrast(greyscale)
# po.first_image.segmentation()
# po.first_image.get_crop_coordinates()
# po.first_image.shape_number = 1
# # po.fast_image_segmentation(True)
# po.videos = OneVideoPerBlob(po.first_image, starting_blob_hsize_in_pixels=2, raw_images=False)
# are_gravity_centers_moving = True
# img_list = po.data_list
# color_space_combination = {"logical": 'None', "bgr": np.ones(3, dtype=np.uint8)}
# po.videos.get_bounding_boxes(are_gravity_centers_moving, img_list, color_space_combination)
# ##### Current pb: The algorithm does not work when it does not detect a shape at each frame.
# image = several_arenas_img
# po = ProgramOrganizer()
# po.first_image = OneImageAnalysis(image, shape_number=6)
# color_number=2
# sample_size=2
# are_gravity_centers_moving = True
# img_list = [image, image]
# all_specimens_have_same_direction = True
# never_closer_than_2_pixels = False
# display = False
# filter_spec = None
# color_space_combination = {"logical": 'None', "bgr": np.ones(3, dtype=np.uint8)}
# po.first_image.validated_shapes = several_arenas_bin_img
# po.videos = OneVideoPerBlob(po.first_image, starting_blob_hsize_in_pixels=2, raw_images=False)
# self=po.videos
# po.videos.get_bounding_boxes(True, img_list, color_space_combination, sample_size=sample_size)
# po.videos.top
# po.videos.bot
# po.videos.left
# po.videos.right

class TestOneImageAnalysis(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.color_space_combination = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}

        cls.img_list = several_arenas_vid
        cls.image = several_arenas_vid[0]
        cls.shape_number = 2
        cls.po = ProgramOrganizer()
        cls.po.first_image = OneImageAnalysis(cls.image, cls.shape_number)
        cls.po.first_image.validated_shapes = several_arenas_bin_vid[0]
        cls.po.videos = OneVideoPerBlob(cls.po.first_image, starting_blob_hsize_in_pixels=2, raw_images=False)


        cls.image2 = several_arenas_img
        cls.shape_number2 = 6
        cls.po2 = ProgramOrganizer()
        cls.po2.first_image = OneImageAnalysis(cls.image2, cls.shape_number2)
        cls.po2.first_image.validated_shapes = several_arenas_bin_img
        cls.po2.videos = OneVideoPerBlob(cls.po2.first_image, starting_blob_hsize_in_pixels=2, raw_images=False)

    def test_get_bounding_boxes_with_moving_centers(self):
        are_gravity_centers_moving = True
        img_list = self.img_list
        color_number = 2
        sample_size = 4
        all_specimens_have_same_direction = True
        display = False
        filter_spec = None
        color_space_combination = self.color_space_combination
        self.po.videos.get_bounding_boxes(are_gravity_centers_moving, img_list, self.color_space_combination,
                                          sample_size=sample_size)
        visited_pixels = np.any(several_arenas_bin_vid[:-1, ...], axis=0)
        Y, X = np.nonzero(visited_pixels)
        self.assertTrue(self.po.videos.top.min() <= Y.min())
        self.assertTrue(self.po.videos.left.min() <= X.min())
        self.assertTrue(self.po.videos.bot.max() >= Y.max())
        self.assertTrue(self.po.videos.right.max() >= X.max())


    def test_get_bounding_boxes_with_moving_centers_with_close_shapes(self):
        are_gravity_centers_moving = True
        img_list = [self.image2, self.image2]
        color_number = 2
        sample_size = 2
        all_specimens_have_same_direction = True
        display = False
        filter_spec = None
        color_space_combination = self.color_space_combination
        self.po2.videos.get_bounding_boxes(are_gravity_centers_moving, img_list, self.color_space_combination,
                                          sample_size=sample_size)
        self.assertGreater(self.po2.videos.top.sum(), 0)
        self.assertGreater(self.po2.videos.bot.sum(), 0)
        self.assertGreater(self.po2.videos.left.sum(), 0)
        self.assertGreater(self.po2.videos.right.sum(), 0)


    # def test_get_bounding_boxes_with_moving_centers(self):
    #     are_gravity_centers_moving = True
    #     img_list = [self.image]
    #     color_number = 2
    #     sample_size = 5
    #     all_specimens_have_same_direction = True,
    #     display = False
    #     filter_spec = None
    #     self.po.videos.get_bounding_boxes(are_gravity_centers_moving, img_list, color_space_combination)
    #
    #     self.assertEqual(self.po.videos.top, 0)
    #     self.assertEqual(self.po.videos.left, 0)
    #     self.assertEqual(self.po.videos.bot, 11)
    #     self.assertEqual(self.po.videos.right, 11)
    # def test_get_bounding_boxes_without_moving_centers(self):
    #     """Test get_bounding_boxes without moving centers and default parameters."""
    #     # Create a mock image list for testing
    #     img_list = ["image1", "image2"]
    #     color_space_combination = {"combination": "default"}
    #
    #     self.po.videos.get_bounding_boxes(
    #         are_gravity_centers_moving=False,
    #         img_list=img_list,
    #         color_space_combination=color_space_combination
    #     )
    #
    #     # Check that the method ran without error and updated attributes
    #     self.assertIsNotNone(self.instance.ordered_first_image)
    #     self.assertTrue(hasattr(self.instance, "top"))
    #     self.assertTrue(hasattr(self.instance, "bot"))
    #     self.assertTrue(hasattr(self.instance, "left"))
    #     self.assertTrue(hasattr(self.instance, "right"))
    #
    # def test_get_bounding_boxes_with_moving_centers(self):
    #     """Test get_bounding_boxes with moving centers and sample images."""
    #     # Create a mock image list for testing
    #     img_list = ["img1", "img2"]
    #     color_space_combination = {"combination": "default"}
    #
    #     self.instance.get_bounding_boxes(
    #         are_gravity_centers_moving=True,
    #         img_list=img_list,
    #         color_space_combination=color_space_combination
    #     )
    #
    #     # Check that the method ran without error and updated attributes
    #     self.assertIsNotNone(self.instance.ordered_first_image)
    #     self.assertTrue(hasattr(self.instance, "motion_list"))
    #
    # def test_standardize_video_sizes_with_valid_input(self):
    #     """Test standardize_video_sizes with valid input data."""
    #     # Set up mock bounding box values that stay within bounds
    #     shape_height = 100
    #     shape_width = 200
    #
    #     self.top = np.array([10] * 5, dtype=np.int64)
    #     self.bot = np.array([shape_height - 10] * 5, dtype=np.int64)
    #     self.left = np.array([10] * 5, dtype=np.int64)
    #     self.right = np.array([shape_width - 10] * 5, dtype=np.int64)
    #
    #     # Run the method
    #     self.instance.standardize_video_sizes()
    #
    #     # Check that standard has been calculated correctly and stays within bounds
    #     self.assertTrue(np.all(self.top < self.instance.ordered_first_image.shape[0]))
    #     self.assertTrue(np.all(self.bot <= self.instance.ordered_first_image.shape[0] - 1))
    #     self.assertTrue(np.all(self.left > 0))
    #     self.assertTrue(np.all(self.right <= self.instance.ordered_first_image.shape[1] - 1))
    #
    # def test_standardize_video_sizes_with_boundary_overflow(self):
    #     """Test standardize_video_sizes with overflowing values."""
    #     # Set up mock bounding box values that overflow
    #     shape_height = 100
    #     shape_width = 200
    #
    #     self.top = np.array([shape_height - 95] * 5, dtype=np.int64)
    #     self.bot = np.array([shape_height + 5] * 5, dtype=np.int64)
    #     self.left = np.array([-5] * 5, dtype=np.int64)
    #     self.right = np.array([shape_width + 5] * 5, dtype=np.int64)
    #
    #     # Run the method
    #     self.instance.standardize_video_sizes()
    #
    #     # Check that overflowing values have been corrected or shapes removed
    #     if hasattr(self.instance, "shapes_to_remove"):
    #         self.assertTrue(len(self.instance.shapes_to_remove) > 0)
    #
    # def test_prepare_video_writing_valid_input(self):
    #     """Test prepare_video_writing with valid inputs."""
    #     img_list = ["img1", "img2"]
    #
    #     result = self.instance.prepare_video_writing(
    #         img_list=img_list,
    #         min_ram_free=2.0
    #     )
    #
    #     # Check that the method returns expected values without errors
    #     self.assertIsInstance(result, tuple)
    #     bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining = result
    #
    #     self.assertIsInstance(bunch_nb, int)
    #     self.assertTrue(np.issubdtype(video_nb_per_bunch.dtype, np.integer))
    #     self.assertIsInstance(sizes, np.ndarray)
    #     if bunch_nb == 1:
    #         self.assertIsInstance(video_bunch, (list, np.ndarray))
    #
    # def test_write_videos_as_np_arrays_valid_input(self):
    #     """Test write_videos_as_np_arrays with valid inputs."""
    #     img_list = ["img1", "img2"]
    #
    #     # Prepare by calling prepare_video_writing first
    #     self.instance.prepare_video_writing(
    #         img_list=img_list,
    #         min_ram_free=2.0
    #     )
    #
    #     result = self.instance.write_videos_as_np_arrays(
    #         img_list=img_list,
    #         min_ram_free=2.0
    #     )
    #
    #     # Check that the method runs without error and creates files
    #     vid_names = [f"ind_{i + 1}.npy" for i in range(self.instance.first_image.shape_number)]
    #     expected_files = [self.path_output / name for name in vid_names]
    #
    #     for file_path in expected_files:
    #         if os.path.exists(file_path):
    #             self.assertTrue(os.path.getsize(file_path) > 0)
    #
    # def test__get_quick_bb_normal_operation(self):
    #     """Test _get_quick_bb with normal input data."""
    #     # Set up mock data that should work normally
    #     shape_number = 2
    #
    #     self.top = np.array([10, 30], dtype=np.int64)
    #     self.bot = np.array([80, 150], dtype=np.int64)
    #     self.left = np.array([10, 70], dtype=np.int64)
    #     self.right = np.array([90, 200], dtype=np.int64)
    #
    #     # Run the method
    #     self.instance._get_quick_bb()
    #
    #     # Check that it runs without error and updates attributes correctly
    #     self.assertTrue(hasattr(self.instance, "left"))
    #     self.assertTrue(hasattr(self.instance, "right"))
    #     self.assertTrue(hasattr(self.instance, "top"))
    #     self.assertTrue(hasattr(self.instance, "bot"))
    #
    # def test__get_bb_with_moving_centers_normal_operation(self):
    #     """Test _get_bb_with_moving_centers with normal input data."""
    #     # Set up mock image list and parameters that should work normally
    #     img_list = ["img1", "img2"]
    #     color_space_combination = {"combination": "default"}
    #
    #     self.instance._get_bb_with_moving_centers(
    #         img_list=img_list,
    #         color_space_combination=color_space_combination
    #     )
    #
    #     # Check that the method runs without error and updates attributes correctly
    #     self.assertTrue(hasattr(self.instance, "motion_list"))
    #
    # def test__segment_blob_motion_valid_input(self):
    #     """Test _segment_blob_motion with valid input data."""
    #     image = np.zeros((100, 200))
    #     color_space_combination = {"combination": "default"}
    #
    #     result = self.instance._segment_blob_motion(
    #         image=image,
    #         color_space_combination=color_space_combination,
    #         color_number=3
    #     )
    #
    #     # Check that the method returns a binary image without errors
    #     self.assertIsInstance(result, np.ndarray)
    #     self.assertEqual(len(result.shape), 2)  # Binary image should be 2D


if __name__ == '__main__':
    unittest.main()