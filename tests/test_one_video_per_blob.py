#!/usr/bin/env python3
"""
This script contains all unit tests of the one_video_per_blob script
"""
import unittest
from cellects.core.one_video_per_blob import *
from cellects.core.program_organizer import ProgramOrganizer
from cellects.image_analysis.morphological_operations import rhombus_55
from tests._base import CellectsUnitTest, several_arenas_img, several_arenas_bin_img, several_arenas_vid, several_arenas_bin_vid
import numpy as np
import cv2
import os

class TestOneVideoPerBlob(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        """Initialize two data sets for testing"""
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
        """Test the get bounding boxes algorithm when the centroids of the shapes move throughout the video"""
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

    def test_get_bounding_boxes_with_moving_centers_large_shapes(self):
        """Test get_bounding_boxes when the shapes are large"""
        image = np.zeros((70, 70), dtype=np.uint8)
        image[10, 8] = 1
        image[35, 8] = 1
        image[60, 8] = 1
        image[10, 35] = 1
        image[10, 60] = 1
        # image[35, 35] = 1
        image[60, 35] = 1
        image[35, 60] = 1
        image[61, 61] = 1
        image = cv2.dilate(image, rhombus_55, iterations=3)
        shape_number = 8
        first_image = OneImageAnalysis(image, shape_number)
        first_image.validated_shapes = image
        videos = OneVideoPerBlob(first_image, starting_blob_hsize_in_pixels=10, raw_images=False)
        videos.get_bounding_boxes(are_gravity_centers_moving=True, img_list=[image, image], sample_size=2,
                                  color_space_combination=self.color_space_combination)
        self.assertTrue(np.sum(videos.top) > 0)
        self.assertTrue(np.sum(videos.bot) > 0)
        self.assertTrue(np.sum(videos.left) > 0)
        self.assertTrue(np.sum(videos.right) > 0)

    def test_get_quick_bounding_boxes(self):
        """Test get_bounding_boxes using a fast algorithm"""
        self.po.first_image.validated_shapes = several_arenas_bin_vid[0]
        self.po.videos.get_bounding_boxes(are_gravity_centers_moving=False, img_list=self.img_list,
                                          color_space_combination=self.color_space_combination)
        self.assertTrue(np.sum(self.po.videos.top) > 0)
        self.assertTrue(np.sum(self.po.videos.bot) > 0)
        self.assertTrue(np.sum(self.po.videos.left) > 0)
        self.assertTrue(np.sum(self.po.videos.right) > 0)


    def test_get_bounding_boxes_with_no_shapes(self):
        """Test get_bounding_boxes when no shapes are detected"""
        self.po.first_image.validated_shapes = np.zeros_like(several_arenas_bin_vid[0])
        self.po.videos.get_bounding_boxes(are_gravity_centers_moving=False, img_list=self.img_list,
                                          color_space_combination=self.color_space_combination)
        self.assertEqual(self.po.videos.top, 0)
        self.assertEqual(self.po.videos.bot, 20)
        self.assertEqual(self.po.videos.left, 0)
        self.assertEqual(self.po.videos.right, 20)

    def test_get_bounding_boxes_with_moving_centers_with_close_shapes(self):
        """Test get_bounding_boxes when shapes are very close to each other"""
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

    def test_write_videos_with_different_video_dimensions(self):
        """Test write_videos_as_np_arrays when the dimensions of each video are different"""
        min_ram_free = 0
        img_list = [self.path_experiment / "image1.tif", self.path_experiment / "image2.tif"]
        self.po.videos.top=np.array([0, 3])
        self.po.videos.bot=np.array([2, 6])
        self.po.videos.left=np.array([0, 3])
        self.po.videos.right=np.array([3, 5])
        self.po.videos.first_image.shape_number = 2
        self.po.videos.write_videos_as_np_arrays(img_list, min_ram_free, in_colors=True, pathway=str(self.path_output) + "/")
        self.assertTrue(os.path.isfile(self.path_output / f"ind_1.npy"))

    def test_prepare_video_writing_using_too_much_memory(self):
        """Test prepare_video_writing when writing all videos at the same time is not possible with current memory"""
        min_ram_free = (psutil.virtual_memory().available >> 30) - 5
        if min_ram_free > 0:
            img_list = [np.zeros((1000, 1000), dtype=np.uint8) for i in range(5000)]
            self.po.videos.top=np.array([0, 500])
            self.po.videos.left=np.array([0, 500])
            self.po.videos.bot=np.array([500, 1000])
            self.po.videos.right=np.array([500, 1000])
            bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining  = self.po.videos.prepare_video_writing(img_list, min_ram_free)
            self.assertGreater(bunch_nb, 0)

    def test_segment_blob_motion(self):
        """Test _segment_blob_motion usual behavior"""
        bin_img = self.po.videos._segment_blob_motion(str(self.path_experiment / "image1.tif"), self.color_space_combination, 2, None)
        self.assertIsInstance(bin_img, np.ndarray)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_output / f"ind_1.npy"):
            os.remove(self.path_output / f"ind_1.npy")


if __name__ == '__main__':
    unittest.main()