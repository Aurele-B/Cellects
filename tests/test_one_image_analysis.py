#!/usr/bin/env python3
"""
This script contains all unit tests of the one_image_analysis script
"""
import unittest
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.core.one_video_per_blob import OneVideoPerBlob
from tests.test_based_run import load_test_folder, run_image_analysis_for_testing
from tests._base import CellectsUnitTest, several_arenas_img, several_arenas_bin_img
import numpy as np
import cv2
import os

class TestOneImageAnalysis(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = several_arenas_img
        cls.oia = OneImageAnalysis(cls.image)

    def test_find_first_im_csc(self):
        """test find_first_im_csc main functionality"""
        sample_number = None
        several_blob_per_arena = True
        spot_shape = None
        spot_size = None
        kmeans_clust_nb = None
        biomask = None
        backmask = None
        color_space_dictionaries = None
        carefully = False
        self.oia.find_first_im_csc()
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_first_im_csc_zeros_image(self):
        """test find_first_im_csc with zeros image"""
        oia = OneImageAnalysis(np.zeros((3, 3, 3), dtype=np.uint8))
        oia.find_first_im_csc()
        self.assertEqual(oia.saved_csc_nb, 0)

    def test_find_first_im_csc_with_sample_number_carefully(self):
        """test find_first_im_csc with sample number"""
        sample_number = 6
        carefully = True
        self.oia.find_first_im_csc(sample_number=sample_number, carefully=carefully)
        self.assertGreater(self.oia.saved_csc_nb, 0)
        self.oia.update_current_images(0)
        self.assertIsInstance(self.oia.validated_shapes, np.ndarray)

    def test_find_first_im_csc_with_backmask(self):
        """test find_first_im_csc with background mask"""
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        self.oia.find_first_im_csc(backmask=backmask)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_first_im_csc_with_biomask(self):
        """test find_first_im_csc with bio mask"""
        biomask = several_arenas_bin_img
        self.oia.find_first_im_csc(biomask=biomask)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_first_im_csc_with_bio_and_back_mask(self):
        """test find_first_im_csc with bio and back mask"""
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        biomask = several_arenas_bin_img
        self.oia.find_first_im_csc(biomask=biomask, backmask=backmask)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc(self):
        """test find_last_im_csc main functionality"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        out_of_arenas = None
        ref_image = None
        subtract_background = None
        kmeans_clust_nb = None
        biomask = None
        backmask = None
        color_space_dictionaries = None
        carefully = False
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_zeros_image(self):
        """test find_first_im_csc with zeros image"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        oia = OneImageAnalysis(np.zeros((3, 3, 3), dtype=np.uint8))
        oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size)
        self.assertEqual(oia.saved_csc_nb, 0)

    def test_find_last_im_csc_carefully(self):
        """test find_last_im_csc carefully"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        carefully = True
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, carefully=carefully)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_with_backmask(self):
        """test find_last_im_csc with background mask"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, backmask=backmask)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_with_biomask(self):
        """test find_last_im_csc with bio mask"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        biomask = several_arenas_bin_img
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, biomask=biomask)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_with_bio_and_back_mask(self):
        """test find_last_im_csc with bio and back mask"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        biomask = several_arenas_bin_img
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, biomask=biomask, backmask=backmask)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_with_kmeans(self):
        """test find_last_im_csc with kmeans"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        kmeans_clust_nb = 3
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, kmeans_clust_nb=kmeans_clust_nb)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_convert_and_segment_kmeans_with_two_images(self):
        c_space_dict = dict()
        c_space_dict['bgr'] = np.ones(3, dtype=np.uint8)
        c_space_dict['hsv2'] = np.array((0, 1, 0), dtype=np.uint8)
        c_space_dict['logical'] = "and"
        color_number = 3
        self.oia.convert_and_segment(c_space_dict, color_number)
        self.assertTrue(self.oia.binary_image.any())

        # oia.find_first_im_csc(sample_number=6, several_blob_per_arena=False, carefully=True)
        # self.oia.find_first_im_csc(sample_number=6, kmeans_clust_nb=None,several_blob_per_arena=False)
        # oia.saved_csc_nb

    # def test_find_first_im_csc(self):
    #     """test main functionality"""

    #     self.oia.find_first_im_csc()
    #     self.assertTrue(self.oia.saved_csc_nb > 0)
    #     self.assertGreater(self.oia.saved_csc_nb, 0)


# class TestOneImageAnalysis(CellectsUnitTest):
#     """Test suite for OneImageAnalysis class"""
#
#     @classmethod
#     def setUpClass(cls):
#         super().setUpClass()
#         cls.i = 0
#         cls.po = load_test_folder(str(cls.path_experiment), 1)
#         cls.po.get_first_image()
#         cls.po.first_image = OneImageAnalysis(cls.po.first_im)
#         cls.po.get_last_image()

    # def test_convert_and_segment_basic_operation(self):
    #     """Test basic color space conversion and segmentation with valid input"""
    #     self.po.first_image.convert_and_segment(self.c_space_dict)
    #
    #     # Verify segmentation was called
    #     self.assertTrue(hasattr(instance, "binary_image"))
    #     self.assertEqual(instance.binary_image.shape, (100, 100))
    #
    # def test_convert_and_segment_already_greyscale(self):
    #     """Test behavior when image is already greyscale"""
    #     instance = OneImageAnalysis(np.zeros((100, 100), dtype=np.uint8))
    #
    #     with mock.patch.object(instance, "segmentation") as mock_seg:
    #         instance.convert_and_segment({"logical": "Or"})
    #         mock_seg.assert_called_once()
    #
    # # --- segmentation Tests ---
    #
    # def test_segmentation_kmeans_call(self):
    #     """Test kmeans-based segmentation when color_number > 2"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     with mock.patch.object(instance, "kmeans") as mock_kmeans:
    #         instance.segmentation(color_number=3)
    #         mock_kmeans.assert_called_once()
    #
    # def test_segmentation_invalid_filter_spec(self):
    #     """Test error handling for invalid filter specifications"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     with self.assertRaises(ValueError):
    #         instance.segmentation(filter_spec={"filter1_type": "invalid_filter_type"})
    #
    # # --- generate_subtract_background Tests ---
    #
    # def test_generate_subtract_background_creates_all_c_spaces(self):
    #     """Test that all_c_spaces is created when empty"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     self.assertEqual(len(instance.all_c_spaces), 0)
    #
    #     with mock.patch("cellects.image_analysis.one_image_analysis.get_color_spaces") as mock_get:
    #         mock_get.return_value = {
    #             "bgr": self.test_image,
    #             "lab": np.zeros((100, 100), dtype=np.uint8),
    #             "hsv": np.zeros((100, 100), dtype=np.uint8)
    #         }
    #
    #         instance.generate_subtract_background(self.mock_c_space_dict)
    #
    #         mock_get.assert_called_once_with(instance.bgr)
    #
    # # --- check_if_image_border_attest_drift_correction Tests ---
    #
    # def test_check_border_conditions_black_borders_true(self):
    #     """Test returns True when opposite borders are black"""
    #     instance = OneImageAnalysis(np.zeros((100, 100), dtype=np.uint8))
    #
    #     with mock.patch.object(instance, "binary_image",
    #                            np.array([[0] * 100] + [[0] * 100 for _ in range(98)] + [[0] * 100], dtype=bool)):
    #         result = instance.check_if_image_border_attest_drift_correction()
    #         self.assertTrue(result)
    #
    # # --- adjust_to_drift_correction Tests ---
    #
    # def test_adjust_with_empty_binary_image(self):
    #     """Test drift correction with empty binary image"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     instance.binary_image = np.zeros((100, 100), dtype=bool)
    #
    #     instance.adjust_to_drift_correction(logical="Or")
    #
    #     # Verify the attributes were modified
    #     self.assertFalse(np.array_equal(instance.image, self.test_image))
    #
    # # --- find_first_im_csc Tests ---
    #
    # def test_find_first_im_csc_saves_combinations(self):
    #     """Test that color space combinations are saved correctly"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     with mock.patch.object(instance, "ProcessFirstImage") as mock_process:
    #         instance.find_first_im_csc()
    #
    #         self.assertTrue(len(instance.im_combinations) > 0)
    #         self.assertIsInstance(instance.im_combinations[0], dict)
    #
    # # --- save_combination_features Tests ---
    #
    # def test_save_combination_features_sets_attributes(self):
    #     """Test that combination features are saved correctly"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     mock_process_i = mock.MagicMock()
    #     mock_process_i.image = self.test_image
    #     mock_process_i.validated_shapes = np.zeros((100, 100), dtype=bool)
    #     mock_process_i.csc_dict = {"bgr": [1, 0, 0]}
    #
    #     instance.save_combination_features(mock_process_i)
    #
    #     # Verify that saved attributes are updated
    #     self.assertTrue(instance.saved_images_list[0].shape == (100, 100))
    #
    # # --- update_current_images Tests ---
    #
    # def test_update_current_images_sets_attributes(self):
    #     """Test updating current images with valid combination ID"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     instance.im_combinations = [{
    #         "converted_image": self.test_image,
    #         "binary_image": np.zeros((100, 100), dtype=bool)
    #     }]
    #
    #     instance.update_current_images(0)
    #
    #     self.assertTrue(np.array_equal(instance.image, self.test_image))
    #
    # # --- kmeans Tests ---
    #
    # def test_kmeans_invalid_cluster_number(self):
    #     """Test error handling for invalid cluster number"""
    #     instance = OneImageAnalysis(self.test_image)
    #
    #     with self.assertRaises(ValueError):
    #         instance.kmeans(cluster_number=1)



if __name__ == '__main__':
    unittest.main()