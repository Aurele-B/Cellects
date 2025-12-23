#!/usr/bin/env python3
"""
This script contains all unit tests of the one_image_analysis script
"""
import unittest

import cv2

from cellects.core.one_image_analysis import *
from cellects.image_analysis.image_segmentation import get_color_spaces, combine_color_spaces
from cellects.image_analysis.morphological_operations import image_borders
from tests._base import *
import numpy as np

class TestOneImageAnalysisBasicOperations(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = rgb_several_arenas_img
        cls.oia = OneImageAnalysis(rgb_several_arenas_img)
        cls.oia.all_c_spaces = get_color_spaces(rgb_several_arenas_img, space_names=["bgr", "hsv"])

    def test_subtract_background(self):
        c_space_dict = dict()
        c_space_dict['bgr'] = np.ones(3, dtype=np.uint8)
        c_space_dict['logical'] = "and"
        c_space_dict['hsv2'] = np.ones(3, dtype=np.uint8)
        self.oia.generate_subtract_background(c_space_dict)
        self.assertTrue(self.oia.subtract_background.any())

    def test_whether_image_border_attest_drift_correction(self):
        self.oia.binary_image = image_borders(rgb_several_arenas_img.shape[:2])
        result0 = self.oia.check_if_image_border_attest_drift_correction()
        self.assertFalse(result0)
        self.oia.binary_image = 1 - image_borders(rgb_several_arenas_img.shape[:2])
        result1 = self.oia.check_if_image_border_attest_drift_correction()
        self.assertTrue(result1)
        self.oia.binary_image[4, :] = 0
        result2 = self.oia.check_if_image_border_attest_drift_correction()
        self.assertTrue(result2)
        self.assertTrue(self.oia.drift_mask_coord == (np.int64(1), np.int64(10), np.int64(0), np.int64(11)))

    def test_adjust_to_drift_correction(self):
        self.oia.image = rgb_video_test[5, :, :, 0]
        self.oia.image2 = rgb_video_test[5, :, :, 2]
        self.oia.binary_image = 1 - image_borders(rgb_video_test.shape[1:3])
        self.oia.binary_image2 = 1 - image_borders(rgb_video_test.shape[1:3])
        self.oia.binary_image2[:7,4:7]=1
        self.oia.adjust_to_drift_correction("And")
        self.oia.drift_correction_already_adjusted = False
        self.oia.image = rgb_video_test[5, :, :, 0]
        self.oia.image2 = rgb_video_test[5, :, :, 2]
        self.oia.binary_image = 1 - image_borders(rgb_video_test.shape[1:3])
        self.oia.binary_image2 = 1 - image_borders(rgb_video_test.shape[1:3])
        self.oia.adjust_to_drift_correction("Or")
        self.oia.drift_correction_already_adjusted = False
        self.oia.image = rgb_video_test[5, :, :, 0]
        self.oia.image2 = rgb_video_test[5, :, :, 2]
        self.oia.binary_image = 1 - image_borders(rgb_video_test.shape[1:3])
        self.oia.binary_image2 = 1 - image_borders(rgb_video_test.shape[1:3])
        self.oia.adjust_to_drift_correction("Xor")
        self.assertIsInstance(self.oia.binary_image, np.ndarray)

    def test_get_crop_coordinates(self):
        self.oia.validated_shapes = np.vstack((np.repeat(0, 18), np.tile([0, 0, 1], 6), np.repeat(0, 18), np.tile([1, 0, 0], 6), np.repeat(0, 18), np.tile([0, 0, 1], 6), np.repeat(0, 18), np.tile([1, 0, 0], 6), np.repeat(0, 18), np.tile([0, 0, 1], 6), np.repeat(0, 18)))
        self.oia.get_crop_coordinates()
        self.assertTrue(self.oia.crop_coord is not None)
        self.oia.validated_shapes = np.vstack((np.tile([0, 0, 0, 1], 4), np.tile([0, 1, 0, 0], 4), np.tile([0, 0, 0, 1], 4), np.tile([0, 1, 0, 0], 4), np.tile([0, 0, 0, 1], 4), np.tile([0, 1, 0, 0], 4)))
        self.oia.get_crop_coordinates()
        self.assertTrue(self.oia.crop_coord is not None)

    def test_automatically_crop(self):
        self.oia.greyscale2 = self.oia.image
        self.oia.image2 = self.oia.image.copy()
        self.oia.binary_image2 = several_arenas_bin_img.copy()
        self.oia.subtract_background = self.oia.image.copy()
        self.oia.subtract_background2 = self.oia.image.copy()
        self.oia.automatically_crop([0, 5, 0, 5])
        self.assertTrue(self.oia.y_boundaries is not None)


class TestOneImageAnalysisConvertAndSegment(CellectsUnitTest):
    """Test suite for the convert_and_segment method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = rgb_several_arenas_img
        cls.oia = OneImageAnalysis(rgb_several_arenas_img)
        cls.oia.all_c_spaces = get_color_spaces(rgb_several_arenas_img, space_names=["bgr", "hsv"])

    def test_convert_and_segment_filtered_otsu_with_two_images_and_previous_bin(self):
        c_space_dict = dict()
        c_space_dict['bgr'] = np.ones(3, dtype=np.uint8)
        c_space_dict['hsv2'] = np.array((0, 1, 0), dtype=np.uint8)
        c_space_dict['logical'] = "And"
        self.oia.convert_and_segment(c_space_dict)
        print(rgb_several_arenas_img.shape)
        self.assertIsInstance(self.oia.binary_image, np.ndarray)
        c_space_dict['logical'] = "Or"
        self.oia.convert_and_segment(c_space_dict)
        self.assertIsInstance(self.oia.binary_image, np.ndarray)
        c_space_dict['logical'] = "Xor"
        self.oia.previous_binary_image = several_arenas_bin_img
        self.oia.convert_and_segment(c_space_dict, filter_spec={'filter1_type': "Gaussian", 'filter1_param': [1., 1.], 'filter2_type': "Median", 'filter2_param': [1., 1.]})
        self.assertIsInstance(self.oia.binary_image, np.ndarray)

    def test_convert_and_segment_kmeans_with_two_images(self):
        c_space_dict = dict()
        c_space_dict['bgr'] = np.ones(3, dtype=np.uint8)
        c_space_dict['hsv2'] = np.array((0, 1, 0), dtype=np.uint8)
        c_space_dict['logical'] = "and"
        color_number = 3
        self.oia.convert_and_segment(c_space_dict, color_number)
        self.assertTrue(self.oia.binary_image.any())

class TestSegmentBlobOneLargeCentralBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = blob_vary_rgb_one_large_central_blob
        cls.oia = OneImageAnalysis(blob_vary_rgb_one_large_central_blob)
        cls.oia.all_c_spaces = get_color_spaces(blob_vary_rgb_one_large_central_blob)
        cls.params = init_params()
        cls.params['several_blob_per_arena'] = False
        cls.params['is_first_image'] = True
        cls.params['blob_nb'] = 1
        cls.params['blob_size'] = large_size
        cls.params['kmeans_clust_nb'] = 2
        cls.params['bio_mask'] = cv2.erode(one_large_central_blob, rhombus_55, iterations=10)
        cls.params['ref_image'] = cls.params['bio_mask']
        cls.params['arenas_mask'] = cv2.dilate(one_large_central_blob, cross_33, iterations=10)
        cls.params['back_mask'] = np.zeros_like(one_large_central_blob)
        cls.params['back_mask'][:, :200] = 1
        cls.params['back_mask'][:200, :] = 1
        cls.params['back_mask'][:, -200:] = 1
        cls.params['back_mask'][-200:, :] = 1

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())

    def test_find_csc_as_first_image(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations(self.params)
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())

    def test_find_csc_as_any_image(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.params['is_first_image'] = False
        self.oia.find_color_space_combinations(self.params)
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())

    def test_network_detection(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.network_detection(self.params['arenas_mask'])
        self.assertTrue(len(self.oia.im_combinations) > 0)
        self.assertTrue(self.oia.im_combinations[0]['binary_image'].sum() > 0)


class TestSegmentBlobOneLargeSideBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = blob_vary_rgb_one_large_side_blob
        cls.oia = OneImageAnalysis(blob_vary_rgb_one_large_side_blob)
        cls.oia.all_c_spaces = get_color_spaces(blob_vary_rgb_one_large_side_blob)
        cls.params = init_params()
        cls.params['several_blob_per_arena'] = False
        cls.params['is_first_image'] = True
        cls.params['blob_nb'] = 1

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations(self.params)
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())


class TestSegmentBlobOneSmallCentralBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = blob_vary_rgb_one_small_central_blob
        cls.oia = OneImageAnalysis(blob_vary_rgb_one_small_central_blob)
        cls.oia.all_c_spaces = get_color_spaces(blob_vary_rgb_one_small_central_blob)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        params = init_params()
        params['blob_nb'] = 1
        self.oia.find_color_space_combinations(params)
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())


class TestSegmentBlobOneSmallSideBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = blob_vary_gb_one_small_side_blob
        cls.oia = OneImageAnalysis(blob_vary_gb_one_small_side_blob)
        cls.oia.all_c_spaces = get_color_spaces(blob_vary_gb_one_small_side_blob)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())


class TestSegmentBlobManySmallBlobs(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = blob_vary_rgb_many_small_blobs
        cls.oia = OneImageAnalysis(blob_vary_rgb_many_small_blobs)
        cls.oia.all_c_spaces = get_color_spaces(blob_vary_rgb_many_small_blobs)
        cls.params = init_params()
        cls.params['is_first_image'] = False
        cls.params['kmeans_clust_nb'] = 2
        cls.params['bio_mask'] = cv2.erode(many_small_blobs, rhombus_55, iterations=1)
        cls.params['ref_image'] = cls.params['bio_mask']
        cls.params['arenas_mask'] = cv2.dilate(many_small_blobs, cross_33, iterations=10)
        cls.params['back_mask'] = np.zeros_like(many_small_blobs)
        cls.params['back_mask'][:, :1] = 1
        cls.params['back_mask'][:1, :] = 1
        cls.params['back_mask'][:, -1:] = 1
        cls.params['back_mask'][-1:, :] = 1

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations(self.params)
        self.assertTrue((self.oia.combination_features['blob_nb'] == small_blob_nb).any())


class TestSegmentBlobManyMediumBlobs(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = blob_vary_rgb_many_medium_blobs
        cls.oia = OneImageAnalysis(blob_vary_rgb_many_medium_blobs)
        cls.oia.all_c_spaces = get_color_spaces(blob_vary_rgb_many_medium_blobs)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == medium_blob_nb).any())

    def test_with_other_params(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        params = init_params()
        params['kmeans_clust_nb'] = 2
        self.oia.find_color_space_combinations(params)
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == medium_blob_nb).any())

class TestSegmentBlobManyVaryingBlobs(CellectsUnitTest):
    """Second test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = blob_vary_rgb_many_varying_blobs
        cls.oia = OneImageAnalysis(blob_vary_rgb_many_varying_blobs)
        cls.oia.all_c_spaces = get_color_spaces(blob_vary_rgb_many_varying_blobs)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == medium_blob_nb).any())


class TestSegmentBackOneLargeCentralBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = back_vary_rgb_one_large_central_blob
        cls.oia = OneImageAnalysis(back_vary_rgb_one_large_central_blob)
        cls.oia.all_c_spaces = get_color_spaces(back_vary_rgb_one_large_central_blob)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())


class TestSegmentBackOneLargeSideBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = back_vary_rgb_one_large_side_blob
        cls.oia = OneImageAnalysis(back_vary_rgb_one_large_side_blob)
        cls.oia.all_c_spaces = get_color_spaces(back_vary_rgb_one_large_side_blob)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())


class TestSegmentBackOneSmallCentralBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = back_vary_rgb_one_small_central_blob
        cls.oia = OneImageAnalysis(back_vary_rgb_one_small_central_blob)
        cls.oia.all_c_spaces = get_color_spaces(back_vary_rgb_one_small_central_blob)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        params = init_params()
        params['blob_nb'] = 1
        self.oia.find_color_space_combinations(params)
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())


class TestSegmentBackOneSmallSideBlob(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = back_vary_rgb_one_small_side_blob
        cls.oia = OneImageAnalysis(back_vary_rgb_one_small_side_blob)
        cls.oia.all_c_spaces = get_color_spaces(back_vary_rgb_one_small_side_blob)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        params = init_params()
        params['blob_nb'] = 1
        self.oia.find_color_space_combinations(params)
        self.assertTrue((self.oia.combination_features['blob_nb'] == 1).any())


class TestSegmentBackManySmallBlobs(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = back_vary_rgb_many_small_blobs
        cls.oia = OneImageAnalysis(back_vary_rgb_many_small_blobs)
        cls.oia.all_c_spaces = get_color_spaces(back_vary_rgb_many_small_blobs)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == small_blob_nb).any())

class TestSegmentBackManyMediumBlobs(CellectsUnitTest):
    """First test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = back_vary_rgb_many_medium_blobs
        cls.oia = OneImageAnalysis(back_vary_rgb_many_medium_blobs)
        cls.oia.all_c_spaces = get_color_spaces(back_vary_rgb_many_medium_blobs)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == medium_blob_nb).any())

class TestSegmentBackManyVaryingBlobs(CellectsUnitTest):
    """Second test suite for the find_color_space_combinations method of the OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = back_vary_rgb_many_varying_blobs
        cls.oia = OneImageAnalysis(back_vary_rgb_many_varying_blobs)
        cls.oia.all_c_spaces = get_color_spaces(back_vary_rgb_many_varying_blobs)

    def test_find_color_space_combinations(self):
        """test if the number of detected connected components is the same as the number of connected components used to create the image"""
        self.oia.find_color_space_combinations()
        self.assertTrue((self.oia.combination_features['blob_nb'] == medium_blob_nb).any())


if __name__ == '__main__':
    unittest.main()