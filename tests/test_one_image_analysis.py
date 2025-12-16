#!/usr/bin/env python3
"""
This script contains all unit tests of the one_image_analysis script
"""
import unittest
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.image_segmentation import get_color_spaces, combine_color_spaces
from cellects.image_analysis.morphological_operations import image_borders
from tests._base import CellectsUnitTest, several_arenas_img, several_arenas_bin_img, video_test, binary_video_test
import numpy as np

class TestOneImageAnalysisBasicOperations(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = several_arenas_img
        cls.oia = OneImageAnalysis(several_arenas_img)
        cls.oia.all_c_spaces = get_color_spaces(several_arenas_img, space_names=["bgr", "hsv"])

    def test_subtract_background(self):
        c_space_dict = dict()
        c_space_dict['bgr'] = np.ones(3, dtype=np.uint8)
        c_space_dict['logical'] = "and"
        c_space_dict['hsv2'] = np.ones(3, dtype=np.uint8)
        self.oia.generate_subtract_background(c_space_dict)
        self.assertTrue(self.oia.subtract_background.any())

    def test_whether_image_border_attest_drift_correction(self):
        self.oia.binary_image = image_borders(several_arenas_img.shape[:2])
        result0 = self.oia.check_if_image_border_attest_drift_correction()
        self.assertFalse(result0)
        self.oia.binary_image = 1 - image_borders(several_arenas_img.shape[:2])
        result1 = self.oia.check_if_image_border_attest_drift_correction()
        self.assertTrue(result1)
        self.oia.binary_image[4, :] = 0
        result2 = self.oia.check_if_image_border_attest_drift_correction()
        self.assertTrue(result2)
        self.assertTrue(self.oia.drift_mask_coord == (np.int64(1), np.int64(10), np.int64(0), np.int64(11)))

    def test_adjust_to_drift_correction(self):
        self.oia.image = video_test[5, :, :, 0]
        self.oia.image2 = video_test[5, :, :, 2]
        self.oia.binary_image = 1 - image_borders(video_test.shape[1:3])
        self.oia.binary_image2 = 1 - image_borders(video_test.shape[1:3])
        self.oia.binary_image2[:7,4:7]=1
        self.oia.adjust_to_drift_correction("And")
        self.oia.drift_correction_already_adjusted = False
        self.oia.image = video_test[5, :, :, 0]
        self.oia.image2 = video_test[5, :, :, 2]
        self.oia.binary_image = 1 - image_borders(video_test.shape[1:3])
        self.oia.binary_image2 = 1 - image_borders(video_test.shape[1:3])
        self.oia.adjust_to_drift_correction("Or")
        self.oia.drift_correction_already_adjusted = False
        self.oia.image = video_test[5, :, :, 0]
        self.oia.image2 = video_test[5, :, :, 2]
        self.oia.binary_image = 1 - image_borders(video_test.shape[1:3])
        self.oia.binary_image2 = 1 - image_borders(video_test.shape[1:3])
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
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = several_arenas_img
        cls.oia = OneImageAnalysis(several_arenas_img)
        cls.oia.all_c_spaces = get_color_spaces(several_arenas_img, space_names=["bgr", "hsv"])

    def test_convert_and_segment_filtered_otsu_with_two_images_and_previous_bin(self):
        c_space_dict = dict()
        c_space_dict['bgr'] = np.ones(3, dtype=np.uint8)
        c_space_dict['hsv2'] = np.array((0, 1, 0), dtype=np.uint8)
        c_space_dict['logical'] = "And"
        self.oia.convert_and_segment(c_space_dict)
        print(several_arenas_img.shape)
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


class TestOneImageAnalysisFindCSC(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = several_arenas_img
        cls.oia = OneImageAnalysis(several_arenas_img)
        cls.oia.all_c_spaces = get_color_spaces(several_arenas_img)

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
        self.oia.find_first_im_csc(basic=False)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_first_im_csc_zeros_image(self):
        """test find_first_im_csc with zeros image"""
        oia = OneImageAnalysis(np.zeros((3, 3, 3), dtype=np.uint8))
        oia.find_first_im_csc(basic=False)
        self.assertEqual(oia.saved_csc_nb, 0)

    def test_find_first_im_csc_with_sample_number_basic(self):
        """test find_first_im_csc with sample number"""
        sample_number = 6
        self.oia.find_first_im_csc(sample_number=sample_number, basic=False)
        self.assertGreater(self.oia.saved_csc_nb, 0)
        self.oia.update_current_images(0)
        self.assertIsInstance(self.oia.validated_shapes, np.ndarray)

    def test_find_first_im_csc_with_backmask(self):
        """test find_first_im_csc with background mask"""
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        self.oia.find_first_im_csc(backmask=backmask, basic=False)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_first_im_csc_with_biomask(self):
        """test find_first_im_csc with bio mask"""
        biomask = several_arenas_bin_img
        self.oia.find_first_im_csc(biomask=biomask, basic=False)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_first_im_csc_with_bio_and_back_mask(self):
        """test find_first_im_csc with bio and back mask"""
        # self.oia.image = video_test[5, :, :, :] # binary_video_test[5, :, :]
        # self.oia.all_c_spaces = {}
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        backmask[:, 0] = 1
        # backmask[5:6, :] = 1
        biomask = several_arenas_bin_img
        # biomask = binary_video_test[5, :, :]
        # biomask[3:5, :] = 0
        self.oia.find_first_im_csc(biomask=biomask, backmask=backmask, basic=False)
        self.assertGreater(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc(self):
        """test find_last_im_csc main functionality"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        arenas_mask = None
        ref_image = None
        subtract_background = None
        kmeans_clust_nb = None
        biomask = None
        backmask = None
        color_space_dictionaries = None
        basic = False
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, basic=False)
        self.assertGreaterEqual(self.oia.saved_csc_nb, 1)

    def test_find_last_im_csc_zeros_image(self):
        """test find_first_im_csc with zeros image"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        zeros_image = np.zeros((3, 3, 3), dtype=np.uint8)
        oia = OneImageAnalysis(zeros_image)
        oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, arenas_mask=zeros_image + 1, ref_image=zeros_image, basic=False)
        self.assertEqual(oia.saved_csc_nb, 0)

    def test_find_last_im_csc_basic(self):
        """test find_last_im_csc basic"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        basic = True
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, basic=basic)
        self.assertGreaterEqual(self.oia.saved_csc_nb, 1)

    def test_find_last_im_csc_with_backmask(self):
        """test find_last_im_csc with background mask"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, backmask=backmask, basic=False)
        self.assertGreaterEqual(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_with_biomask(self):
        """test find_last_im_csc with bio mask"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        biomask = several_arenas_bin_img
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, biomask=biomask, basic=False)
        self.assertGreaterEqual(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_with_bio_and_back_mask(self):
        """test find_last_im_csc with bio and back mask"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        biomask = several_arenas_bin_img
        basic = True
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, biomask=biomask, backmask=backmask, basic=basic)
        self.assertGreaterEqual(self.oia.saved_csc_nb, 0)

    def test_find_last_im_csc_with_kmeans(self):
        """test find_last_im_csc with kmeans"""
        total_surfarea = self.image.size
        concomp_nb =[6, 20*6]
        max_shape_size = 10
        backmask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        backmask[:, 0] = 1
        kmeans_clust_nb = 3
        self.oia.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, kmeans_clust_nb=kmeans_clust_nb, backmask=backmask, basic=False)
        self.assertGreaterEqual(self.oia.saved_csc_nb, 0)


if __name__ == '__main__':
    unittest.main()