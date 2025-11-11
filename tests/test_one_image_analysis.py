#!/usr/bin/env python3
"""
This script contains all unit tests of the one_image_analysis script
"""
import unittest
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.image_segmentation import get_color_spaces, combine_color_spaces
from tests._base import CellectsUnitTest, several_arenas_img, several_arenas_bin_img
import numpy as np

class TestOneImageAnalysis(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.image = several_arenas_img
        cls.oia = OneImageAnalysis(cls.image)
        cls.oia.all_c_spaces = get_color_spaces(cls.image)
        # csc = Dict()
        # csc['lab'] = np.array((0,0,1), np.uint8)
        # combine_color_spaces(csc, oia.all_c_spaces)

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



if __name__ == '__main__':
    unittest.main()