#!/usr/bin/env python3
"""
This script contains all unit tests of the image filtering script
"""
from tests._base import CellectsUnitTest
from cellects.image.image_filtering import *
import numpy as np
from numba.typed import Dict, List

class TestMaskedVesselFilters(CellectsUnitTest):
    """Test suite for masked_vessel_filters() method"""
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures."""
        super().setUpClass()
        cls.image = np.array([[191, 231, 173], [ 97,  51,  73], [242,  65,  40]], dtype=np.uint8)
        cls.sigmas = [[1.0],  [2.0], [1.0, 2.0], [0.5, 1.0, 2.0], [1.37, 2.81, 5.23]]

    def test_masked_frangi_filter(self):
        """Test Frangi filter by comparing its output with skimage v0.26.0 using different sigmas"""
        skimage_0_26_0_frangi = [np.array([[0.22929587584135777, 0.03609335656717714, 0.0],
                                             [0.7532077275049168, 0.781042738270261, 0.30977595430229865],
                                             [0.43743166112919624, 0.8403551956660499, 0.45728979925864677]]),
                                np.array([[0.07455677359759498, 0.17788745182642302, 0.12395079374712724],
                                             [0.2235025828989898, 0.3372922896194513, 0.29237434217922076],
                                             [0.16033166548395297, 0.303358682249759, 0.2676312193205446]]),
                                np.array([[0.22929587584135777, 0.03609335656717714, 0.0004080565277844023],
                                         [0.7532077275049168, 0.781042738270261, 0.30977595430229865],
                                         [0.43743166112919624, 0.8403551956660499, 0.45728979925864677]]),
                                np.array([[0.6477118469771903, 0.1821490595931513, 0.015907980148264295],
                                             [0.9153997719123184, 0.9497320296654271, 0.7973784157978395],
                                             [0.7455310698356454, 0.9718856099641341, 0.7436897746641659]]),
                                np.array([[0.15191699150051644, 0.167872165568678, 0.07088721956780543],
                                         [0.6127128460771727, 0.7113040453657281, 0.4972607201745982],
                                         [0.38551758198829056, 0.7162362376145737, 0.5468893797607679]])]
        for sigma, frangi_ref in zip(self.sigmas, skimage_0_26_0_frangi):
            frangi_home = frangi_filter(self.image, sigma)
            self.assertLess(np.abs(frangi_home - frangi_ref).max(), 10**-15)
            self.assertGreater(frangi_home.max(), 0)

    def test_masked_sato_filter(self):
        """Test Sato filter by comparing its output with skimage v0.26.0 using different sigmas"""
        skimage_0_26_0_sato = [np.array([[20.064058841308515, 11.564236475501689, 0.10080762142557076], [43.10925404424466, 43.4634635132972, 22.257883313720868], [29.394271220244605, 47.02259675121556, 30.528831482583918]]),
                               np.array([[3.4968313449163917, 6.963551201100796, 3.75368897492622], [4.527677168208166, 8.027796084151312, 4.725218134439233], [4.153895687174275, 7.88141881054044, 4.595374057670962]]),
                               np.array([[20.064058841308515, 11.564236475501689, 3.75368897492622], [43.10925404424466, 43.4634635132972, 22.257883313720868], [29.394271220244605, 47.02259675121556, 30.528831482583918]]),
                               np.array([[20.064058841308515, 11.564236475501689, 3.75368897492622], [43.10925404424466, 43.4634635132972, 22.257883313720868], [29.394271220244605, 47.02259675121556, 30.528831482583918]]),
                               np.array([[12.245701954875065, 16.340585542501923, 7.693541788628687], [24.85440764245779, 31.079710162355013, 19.490669264124474], [19.05239365276853, 31.698144527094133, 21.130972644436397]])]

        for sigma, sato_ref in zip(self.sigmas, skimage_0_26_0_sato):
            sato_home = sato_filter(self.image, sigma)
            self.assertLess(np.abs(sato_home - sato_ref).max(), 10**-15)
            self.assertGreater(sato_home.max(), 0)