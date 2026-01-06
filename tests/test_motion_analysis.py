#!/usr/bin/env python3
"""
This script contains all unit tests of the one_image_analysis script
"""
import unittest

from cellects.config.all_vars_dict import DefaultDicts
from cellects.core.motion_analysis import *
from tests._base import CellectsUnitTest, rgb_video_test, binary_video_test, several_arenas_vid, several_arenas_bin_vid
import numpy as np


class TestFullMotionAnalysis(CellectsUnitTest):
    """Test the full pipeline of the MotionAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_output)
        cls.color_space_combination = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
        cls.videos_already_in_ram = [rgb_video_test[:, :, :, :], rgb_video_test[:, :, :, 0]]
        cls.i = 0
        cls.vars = DefaultDicts().vars
        cls.vars['origin_list'] = [np.nonzero(binary_video_test[0])]
        cls.vars['background_list'] = []
        cls.vars['lighter_background'] = False
        cls.vars['first_move_threshold'] = 1
        cls.vars['average_pixel_size'] = 1.
        cls.l = [cls.i, cls.i + 1, cls.vars, True, True, False, cls.videos_already_in_ram]

    def test_simple_motion_analysis(self):
        self.ma = MotionAnalysis(self.l)

class TestMotionAnalysisWithOneFrame(CellectsUnitTest):
    """Test the full pipeline of the MotionAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_output)
        cls.color_space_combination = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
        cls.videos_already_in_ram = [rgb_video_test[0, :, :, :][None, :, :, :], rgb_video_test[0, :, :, 0][None, :, :]]
        cls.i = 0
        cls.vars = DefaultDicts().vars
        cls.vars['origin_list'] = [np.nonzero(binary_video_test[0])]
        cls.vars['background_list'] = []
        cls.vars['lighter_background'] = False
        cls.vars['first_move_threshold'] = 1
        cls.vars['average_pixel_size'] = 1.
        cls.vars['save_coord_network'] = True
        cls.vars['save_graph'] = True
        cls.vars['oscilacyto_analysis'] = True
        cls.vars['save_coord_thickening_slimming'] = True
        cls.vars['fractal_analysis'] = True
        cls.l = [cls.i, cls.i + 1, cls.vars, True, True, False, cls.videos_already_in_ram]

    def test_one_frame_motion_analysis(self):
        self.ma = MotionAnalysis(self.l)


class TestMotionAnalysisWithOneBlob(CellectsUnitTest):
    """Parent for testing the MotionAnalysis class with one blob"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_output)
        cls.color_space_combination = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
        cls.videos_already_in_ram = rgb_video_test[:, :, :, 0]
        cls.i = 0
        cls.dd = DefaultDicts()
        cls.vars = cls.dd.vars
        for k in cls.vars['descriptors'].keys():
            cls.vars['descriptors'][k] = True
        cls.origin_list = [np.nonzero(binary_video_test[0])]
        cls.vars['origin_list'] = cls.origin_list
        cls.vars['already_greyscale'] = True
        cls.vars['drift_already_corrected'] = True # Will automatically become False
        cls.vars['lose_accuracy_to_save_memory'] = False
        cls.vars['first_move_threshold'] = 1
        cls.vars['lighter_background'] = False
        cls.vars['several_blob_per_arena'] = False
        cls.vars['output_in_mm'] = True
        cls.vars['do_fading'] = True
        cls.vars['save_coord_network'] = True
        cls.vars['save_graph'] = True
        cls.vars['oscilacyto_analysis'] = True
        cls.vars['save_coord_thickening_slimming'] = True
        cls.vars['fractal_analysis'] = True
        cls.vars['save_processed_videos'] = True
        cls.vars['filter_spec'] = None
        cls.vars['correct_errors_around_initial'] = True
        cls.vars['prevent_fast_growth_near_periphery'] = True
        cls.vars['origin_state'] = "constant"
        cls.vars['average_pixel_size'] = 1
        cls.vars['save_coord_specimen'] = False
        cls.vars["appearance_detection_method"] = 'largest'
        cls.vars['contour_color']: np.uint8 = 0
        cls.l = [cls.i, cls.i + 1, cls.vars, False, False, False, cls.videos_already_in_ram]

    def test_one_blob_motion_analysis(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)

class TestMotionAnalysisSlopeDetection(TestMotionAnalysisWithOneBlob):
    """Test suite for OneImageAnalysis class"""


    def test_most_central_detection(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.vars["appearance_detection_method"] = 'most_central'
        self.ma.get_origin_shape()
        self.assertTrue(self.ma.start is not None)
        self.ma.get_covering_duration(1) # Put this in init
        self.assertGreater(self.ma.substantial_time, 0)

    def test_detect_slope_only(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = True
        self.ma.vars['do_value_segmentation'] = False
        self.ma.vars['frame_by_frame_segmentation'] = False
        self.ma.detection(False)
        self.assertTrue(np.all(self.ma.segmented.sum((1,2))))

class TestMotionAnalysisAnd(TestMotionAnalysisWithOneBlob):
    """Test suite for OneImageAnalysis class"""

    def test_detect_slope_and_thresh(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = True
        self.ma.vars['do_value_segmentation'] = True
        self.ma.vars['frame_by_frame_segmentation'] = False
        self.ma.vars['true_if_use_light_AND_slope_else_OR'] = True
        self.ma.detection(False)
        self.assertTrue(np.any(self.ma.segmented.sum((1,2))))

class TestMotionAnalysisOr(TestMotionAnalysisWithOneBlob):
    """Test suite for OneImageAnalysis class"""

    def test_detect_slope_or_thresh(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = True
        self.ma.vars['do_value_segmentation'] = True
        self.ma.vars['frame_by_frame_segmentation'] = False
        self.ma.vars['true_if_use_light_AND_slope_else_OR'] = False
        self.ma.detection(False)
        self.assertTrue(np.all(self.ma.segmented.sum((1,2))))

class TestMotionAnalysisStep1(TestMotionAnalysisWithOneBlob):
    """Test suite for OneImageAnalysis class"""

    def test_detect_frame_step1(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = False
        self.ma.vars['do_value_segmentation'] = False
        self.ma.vars['frame_by_frame_segmentation'] = True
        self.ma.step = 1
        self.ma.detection(False)
        self.assertTrue(np.all(self.ma.segmented.sum((1,2))))

class TestMotionAnalysisFullPipeline(TestMotionAnalysisWithOneBlob):
    """Test suite for OneImageAnalysis class"""
    def test_post_processing(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = False
        self.ma.vars['do_value_segmentation'] = False
        self.ma.vars['frame_by_frame_segmentation'] = True
        self.ma.detection(False)
        self.ma.initialize_post_processing()
        self.ma.t = self.ma.start
        while self.ma.t < self.ma.binary.shape[0]:  # 200:
            self.ma.update_shape(False)
        self.assertTrue(np.all(self.ma.binary.sum((1,2))))

        self.ma.get_descriptors_from_binary()
        self.ma.detect_growth_transitions()
        self.ma.check_converted_video_type()
        self.ma.networks_analysis()
        self.ma.study_cytoscillations()
        self.ma.fractal_descriptions()
        self.ma.change_results_of_one_arena(False)
        self.ma.change_results_of_one_arena(False)
        self.ma.save_results()

    def tearDown(self):
        """Remove all written files."""
        file_names = os.listdir(self.path_output)
        for file_name in file_names:
            if os.path.isfile(file_name):
                os.remove(file_name)


class TestMotionAnalysisWithSeveralBlob(CellectsUnitTest):
    """Test suite for OneImageAnalysis class"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_output)
        video = several_arenas_vid.copy()
        # Simulate a drift correction:
        video[:, 0, :, :] = 0
        video[:, :, -1, :] = 0
        video[5, :, -2, :] = 0
        cls.videos_already_in_ram = [video, video[:, :, :, 0], video[:, :, :, 2]]
        cls.i = 0
        cls.dd = DefaultDicts()
        cls.vars = cls.dd.vars
        cls.vars["convert_for_motion"] = {'lab': np.array([0, 0, 1], dtype=np.int8), 'logical': 'And',
                                         'lab2': np.array([0, 0, 1], dtype=np.int8)}
        for k in cls.vars['descriptors'].keys():
            cls.vars['descriptors'][k] = True
        cls.origin_list = [np.nonzero(several_arenas_bin_vid[0])]
        cls.vars['origin_list'] = cls.origin_list
        cls.vars['greyscale2'] = False
        cls.vars['drift_already_corrected'] = True
        cls.vars['already_greyscale'] = False
        cls.vars['first_move_threshold'] = 1
        cls.vars['lose_accuracy_to_save_memory'] = True
        cls.vars['lighter_background'] = True
        cls.vars['several_blob_per_arena'] = True
        cls.vars['output_in_mm'] = True
        cls.vars['do_fading'] = True
        cls.vars['correct_errors_around_initial'] = False
        cls.vars['prevent_fast_growth_near_periphery'] = False
        cls.vars['background_list'] = [several_arenas_vid[:, :, :, 0]]
        cls.vars['background_list2'] = [several_arenas_vid[:, :, :, 2]]
        cls.vars['origin_state'] = "fluctuating"
        cls.vars['filter_spec'] = {'filter1_type': "Gaussian", 'filter1_param': [1., 1.], 'filter2_type': "Median", 'filter2_param': [1., 1.]}
        cls.vars['save_coord_specimen'] = True
        cls.vars['average_pixel_size'] = 1
        cls.vars['study_cytoscillations'] = True
        cls.vars['fractal_analysis'] = True
        cls.l = [cls.i, cls.i + 1, cls.vars, False, False, False, cls.videos_already_in_ram]
        cls.ma = MotionAnalysis(cls.l)
        cls.ma.get_origin_shape()
        cls.ma.get_covering_duration(1)
        cls.ma.detection(True)
        cls.ma.initialize_post_processing()

    def test_get_origin_shape(self):
        self.assertTrue(self.ma.start is not None)

    def test_get_covering_duration(self):
        self.assertGreater(self.ma.substantial_time, 0)
        self.assertLess(self.ma.substantial_image.sum(), self.ma.dims[1] * self.ma.dims[2])
        self.assertGreaterEqual(self.ma.substantial_image.sum(), 0)

    def test_detection(self):
        self.assertIsInstance(self.ma.segmented, np.ndarray)
        self.assertLess(self.ma.segmented[-1].sum(), self.ma.dims[1] * self.ma.dims[2])

    def test_initialize_post_processing_fluctuating(self):
        self.assertIsInstance(self.ma.gravity_field, np.ndarray)

    def test_complete_analysis(self):
        self.ma.t = self.ma.start
        while self.ma.t < self.ma.binary.shape[0]:  # 200:
            self.ma.update_shape(False)
        self.assertTrue(self.ma.binary.any())
        self.ma.get_descriptors_from_binary(False)
        self.assertTrue(np.all(self.ma.surfarea.sum() > 0))
        self.ma.detect_growth_transitions() # Do nothing when several blob
        self.ma.networks_analysis(False) # Do nothing when several blob
        self.ma.study_cytoscillations(False)
        self.ma.fractal_descriptions()
        # self.ma.fractal_descriptions()
        self.ma.save_results()
    def tearDown(self):
        """Remove all written files."""
        file_names = os.listdir(self.path_output)
        for file_name in file_names:
            if os.path.isfile(file_name):
                os.remove(file_name)


if __name__ == '__main__':
    unittest.main()