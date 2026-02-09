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
        cls.color_space_combination = {"logical": 'None', "PCA": [1, 1, 1]}
        cls.videos_already_in_ram = [rgb_video_test[:, :, :, :], rgb_video_test[:, :, :, 0]]
        cls.i = 0
        cls.vars = DefaultDicts().vars
        write_h5(f'ind_{1}.h5', np.array(np.nonzero(binary_video_test[0])), 'origin_coord')
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
        cls.color_space_combination = {"logical": 'None', "PCA": [1, 1, 1]}
        cls.videos_already_in_ram = [rgb_video_test[0, :, :, :][None, :, :, :], rgb_video_test[0, :, :, 0][None, :, :]]
        cls.i = 0
        cls.vars = DefaultDicts().vars
        write_h5(f'ind_{1}.h5', np.array(np.nonzero(binary_video_test[0])), 'origin_coord')
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

class TestMotionAnalysisWithMP4(CellectsUnitTest):
    """Test MotionAnalysis with a .mp4 file"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_experiment)
        image = readim('image1.tif')
        cls.bin_image = (image[:, :, 2] < 80).astype(np.uint8)
        cls.dims = image.shape[:2]
        cls.vars = DefaultDicts().vars
        cls.vars['crop_coord'] = [0, cls.dims[0], 0, cls.dims[1]]
        cls.vars['arenas_coord'] = [[0], [cls.dims[0]], [0], [cls.dims[1]]]
        cls.videos_already_in_ram = None
        cls.i = 0

    def test_mp4_loading(self):
        write_h5(f'ind_{1}.h5', self.bin_image, 'origin_coord')
        self.vars['video_list'] = ['video.mp4']
        image = readim('image1.tif')
        self.color_space_combination = {"logical": 'None', "PCA": [1, 1, 1]}
        self.l = [self.i, self.i + 1, self.vars, False, False, False, self.videos_already_in_ram]
        self.ma = MotionAnalysis(self.l)

    def test_mp4_loading_of_a_greyscale_video(self):
        write_h5(f'ind_{1}.h5', self.bin_image, 'origin_coord')
        self.vars['convert_for_motion'] = {'lab': [0, 0, 1], 'logical': 'Or', 'luv2': [0, 0, 1]}
        vid, grey_vid = video2numpy('video.mp4', self.vars['convert_for_motion'])
        write_h5('video_grey.h5', grey_vid, 'video')
        self.vars['video_list'] = ['video_grey.h5']
        self.vars['already_greyscale'] = True
        self.l = [self.i, self.i + 1, self.vars, False, False, False, self.videos_already_in_ram]
        self.ma = MotionAnalysis(self.l)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(f'ind_{1}.h5'):
            os.remove(f'ind_{1}.h5')
        if os.path.isfile(f'video_grey.h5'):
            os.remove(f'video_grey.h5')


class TestMotionAnalysisWithOneBlob(CellectsUnitTest):
    """Parent for testing the MotionAnalysis class with one blob"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_output)
        cls.color_space_combination = {"logical": 'None', "PCA": [1, 1, 1]}
        cls.videos_already_in_ram = rgb_video_test[:, :, :, 0]
        cls.i = 0
        cls.dd = DefaultDicts()
        cls.vars = cls.dd.vars
        for k in cls.vars['descriptors'].keys():
            cls.vars['descriptors'][k] = True
        write_h5(f'ind_{1}.h5', np.array(np.nonzero(binary_video_test[0])), 'origin_coord')
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
    """Test suite for MotionAnalysis class"""

    def test_most_central_detection(self):
        self.ma = MotionAnalysis(self.l)
        self.ma.vars["appearance_detection_method"] = 'most_central'
        self.ma.get_origin_shape()
        self.assertTrue(self.ma.start is not None)
        self.ma.get_covering_duration(1) # Put this in init
        self.assertGreater(self.ma.substantial_time, 0)

class TestMotionAnalysisAnd(TestMotionAnalysisWithOneBlob):
    """Test suite for MotionAnalysis class"""

    def test_detect_slope_and_thresh(self):
        self.vars['lose_accuracy_to_save_memory'] = True
        self.vars['repeat_video_smoothing'] = 2
        self.l = [self.i, self.i + 1, self.vars, False, False, False, self.videos_already_in_ram]
        self.ma = MotionAnalysis(self.l)
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
    """Test suite for MotionAnalysis class"""

    def test_detect_slope_or_thresh(self):
        self.vars['lose_accuracy_to_save_memory'] = False
        self.vars['repeat_video_smoothing'] = 2
        self.l = [self.i, self.i + 1, self.vars, False, False, False, self.videos_already_in_ram]
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
    """Test suite for MotionAnalysis class"""

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


class TestMotionAnalysisWith2Conversions(CellectsUnitTest):
    """Test suite for MotionAnalysis with 2 conversions"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.chdir(cls.path_output)
        cls.videos_already_in_ram = rgb_video_test[:, :, :, :], rgb_video_test[:, :, :, 0], rgb_video_test[:, :, :, 2]
        cls.i = 0
        cls.dd = DefaultDicts()
        cls.vars = cls.dd.vars
        for k in cls.vars['descriptors'].keys():
            cls.vars['descriptors'][k] = True
        write_h5(f'ind_{1}.h5', np.array(np.nonzero(binary_video_test[0])), 'origin_coord')
        cls.vars['already_greyscale'] = False
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

    def test_detect_slope_with_or(self):
        self.vars['convert_for_motion'] = {"logical": 'Or', "lab": [0, 0, 1], "luv2": [0, 0, 1]}
        self.l = [self.i, self.i + 1, self.vars, False, False, False, self.videos_already_in_ram]
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = True
        self.ma.vars['do_value_segmentation'] = False
        self.ma.vars['frame_by_frame_segmentation'] = False
        self.ma.detection(False)
        self.assertTrue(isinstance(self.ma.segmented, np.ndarray))

    def test_detect_slope_with_and(self):
        self.vars['convert_for_motion'] = {"logical": 'And', "lab": [0, 0, 1], "luv2": [0, 0, 1]}
        self.l = [self.i, self.i + 1, self.vars, False, False, False, self.videos_already_in_ram]
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = True
        self.ma.vars['do_value_segmentation'] = False
        self.ma.vars['frame_by_frame_segmentation'] = False
        self.ma.detection(False)
        self.assertTrue(isinstance(self.ma.segmented, np.ndarray))

    def test_detect_slope_with_xor(self):
        self.vars['convert_for_motion'] = {"logical": 'Xor', "lab": [0, 0, 1], "luv2": [0, 0, 1]}
        self.l = [self.i, self.i + 1, self.vars, False, False, False, self.videos_already_in_ram]
        self.ma = MotionAnalysis(self.l)
        self.ma.get_origin_shape()
        self.ma.get_covering_duration(1)
        self.ma.vars['do_slope_segmentation'] = True
        self.ma.vars['do_value_segmentation'] = False
        self.ma.vars['frame_by_frame_segmentation'] = False
        self.ma.detection(False)
        self.assertTrue(isinstance(self.ma.segmented, np.ndarray))

    # def test_detect_thresh_only(self):
    #     self.ma = MotionAnalysis(self.l)
    #     self.ma.get_origin_shape()
    #     self.ma.get_covering_duration(1)
    #     self.ma.vars['do_slope_segmentation'] = True
    #     self.ma.vars['do_value_segmentation'] = False
    #     self.ma.vars['frame_by_frame_segmentation'] = False
    #     self.ma.detection(False)
    #     self.assertTrue(np.all(self.ma.segmented.sum((1,2))))

class TestMotionAnalysisFullPipeline(TestMotionAnalysisWithOneBlob):
    """Test suite for MotionAnalysis class"""
    def test_post_processing(self):
        write_h5(f'ind_{1}.h5', np.array(np.nonzero(binary_video_test[0])), 'origin_coord')
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
    """Test suite for MotionAnalysis class"""

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
        cls.vars["convert_for_motion"] = {'lab': [0, 0, 1], 'logical': 'And', 'lab2': [0, 0, 1]}
        for k in cls.vars['descriptors'].keys():
            cls.vars['descriptors'][k] = True
        cls.origin_coord = np.array(np.nonzero(several_arenas_bin_vid[0]))
        write_h5(f'ind_{1}.h5', cls.origin_coord, 'origin_coord')
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
        write_h5(f'ind_{1}.h5', several_arenas_vid[:, :, :, 0], 'background')
        write_h5(f'ind_{1}.h5', several_arenas_vid[:, :, :, 2], 'background2')
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

    def test_get_origin_shape_invisible(self):
        write_h5(f'ind_{1}.h5', self.origin_coord, 'origin_coord')
        variables = self.vars.copy()
        variables["origin_state"] = "invisible"
        l = [self.i, self.i + 1, variables, False, False, False, self.videos_already_in_ram]
        ma = MotionAnalysis(l)
        ma.get_origin_shape()
        variables['several_blob_per_arena'] = False
        variables['appearance_detection_method'] = 'most_central'
        l = [self.i, self.i + 1, variables, False, False, False, self.videos_already_in_ram]
        ma = MotionAnalysis(l)
        ma.get_origin_shape()

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
        self.ma.save_results(False, False)
        self.ma.save_results()

    def tearDown(self):
        """Remove all written files."""
        file_names = os.listdir(self.path_output)
        for file_name in file_names:
            if os.path.isfile(file_name):
                os.remove(file_name)


if __name__ == '__main__':
    unittest.main()