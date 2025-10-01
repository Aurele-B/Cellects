#!/usr/bin/env python3
"""
This script contains all integration tests of Cellects
"""
import os.path
import unittest
import logging
from glob import glob
from tests._base import CellectsUnitTest
from cellects.core.program_organizer import ProgramOrganizer
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.core.motion_analysis import MotionAnalysis
from cellects.config.all_vars_dict import DefaultDicts
from cellects.utils.load_display_save import PickleRick
import numpy as np
from numba.typed import Dict as TDict


class TestCellects(CellectsUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.po = ProgramOrganizer()
        cls.po.load_variable_dict()

        # use the test data folders provided by _base
        cls.po.all['global_pathway'] = str(cls.path_experiment)
        cls.po.all['first_folder_sample_number'] = 100

        cls.po.look_for_data()
        cls.po.load_data_to_run_cellects_quickly()

        cls.i = 7

    def test_look_for_data(self):
        self.assertEqual(len(self.po.data_list), 25)
        self.assertEqual(self.po.all['folder_number'], 1)

    def test_save_data_to_run_cellects_quickly(self):
        if not os.path.isfile(self.path_experiment / "Data to run Cellects quickly.pkl"):
            coordinates = PickleRick().read_file(self.path_output / "coordinates.pkl")
            dd = DefaultDicts()
            self.po.all = dd.all
            self.po.vars = dd.vars
            for k in self.po.all['descriptors'].keys():
                self.po.all['descriptors'][k] = True
            self.po.vars['oscilacyto_analysis'] = True
            self.po.vars['keep_unaltered_videos'] = True
            self.po.update_output_list()
            self.po.instantiate_tables()
            self.po.vars['maximal_growth_factor'] = 0.25
            self.po.get_first_image()
            backmask = np.zeros(self.po.first_im.shape[:2], np.uint8)
            backmask[-30:, :] = 1
            backmask = np.nonzero(backmask)
            self.po.vars['convert_for_origin'] = {'lab': np.array([0, 0, 1], dtype=np.int8), 'logical': 'None'}
            self.po.vars['convert_for_motion'] = {'lab': np.array([0, 0, 1], dtype=np.int8), 'logical': 'None'}
            self.po.fast_image_segmentation(True, backmask=backmask)
            self.po.all['automatically_crop'] = True
            self.po.cropping(is_first_image=True)
            self.assertTrue(np.array_equal(self.po.first_image.crop_coord, coordinates[0]))
            self.po.all['scale_with_image_or_cells'] = 1
            self.po.all['starting_blob_hsize_in_mm'] = 15
            self.po.get_average_pixel_size()
            self.po.delineate_each_arena()
            self.assertTrue(np.array_equal(self.po.left, coordinates[1]))
            self.assertTrue(np.array_equal(self.po.right, coordinates[2]))
            self.assertTrue(np.array_equal(self.po.top, coordinates[3]))
            self.assertTrue(np.array_equal(self.po.bot, coordinates[4]))
            self.po.get_background_to_subtract()
            self.po.get_origins_and_backgrounds_lists()
            self.po.get_last_image()
            self.po.fast_image_segmentation(is_first_image=False)
            self.po.find_if_lighter_background()
            self.assertEqual(self.po.vars['lighter_background'], False)
            self.po.extract_exif()
            for k in self.po.data_to_save.keys():
                self.po.data_to_save[k] = True
            self.po.save_data_to_run_cellects_quickly()

    def test_load_data_to_run_cellects_quickly(self):
        self.test_save_data_to_run_cellects_quickly()
        coordinates = PickleRick().read_file(self.path_output / "coordinates.pkl")
        self.assertTrue(np.array_equal(self.po.first_image.crop_coord, coordinates[0]))
        self.assertTrue(np.array_equal(self.po.left, coordinates[1]))
        self.assertTrue(np.array_equal(self.po.right, coordinates[2]))
        self.assertTrue(np.array_equal(self.po.top, coordinates[3]))
        self.assertTrue(np.array_equal(self.po.bot, coordinates[4]))

    def test_video_writing(self):
        self.test_save_data_to_run_cellects_quickly()
        look_for_existing_videos = glob('ind_' + '*' + '.npy')
        there_already_are_videos = len(look_for_existing_videos) == len(self.po.vars['analyzed_individuals'])
        logging.info(
            f"{len(look_for_existing_videos)} .npy video files found for {len(self.po.vars['analyzed_individuals'])} arenas to analyze")
        do_write_videos = not there_already_are_videos or (
                there_already_are_videos and self.po.all['overwrite_unaltered_videos'])
        if do_write_videos:
            self.po.videos = OneVideoPerBlob(self.po.first_image, self.po.starting_blob_hsize_in_pixels, self.po.all['raw_images'])
            self.po.videos.left = self.po.left
            self.po.videos.right = self.po.right
            self.po.videos.top = self.po.top
            self.po.videos.bot = self.po.bot
            self.po.videos.first_image.shape_number = self.po.sample_number
            self.po.videos.write_videos_as_np_arrays(
                self.po.data_list, self.po.vars['min_ram_free'], not self.po.vars['already_greyscale'], self.po.reduce_image_dim)
        for video_i in np.arange(len(self.po.vars['analyzed_individuals'])) + 1:
            self.assertTrue(os.path.isfile(f"ind_{video_i}.npy"))

    def test_thresh_detection(self):
        self.test_save_data_to_run_cellects_quickly()
        self.test_video_writing()
        self.po.vars['do_value_segmentation'] = True
        self.po.vars['do_slope_segmentation'] = False
        self.po.vars['frame_by_frame_segmentation'] = False
        analysis_thresh = MotionAnalysis([self.i, self.i + 1, self.po.vars, True, True, True, None])
        # PickleRick().write_file(analysis_thresh.whole_shape_descriptors, self.path_output / "motion_analysis_thresh.pkl")
        reference = PickleRick().read_file(self.path_output / "motion_analysis_thresh.pkl")
        self.assertTrue(analysis_thresh.whole_shape_descriptors.equals(reference))

    def test_slope_detection(self):
        self.test_save_data_to_run_cellects_quickly()
        self.test_video_writing()
        self.po.vars['do_slope_segmentation'] = True
        self.po.vars['do_value_segmentation'] = False
        self.po.vars['frame_by_frame_segmentation'] = False
        analysis_slope = MotionAnalysis([self.i, self.i + 1, self.po.vars, True, True, True, None])
        # PickleRick().write_file(analysis_slope.whole_shape_descriptors, self.path_output / "motion_analysis_slope.pkl")
        reference = PickleRick().read_file(self.path_output / "motion_analysis_slope.pkl")
        self.assertTrue(analysis_slope.whole_shape_descriptors.equals(reference))

    def test_t_or_s_detection(self):
        self.test_save_data_to_run_cellects_quickly()
        self.test_video_writing()
        self.po.vars['do_value_segmentation'] = True
        self.po.vars['do_slope_segmentation'] = True
        self.po.vars['frame_by_frame_segmentation'] = False
        self.po.vars['true_if_use_light_AND_slope_else_OR'] = False
        analysis_t_or_s = MotionAnalysis([self.i, self.i + 1, self.po.vars, True, True, True, None])
        # PickleRick().write_file(analysis_t_or_s.whole_shape_descriptors, self.path_output / "motion_analysis_t_or_s.pkl")
        reference = PickleRick().read_file(self.path_output / "motion_analysis_t_or_s.pkl")
        self.assertTrue(analysis_t_or_s.whole_shape_descriptors.equals(reference))

    def test_t_and_s_detection(self):
        self.test_save_data_to_run_cellects_quickly()
        self.test_video_writing()
        self.po.vars['do_value_segmentation'] = True
        self.po.vars['do_slope_segmentation'] = True
        self.po.vars['frame_by_frame_segmentation'] = False
        self.po.vars['true_if_use_light_AND_slope_else_OR'] = True
        analysis_t_and_s = MotionAnalysis([self.i, self.i + 1, self.po.vars, True, True, True, None])
        # PickleRick().write_file(analysis_t_and_s.whole_shape_descriptors, self.path_output / "motion_analysis_t_and_s.pkl")
        reference = PickleRick().read_file(self.path_output / "motion_analysis_t_and_s.pkl")
        self.assertTrue(analysis_t_and_s.whole_shape_descriptors.equals(reference))

    def test_frame_detection(self):
        self.test_save_data_to_run_cellects_quickly()
        self.test_video_writing()
        self.po.vars['do_slope_segmentation'] = False
        self.po.vars['do_value_segmentation'] = False
        self.po.vars['frame_by_frame_segmentation'] = True
        analysis_frame = MotionAnalysis([self.i, self.i + 1, self.po.vars, True, True, True, None])
        # PickleRick().write_file(analysis_frame.whole_shape_descriptors, self.path_output / "motion_analysis_frame.pkl")
        reference = PickleRick().read_file(self.path_output / "motion_analysis_frame.pkl")
        self.assertTrue(analysis_frame.whole_shape_descriptors.equals(reference))

    def test_run_all_arenas(self):
        self.test_save_data_to_run_cellects_quickly()
        self.test_video_writing()
        self.po.instantiate_tables()
        self.po.vars['maximal_growth_factor'] = 0.25
        self.po.vars['do_slope_segmentation'] = False
        self.po.vars['do_value_segmentation'] = False
        self.po.vars['frame_by_frame_segmentation'] = True
        self.po.vars['keep_unaltered_videos'] = False
        for i, arena in enumerate(self.po.vars['analyzed_individuals']):
            # i = 47; arena = i + 1
            l = [i, arena, self.po.vars, True, True, True, None]
            # l = [i, arena, self.po.vars, True, False, False, None]
            analysis_i = MotionAnalysis(l)
            self.assertTrue(os.path.isfile(f"ind_{arena}.mp4"))
            self.po.add_analysis_visualization_to_first_and_last_images(i, analysis_i.efficiency_test_1,
                                                                     analysis_i.efficiency_test_2)
        self.po.save_tables()
        self.assertTrue(os.path.isfile(f"one_row_per_arena.csv"))
        self.assertTrue(os.path.isfile(f"one_row_per_frame.csv"))
        self.assertTrue(os.path.isfile(f"one_row_per_oscillating_cluster.csv"))
        self.assertTrue(os.path.isfile(f"software_settings.csv"))
        self.assertTrue(os.path.isfile(f"Analysis efficiency, 3th image.jpg"))
        self.assertTrue(os.path.isfile(f"Analysis efficiency, last image.jpg"))

    def tearDown(self):

        if os.path.isfile(self.path_experiment / f"Data to run Cellects quickly.pkl"):
            os.remove(self.path_experiment / f"Data to run Cellects quickly.pkl")

        if os.path.isfile(self.path_experiment / f"one_row_per_arena.csv"):
            os.remove(self.path_experiment / f"one_row_per_arena.csv")
        if os.path.isfile(self.path_experiment / f"one_row_per_frame.csv"):
            os.remove(self.path_experiment / f"one_row_per_frame.csv")
        if os.path.isfile(self.path_experiment / f"one_row_per_oscillating_cluster.csv"):
            os.remove(self.path_experiment / f"one_row_per_oscillating_cluster.csv")
        if os.path.isfile(self.path_experiment / f"software_settings.csv"):
            os.remove(self.path_experiment / f"software_settings.csv")

        if os.path.isfile(self.path_experiment / f"Analysis efficiency, 3th image.jpg"):
            os.remove(self.path_experiment / f"Analysis efficiency, 3th image.jpg")
        if os.path.isfile(self.path_experiment / f"Analysis efficiency, last image.jpg"):
            os.remove(self.path_experiment / f"Analysis efficiency, last image.jpg")

        for arena in self.po.vars['analyzed_individuals']:
            if os.path.isfile(self.path_experiment / f"ind_{arena}.npy"):
                os.remove(self.path_experiment / f"ind_{arena}.npy")
            if os.path.isfile(self.path_experiment / f"ind_{arena}.mp4"):
                os.remove(self.path_experiment / f"ind_{arena}.mp4")


if __name__ == '__main__':
    unittest.main()
