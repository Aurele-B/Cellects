#!/usr/bin/env python3
"""
This script contains all integration tests of Cellects
"""
import os.path
import unittest
from tests._base import CellectsUnitTest
from tests.test_based_run import *
from cellects.config.all_vars_dict import DefaultDicts
from cellects.utils.load_display_save import PickleRick
import numpy as np
from numba.typed import Dict as TDict


class TestCellects(CellectsUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.i = 0
        cls.po = load_test_folder(str(cls.path_experiment), 1)
        cls.po = run_image_analysis_for_testing(cls.po)
        cls.descriptor_nb = np.sum([des for des in cls.po.vars['descriptors'].values()])

    def test_look_for_data(self):
        self.assertEqual(len(self.po.data_list), 25)
        self.assertEqual(self.po.all['folder_number'], 1)

    def test_save_data_to_run_cellects_quickly(self):
        if not os.path.isfile(self.path_experiment / "Data to run Cellects quickly.pkl"):
            coordinates = PickleRick().read_file(self.path_input / "coordinates.pkl")
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
        coordinates = PickleRick().read_file(self.path_input / "coordinates.pkl")
        self.assertTrue(np.array_equal(self.po.first_image.crop_coord, coordinates[0]))
        self.assertTrue(np.array_equal(self.po.left, coordinates[1]))
        self.assertTrue(np.array_equal(self.po.right, coordinates[2]))
        self.assertTrue(np.array_equal(self.po.top, coordinates[3]))
        self.assertTrue(np.array_equal(self.po.bot, coordinates[4]))

    def test_run_image_analysis(self):
        self.assertTrue(self.po.first_image.binary_image.sum() < self.po.last_image.binary_image.sum())

    def test_video_writing(self):
        self.po = run_write_videos_for_testing(self.po)
        for video_i in np.arange(len(self.po.vars['analyzed_individuals'])) + 1:
            self.assertTrue(os.path.isfile(f"ind_{video_i}.npy"))

    def run_all_arenas_for_testing(self):
        self.po.instantiate_tables()
        self.po.vars['do_slope_segmentation'] = False
        self.po.vars['do_value_segmentation'] = False
        self.po.vars['frame_by_frame_segmentation'] = True
        self.po.vars['keep_unaltered_videos'] = True
        self.po = run_write_videos_for_testing(self.po)
        run_all_arenas_for_testing(self.po)
        for i, arena in enumerate(self.po.vars['analyzed_individuals']):
            self.assertTrue(os.path.isfile(f"ind_{arena}.mp4"))
        self.assertTrue(os.path.isfile(f"one_row_per_arena.csv"))
        self.assertTrue(os.path.isfile(f"one_row_per_frame.csv"))
        self.assertTrue(os.path.isfile(f"software_settings.csv"))
        self.assertTrue(os.path.isfile(f"Analysis efficiency, 3th image.jpg"))
        self.assertTrue(os.path.isfile(f"Analysis efficiency, last image.jpg"))

    def test_detection(self):
        self.po = run_write_videos_for_testing(self.po)
        MA = run_one_video_analysis_for_testing(self.po)
        self.assertTrue(MA.one_row_per_frame.shape[0] == 25)
        self.assertTrue(MA.one_row_per_frame.shape[1] == self.descriptor_nb + 2)
        self.assertTrue(np.any(MA.one_row_per_frame.iloc[:, 2:]))

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
