#!/usr/bin/env python3
"""
This script contains all unit tests of the one_video_per_blob script
"""
import unittest
import psutil
from cellects.core.program_organizer import ProgramOrganizer
from cellects.core.motion_analysis import MotionAnalysis
from cellects.config.all_vars_dict import DefaultDicts
from cellects.image_analysis.morphological_operations import rhombus_55
from cellects.utils.load_display_save import write_video_sets, PickleRick
from cellects.core.cellects_paths import ALL_VARS_PKL_FILE
from tests._base import CellectsUnitTest, rgb_several_arenas_img, several_arenas_bin_img, several_arenas_vid, several_arenas_bin_vid
import numpy as np
import cv2
import os


class TestProgramOrganizerLoading(CellectsUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.i = 0
        cls.po = ProgramOrganizer()
        cls.po.update_variable_dict()
        cls.po.all['global_pathway'] = cls.path_experiment
        cls.po.look_for_data()

    def test_save_as_pkl_and_load_vars_exception(self):
        dd = DefaultDicts()
        dd.save_as_pkl()
        with open(ALL_VARS_PKL_FILE, 'rb') as source:
            data = bytearray(source.read())
            if len(data) < 10:  # Avoid truncating too much
                raise ValueError("Original pickle is too small to safely truncate")
            truncated_data = data[:-10]  # Remove last 10 bytes

        with open(ALL_VARS_PKL_FILE, 'wb') as broken_file:
            broken_file.write(truncated_data)
        po = ProgramOrganizer()
        po.load_variable_dict()
        dd.save_as_pkl(self.po)
        self.assertTrue(os.path.isfile(ALL_VARS_PKL_FILE))

    def test_look_for_data(self):
        self.assertEqual(len(self.po.data_list), 25)
        self.assertEqual(self.po.all['folder_number'], 1)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(ALL_VARS_PKL_FILE):
            os.remove(ALL_VARS_PKL_FILE)
        if os.path.isfile(self.path_experiment / "Data to run Cellects quickly.pkl"):
            os.remove(self.path_experiment / "Data to run Cellects quickly.pkl")


class TestProgramOrganizerPickleDeletion(CellectsUnitTest):
    def test_pickle_rick_deletion(self):
        os.chdir(self.path_output)
        PickleRick()._write_pickle_rick()
        PickleRick(0)._write_pickle_rick()
        po = ProgramOrganizer()
        self.assertFalse(os.path.isfile('PickleRick.pkl'))
        self.assertFalse(os.path.isfile('PickleRick0.pkl'))

class TestProgramOrganizerUpdateVariableDict(CellectsUnitTest):
    def test_update_variable_dict(self):
        po = ProgramOrganizer()
        po.all['descriptors'] = {}
        po.vars['descriptors'] = {}
        po.update_variable_dict()
        self.assertGreater(len(po.all['descriptors']), 0)
        self.assertGreater(len(po.vars['descriptors']), 0)


class TestProgramOrganizerSegmentation(CellectsUnitTest):
    """Test suite for segmenting images using ProgramOrganizer class"""

    @classmethod
    def setUpClass(cls):
        """Initialize two data sets for testing"""
        super().setUpClass()
        cls.color_space_combination = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
        cls.img_list = several_arenas_vid
        cls.image = several_arenas_vid[0]
        cls.shape_number = 2

        cls.po = ProgramOrganizer()
        cls.po.get_first_image(several_arenas_vid[0], sample_number=2)
        cls.po.get_last_image(several_arenas_vid[0])
        cls.po.vars["convert_for_origin"] = cls.color_space_combination
        cls.po.vars["convert_for_motion"] = cls.color_space_combination
        cls.po.update_variable_dict()

    def test_save_and_load_variable_dict(self):
        self.po.save_variable_dict()
        self.assertTrue(os.path.isfile(ALL_VARS_PKL_FILE))
        self.po.load_variable_dict()
        if os.path.isfile(ALL_VARS_PKL_FILE):
            os.remove(ALL_VARS_PKL_FILE)
        self.po.load_variable_dict()

    def test_look_for_data_in_sub_folder(self):
        po = ProgramOrganizer()
        po.all['global_pathway'] = self.d / "multiple_experiments"
        po.all['radical'] = "im"
        po.all['extension'] = "tif"
        po.all['sample_number_per_folder'] = 1
        po.all['im_or_vid'] = 1
        po.look_for_data()
        po.update_folder_id(sample_number=1, folder_name="f1")
        po.all['im_or_vid'] = 0
        po.look_for_data()
        po.update_folder_id(sample_number=1, folder_name="f1")
        self.assertEqual(len(po.all['folder_list']), 2)
        po.update_folder_id(sample_number=1, folder_name="f1")
        po.load_data_to_run_cellects_quickly()
        po.get_first_image()
        back_mask = np.zeros(po.first_im.shape[:2], np.uint8)
        back_mask[-30:, :] = 1
        po.all['back_mask'] = np.nonzero(back_mask)
        po.vars['convert_for_origin'] = {'PCA': np.array([0, 0, 1], dtype=np.int8), 'logical': 'None'}
        po.vars['convert_for_motion'] = po.vars['convert_for_origin']
        po.all['automatically_crop'] = True

        po.fast_first_image_segmentation()
        po.cropping(is_first_image=True)

        self.assertEqual(len(po.first_image.crop_coord), 4)
        po.all['scale_with_image_or_cells'] = 1
        po.all['starting_blob_hsize_in_mm'] = 15
        po.get_average_pixel_size()
        info = po.delineate_each_arena()
        self.assertTrue(info['continue'])
        self.assertEqual(len(po.left), 1)
        self.assertEqual(len(po.right), 1)
        self.assertEqual(len(po.top), 1)
        self.assertEqual(len(po.bot), 1)
        po.get_background_to_subtract()
        po.get_origins_and_backgrounds_lists()
        self.assertTrue(len(po.vars['origin_list'][0]) > 0)
        po.get_last_image()
        po.fast_last_image_segmentation()
        self.assertTrue(po.last_image.binary_image.any())
        po.cropping(is_first_image=False)
        po.find_if_lighter_background()
        self.assertEqual(po.vars['lighter_background'], False)
        for k in po.all['descriptors'].keys():
            po.all['descriptors'][k] = True
        po.vars['oscilacyto_analysis'] = True
        po.vars['keep_unaltered_videos'] = True
        po.update_output_list()
        po.instantiate_tables()
        po.vars['maximal_growth_factor'] = 0.25
        for k in po.data_to_save.keys():
            po.data_to_save[k] = True
        po.save_variable_dict()
        po.save_data_to_run_cellects_quickly()
        po.load_data_to_run_cellects_quickly()
        po = ProgramOrganizer()
        po.all['global_pathway'] = self.d / "multiple_experiments"
        po.load_variable_dict()
        po.update_folder_id(sample_number=1, folder_name="f1")
        po.load_data_to_run_cellects_quickly()

    def test_simple_pipeline(self):
        # Simulate a drift correction:
        po = ProgramOrganizer()
        image = several_arenas_vid[0].copy()
        image[0, :, :] = 0
        image[:, -1, :] = 0
        image[:, -2, :] = 0
        sample_number = 2
        po.get_first_image(image, sample_number)
        po.get_last_image(image)
        po.fast_first_image_segmentation()
        po.fast_last_image_segmentation()
        print(po.all['automatically_crop']) # False currently
        po.cropping(True)
        po.get_average_pixel_size()
        po.all['scale_with_image_or_cells'] = 0
        po.get_average_pixel_size()
        po.vars['subtract_background'] = True
        po.get_background_to_subtract()
        po.vars['output_in_mm'] = True

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(ALL_VARS_PKL_FILE):
            os.remove(ALL_VARS_PKL_FILE)
        if os.path.isfile(self.d / "multiple_experiments" / "f1" / "Data to run Cellects quickly.pkl"):
            os.remove(self.d / "multiple_experiments" / "f1" / "Data to run Cellects quickly.pkl")

class TestProgramOrganizerArenaDelineation(CellectsUnitTest):
    """Test suite for delineating arenas using ProgramOrganizer class"""

    @classmethod
    def setUpClass(cls):
        """Initialize two data sets for testing"""
        super().setUpClass()
        cls.color_space_combination = {"logical": "None", "PCA": np.ones(3, dtype=np.uint8)}
        cls.img_list = several_arenas_vid
        cls.image = several_arenas_vid[0]
        # Simulate a drift correction:
        cls.image[0, :, :] = 0
        cls.image[:, -1, :] = 0
        cls.image[:, -2, :] = 0
        cls.po = ProgramOrganizer()
        cls.po.get_first_image(cls.image, sample_number=2)
        cls.po.get_last_image(cls.image)
        cls.po.fast_first_image_segmentation()
        cls.po.fast_last_image_segmentation()
        # print(cls.po.all['automatically_crop']) # False currently
        # cls.po.save_data_to_run_cellects_quickly()
        # cls.po.all['overwrite_unaltered_videos'] = False
        cls.po.cropping(True)
        cls.po.get_average_pixel_size()
        cls.po.all['scale_with_image_or_cells'] = 0
        cls.po.get_average_pixel_size()
        cls.po.vars['subtract_background'] = True
        cls.po.get_background_to_subtract()
        # print(cls.po.first_image.validated_shapes)
        # cls.po.get_first_image(several_arenas_vid[0], sample_number=2)
        # cls.po.vars["convert_for_origin"] = cls.color_space_combination
        # cls.po.vars["convert_for_motion"] = cls.color_space_combination
        # cls.po.update_variable_dict()
        # cls.po.first_image.validated_shapes = several_arenas_bin_vid[0]
        # cls.po.data_list = [several_arenas_vid]
        #
        cls.image2 = rgb_several_arenas_img[:, :, 0]
        cls.shape_number2 = 6
        cls.po2 = ProgramOrganizer()
        cls.po2.update_variable_dict()
        cls.po2.get_first_image(cls.image2, sample_number=6)
        cls.po2.get_last_image(cls.image2)
        cls.po2.vars["convert_for_motion"] = cls.color_space_combination
        cls.po2.first_image.validated_shapes = several_arenas_bin_img
        cls.po2.first_image.shape_number = 6

    def test_get_bounding_boxes_with_moving_centers(self):
        """Test the get bounding boxes algorithm when the centroids of the shapes move throughout the video"""
        are_gravity_centers_moving = True
        sample_size = 1
        all_specimens_have_same_direction = True
        motion_list = self.po._segment_blob_motion(sample_size=sample_size)
        self.po.get_bounding_boxes(are_gravity_centers_moving, motion_list, all_specimens_have_same_direction,
                                   original_shape_hsize=2)
        visited_pixels = np.any(several_arenas_bin_vid[:-1, ...], axis=0)
        Y, X = np.nonzero(visited_pixels)
        self.assertTrue(self.po.top.min() <= Y.min())
        self.assertTrue(self.po.left.min() <= X.min())
        self.assertTrue(self.po.bot.max() >= Y.max())
        self.assertTrue(self.po.right.max() >= self.po.left.min())

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
        po = ProgramOrganizer()
        po.get_first_image(image, sample_number=shape_number)
        po.first_image.validated_shapes = image
        po.first_image.shape_number = po.sample_number
        po.update_variable_dict()
        po.vars["convert_for_motion"] = {"logical": "None", "PCA": np.ones(3, dtype=np.uint8)}
        motion_list = [image]
        are_gravity_centers_moving=True
        all_specimens_have_same_direction=True
        original_shape_hsize = 10
        po.get_bounding_boxes(are_gravity_centers_moving, motion_list, all_specimens_have_same_direction,
                                   original_shape_hsize)
        self.assertTrue(np.sum(po.top) > 0)
        self.assertTrue(np.sum(po.bot) > 0)
        self.assertTrue(np.sum(po.left) > 0)
        self.assertTrue(np.sum(po.right) > 0)

    def test_get_quick_bounding_boxes(self):
        """Test get_bounding_boxes using a fast algorithm"""
        self.po.first_image.validated_shapes = several_arenas_bin_vid[0]
        are_gravity_centers_moving = False
        self.po.get_bounding_boxes(are_gravity_centers_moving)
        self.po.get_origins_and_backgrounds_lists()
        self.assertTrue(np.sum(self.po.top) > 0)
        self.assertTrue(np.sum(self.po.bot) > 0)
        self.assertTrue(np.sum(self.po.left) > 0)
        self.assertTrue(np.sum(self.po.right) > 0)

    def test_get_bounding_boxes_with_no_shapes(self):
        """Test get_bounding_boxes when no shapes are detected"""
        self.po.first_image.validated_shapes = np.zeros_like(several_arenas_bin_vid[0])
        are_gravity_centers_moving = False
        self.po.get_bounding_boxes(are_gravity_centers_moving)
        self.assertEqual(self.po.top, 0)
        self.assertEqual(self.po.bot, 20)
        self.assertEqual(self.po.left, 0)
        self.assertEqual(self.po.right, 20)

    def test_get_bounding_boxes_with_moving_centers_with_close_shapes(self):
        """Test get_bounding_boxes when shapes are very close to each other"""
        are_gravity_centers_moving = True
        img_list = [self.image2, self.image2]
        color_number = 2
        sample_size = 2
        all_specimens_have_same_direction = True
        display = False
        filter_spec = None
        self.po2.data_list = [self.image2, self.image2]
        motion_list = self.po2._segment_blob_motion(sample_size=2)
        self.po2.get_bounding_boxes(are_gravity_centers_moving, motion_list, all_specimens_have_same_direction,
                                          original_shape_hsize=2)
        self.assertGreater(self.po2.top.sum(), 0)
        self.assertGreater(self.po2.bot.sum(), 0)
        self.assertGreater(self.po2.left.sum(), 0)
        self.assertGreater(self.po2.right.sum(), 0)

    def test_write_videos_with_different_video_dimensions(self):
        """Test write_videos_as_np_arrays when the dimensions of each video are different"""
        os.chdir(self.path_experiment)
        img_list = [self.path_experiment / "image1.tif", self.path_experiment / "image2.tif"]
        self.po.top=np.array([0, 3])
        self.po.bot=np.array([2, 6])
        self.po.left=np.array([0, 3])
        self.po.right=np.array([3, 5])
        self.po.first_image.shape_number = 2
        in_colors = not self.po.vars['already_greyscale']
        min_ram_free = self.po.vars['min_ram_free']
        bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining, use_list_of_vid, is_landscape = self.po.prepare_video_writing(
            img_list, min_ram_free, in_colors)
        write_video_sets(self.po.data_list, sizes, vid_names, self.po.first_image.crop_coord,
                         (self.po.top, self.po.bot, self.po.left, self.po.right), bunch_nb, video_nb_per_bunch,
                         remaining, self.po.all["raw_images"], is_landscape, use_list_of_vid, in_colors, self.po.reduce_image_dim,
                         pathway="")
        self.assertTrue(os.path.isfile(self.path_experiment / f"ind_1.npy"))
        self.assertTrue(os.path.isfile(self.path_experiment / f"ind_2.npy"))
        self.po.get_origins_and_backgrounds_lists()
        self.po.vars['bb_coord'] = 0, self.po.first_image.image.shape[0], 0, self.po.first_image.image.shape[0], self.po.top, self.po.bot, self.po.left, self.po.right
        self.po.vars['convert_for_motion']['PCA2'] = np.ones(3)
        self.po.vars['filter_spec'] = {'filter1_type': 'Gaussian', 'filter1_param': [.5, 1.], 'filter2_type': "Median", 'filter2_param': [.5, 1.]}
        self.l = [0, 1, self.po.vars, False, False, False, None]
        self.ma = MotionAnalysis(self.l)

    def test_prepare_video_writing_using_too_much_memory(self):
        """Test prepare_video_writing when writing all videos at the same time is not possible with current memory"""
        min_ram_free = (psutil.virtual_memory().available >> 30) - 5
        if min_ram_free > 0:
            img_list = [np.zeros((1000, 1000), dtype=np.uint8) for i in range(5000)]
            self.po.top=np.array([0, 500])
            self.po.left=np.array([0, 500])
            self.po.bot=np.array([500, 1000])
            self.po.right=np.array([500, 1000])
            bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining, use_list_of_vid, is_landscape  = self.po.prepare_video_writing(img_list, min_ram_free)
            self.assertGreater(bunch_nb, 0)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_experiment / f"ind_1.npy"):
            os.remove(self.path_experiment / f"ind_1.npy")
        if os.path.isfile(self.path_experiment / f"ind_2.npy"):
            os.remove(self.path_experiment / f"ind_2.npy")


class TestProgramOrganizerWithVideo(CellectsUnitTest):
    """Test suite for analyzing videos using ProgramOrganizer class"""
    @classmethod
    def setUpClass(cls):
        """Initialize data set for testing"""
        super().setUpClass()
        cls.po = ProgramOrganizer()
        cls.po.all['global_pathway'] = cls.d / "single_experiment"
        cls.po.all['radical'] = "vid"
        cls.po.all['extension'] = "mp4"
        cls.po.all['sample_number_per_folder'] = 1
        cls.po.all['im_or_vid'] = 1

    def test_with_video(self):
        self.po.update_variable_dict()
        self.po.look_for_data()
        self.po.get_first_image()
        self.assertTrue(self.po.first_image.image.any())
        self.po.get_last_image()
        self.assertTrue(self.po.last_image.image.any())
        timings = self.po.extract_exif()
        self.assertTrue(timings.any())
        self.po.fast_first_image_segmentation()
        self.po.fast_last_image_segmentation()
        self.po.cropping(True)
        self.po.cropping(False)
        self.po.get_average_pixel_size()
        self.po.get_background_to_subtract()
        self.po.find_if_lighter_background()
        info = self.po.delineate_each_arena()
        self.assertTrue(info['continue'])
        self.assertEqual(len(self.po.left), 1)
        self.assertEqual(len(self.po.right), 1)
        self.assertEqual(len(self.po.top), 1)
        self.assertEqual(len(self.po.bot), 1)
        self.po.vars['exif'] = np.arange(len(self.po.data_list))
        self.po.vars['do_fading'] = True
        self.po.complete_image_analysis()
        self.assertTrue(os.path.isfile(self.path_experiment / f"one_row_per_frame.csv"))
        self.assertTrue(os.path.isfile(self.path_experiment / f"one_row_per_arena.csv"))
        self.po.get_origins_and_backgrounds_lists()
        self.po.vars['save_coord_network'] = True
        self.po.vars['save_graph'] = True
        self.po.vars['study_cytoscillations'] = True
        self.po.vars['save_coord_thickening_slimming'] = True
        self.po.vars['fractal_analysis'] = True
        self.l = [0, 1, self.po.vars, True, True, False, None]
        # self.l = [0, 1, self.po.vars, False, False, False, None]
        self.ma = MotionAnalysis(self.l)

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile(self.path_experiment / f"ind_1.mp4"):
            os.remove(self.path_experiment / f"ind_1.mp4")
        if os.path.isfile(self.path_experiment / f"one_row_per_frame.csv"):
            os.remove(self.path_experiment / f"one_row_per_frame.csv")
        if os.path.isfile(self.path_experiment / f"one_row_per_arena.csv"):
            os.remove(self.path_experiment / f"one_row_per_arena.csv")
        if os.path.isfile(self.path_experiment / f"Analysis efficiency, 3th image.JPG"):
            os.remove(self.path_experiment / f"Analysis efficiency, 3th image.JPG")
        if os.path.isfile(self.path_experiment / f"Analysis efficiency, last image.JPG"):
            os.remove(self.path_experiment / f"Analysis efficiency, last image.JPG")
        if os.path.isfile(self.path_experiment / f"software_settings.csv"):
            os.remove(self.path_experiment / f"software_settings.csv")
        if os.path.isfile(self.path_experiment / f"coord_network1_t25_y244_x300.npy"):
            os.remove(self.path_experiment / f"coord_network1_t25_y244_x300.npy")
        if os.path.isfile(self.path_experiment / f"coord_pseudopods1_t25_y244_x300.npy"):
            os.remove(self.path_experiment / f"coord_pseudopods1_t25_y244_x300.npy")
        if os.path.isfile(self.path_experiment / f"vertex_table1_t25_y244_x300.csv"):
            os.remove(self.path_experiment / f"vertex_table1_t25_y244_x300.csv")
        if os.path.isfile(self.path_experiment / f"edge_table1_t25_y244_x300.csv"):
            os.remove(self.path_experiment / f"edge_table1_t25_y244_x300.csv")


if __name__ == '__main__':
    unittest.main()