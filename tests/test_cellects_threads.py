#!/usr/bin/env python3
"""
This script contains all unit tests of the cellects_threads script
"""
from cellects.utils.utilitarian import insensitive_glob
from cellects.core.cellects_threads import *
from cellects.core.cellects_paths import CONFIG_DIR
from tests._base import CellectsUnitTest
import unittest


class TestPrecompileNJITThread(CellectsUnitTest):
    """Test suite for PrecompileNJITThread class."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.precompile = PrecompileNJITThread()

    def test_precompile_n_jit_methods(self):
        """Test basic behavior."""
        self.precompile.run()
        self.assertFalse(self.precompile.isRunning())
        self.assertFalse(self.precompile.isInterruptionRequested())


class TestOnSeveralFolders(CellectsUnitTest):
    """Test suite for threads to analyze several folders ."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.po = ProgramOrganizer()
        cls.po.all['global_pathway'] = cls.d + '/multiple_experiments'
        cls.po.all['radical'] = 'im'
        cls.po.all['extension'] = 'tif'
        cls.po.all['im_or_vid'] = 0
        cls.po.sample_number = 1
        cls.po.all['first_folder_sample_number'] = 1
        cls.po.all['sample_number_per_folder'] = [1]
        cls.look_for_data_thread = LookForDataThreadInFirstW(cls.po)
        cls.look_for_data_thread.run()
        cls.load_if_several_thread = LoadFirstFolderIfSeveralThread(cls.po)
        cls.po.update_folder_id(1, 'f1')
        cls.load_if_several_thread.run()
        cls.save_all_vars_thread = SaveAllVarsThread(cls.po)
        cls.video_tracking_thread = VideoTrackingThread(cls.po)

    def test_load_first_folder_if_several_thread(self):
        """Test the basic behavior of the LoadFirstFolderIfSeveralThread class."""
        self.assertIsNotNone(self.po.first_im)

    def test_save_all_vars_thread(self):
        """Test the behavior of the SaveAllVarsThread class with several folders."""
        self.save_all_vars_thread.run()
        self.assertTrue(os.path.isfile("cellects_settings.json"))

    def test_one_arena_video_tracking_thread(self):
        """Test the behavior of the VideoTrackingThread class with several folders."""
        self.po.video_task = 'one_arena'
        self.po.all['compute_all_options'] = True
        self.po.load_quick_full = 2
        self.video_tracking_thread.run()
        self.assertIsNotNone(self.po.motion.segmented)

    def test_all_video_tracking_thread(self):
        """Test running all arenas of all folders with the VideoTrackingThread class."""
        self.po.video_task = 'all'
        self.video_tracking_thread.run()
        self.assertTrue(os.path.isfile( f"one_row_per_arena.csv"))

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile("cellects_settings.json"):
            os.remove("cellects_settings.json")
        files_to_remove = insensitive_glob('*.h5') + insensitive_glob('Analysis efficiency*') + insensitive_glob('one_*.csv') + insensitive_glob('ind_1.mp4')
        for file in files_to_remove:
            os.remove(file)


class TestCellectsThreads(CellectsUnitTest):
    """Test suite for cellects basic threads."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.po = ProgramOrganizer()
        cls.po.all['global_pathway'] = cls.d + '/single_experiment'
        cls.po.all['radical'] = 'im'
        cls.po.all['extension'] = 'tif'
        cls.po.all['im_or_vid'] = 0
        cls.po.all['first_folder_sample_number'] = 1
        cls.po.sample_number = 1
        cls.load_quick_thread = LoadDataToRunCellectsQuicklyThread(cls.po)
        cls.look_for_data_thread = LookForDataThreadInFirstW(cls.po)
        cls.get_first_im_thread = GetFirstImThread(cls.po)
        cls.get_last_im_thread = GetLastImThread(cls.po)
        cls.load_quick_thread.run()
        cls.get_first_im_thread.run()
        cls.get_last_im_thread.run()
        cls.po.drawn_image = cls.po.first_image.image.copy()
        cls.po.current_image = cls.po.first_image.image.copy()
        cls.update_image_thread = UpdateImageThread(cls.po)
        cls.first_image_analysis_thread = FirstImageAnalysisThread(cls.po)
        cls.first_image_analysis_thread.run()
        cls.po.bio_masks_number = 1
        cls.po.back_masks_number = 1
        cls.dims = cls.po.first_image.image.shape[:2]
        cls.po.bio_mask = np.zeros(cls.dims, dtype=np.uint8)
        cls.po.back_mask = np.zeros(cls.dims, dtype=np.uint8)
        cls.po.back_mask[0, :] = 1
        cls.po.bio_mask[cls.dims[0] // 2, cls.dims[1] // 2] = 1
        cls.last_image_analysis_thread = LastImageAnalysisThread(cls.po)
        cls.crop_scale_subtract_delineate_thread = CropScaleSubtractDelineateThread(cls.po)
        cls.po.arena_mask = np.ones(cls.dims, dtype=np.uint8)
        cls.save_manual_delineation_thread = SaveManualDelineationThread(cls.po)
        cls.save_manual_delineation_thread.run()
        cls.get_exif_data_thread = GetExifDataThread(cls.po)
        cls.get_exif_data_thread.run()
        cls.complete_image_analysis_thread = CompleteImageAnalysisThread(cls.po)
        cls.po.vars['convert_for_motion'] = {'lab': [0, 0, 1], 'logical': 'None'}
        cls.prepare_video_analysis_thread = PrepareVideoAnalysisThread(cls.po)
        cls.save_all_vars_thread = SaveAllVarsThread(cls.po)
        cls.video_tracking_thread = VideoTrackingThread(cls.po)

    def test_load_data_to_run_cellects_quickly_thread(self):
        """Test the basic behavior of the LoadDataToRunCellectsQuicklyThread class."""
        self.assertIsNotNone(self.po.first_im)

    def test_look_for_data_thread_in_first_w_thread(self):
        """Test the basic behavior of the LookForDataThreadInFirstW class."""
        self.look_for_data_thread.run()
        self.assertGreater(len(self.po.data_list), 0)

    def test_get_first_im_thread(self):
        """Test the basic behavior of the GetFirstImThread class."""
        self.assertIsNotNone(self.po.first_im)

    def test_get_last_im_thread(self):
        """Test the basic behavior of the GetLastImThread class."""
        self.assertIsNotNone(self.po.last_im)

    def test_update_image_thread(self):
        """Test the basic behavior of the UpdateImageThread class."""
        self.update_image_thread.run()
        self.assertTrue(self.po.drawn_image.any())

    def test_first_image_analysis_thread(self):
        """Test the basic behavior of the FirstImageAnalysisThread class."""
        self.assertTrue(self.po.first_image.binary_image.any())

    def test_last_image_analysis_thread(self):
        """Test the basic behavior of the LastImageAnalysisThread class."""
        self.last_image_analysis_thread.run()
        self.assertTrue(self.po.last_image.binary_image.any())

    def test_crop_scale_subtract_delineate_thread(self):
        """Test the basic behavior of the CropScaleSubtractDelineateThread class."""
        self.crop_scale_subtract_delineate_thread.run()
        self.assertIsNotNone(self.po.top)

    def test_get_exif_data_thread(self):
        """Test the basic behavior of the GetExifDataThread class."""
        self.assertTrue('exif' in self.po.vars)

    def test_complete_image_analysis_thread(self):
        """Test the basic behavior of the CompleteImageAnalysisThread class."""
        self.po.get_average_pixel_size()
        self.complete_image_analysis_thread.run()
        self.assertTrue(os.path.isfile('one_row_per_arena.csv'))
        # self.assertTrue(os.path.isfile(f"Analysis efficiency, last image.jpg"))

    def test_prepare_video_analysis_thread(self):
        """Test the basic behavior of the PrepareVideoAnalysisThread class."""
        self.prepare_video_analysis_thread.run()
        self.assertTrue('lighter_background' in self.po.vars)
        self.assertTrue(os.path.isfile(f'ind_{1}.h5'))

    def test_save_all_vars_thread(self):
        """Test the basic behavior of the SaveAllVarsThread class."""
        self.save_all_vars_thread.run()
        self.assertFalse(os.path.isfile(CONFIG_DIR / 'masks.h5'))

    def test_one_arena_video_tracking_thread(self):
        """Test the behavior of the VideoTrackingThread class."""
        self.po.video_task = 'one_arena'
        self.video_tracking_thread.run()
        self.assertTrue(len(self.po.motion.one_descriptor_per_arena) > 0)
        self.po.video_task = 'change_one_arena_result'
        self.video_tracking_thread.run()
        self.assertTrue(os.path.isfile('one_row_per_arena.csv'))

    def test_all_video_tracking_thread(self):
        """Test running all arenas of all folders with the VideoTrackingThread class."""
        self.po.video_task = 'all'
        self.po.all['do_multiprocessing'] = True
        os.remove('cellects_settings.json')
        self.video_tracking_thread.run()
        self.assertTrue(os.path.isfile( f"one_row_per_arena.csv"))

    def tearDown(self):
        """Remove all written files."""
        if os.path.isfile("cellects_settings.json"):
            os.remove("cellects_settings.json")
        files_to_remove = insensitive_glob('*.h5') + insensitive_glob('Analysis efficiency*') + insensitive_glob('*.csv') + insensitive_glob('ind_1.mp4')
        for file in files_to_remove:
            os.remove(file)

if __name__ == '__main__':
    unittest.main()
