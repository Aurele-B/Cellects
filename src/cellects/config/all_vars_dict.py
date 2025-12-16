#!/usr/bin/env python3
"""
This script generates the default parameters of the GUI of Cellects.
It can be used to write these parameters in a file named all_vars_dict.
Then, the gui updates this file as users adjust the GUI parameters.
These parameters are stored in a dictionary with keys corresponding to the parameter's name and values to its tunable
 value.
"""

import os
from cellects.image_analysis.shape_descriptors import descriptors_categories, descriptors
import numpy as np
from cellects.core.cellects_paths import ALL_VARS_PKL_FILE
from cellects.utils.load_display_save import PickleRick
from cellects.core.cellects_paths import EXPERIMENTS_DIR


class DefaultDicts:
    def __init__(self):
        # po.load_variable_dict()
        self.all = {
            # Interface settings:
            'compute_all_options': True,
            'expert_mode': False,
            'is_auto': False,
            'night_mode': False,
            'arena': 1,
            'video_option': 0,

            # Analysis settings:
            'are_gravity_centers_moving': 0,
            'are_zigzag': 'columns',
            'automatic_size_thresholding': True,
            'color_number': 2,
            'cores': 1,
            'automatically_crop': False,
            'descriptors': descriptors_categories,
            'display_shortcuts': False,
            'connect_distant_shape_during_segmentation': False,
            'all_specimens_have_same_direction': True,
            'extract_time_interval': True,
            'do_multiprocessing': False,
            'extension': '.tif',
            'folder_number': 1,
            'first_folder_sample_number': 1,
            'first_move_threshold_in_mm²': 10,
            'folder_list': [],
            'global_pathway': str(EXPERIMENTS_DIR),
            'im_or_vid': 0,
            'image_horizontal_size_in_mm': 700,
            'minimal_appearance_size': 10,
            'more_than_two_colors': False,
            'bio_mask': None,
            'back_mask': None,
            'keep_cell_and_back_for_all_folders': False,

            # 'overwrite_cellects_data': True,
            'overwrite_unaltered_videos': False,
            'radical': 'im',
            'raw_images': False,
            'sample_number_per_folder': [1],
            'scale_with_image_or_cells': 1,
            'set_spot_shape': True,
            'set_spot_size': True,
            'starting_blob_hsize_in_mm': 15,
            'starting_blob_shape': None,
            'auto_mesh_side_length': True,
            'auto_mesh_step_length': True,
            'auto_mesh_min_int_var': True,
        }

        self.vars = {
            'video_list': None,
            'analyzed_individuals': np.array([1], dtype=np.uint16),
            'arena_shape': 'rectangle', # 'circle',
            'bio_label': 1,
            'bio_label2': 1,
            'color_number': 2,
            'convert_for_motion': {
                'lab': np.array((0, 0, 1), np.int8),
                'logical': 'None'},
            'convert_for_origin': {
                'lab': np.array((0, 0, 1), np.int8),
                'logical': 'None'},
            'detection_range_factor': 2,
            'first_move_threshold': None,
            'img_number': 0,
            'iso_digi_analysis': True,
            'first_detection_frame': 0,
            'luminosity_threshold': 127,
            'max_size_for_connection': 300,
            'min_size_for_connection': 20,
            'origin_state': 'fluctuating',
            'oscilacyto_analysis': False,

            'rolling_window_segmentation': {'do': False, 'side_len': None, 'step': None, 'min_int_var': None},
            'grid_segmentation': False,
            'mesh_side_length': 4,
            'mesh_step_length': 2,
            'mesh_min_int_var': 20,

            'fractal_analysis': False,
            'fractal_box_side_threshold': 32,
            'fractal_zoom_step': 0,
            'subtract_background': False,
            'correct_errors_around_initial': False,
            'prevent_fast_growth_near_periphery': False,
            'periphery_width': 40,
            'max_periphery_growth': 20,
            # According to Smith and Saldana (1992),
            # P. polycephalum shuttle streaming has a period of 100-200s
            'already_greyscale': False,
            'descriptors_in_long_format': True,
            'do_slope_segmentation': False,
            'do_threshold_segmentation': True,
            'drift_already_corrected': False,
            'appearance_detection_method': 'largest',
            'frame_by_frame_segmentation': False,
            'repeat_video_smoothing': 1,
            'keep_unaltered_videos': False,
            'maximal_growth_factor': 0.05,
            'min_ram_free': 0.87,
            'expected_oscillation_period': 2,  # (min)
            'minimal_oscillating_cluster_size': 50,  # (pixels)
            'output_in_mm': True,
            'save_processed_videos': True,
            'several_blob_per_arena': False,
            'time_step': 1,
            'time_step_is_arbitrary': True,
            'true_if_use_light_AND_slope_else_OR': False,
            'do_fading': False,
            'fading': 0,
            'video_fps': 60,
            'videos_extension': '.mp4',
            'exif': [],
            'lose_accuracy_to_save_memory': True,
            'save_coord_specimen': False,
            'save_graph': False,
            'save_coord_thickening_slimming': False,
            'save_coord_network': False,
            # Data stored during analysis:
            'descriptors': descriptors,
            'filter_spec': {'filter1_type': "", 'filter1_param': [.5, 1.], 'filter2_type': "", 'filter2_param': [.5, 1.]},
        }

    def save_as_pkl(self, po=None):
        if po is None:
            if os.path.isfile('PickleRick0.pkl'):
                os.remove('PickleRick0.pkl')
            pickle_rick = PickleRick(0)
            pickle_rick.write_file(self.all, ALL_VARS_PKL_FILE)
        else:
            po = po
            po.all = self.all
            po.vars = self.vars
            po.save_variable_dict()
