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
from cellects.core.cellects_paths import ALL_VARS_JSON_FILE
from cellects.utils.load_display_save import write_json, read_json
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
            'initial_bio_mask': None,
            'initial_back_mask': None,
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
            # Main image analysis params:
            'several_blob_per_arena': False,
            'convert_for_motion': {
                'lab': [0, 0, 1],
                'logical': 'None'},
            'convert_for_origin': {
                'lab': [0, 0, 1],
                'logical': 'None'},
            'arena_shape': 'rectangle', # 'circle',
            'subtract_background': False,
            'filter_spec': {'filter1_type': "", 'filter1_param': [.5, 1.], 'filter2_type': "", 'filter2_param': [.5, 1.]},

            # Main video tracking params:
            'frame_by_frame_segmentation': False,
            'do_slope_segmentation': False,
            'do_threshold_segmentation': True,
            'true_if_use_light_AND_slope_else_OR': False,
            'maximal_growth_factor': 0.05,
            'repeat_video_smoothing': 1,

            # Post-processing params
            'specimen_activity': 'grow',
            'sliding_window_segmentation': True,
            'morphological_opening': True,
            'morphological_closing': True,
            'fading': 0,
            'detection_range_factor': 2,
            'max_size_for_connection': 300,
            'min_size_for_connection': 20,
            'correct_errors_around_initial': False,
            'prevent_fast_growth_near_periphery': False,
            'appearance_detection_method': 'largest',
            'periphery_width': 40,
            'max_periphery_growth': 20,

            # Segmentation params:
            'color_number': 2,
            'rolling_window_segmentation': {'do': False, 'side_len': None, 'step': None, 'min_int_var': None},
            'grid_segmentation': False,
            'mesh_side_length': 4,
            'mesh_step_length': 2,
            'mesh_min_int_var': 20,
            'first_detection_frame': 0,
            'luminosity_threshold': 127,

            # Output params:
            'output_in_mm': True,
            'save_coord_specimen': False,
            'save_graph': False,
            'save_coord_thickening_slimming': False,
            'save_coord_network': False,
            'oscilacyto_analysis': False,
            'fractal_analysis': False,
            'fractal_box_side_threshold': 32,
            'fractal_zoom_step': 0,
            'expected_oscillation_period': 2,  # (min)
            'minimal_oscillating_cluster_size': 50,  # (pixels)
            'iso_digi_analysis': True,
            # Data stored during analysis:
            'descriptors': descriptors,

            # Hard params:
            'keep_unaltered_videos': False,
            'min_ram_free': 1.,
            'save_processed_videos': True,
            'video_fps': 60,
            'videos_extension': '.mp4',
            'lose_accuracy_to_save_memory': True,

            # Automatically determined params:
            'video_list': None,
            'analyzed_individuals': [1],
            'bio_label': 1,
            'bio_label2': 1,
            'first_move_threshold': None,
            'img_number': 0,
            'origin_state': 'fluctuating',
            'already_greyscale': False,
            'drift_already_corrected': False,
            'time_step': 1,
            'time_step_is_arbitrary': True,
            'exif': [],
        }

    def save_as_json(self, po=None, reset_params: bool=False):
        if po is None:
            write_json('cellects_settings.json', self.all)
        else:
            if reset_params:
                po.all = self.all
                po.vars = self.vars
            po.save_variable_dict()
