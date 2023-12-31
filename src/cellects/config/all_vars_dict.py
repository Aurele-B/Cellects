import os
from cellects.image_analysis.shape_descriptors import descriptors_categories, descriptors
from numpy import min, max, all, any, array, int8, empty, uint16
from cellects.core.cellects_paths import ALL_VARS_PKL_FILE
from cellects.utils.load_display_save import PickleRick
from cellects.core.cellects_paths import TEST_DIR


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
            'crop_images': False,
            'descriptors': descriptors_categories,
            'display_shortcuts': False,
            'do_distant_shape_integration_during_segmentation': False,
            'all_same_direction': True,
            'do_multiprocessing': False,
            'extension': '.jpg',
            'first_detection_frame': 1,
            'folder_number': 1,
            'first_folder_sample_number': 1,
            'first_move_threshold_in_mm²': 10,
            'folder_list': [],
            'global_pathway': str(TEST_DIR / "experiment"),
            'im_or_vid': 0,
            'image_horizontal_size_in_mm': 700,
            'minimal_appearance_size': 10,
            'more_than_two_colors': False,
            # 'overwrite_cellects_data': True,
            'overwrite_unaltered_videos': False,
            'radical': 'IMG_',
            'raw_images': False,
            'sample_number_per_folder': [1],
            'scale_with_image_or_cells': 0,
            'set_spot_shape': True,
            'set_spot_size': True,
            'starting_blob_hsize_in_mm': 15,
            'starting_blob_shape': None
        }

        self.vars = {
            'analyzed_individuals': empty(0, dtype=uint16),
            'arena_shape': 'circle',
            'bio_label': 1,
            'bio_label2': 1,
            'color_number': 2,
            'convert_for_motion': {
                'lab': array((0, 0, 1), int8),
                'logical': 'None'},
            'convert_for_origin': {
                'lab': array((0, 0, 1), int8),
                'logical': 'None'},
            'ease_distant_shape_integration': 2,
            'first_move_threshold': None,
            'img_number': 0,
            'iso_digi_analysis': True,
            'luminosity_threshold': 127,
            'max_distant_shape_size': 300,
            'min_distant_shape_size': 20,
            'origin_state': 'fluctuating',
            'oscilacyto_analysis': False,
            'network_detection': False,
            'fractal_analysis': False,
            'fractal_threshold_detection': 100,
            'subtract_background': False,
            'ring_correction': False,
            # According to Smith and Saldana (1992),
            # P. polycephalum shuttle streaming has a period of 100-200s
            'already_greyscale': False,
            'descriptors_in_long_format': True,
            'do_slope_segmentation': False,
            'do_value_segmentation': True,
            'drift_already_corrected': False,
            'first_detection_method': 'largest',
            'frame_by_frame_segmentation': False,
            'iterate_smoothing': 1,
            'keep_unaltered_videos': False,
            'max_growth_per_frame': 0.05,
            'min_ram_free': 0.87,
            'oscillation_period': 2,  # (min)
            'output_in_mm': True,
            'save_processed_videos': True,
            'several_blob_per_arena': False,
            'time_step': 1,
            'true_if_use_light_AND_slope_else_OR': False,
            'do_fading': False,
            'fading': 0,
            'video_fps': 60,
            'videos_extension': '.mp4',
            'exif': [],
            'lose_accuracy_to_save_memory': True,

            # Data stored during analysis:
            'descriptors': descriptors,
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



if __name__ == "__main__":
    DefaultDicts().save_as_pkl()
