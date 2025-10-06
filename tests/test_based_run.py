#!/usr/bin/env python3
"""This file contains lines to run Cellects without user interface"""

import logging
import numpy as np
import pandas as pd
import cv2
from cellects.core.program_organizer import ProgramOrganizer
from cellects.utils.utilitarian import insensitive_glob
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.core.motion_analysis import MotionAnalysis
from cellects.config.all_vars_dict import DefaultDicts


def load_test_folder(pathway, sample_number):
    po = ProgramOrganizer()
    dd = DefaultDicts()
    po.all = dd.all
    po.vars = dd.vars
    po.all['global_pathway'] = pathway
    po.all['first_folder_sample_number'] = sample_number
    po.look_for_data()
    return po


def run_image_analysis_for_testing(po):
    if not po.first_exp_ready_to_run:
        po.get_first_image()
        po.fast_image_segmentation(True)
        po.cropping(is_first_image=True)
        po.get_average_pixel_size()
        po.delineate_each_arena()
        po.get_background_to_subtract()
        po.get_origins_and_backgrounds_lists()
        po.get_last_image()
        po.fast_image_segmentation(is_first_image=False)
        po.find_if_lighter_background()
        po.extract_exif()
    else:
        print('Image analysis already done, run video analysis')
    return po


def run_write_videos_for_testing(po):
    po.update_output_list()
    look_for_existing_videos = insensitive_glob('ind_' + '*' + '.npy')
    there_already_are_videos = len(look_for_existing_videos) == len(po.vars['analyzed_individuals'])
    logging.info(
        f"{len(look_for_existing_videos)} .npy video files found for {len(po.vars['analyzed_individuals'])} arenas to analyze")
    do_write_videos = not there_already_are_videos or (
            there_already_are_videos and po.all['overwrite_unaltered_videos'])
    if do_write_videos:
        po.videos = OneVideoPerBlob(po.first_image, po.starting_blob_hsize_in_pixels, po.all['raw_images'])
        po.videos.left = po.left
        po.videos.right = po.right
        po.videos.top = po.top
        po.videos.bot = po.bot
        po.videos.first_image.shape_number = po.sample_number
        po.videos.write_videos_as_np_arrays(
            po.data_list, po.vars['min_ram_free'], not po.vars['already_greyscale'], po.reduce_image_dim)
    po.instantiate_tables()
    return po


def run_one_video_analysis_for_testing(po):
    i=0
    show_seg= False
    l = [i, i + 1, po.vars, True, False, show_seg, None]
    MA = MotionAnalysis(l)
    MA.get_descriptors_from_binary()
    MA.detect_growth_transitions()
    MA.networks_detection(show_seg)
    MA.study_cytoscillations(show_seg)
    return MA


def run_all_arenas_for_testing(po):
    po.instantiate_tables()
    for i, arena in enumerate(po.vars['analyzed_individuals']):
        l = [i, arena, po.vars, True, True, False, None]
        analysis_i = MotionAnalysis(l)
        po.add_analysis_visualization_to_first_and_last_images(i, analysis_i.efficiency_test_1,
                                                                    analysis_i.efficiency_test_2)
        if not po.vars['several_blob_per_arena']:
            # Save basic statistics
            po.update_one_row_per_arena(i, analysis_i.one_descriptor_per_arena)

            # Save descriptors in long_format
            po.update_one_row_per_frame(i * po.vars['img_number'],
                                                      arena * po.vars['img_number'],
                                                      analysis_i.one_row_per_frame)
            # Save cytosol_oscillations
        if not pd.isna(analysis_i.one_descriptor_per_arena["first_move"]):
            if po.vars['oscilacyto_analysis']:
                oscil_i = pd.DataFrame(
                    np.c_[np.repeat(arena,
                                    analysis_i.clusters_final_data.shape[0]), analysis_i.clusters_final_data],
                    columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance', 'coord_y',
                             'coord_x'])
                if po.one_row_per_oscillating_cluster is None:
                    po.one_row_per_oscillating_cluster = oscil_i
                else:
                    po.one_row_per_oscillating_cluster = pd.concat((po.one_row_per_oscillating_cluster, oscil_i))
    po.save_tables()
    cv2.imwrite(f"Analysis efficiency, last image.jpg", po.last_image.bgr)
    cv2.imwrite(f"Analysis efficiency, {np.ceil(po.vars['img_number'] / 10).astype(np.uint64)}th image.jpg",
        po.first_image.bgr)
