#!/usr/bin/env python3
"""This file contains lines to run Cellects without user interface"""

import logging
import numpy as np
import pandas as pd
import cv2
from cellects.core.program_organizer import ProgramOrganizer
from cellects.utils.utilitarian import insensitive_glob
from cellects.core.motion_analysis import MotionAnalysis
from cellects.config.all_vars_dict import DefaultDicts
from cellects.utils.load_display_save import write_video_sets

"""
1. Browse po and ma to make a list of factors and their level
2. Draw a tree allowing to cover each
3. Make as many classes as necessary
"""


def load_test_folder(pathway, sample_number):
    # pathway="/Users/Directory/Scripts/python/Cellects/data/single_experiment"
    # sample_number=1
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
        po.vars['convert_for_motion'] = {'lab': np.array([0, 0, 1], dtype=np.int8), 'logical': 'Or',
                                         'luv2': np.array([0, 0, 1], dtype=np.int8)}
        po.fast_first_image_segmentation()
        po.cropping(is_first_image=True)
        po.get_average_pixel_size()
        analysis_status = po.delineate_each_arena()
        po.vars['subtract_background'] = True
        po.get_background_to_subtract()
        po.get_origins_and_backgrounds_lists()
        po.get_last_image()
        po.fast_last_image_segmentation()
        po.find_if_lighter_background()
        timing = po.extract_exif()
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
        po.first_image.shape_number = po.sample_number
        in_colors = not po.vars['already_greyscale']
        bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining, use_list_of_vid, is_landscape = po.prepare_video_writing(
            po.data_list, po.vars['min_ram_free'], in_colors)
        write_video_sets(po.data_list, sizes, vid_names, po.first_image.crop_coord,
                         (po.top, po.bot, po.left, po.right), bunch_nb, video_nb_per_bunch,
                         remaining, po.all["raw_images"], is_landscape, use_list_of_vid, in_colors,
                         po.reduce_image_dim,
                         pathway="")
    po.instantiate_tables()
    return po

def run_one_video_analysis_for_testing(po):
    i=0
    show_seg= False
    l = [i, i + 1, po.vars, True, False, show_seg, None]
    # l = [i, i + 1, po.vars, False, False, show_seg, None]
    MA = MotionAnalysis(l)
    MA.get_descriptors_from_binary()
    MA.detect_growth_transitions()
    MA.networks_analysis(show_seg)
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
    po.save_tables()
    cv2.imwrite(f"Analysis efficiency, last image.jpg", po.last_image.bgr)
    cv2.imwrite(f"Analysis efficiency, {np.ceil(po.vars['img_number'] / 10).astype(np.uint64)}th image.jpg",
        po.first_image.bgr)
