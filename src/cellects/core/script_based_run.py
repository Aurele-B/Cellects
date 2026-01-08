#!/usr/bin/env python3
"""This file contains lines to run Cellects without user interface"""

import logging
import os
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import cv2
from cellects.core.program_organizer import ProgramOrganizer
from cellects.utils.utilitarian import insensitive_glob
from cellects.core.motion_analysis import MotionAnalysis
from cellects.image_analysis.morphological_operations import create_ellipse
from cellects.image_analysis.image_segmentation import convert_subtract_and_filter_video
from cellects.core.one_image_analysis import init_params
from cellects.utils.load_display_save import write_video_sets, readim, display_network_methods
from cellects.image_analysis.network_functions import NetworkDetection

def generate_colony_like_video():
    ellipse = create_ellipse(7, 7).astype(np.uint8)
    binary_video = np.zeros((20, 1000, 1000), dtype=np.uint8)
    binary_video[0, np.random.randint(100, 900, 20), np.random.randint(100, 900, 20)] = 1
    binary_video[0, ...] = cv2.dilate(binary_video[0, ...], ellipse)
    for t in range(1, binary_video.shape[0]):
        binary_video[t, ...] = cv2.dilate(binary_video[t - 1, ...], ellipse, iterations=1)
    rgb_video = np.zeros((binary_video.shape[0], binary_video.shape[1], binary_video.shape[2], 3), dtype=np.uint8)
    for c_ in range(3):
        rgb_video[:, :, :, c_][binary_video > 0] = np.random.randint(150 + c_ * 20, 250 - c_ * 20, binary_video.sum())
        rgb_video[:, :, :, c_][binary_video == 0] = np.random.randint(5 ,20 ,
                                                               binary_video.size - binary_video.sum())
    return rgb_video

def load_data(rgb_video: NDArray=None, pathway: str='', sample_number:int=None, radical: str='', extension: str='', im_or_vid: int=0):
    po = ProgramOrganizer()
    if rgb_video is None:
        if len(pathway) == 0:
            pathway = Path(os.getcwd() + "/data/single_experiment")
        po.all['global_pathway'] = pathway
        po.all['first_folder_sample_number'] = sample_number
        po.all['radical'] = radical
        po.all['extension'] = extension
        po.all['im_or_vid'] = im_or_vid
        po.look_for_data()
        po.load_data_to_run_cellects_quickly()
        po.get_first_image()
    else:
        po.get_first_image(rgb_video[0,...])
        po.analysis_instance = rgb_video
        po.all['im_or_vid'] = 1
    return po

def run_image_analysis(po, PCA: bool=True, last_im:NDArray=None):
    if not po.first_exp_ready_to_run:
        if PCA:
            po.fast_first_image_segmentation()
        else:
            params = init_params()
            params['is_first_image'] = True
            po.first_image.find_color_space_combinations(params)
        po.cropping(is_first_image=True)
        po.get_average_pixel_size()
        po.delineate_each_arena()
        po.get_background_to_subtract()
        po.get_origins_and_backgrounds_lists()
        po.get_last_image(last_im)
        po.fast_last_image_segmentation()
        po.find_if_lighter_background()
        po.extract_exif()
    else:
        print('Image analysis already done, run video analysis')
    return po

def run_one_video_analysis(po, with_video_in_ram: bool=False):
    i=0
    show_seg= False
    po.vars['frame_by_frame_segmentation'] = True
    po.vars['do_threshold_segmentation'] = False
    po.vars['do_slope_segmentation'] = False
    if po.vars['convert_for_motion'] is None:
        po.vars['convert_for_motion'] = po.vars['convert_for_origin']
    videos_already_in_ram = None
    if with_video_in_ram:
        converted_video, _ = convert_subtract_and_filter_video(po.analysis_instance, po.vars['convert_for_motion'])
        videos_already_in_ram = [po.analysis_instance, converted_video]
    segment: bool = True
    l = [i, i + 1, po.vars, segment, False, show_seg, videos_already_in_ram]
    MA = MotionAnalysis(l)
    MA.get_descriptors_from_binary()
    if os.path.isfile('colony_centroids1_20col_t20_y1000_x1000.csv'):
        os.remove('colony_centroids1_20col_t20_y1000_x1000.csv')
    if os.path.isfile('data/single_experiment/colony_centroids1_20col_t20_y1000_x1000.csv'):
        os.remove('data/single_experiment/colony_centroids1_20col_t20_y1000_x1000.csv')
    # MA.detect_growth_transitions()
    # MA.networks_analysis(show_seg)
    # MA.study_cytoscillations(show_seg)
    return MA

def write_videos(po):
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
                         remaining, po.all["raw_images"], is_landscape, use_list_of_vid, in_colors, po.reduce_image_dim,
                         pathway="")
    po.instantiate_tables()
    return po

def run_all_arenas(po):
    po.instantiate_tables()
    for i, arena in enumerate(po.vars['analyzed_individuals']):
        l = [i, arena, po.vars, True, True, False, None]
        # l = [i, arena, po.vars, False, True, False, None]
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
    # Keep the tables
    one_row_per_arena = po.one_row_per_arena
    one_row_per_frame = po.one_row_per_frame
    po.save_tables()
    po.one_row_per_arena = one_row_per_arena
    po.one_row_per_frame = one_row_per_frame
    cv2.imwrite(f"Analysis efficiency, last image.jpg", po.last_image.bgr)
    cv2.imwrite(f"Analysis efficiency, {np.ceil(po.vars['img_number'] / 10).astype(np.uint64)}th image.jpg",
        po.first_image.bgr)
    return po

def detect_network_in_one_image(im_path, save_path=None):
    im = readim(im_path)
    # im = im[100:870, 200:1000]
    greyscale_image = im.mean(axis=2)
    net = NetworkDetection(greyscale_image, add_rolling_window=True)
    net.get_best_network_detection_method()
    display_network_methods(net, save_path)
