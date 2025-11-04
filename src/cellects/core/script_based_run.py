#!/usr/bin/env python3
"""This file contains lines to run Cellects without user interface"""

import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from cellects.core.program_organizer import ProgramOrganizer
from cellects.utils.utilitarian import insensitive_glob
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.core.motion_analysis import MotionAnalysis
from cellects.config.all_vars_dict import DefaultDicts
from cellects.utils.load_display_save import show


def load_one_folder(pathway, sample_number):
    po = ProgramOrganizer()
    po.load_variable_dict()
    # dd = DefaultDicts()
    # po.all = dd.all
    # po.vars = dd.vars
    po.all['global_pathway'] = pathway
    po.all['first_folder_sample_number'] = sample_number
    # po.all['first_folder_sample_number'] = 6
    # po.all['radical'] = "IMG"
    # po.all['extension'] = ".jpg"
    # po.all['im_or_vid'] = 0
    po.look_for_data()
    po.load_data_to_run_cellects_quickly()
    return po

def run_image_analysis(po):
    if not po.first_exp_ready_to_run:
        po.get_first_image()
        po.fast_image_segmentation(True)
        # po.first_image.find_first_im_csc(sample_number=po.sample_number,
        #                                    several_blob_per_arena=None,
        #                                    spot_shape=None, spot_size=None,
        #                                    kmeans_clust_nb=2,
        #                                    biomask=None, backmask=None,
        #                                    color_space_dictionaries=None,
        #                                    carefully=True)
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


def write_videos(po):
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


def run_one_video_analysis(po):
    i=0
    show_seg= False
    # if os.path.isfile(f"coord_specimen{i + 1}_t720_y1475_x1477.npy"):
    #     binary_coord = np.load(f"coord_specimen{i + 1}_t720_y1475_x1477.npy")
    #     l = [i, i + 1, po.vars, False, False, show_seg, None]
    #     MA = MotionAnalysis(l)
    #     MA.binary = np.zeros((720, 1475, 1477), dtype=np.uint8)
    #     MA.binary[binary_coord[0, :], binary_coord[1, :], binary_coord[2, :]] = 1
    # else:
    l = [i, i + 1, po.vars, True, False, show_seg, None]
    MA = MotionAnalysis(l)
    MA.get_descriptors_from_binary()
    MA.detect_growth_transitions()
    MA.networks_detection(show_seg)
    MA.study_cytoscillations(show_seg)
    return MA


def run_all_arenas(po):
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



if __name__ == "__main__":
    po = load_one_folder(Path("/data/experiment"), 1)
    po = run_image_analysis(po)
    po = write_videos(po)
    # MA = run_one_video_analysis(po)
    run_all_arenas(po)

    # MA.one_row_per_frame.to_csv(
    #     "/Users/Directory/Scripts/python/Cellects/tests/data/experiment/motion_analysis_thresh.csv")

    # path = Path("/Users/Directory/Scripts/python/Cellects/tests/data/experiment")
    # po.load_variable_dict()
    # run_image_analysis(po)
    # os.chdir(path)
    # from glob import glob
    # from cellects.utils.load_display_save import readim
    # im_names = np.sort(glob("*.JPG"))
    # for i, im_name in enumerate(im_names): #  im_name = im_names[-1]
    #     im = readim(im_name)
    #     cv2.imwrite(f"image{i + 1}.tif", im[2925:3170, 1200:1500, :])
    #
    #     show(im[2925:3170,1200:1500, :])
