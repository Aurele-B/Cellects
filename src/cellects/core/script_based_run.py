#!/usr/bin/env python3
"""This file contains lines to run Cellects without user interface"""

import logging
import os
from pathlib import Path
from numpy import (
    median, stack, ceil, all, any, equal, pi, min, round, mean, diff, sum, multiply, square,
    sqrt, zeros, array, arange, ones_like, isin, sort, repeat, uint8, uint32, unique, vstack, hstack,
    uint16, uint64, delete, savetxt, nonzero, max, absolute, load, logical_or)
from cellects.core.program_organizer import ProgramOrganizer
from cellects.utils.utilitarian import insensitive_glob
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.core.motion_analysis import MotionAnalysis


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

def run_one_video_analysis(po):
    i=2
    show_seg= True
    if os.path.isfile(f"coord_specimen{i + 1}_t720_y1475_x1477.npy"):
        binary_coord = np.load(f"coord_specimen{i + 1}_t720_y1475_x1477.npy")
        l = [i, i + 1, po.vars, False, False, show_seg, None]
        self = MotionAnalysis(l)
        self.binary = np.zeros((720, 1475, 1477), dtype=np.uint8)
        self.binary[binary_coord[0, :], binary_coord[1, :], binary_coord[2, :]] = 1
    else:
        l = [i, i + 1, po.vars, True, False, show_seg, None]
        self = MotionAnalysis(l)
    self.get_descriptors_from_binary()
    self.detect_growth_transitions()
    self.networks_detection(show_seg)
    self.study_cytoscillations(show_seg)

if __name__ == "__main__":
    po = load_one_folder(Path("/Users/Directory/Data/dossier1"), 6)
    run_image_analysis(po)
    write_videos(po)
    run_one_video_analysis(po)