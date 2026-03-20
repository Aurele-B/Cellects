#!/usr/bin/env python3
"""This file contains lines to run Cellects without user interface"""

import logging
import os
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import cv2
from cellects.core.cellects_paths import DATA_DIR
from cellects.core.program_organizer import ProgramOrganizer
from cellects.utils.utilitarian import insensitive_glob
from cellects.core.motion_analysis import MotionAnalysis
from cellects.image_analysis.morphological_operations import create_ellipse
from cellects.image_analysis.image_segmentation import convert_subtract_and_filter_video
from cellects.core.one_image_analysis import init_params
from cellects.utils.load_display_save import write_video_sets, readim, display_network_methods, video_writing_decision
from cellects.image_analysis.network_functions import NetworkDetection

def generate_colony_like_video():
    """
    Generate a colony-like video by applying dilation operations and random color filling.
    This function creates a binary video with randomized initial frames, dilates the
    frames using a circular kernel to simulate colony growth over time, and then converts
    the binary video into a colored RGB video.

    Parameters
    ----------
    None

    Other Parameters
    ----------------
    seed : int, optional
        The seed for the random number generator. Defaults to 42.

    ellipse_shape : tuple of int, optional
        The shape of the ellipse used for dilation. Defaults to (7, 7).

    binary_video_shape : tuple of int, optional
        The shape of the binary video. Defaults to (20, 1000, 1000).

    returns
    -------
    rgb_video : numpy.ndarray
        A video with shape `(20, 1000, 1000, 3)` where each frame is represented in RGB format.
        The video shows the growth and coloration of the colony over time.

    Examples
    --------
    >>> rgb_video = generate_colony_like_video()
    >>> print(rgb_video.shape)
    (20, 1000, 1000, 3)
    """
    np.random.seed(42)
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
    """
    Load data from a video (a numpy array of at least 3 dimensions) or using saved timelapse in a specified pathway.

    Parameters
    ----------
    rgb_video : NDArray, optional
        Numpy array representing RGB or greyscale video frames. Default is None.
    pathway : str, optional
        Path to the data directory. Default is an empty string leading to the data saved in Cellects' repository.
    sample_number : int, optional
        The number of arenas to detect in the first folder. Default is None.
    radical : str, optional
        The image or video pattern to look for at the beginning of their names. Default is an empty string taking all possibilities.
    extension : str, optional
        The image or video extension to look for. Default is an empty string taking all possibilities.
    im_or_vid : int, optional
        Indicator whether data is from image or video. 0 for image and 1 for video.
        Default is 0.

    Returns
    -------
    ProgramOrganizer
        An instance of the ProgramOrganizer class with loaded data.

    Examples
    --------
    >>> po = load_data(pathway="data/single_experiment", sample_number=1, radical="test", extension="jpg")
    >>> print(po.all)
    {'global_pathway': 'data/single_experiment', 'first_folder_sample_number': 1, 'radical': 'test', 'extension': 'jpg', 'im_or_vid': 0}
    """
    po = ProgramOrganizer()
    if rgb_video is None:
        if len(pathway) == 0:
            pathway = str(DATA_DIR / "single_experiment")
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

def run_image_analysis(po, run_automatic_color_space_finding: bool=False, last_im:NDArray=None):
    """
    Use the first image of the timelapse to extract all necessary information for video tracking.

    Do nothing if the experiment is already ready to run. Compute the Complete image analysis otherwise.

    Parameters
    ----------
    po : object
        The object containing current analysis parameters and connecting all methods of the software.
    run_automatic_color_space_finding : bool, optional
        Whether to perform automatic color space finding. Default is False.
    last_im : NDArray, optional
        The last image to be analyzed. Default is None, will read the last image in the folder.

    Returns
    -------
    po : object
        The modified object containing current analysis parameters and connecting all methods of the software.

    Notes
    -----
    This function modifies the `po` object in place to perform various
    image analysis tasks such as cropping, delineating arenas,
    finding background, saving origins and backgrounds lists,
    segmenting the first and last images, and finding lighter
    background.

    """
    if not po.first_exp_ready_to_run:
        if run_automatic_color_space_finding:
            params = init_params()
            params['is_first_image'] = True
            po.first_image.find_color_space_combinations(params)
        else:
            po.fast_first_image_segmentation()
        po.cropping(is_first_image=True)
        po.get_average_pixel_size()
        po.delineate_each_arena()
        po.get_background_to_subtract()
        po.save_origins_and_backgrounds_lists()
        po.get_last_image(last_im)
        po.fast_last_image_segmentation()
        po.find_if_lighter_background()
        po.save_exif()
    else:
        print('Image analysis already done, run video analysis')
    return po

def run_one_video_analysis(po, arena_id: int=1, do_segmentation: bool= True, with_video_in_ram: bool=False, remove_files: bool=False):
    """
    Load the video of one arena and (if required) runs motion analysis on it.

    Parameters
    ----------
    po : object
        The object containing current analysis parameters and connecting all methods of the software.
    arena_id : int, optional
        Arena ID to process. Default and first is 1.
    do_segmentation : bool, optional
        Whether to perform segmentation during analysis. Default is True.
    with_video_in_ram : bool, optional
        Whether the video is already in RAM or need to be loaded. Default is False.
    remove_files : bool, optional
        Whether to remove output files after processing. Default is False.

    Returns
    -------
    MotionAnalysis
        The Motion Analysis object containing the videos and its segmentation (if required).
    """
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
    l = [arena_id - 1, arena_id, po.vars, do_segmentation, False, show_seg, videos_already_in_ram]
    MA = MotionAnalysis(l)
    if MA.binary is None:
        return MA
    MA.get_descriptors_from_binary()
    if remove_files:
        files = insensitive_glob("colony_centroids*") + insensitive_glob("ind_*")
        for f in files:
            os.remove(f)
        if os.path.isfile('cellects_data.h5'):
            os.remove('cellects_data.h5')
    # MA.detect_growth_transitions()
    # MA.networks_analysis(show_seg)
    # MA.study_cytoscillations(show_seg)
    return MA

def write_videos(po: object):
    """
    Write one video per arena in the current folder.

    This method requires the first_exp_ready_to_run argument of the ProgramOrganizer instance (po) to be true.

    Parameters
    ----------
    po : object
        The object containing current analysis parameters and connecting all methods of the software.

    Returns
    -------
    po : object
        The modified object containing current analysis parameters and connecting all methods of the software.

    Raises
    ------
    ValueError
        If there is an issue with writing videos (e.g., insufficient memory, invalid parameters).

    Notes
    -----
    This function updates the output list and makes a decision on whether to write videos based on certain conditions. If the decision is positive, it prepares and writes video sets.

    """
    po.update_output_list()
    do_write_videos = video_writing_decision(len(po.vars['analyzed_individuals']), po.all['im_or_vid'], po.all['overwrite_unaltered_videos'])
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
    """
    Run analysis on all arenas and save the results.

    Performs motion analysis on each arena, updates visualization,
    saves basic statistics and descriptors in long format, and saves the tables.

    Parameters
    ----------
    po : object
        The object containing current analysis parameters and connecting all methods of the software.

    Returns
    -------
    po : object
        The modified object containing current analysis parameters and connecting all methods of the software.
    """
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

def detect_network_in_one_image(im_path, edge_max_width, save_path=None):
    """
    Detect and visualize the best network detection method for a given image.

    Parameters
    ----------
    im_path : str
        Path to the input image.
    edge_max_width : int
        Maximum width of edges in pixels for network detection.
    save_path : str or None, optional
        Path to save the visualization of detected networks. If `None`, the visualization is not saved.

    Notes
    -----
    This function uses image processing techniques to detect networks in a given grayscale image.
    It then visualizes the detection results and saves them if the `save_path` is specified.
    """
    im = readim(im_path)
    # im = im[100:870, 200:1000]
    greyscale_image = im.mean(axis=2)
    net = NetworkDetection(greyscale_image, add_rolling_window=True, edge_max_width=edge_max_width, morphological_closing=True)
    net.get_best_network_detection_method()
    display_network_methods(net, save_path)
