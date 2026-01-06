#!/usr/bin/env python3
"""Module for analyzing motion, growth patterns, and structural properties of biological specimens in video data.

This module provides comprehensive tools to analyze videos of biological samples (e.g., cell colonies) by:
1. Loading and converting RGB videos to grayscale using configurable color space combinations
2. Performing multi-strategy segmentation (frame-by-frame, intensity thresholding, derivative-based detection)
3. Applying post-processing steps including error correction algorithms for shape continuity
4. Computing morphological descriptors over time (area, perimeter, fractal dimension, etc.)
5. Detecting network structures and oscillatory behavior in dynamic biological systems

Classes
-------
MotionAnalysis : Processes video data to analyze specimen motion, growth patterns, and structural properties.
    Provides methods for loading videos, performing segmentation using multiple algorithms,
    post-processing results with error correction, extracting morphological descriptors,
    detecting network structures, analyzing oscillations, and saving processed outputs.

Functions
---------
load_images_and_videos : Loads and converts video files to appropriate format for analysis.
get_converted_video : Converts RGB video to grayscale based on specified color space parameters.
detection : Performs multi-strategy segmentation of the specimen across all frames.
update_shape : Updates segmented shape with post-processing steps like noise filtering and hole filling.
save_results : Saves processed data, efficiency tests, and annotated videos.

Notes
-----
The features of this module include:
- Processes large video datasets with memory optimization strategies including typed arrays (NumPy)
  and progressive processing techniques.
- The module supports both single-specimen and multi-specimen analysis through configurable parameters.
- Segmentation strategies include intensity-based thresholding, gradient detection, and combinations thereof.
- Post-processing includes morphological operations to refine segmented regions and error correction for specific use cases (e.g., Physarum polycephalum).
- Biological network detection and graph extraction is available to represent network structures as vertex-edge tables.
- Biological oscillatory pattern detection
- Fractal dimension calculation
"""

import weakref
from gc import collect
import numpy as np
from numba.typed import Dict as TDict
from psutil import virtual_memory
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.cell_leaving_detection import cell_leaving_detection
from cellects.image_analysis.oscillations_functions import detect_oscillations_dynamics
from cellects.image_analysis.image_segmentation import segment_with_lum_value, convert_subtract_and_filter_video
from cellects.image_analysis.morphological_operations import (find_major_incline, create_ellipse, draw_me_a_sun,
                                                              inverted_distance_transform, dynamically_expand_to_fill_holes,
                                                              box_counting_dimension, prepare_box_counting, cc)
from cellects.image_analysis.network_functions import *
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.image_analysis.shape_descriptors import compute_one_descriptor_per_frame, compute_one_descriptor_per_colony, scale_descriptors, ShapeDescriptors, from_shape_descriptors_class
from cellects.utils.utilitarian import smallest_memory_array
from cellects.utils.formulas import detect_first_move


class MotionAnalysis:

    def __init__(self, l: list):

        """
        Analyzes motion in a given arena using video data.

        This class processes video frames to analyze motion within a specified area,
        detecting shapes, covering durations, and generating descriptors for further
        analysis.

        Args:
            l (list): A list containing various parameters and flags necessary for the motion
                analysis.

        Args:
            l[0] (int): Arena index.
            l[1] (str): Arena identifier or name, stored in one_descriptor_per_arena['arena'].
            l[2] (dict): Variables required for the analysis, stored in vars.
            l[3] (bool): Flag to detect shape.
            l[4] (bool): Flag to analyze shape.
            l[5] (bool): Flag to show segmentation.
            l[6] (None or list): Videos already in RAM.

        Attributes:
            vars (dict): Variables required for the analysis.
            visu (None): Placeholder for visualization data.
            binary (None): Placeholder for binary segmentation data.
            origin_idx (None): Placeholder for the index of the first frame.
            smoothing_flag (bool): Flag to indicate if smoothing should be applied.
            dims (tuple): Dimensions of the converted video.
            segmentation (ndarray): Array to store segmentation data.
            covering_intensity (ndarray): Intensity values for covering analysis.
            mean_intensity_per_frame (ndarray): Mean intensity per frame.
            borders (object): Borders of the arena.
            pixel_ring_depth (int): Depth of the pixel ring for analysis, default is 9.
            step (int): Step size for processing, default is 10.
            lost_frames (int): Number of lost frames to account for, default is 10.
            start (None or int): Starting frame index for the analysis.

        Methods:
            load_images_and_videos(videos_already_in_ram, arena_idx): Loads images and videos
                for the specified arena index.
            update_ring_width(): Updates the width of the pixel ring for analysis.
            get_origin_shape(): Detects the origin shape in the video frames.
            get_covering_duration(step): Calculates the covering duration based on a step size.
            detection(): Performs motion detection within the arena.
            initialize_post_processing(): Initializes post-processing steps.
            update_shape(show_seg): Updates the shape based on segmentation and visualization flags.
            get_descriptors_from_binary(): Extracts descriptors from binary data.
            detect_growth_transitions(): Detects growth transitions in the data.
            networks_analysis(show_seg): Detected networks within the arena based on segmentation
                visualization.
            study_cytoscillations(show_seg): Studies cytoscillations within the arena with
                segmentation visualization.
            fractal_descriptions(): Generates fractal descriptions of the analyzed data.
            get_descriptors_summary(): Summarizes the descriptors obtained from the analysis.
            save_results(): Saves the results of the analysis.

        """
        self.one_descriptor_per_arena = {}
        self.one_descriptor_per_arena['arena'] = l[1]
        vars = l[2]
        detect_shape = l[3]
        analyse_shape = l[4]
        show_seg = l[5]
        videos_already_in_ram = l[6]
        self.visu = None
        self.binary = None
        self.origin_idx = None
        self.smoothing_flag: bool = False
        self.drift_mask_coord = None
        self.coord_network = None
        logging.info(f"Start the motion analysis of the arena n°{self.one_descriptor_per_arena['arena']}")

        self.vars = vars
        if not 'contour_color' in self.vars:
            self.vars['contour_color']: np.uint8 = 0
        if not 'background_list' in self.vars:
            self.vars['background_list'] = []
        self.load_images_and_videos(videos_already_in_ram, l[0])

        self.dims = self.converted_video.shape
        self.segmented = np.zeros(self.dims, dtype=np.uint8)

        self.covering_intensity = np.zeros(self.dims[1:], dtype=np.float64)
        self.mean_intensity_per_frame = np.mean(self.converted_video, (1, 2))

        self.borders = image_borders(self.dims[1:], shape=self.vars['arena_shape'])
        self.pixel_ring_depth = 9
        self.step: int = 10
        self.lost_frames = 10
        self.update_ring_width()

        self.start = None
        if detect_shape:
            self.assess_motion_detection()
            if self.start is not None:
                self.detection()
                self.initialize_post_processing()
                self.t = self.start
                while self.t < self.dims[0]:  #200:
                    self.update_shape(show_seg)
                #

            if analyse_shape:
                self.get_descriptors_from_binary()
                self.detect_growth_transitions()
                self.networks_analysis(show_seg)
                self.study_cytoscillations(show_seg)
                self.fractal_descriptions()
                if videos_already_in_ram is None:
                    self.save_results()

    def load_images_and_videos(self, videos_already_in_ram, i: int):
        """

        Load images and videos from disk or RAM.

        Parameters
        ----------
        videos_already_in_ram : numpy.ndarray or None
            Video data that is already loaded into RAM. If `None`, videos will be
            loaded from disk.
        i : int
            Index used to select the origin and background data.

        Notes
        -----
        This method logs information about the arena number and loads necessary data
        from disk or RAM based on whether videos are already in memory. It sets various
        attributes like `self.origin`, `self.background`, and `self.converted_video`.

        """
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Load images and videos")
        if 'bb_coord' in self.vars:
            crop_top, crop_bot, crop_left, crop_right, top, bot, left, right = self.vars['bb_coord']
        elif videos_already_in_ram is not None:
            if isinstance(videos_already_in_ram, list):
                crop_bot, crop_right = videos_already_in_ram[0].shape[1], videos_already_in_ram[0].shape[2]
            else:
                crop_bot, crop_right = videos_already_in_ram.shape[1], videos_already_in_ram.shape[2]
            crop_top, crop_left, top, bot, left, right = 0, 0, [0], [crop_bot], [0], [crop_right]
        if isinstance(self.vars['origin_list'][i], Tuple):
            self.origin_idx = self.vars['origin_list'][i]
            frame_height = bot[i] - top[i]
            true_frame_width = right[i] - left[i]
            self.origin = np.zeros((frame_height, true_frame_width), dtype=np.uint8)
            self.origin[self.origin_idx[0], self.origin_idx[1]] = 1
        else:
            self.origin = self.vars['origin_list'][i]
            frame_height = self.origin.shape[0]
            true_frame_width = self.origin.shape[1]

        vid_name = None
        if self.vars['video_list'] is not None:
            vid_name = self.vars['video_list'][i]
        self.background = None
        if len(self.vars['background_list']) > 0:
            self.background = self.vars['background_list'][i]
        self.background2 = None
        if 'background_list2' in self.vars and len(self.vars['background_list2']) > 0:
            self.background2 = self.vars['background_list2'][i]
        vids = read_one_arena(self.one_descriptor_per_arena['arena'], self.vars['already_greyscale'],
                              self.vars['convert_for_motion'], videos_already_in_ram, true_frame_width, vid_name,
                              self.background, self.background2)
        self.visu, self.converted_video, self.converted_video2 = vids
        # When the video(s) already exists (not just written as .pny), they need to be sliced:
        if self.visu is not None:
            if self.visu.shape[1] != frame_height or self.visu.shape[2] != true_frame_width:
                self.visu = self.visu[:, crop_top:crop_bot, crop_left:crop_right, ...]
                self.visu = self.visu[:, top[i]:bot[i], left[i]:right[i], ...]
                if self.converted_video is not None:
                    self.converted_video = self.converted_video[:, crop_top:crop_bot, crop_left:crop_right]
                    self.converted_video = self.converted_video[:, top[i]:bot[i], left[i]:right[i]]
                    if self.converted_video2 is not None:
                        self.converted_video2 = self.converted_video2[:, crop_top:crop_bot, crop_left:crop_right]
                        self.converted_video2 = self.converted_video2[:, top[i]:bot[i], left[i]:right[i]]

        if self.converted_video is None:
            logging.info(
                f"Arena n°{self.one_descriptor_per_arena['arena']}. Convert the RGB visu video into a greyscale image using the color space combination: {self.vars['convert_for_motion']}")
            vids = convert_subtract_and_filter_video(self.visu, self.vars['convert_for_motion'],
                                                     self.background, self.background2,
                                                     self.vars['lose_accuracy_to_save_memory'],
                                                     self.vars['filter_spec'])
            self.converted_video, self.converted_video2 = vids

    def assess_motion_detection(self):
        """
        Assess if a motion can be detected using the current parameters.

        Validate the specimen(s) detected in the first frame and evaluate roughly how growth occurs during the video.
        """
        # Here to conditional layers allow to detect if an expansion/exploration occured
        self.get_origin_shape()
        # The first, user-defined is the 'first_move_threshold' and the second is the detection of the
        # substantial image: if any of them is not detected, the program considers there is no motion.
        if self.dims[0] >= 40:
            step = self.dims[0] // 20
        else:
            step = 1
        if self.dims[0] == 1 or self.start >= (self.dims[0] - step - 1):
            self.start = None
            self.binary = np.repeat(np.expand_dims(self.origin, 0), self.converted_video.shape[0], axis=0)
        else:
            self.get_covering_duration(step)

    def get_origin_shape(self):
        """
        Determine the origin shape and initialize variables based on the state of the current analysis.

        This method analyzes the initial frame or frames to determine the origin shape
        of an object in a video, initializing necessary variables and matrices for
        further processing.


        Attributes Modified:
            start: (int) Indicates the starting frame index.
            origin_idx: (np.ndarray) The indices of non-zero values in the origin matrix.
            covering_intensity: (np.ndarray) Matrix used for pixel fading intensity.
            substantial_growth: (int) Represents a significant growth measure based on the origin.

        Notes:
            - The method behavior varies if 'origin_state' is set to "constant" or not.
            - If the background is lighter, 'covering_intensity' matrix is initialized.
            - Uses connected components to determine which shape is closest to the center
              or largest, based on 'appearance_detection_method'.
        """
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Make sure of origin shape")
        if self.vars['drift_already_corrected']:
            self.drift_mask_coord = np.zeros((self.dims[0], 4), dtype=np.uint32)
            for frame_i in np.arange(self.dims[0]):  # 100):#
                true_pixels = np.nonzero(self.converted_video[frame_i, ...])
                self.drift_mask_coord[frame_i, :] = np.min(true_pixels[0]), np.max(true_pixels[0]) + 1, np.min(true_pixels[1]), np.max(true_pixels[1]) + 1
            if np.all(self.drift_mask_coord[:, 0] == 0) and np.all(self.drift_mask_coord[:, 1] == self.dims[1] - 1) and np.all(
                    self.drift_mask_coord[:, 2] == 0) and np.all(self.drift_mask_coord[:, 3] == self.dims[2] - 1):
                logging.error(f"Drift correction has been wrongly detected. Images do not contain zero-valued pixels")
                self.vars['drift_already_corrected'] = False
        self.start = 1
        if self.vars['origin_state'] == "invisible":
            self.start += self.vars['first_detection_frame']
            analysisi = self.frame_by_frame_segmentation(self.start, self.origin)
            # Use connected components to find which shape is the nearest from the image center.
            if self.vars['several_blob_per_arena']:
                self.origin = analysisi.binary_image
            else:
                if self.vars['appearance_detection_method'] == 'largest':
                    self.origin = keep_one_connected_component(analysisi.binary_image)
                elif self.vars['appearance_detection_method'] == 'most_central':
                    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(analysisi.binary_image,
                                                                                               connectivity=8)
                    center = np.array((self.dims[2] // 2, self.dims[1] // 2))
                    stats = np.zeros(nb_components - 1)
                    for shape_i in np.arange(1, nb_components):
                        stats[shape_i - 1] = eudist(center, centroids[shape_i, :])
                    # The shape having the minimal euclidean distance from the center will be the original shape
                    self.origin = np.zeros((self.dims[1], self.dims[2]), dtype=np.uint8)
                    self.origin[output == (np.argmin(stats) + 1)] = 1
        self.origin_idx = np.nonzero(self.origin)
        if self.vars['origin_state'] == "constant":
            if self.vars['lighter_background']:
                # Initialize the covering_intensity matrix as a reference for pixel fading
                self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = 200
        self.substantial_growth = np.min((1.2 * self.origin.sum(), self.origin.sum() + 250))

    def get_covering_duration(self, step: int):
        """
        Determine the number of frames necessary for a pixel to get covered.

        This function identifies the time when significant growth or motion occurs
        in a video and calculates the number of frames needed for a pixel to be
        completely covered. It also handles noise and ensures that the calculated
        step value is reasonable.

        Parameters
        ----------
        step : int
            The initial step size for frame analysis.

        Raises
        ------
        Exception
            If an error occurs during the calculation process.

        Notes
        -----
        This function may modify several instance attributes including
        `substantial_time`, `step`, and `start`.
        """
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Find a frame with a significant growth/motion and determine the number of frames necessary for a pixel to get covered")
        ## Find the time at which growth reached a substantial growth.
        self.substantial_time = self.start
        # To avoid noisy images to have deleterious effects, make sure that area area reaches the threshold thrice.
        occurrence = 0
        allowed_window = None
        if self.vars['drift_already_corrected']:
            allowed_window = self.drift_mask_coord[:, 0].max(), self.drift_mask_coord[:, 1].min(), self.drift_mask_coord[:, 2].max(), self.drift_mask_coord[:, 3].min()
        prev_bin_im = self.origin
        while np.logical_and(occurrence < 3, self.substantial_time < (self.dims[0] - step - 1)):
            self.substantial_time += step
            growth_vision = self.frame_by_frame_segmentation(self.substantial_time, prev_bin_im)
            prev_bin_im = growth_vision.binary_image * self.borders
            surfarea = np.sum(prev_bin_im)
            prev_bin_im = np.logical_or(prev_bin_im, self.origin).astype(np.uint8)
            if surfarea > self.substantial_growth:
                occurrence += 1
        # get a rough idea of the area covered during this time
        if (self.substantial_time - self.start) > 20:
            if self.vars['lighter_background']:
                growth = (np.sum(self.converted_video[self.start:(self.start + 10), :, :], 0) / 10) - (np.sum(self.converted_video[(self.substantial_time - 10):self.substantial_time, :, :], 0) / 10)
            else:
                growth = (np.sum(self.converted_video[(self.substantial_time - 10):self.substantial_time, :, :], 0) / 10) - (
                            np.sum(self.converted_video[self.start:(self.start + 10), :, :], 0) / 10)
        else:
            if self.vars['lighter_background']:
                growth = self.converted_video[self.start, ...] - self.converted_video[self.substantial_time, ...]
            else:
                growth = self.converted_video[self.substantial_time, ...] - self.converted_video[self.start, ...]
        intensity_extent = np.ptp(self.converted_video[self.start:self.substantial_time, :, :], axis=0)
        growth[np.logical_or(growth < 0, intensity_extent < np.median(intensity_extent))] = 0
        growth = bracket_to_uint8_image_contrast(growth)
        growth *= self.borders
        growth_vision = OneImageAnalysis(growth)
        growth_vision.segmentation(allowed_window=allowed_window)
        if self.vars['several_blob_per_arena']:
            _, _, stats, _ = cv2.connectedComponentsWithStats(self.origin)
            do_erode = np.any(stats[1:, 4] > 50)
        else:
            do_erode = self.origin.sum() > 50
        if do_erode:
            self.substantial_image = cv2.erode(growth_vision.binary_image, cross_33, iterations=2)
        else:
            self.substantial_image = growth_vision.binary_image

        if np.any(self.substantial_image):
            natural_noise = np.nonzero(intensity_extent == np.min(intensity_extent))
            natural_noise = self.converted_video[self.start:self.substantial_time, natural_noise[0][0], natural_noise[1][0]]
            natural_noise = moving_average(natural_noise, 5)
            natural_noise = np.ptp(natural_noise)
            subst_idx = np.nonzero(self.substantial_image)
            cover_lengths = np.zeros(len(subst_idx[0]), dtype=np.uint32)
            for index in np.arange(len(subst_idx[0])):
                vector = self.converted_video[self.start:self.substantial_time, subst_idx[0][index], subst_idx[1][index]]
                left, right = find_major_incline(vector, natural_noise)
                # If find_major_incline did find a major incline: (otherwise it put 0 to left and 1 to right)
                if not np.logical_and(left == 0, right == 1):
                    cover_lengths[index] = len(vector[left:-right])
            # If this analysis fails put a deterministic step
            if len(cover_lengths[cover_lengths > 0]) > 0:
                self.step = (np.round(np.mean(cover_lengths[cover_lengths > 0])).astype(int) // 2) + 1
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Pre-processing detection: the time for a pixel to get covered is set to {self.step}")
            else:
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Pre-processing detection: could not automatically find the time for a pixel to get covered. Default value is 1 for video length < 40 and 10 otherwise")

            # Make sure to avoid a step overestimation
            if self.step > self.dims[0] // 20:
                self.step: int = self.dims[0] // 20

            if self.step == 0:
                self.step: int = 1
        # When the first_move_threshold is not stringent enough the program may detect a movement due to noise
        # In that case, the substantial_image is empty and there is no reason to proceed further
        else:
            self.start = None

    def detection(self, compute_all_possibilities: bool=False):
        """

            Perform frame-by-frame or luminosity-based segmentation on video data to detect cell motion and growth.

            This function processes video frames using either frame-by-frame segmentation or luminosity-based
            segmentation algorithms to detect cell motion and growth. It handles drift correction, adjusts parameters
            based on configuration settings, and applies logical operations to combine results from different segmentation
            methods.

            Parameters
            ----------
            compute_all_possibilities : bool, optional
                Flag to determine if all segmentation possibilities should be computed, by default False

            Returns
            -------
            None

            Notes
            -----
            This function modifies the instance variables `self.segmented`, `self.converted_video`,
            and potentially `self.luminosity_segmentation` and `self.gradient_segmentation`.
            Depending on the configuration settings, it performs various segmentation algorithms and updates
            the instance variables accordingly.

        """
        if self.start is None:
            self.start = 1
        else:
            self.start = np.max((self.start, 1))
        self.lost_frames = np.min((self.step, self.dims[0] // 4))
        # I/ Image by image segmentation algorithms
        # If images contain a drift correction (zeros at borders of the image,
        # Replace these 0 by normal background values before segmenting
        if self.vars['frame_by_frame_segmentation'] or compute_all_possibilities:
            logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detect cell motion and growth using the frame by frame segmentation algorithm")
            self.segmented = np.zeros(self.dims, dtype=np.uint8)
            for t in np.arange(self.dims[0]):#20):#
                analysisi = self.frame_by_frame_segmentation(t, self.segmented[t - 1, ...])
                self.segmented[t, ...] = analysisi.binary_image

                if self.vars['lose_accuracy_to_save_memory']:
                    self.converted_video[t, ...] = bracket_to_uint8_image_contrast(analysisi.image)
                else:
                    self.converted_video[t, ...] = analysisi.image
                if self.vars['convert_for_motion']['logical'] != 'None':
                    if self.vars['lose_accuracy_to_save_memory']:
                        self.converted_video2[t, ...] = bracket_to_uint8_image_contrast(analysisi.image2)
                    else:
                        self.converted_video2[t, ...] = analysisi.image2

        if self.vars['color_number'] == 2:
            luminosity_segmentation, l_threshold_over_time = self.lum_value_segmentation(self.converted_video, do_threshold_segmentation=self.vars['do_threshold_segmentation'] or compute_all_possibilities)
            self.converted_video = self.smooth_pixel_slopes(self.converted_video)
            gradient_segmentation = None
            if self.vars['do_slope_segmentation'] or compute_all_possibilities:
                gradient_segmentation = self.lum_slope_segmentation(self.converted_video)
                if gradient_segmentation is not None:
                    gradient_segmentation[-self.lost_frames:, ...] = np.repeat(gradient_segmentation[-self.lost_frames, :, :][np.newaxis, :, :], self.lost_frames, axis=0)
            if self.vars['convert_for_motion']['logical'] != 'None':
                if self.vars['do_threshold_segmentation'] or compute_all_possibilities:
                    luminosity_segmentation2, l_threshold_over_time2 = self.lum_value_segmentation(self.converted_video2, do_threshold_segmentation=True)
                    if luminosity_segmentation is None:
                        luminosity_segmentation = luminosity_segmentation2
                    if luminosity_segmentation is not None:
                        if self.vars['convert_for_motion']['logical'] == 'Or':
                            luminosity_segmentation = np.logical_or(luminosity_segmentation, luminosity_segmentation2)
                        elif self.vars['convert_for_motion']['logical'] == 'And':
                            luminosity_segmentation = np.logical_and(luminosity_segmentation, luminosity_segmentation2)
                        elif self.vars['convert_for_motion']['logical'] == 'Xor':
                            luminosity_segmentation = np.logical_xor(luminosity_segmentation, luminosity_segmentation2)
                self.converted_video2 = self.smooth_pixel_slopes(self.converted_video2)
                if self.vars['do_slope_segmentation'] or compute_all_possibilities:
                    gradient_segmentation2 = self.lum_slope_segmentation(self.converted_video2)
                    if gradient_segmentation2 is not None:
                        gradient_segmentation2[-self.lost_frames:, ...] = np.repeat(gradient_segmentation2[-self.lost_frames, :, :][np.newaxis, :, :], self.lost_frames, axis=0)
                    if gradient_segmentation is None:
                        gradient_segmentation = gradient_segmentation2
                    if gradient_segmentation is not None:
                        if self.vars['convert_for_motion']['logical'] == 'Or':
                            gradient_segmentation = np.logical_or(gradient_segmentation, gradient_segmentation2)
                        elif self.vars['convert_for_motion']['logical'] == 'And':
                            gradient_segmentation = np.logical_and(gradient_segmentation, gradient_segmentation2)
                        elif self.vars['convert_for_motion']['logical'] == 'Xor':
                            gradient_segmentation = np.logical_xor(gradient_segmentation, gradient_segmentation2)

            if compute_all_possibilities:
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Compute all options to detect cell motion and growth. Maximal growth per frame: {self.vars['maximal_growth_factor']}")
                if luminosity_segmentation is not None:
                    self.luminosity_segmentation = np.nonzero(luminosity_segmentation)
                if gradient_segmentation is not None:
                    self.gradient_segmentation = np.nonzero(gradient_segmentation)
                if luminosity_segmentation is not None and gradient_segmentation is not None:
                    self.logical_and = np.nonzero(np.logical_and(luminosity_segmentation, gradient_segmentation))
                    self.logical_or = np.nonzero(np.logical_or(luminosity_segmentation, gradient_segmentation))
            elif not self.vars['frame_by_frame_segmentation']:
                if self.vars['do_threshold_segmentation'] and not self.vars['do_slope_segmentation']:
                    logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detect with luminosity threshold segmentation algorithm")
                    self.segmented = luminosity_segmentation
                if self.vars['do_slope_segmentation']:
                    logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detect with luminosity slope segmentation algorithm")
                    self.segmented = gradient_segmentation
                if np.logical_and(self.vars['do_threshold_segmentation'], self.vars['do_slope_segmentation']):
                    if self.vars['true_if_use_light_AND_slope_else_OR']:
                        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detection resuts from threshold AND slope segmentation algorithms")
                        if luminosity_segmentation is not None and gradient_segmentation is not None:
                            self.segmented = np.logical_and(luminosity_segmentation, gradient_segmentation)
                    else:
                        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Detection resuts from threshold OR slope segmentation algorithms")
                        if luminosity_segmentation is not None and gradient_segmentation is not None:
                            self.segmented = np.logical_or(luminosity_segmentation, gradient_segmentation)
                self.segmented = self.segmented.astype(np.uint8)


    def frame_by_frame_segmentation(self, t: int, previous_binary_image: NDArray=None):
        """

        Frame-by-frame segmentation of a video.

        Parameters
        ----------
        t : int
            The time index of the frame to process.
        previous_binary_image : NDArray, optional
            The binary image from the previous frame. Default is `None`.

        Returns
        -------
        OneImageAnalysis
            An object containing the analysis of the current frame.
        """
        contrasted_im = bracket_to_uint8_image_contrast(self.converted_video[t, :, :])
        # 1. Get the mask valid for a number of images around it (step).
        allowed_window = None
        if self.vars['drift_already_corrected']:
            half_step = np.ceil(self.step / 2).astype(int)
            t_start = t - half_step
            t_end = t + half_step
            t_start = np.max((t_start, 0))
            t_end = np.min((t_end, self.dims[0]))
            min_y, max_y = np.max(self.drift_mask_coord[t_start:t_end, 0]), np.min(self.drift_mask_coord[t_start:t_end, 1])
            min_x, max_x = np.max(self.drift_mask_coord[t_start:t_end, 2]), np.min(self.drift_mask_coord[t_start:t_end, 3])
            allowed_window = min_y, max_y, min_x, max_x

        analysisi = OneImageAnalysis(contrasted_im)
        if self.vars['convert_for_motion']['logical'] != 'None':
            contrasted_im2 = bracket_to_uint8_image_contrast(self.converted_video2[t, :, :])
            analysisi.image2 = contrasted_im2

        if previous_binary_image is None or t == 0:
            analysisi.previous_binary_image = self.origin
        else:
            analysisi.previous_binary_image = previous_binary_image

        analysisi.segmentation(self.vars['convert_for_motion']['logical'], self.vars['color_number'],
                               bio_label=self.vars["bio_label"], bio_label2=self.vars["bio_label2"],
                               rolling_window_segmentation=self.vars['rolling_window_segmentation'],
                               lighter_background=self.vars['lighter_background'],
                               allowed_window=allowed_window, filter_spec=self.vars['filter_spec']) # filtering already done when creating converted_video

        return analysisi

    def lum_value_segmentation(self, converted_video: NDArray, do_threshold_segmentation: bool) -> Tuple[NDArray, NDArray]:
        """
        Perform segmentation based on luminosity values from a video.

        Parameters
        ----------
        converted_video : NDArray
            The input video data in a NumPy array format.
        do_threshold_segmentation : bool
            Flag to determine whether threshold segmentation should be applied.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing two NumPy arrays:
            - The first array is the luminosity segmentation of the video.
            - The second array represents the luminosity threshold over time.

        Notes
        -----
        This function operates under the assumption that there is sufficient motion in the video data.
        If no valid thresholds are found for segmentation, the function returns None for
        `luminosity_segmentation`.
        """
        shape_motion_failed: bool = False
        if self.vars['lighter_background']:
            covering_l_values = np.min(converted_video[:self.substantial_time, :, :],
                                             0) * self.substantial_image
        else:
            covering_l_values = np.max(converted_video[:self.substantial_time, :, :],
                                             0) * self.substantial_image
        # Avoid errors by checking whether the covering values are nonzero
        covering_l_values = covering_l_values[covering_l_values != 0]
        if len(covering_l_values) == 0:
            shape_motion_failed = True

        luminosity_segmentation = None
        l_threshold_over_time = None
        if not shape_motion_failed:
            value_segmentation_thresholds = np.arange(0.8, -0.7, -0.1)
            validated_thresholds = np.zeros(value_segmentation_thresholds.shape, dtype=bool)
            counter = 0
            while_condition = True
            max_motion_per_frame = (self.dims[1] * self.dims[2]) * self.vars['maximal_growth_factor'] * 2
            if self.vars['lighter_background']:
                basic_bckgrnd_values = np.quantile(converted_video[:(self.lost_frames + 1), ...], 0.9, axis=(1, 2))
            else:
                basic_bckgrnd_values = np.quantile(converted_video[:(self.lost_frames + 1), ...], 0.1, axis=(1, 2))
            # Try different values of do_threshold_segmentation and keep the one that does not
            # segment more than x percent of the image
            while counter <= 14:
                value_threshold = value_segmentation_thresholds[counter]
                if self.vars['lighter_background']:
                    l_threshold = (1 + value_threshold) * np.max(covering_l_values)
                else:
                    l_threshold = (1 - value_threshold) * np.min(covering_l_values)
                starting_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video[:(self.lost_frames + 1), ...],
                                                               basic_bckgrnd_values, l_threshold,
                                                               self.vars['lighter_background'])

                changing_pixel_number = np.sum(np.absolute(np.diff(starting_segmentation.astype(np.int8), 1, 0)), (1, 2))
                validation = np.max(np.sum(starting_segmentation, (1, 2))) < max_motion_per_frame and (
                        np.max(changing_pixel_number) < max_motion_per_frame)
                validated_thresholds[counter] = validation
                if np.any(validated_thresholds):
                    if not validation:
                        break
                counter += 1
            # If any threshold is accepted, use their average to proceed the final thresholding
            valid_number = validated_thresholds.sum()
            if valid_number > 0:
                if valid_number > 2:
                    index_to_keep = 2
                else:
                    index_to_keep = valid_number - 1
                value_threshold = value_segmentation_thresholds[
                    np.uint8(np.floor(np.mean(np.nonzero(validated_thresholds)[0][index_to_keep])))]
            else:
                value_threshold = 0

            if self.vars['lighter_background']:
                l_threshold = (1 + value_threshold) * np.max(covering_l_values)
            else:
                l_threshold = (1 - value_threshold) * np.min(covering_l_values)
            if do_threshold_segmentation:
                if self.vars['lighter_background']:
                    basic_bckgrnd_values = np.quantile(converted_video, 0.9, axis=(1, 2))
                else:
                    basic_bckgrnd_values = np.quantile(converted_video, 0.1, axis=(1, 2))
                luminosity_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values,
                                                                 l_threshold, self.vars['lighter_background'])
            else:
                luminosity_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video[:(self.lost_frames + 1), ...],
                                                               basic_bckgrnd_values, l_threshold,
                                                               self.vars['lighter_background'])
        return luminosity_segmentation, l_threshold_over_time

    def smooth_pixel_slopes(self, converted_video: NDArray) -> NDArray:
        """
        Apply smoothing to pixel slopes in a video by convolving with a moving average kernel.

        Parameters
        ----------
        converted_video : NDArray
            The input video array to be smoothed.

        Returns
        -------
        NDArray
            Smoothed video array with pixel slopes averaged using a moving average kernel.

        Raises
        ------
        MemoryError
            If there is not enough RAM available to perform the smoothing operation.

        Notes
        -----
        This function applies a moving average kernel to each pixel across the frames of
        the input video. The smoothing operation can be repeated based on user-defined settings.
        The precision of the output array is controlled by a flag that determines whether to
        save memory at the cost of accuracy.

        Examples
        --------
        >>> smoothed = smooth_pixel_slopes(converted_video)
        >>> print(smoothed.shape)  # Expected output will vary depending on the input video shape
        """
        try:
            if self.vars['lose_accuracy_to_save_memory']:
                smoothed_video = np.zeros(self.dims, dtype=np.float32)
                smooth_kernel = np.ones(self.step, dtype=np.float64) / self.step
                for i in np.arange(converted_video.shape[1]):
                    for j in np.arange(converted_video.shape[2]):
                        padded = np.pad(converted_video[:, i, j] / self.mean_intensity_per_frame,
                                     (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                        moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        if self.vars['repeat_video_smoothing'] > 1:
                            for it in np.arange(1, self.vars['repeat_video_smoothing']):
                                padded = np.pad(moving_average,
                                             (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                                moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        smoothed_video[:, i, j] = moving_average.astype(np.float32)
            else:
                smoothed_video = np.zeros(self.dims, dtype=np.float64)
                smooth_kernel = np.ones(self.step) / self.step
                for i in np.arange(converted_video.shape[1]):
                    for j in np.arange(converted_video.shape[2]):
                        padded = np.pad(converted_video[:, i, j] / self.mean_intensity_per_frame,
                                     (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                        moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        if self.vars['repeat_video_smoothing'] > 1:
                            for it in np.arange(1, self.vars['repeat_video_smoothing']):
                                padded = np.pad(moving_average,
                                             (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                                moving_average = np.convolve(padded, smooth_kernel, mode='valid')
                        smoothed_video[:, i, j] = moving_average
            return smoothed_video

        except MemoryError:
            logging.error("Not enough RAM available to smooth pixel curves. Detection may fail.")
            smoothed_video = converted_video
            return smoothed_video

    def lum_slope_segmentation(self, converted_video: NDArray) -> NDArray:
        """
        Perform lum slope segmentation on the given video.

        Parameters
        ----------
        converted_video : NDArray
            The input video array for segmentation processing.

        Returns
        -------
        NDArray
            Segmented gradient array of the video. If segmentation fails,
            returns `None` for the corresponding frames.

        Notes
        -----
        This function may consume significant memory and adjusts
        data types (float32 or float64) based on available RAM.

        Examples
        --------
        >>> result = lum_slope_segmentation(converted_video)
        """
        shape_motion_failed : bool = False
        # 2) Contrast increase
        oridx = np.nonzero(self.origin)
        notoridx = np.nonzero(1 - self.origin)
        do_increase_contrast = np.mean(converted_video[0, oridx[0], oridx[1]]) * 10 > np.mean(
                converted_video[0, notoridx[0], notoridx[1]])
        necessary_memory = self.dims[0] * self.dims[1] * self.dims[2] * 64 * 2 * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        if self.vars['lose_accuracy_to_save_memory']:
            derive = converted_video.astype(np.float32)
        else:
            derive = converted_video.astype(np.float64)
        if necessary_memory > available_memory:
            converted_video = None

        if do_increase_contrast:
            derive = np.square(derive)

        # 3) Get the gradient
        necessary_memory = derive.size * 64 * 4 * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        if necessary_memory > available_memory:
            for cy in np.arange(self.dims[1]):
                for cx in np.arange(self.dims[2]):
                    if self.vars['lose_accuracy_to_save_memory']:
                        derive[:, cy, cx] = np.gradient(derive[:, cy, cx], self.step).astype(np.float32)
                    else:
                        derive[:, cy, cx] = np.gradient(derive[:, cy, cx], self.step)
        else:
            if self.vars['lose_accuracy_to_save_memory']:
                derive = np.gradient(derive, self.step, axis=0).astype(np.float32)
            else:
                derive = np.gradient(derive, self.step, axis=0)

        # 4) Segment
        if self.vars['lighter_background']:
            covering_slopes = np.min(derive[:self.substantial_time, :, :], 0) * self.substantial_image
        else:
            covering_slopes = np.max(derive[:self.substantial_time, :, :], 0) * self.substantial_image
        covering_slopes = covering_slopes[covering_slopes != 0]
        if len(covering_slopes) == 0:
            shape_motion_failed = True

        gradient_segmentation = None
        if not shape_motion_failed:
            gradient_segmentation = np.zeros(self.dims, np.uint8)
            ####
            # ease_slope_segmentation = 0.8
            value_segmentation_thresholds = np.arange(0.8, -0.7, -0.1)
            validated_thresholds = np.zeros(value_segmentation_thresholds.shape, dtype=bool)
            counter = 0
            while_condition = True
            max_motion_per_frame = (self.dims[1] * self.dims[2]) * self.vars['maximal_growth_factor']
            # Try different values of do_slope_segmentation and keep the one that does not
            # segment more than x percent of the image
            while counter < value_segmentation_thresholds.shape[0]:
                ease_slope_segmentation = value_segmentation_thresholds[counter]
                if self.vars['lighter_background']:
                    gradient_threshold = (1 + ease_slope_segmentation) * np.max(covering_slopes)
                    sample = np.less(derive[:self.substantial_time], gradient_threshold)
                else:
                    gradient_threshold = (1 - ease_slope_segmentation) * np.min(covering_slopes)
                    sample = np.greater(derive[:self.substantial_time], gradient_threshold)
                changing_pixel_number = np.sum(np.absolute(np.diff(sample.astype(np.int8), 1, 0)), (1, 2))
                validation = np.max(np.sum(sample, (1, 2))) < max_motion_per_frame and (
                        np.max(changing_pixel_number) < max_motion_per_frame)
                validated_thresholds[counter] = validation
                if np.any(validated_thresholds):
                    if not validation:
                        break
                counter += 1
                # If any threshold is accepted, use their average to proceed the final thresholding
            valid_number = validated_thresholds.sum()
            if valid_number > 0:
                if valid_number > 2:
                    index_to_keep = 2
                else:
                    index_to_keep = valid_number - 1
                ease_slope_segmentation = value_segmentation_thresholds[
                    np.uint8(np.floor(np.mean(np.nonzero(validated_thresholds)[0][index_to_keep])))]

                if self.vars['lighter_background']:
                    gradient_threshold = (1 - ease_slope_segmentation) * np.max(covering_slopes)
                    gradient_segmentation[:-self.lost_frames, :, :] = np.less(derive, gradient_threshold)[
                        self.lost_frames:, :, :]
                else:
                    gradient_threshold = (1 - ease_slope_segmentation) * np.min(covering_slopes)
                    gradient_segmentation[:-self.lost_frames, :, :] = np.greater(derive, gradient_threshold)[
                        self.lost_frames:, :, :]
            else:
                if self.vars['lighter_background']:
                    gradient_segmentation[:-self.lost_frames, :, :] = (derive < (np.min(derive, (1, 2)) * 1.1)[:, None, None])[self.lost_frames:, :, :]
                else:
                    gradient_segmentation[:-self.lost_frames, :, :] = (derive > (np.max(derive, (1, 2)) * 0.1)[:, None, None])[self.lost_frames:, :, :]
        return gradient_segmentation

    def update_ring_width(self):
        """

        Update the `pixel_ring_depth` and create an erodila disk.

        This method ensures that the pixel ring depth is odd and at least 3,
        then creates an erodila disk of that size.
        """
        # Make sure that self.pixels_depths are odd and greater than 3
        if self.pixel_ring_depth <= 3:
            self.pixel_ring_depth = 3
        if self.pixel_ring_depth % 2 == 0:
            self.pixel_ring_depth = self.pixel_ring_depth + 1
        self.erodila_disk = create_ellipse(self.pixel_ring_depth, self.pixel_ring_depth, min_size=3).astype(np.uint8)
        self.max_distance = self.pixel_ring_depth * self.vars['detection_range_factor']

    def initialize_post_processing(self):
        """

        Initialize post-processing for video analysis.

        This function initializes various parameters and prepares the binary
        representation used in post-processing of video data. It logs information about
        the settings, handles initial origin states, sets up segmentation data,
        calculates surface areas, and optionally corrects errors around the initial
        shape or prevents fast growth near the periphery.

        Notes
        -----
        This function performs several initialization steps and logs relevant information,
        including handling different origin states, updating segmentation data, and
        calculating the gravity field based on binary representation.

        """
        ## Initialization
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting Post_processing. Fading detection: {self.vars['do_fading']}: {self.vars['fading']}, Subtract background: {self.vars['subtract_background']}, Correct errors around initial shape: {self.vars['correct_errors_around_initial']}, Connect distant shapes: {self.vars['detection_range_factor'] > 0}, How to select appearing cell(s): {self.vars['appearance_detection_method']}")
        self.binary = np.zeros(self.dims[:3], dtype=np.uint8)
        if self.origin.shape[0] != self.binary[self.start - 1, :, :].shape[0] or self.origin.shape[1] != self.binary[self.start - 1, :, :].shape[1]:
            logging.error("Unaltered videos deprecated, they have been created with different settings.\nDelete .npy videos and Data to run Cellects quickly.pkl and re-run")

        if self.vars['origin_state'] == "invisible":
            self.binary[self.start - 1, :, :] = deepcopy(self.origin)
            self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = self.converted_video[self.start, self.origin_idx[0], self.origin_idx[1]]
        else:
            if self.vars['origin_state'] == "fluctuating":
                self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = np.median(self.converted_video[:self.start, self.origin_idx[0], self.origin_idx[1]], axis=0)

            self.binary[:self.start, :, :] = np.repeat(np.expand_dims(self.origin, 0), self.start, axis=0)
            if self.start < self.step:
                frames_to_assess = self.step
                self.segmented[self.start - 1, ...] = self.binary[self.start - 1, :, :]
                for t in np.arange(self.start, self.lost_frames):
                    # Only keep pixels that are always detected
                    always_found = np.sum(self.segmented[t:(t + frames_to_assess), ...], 0)
                    always_found = always_found == frames_to_assess
                    # Remove too small shapes
                    without_small, stats, centro = cc(always_found.astype(np.uint8))
                    large_enough = np.nonzero(stats[1:, 4] > ((self.vars['first_move_threshold'] + 1) // 2))[0]
                    if len(large_enough) > 0:
                        always_found *= np.isin(always_found, large_enough + 1)
                        always_found = np.logical_or(always_found, self.segmented[t - 1, ...])
                        self.segmented[t, ...] *= always_found
                    else:
                        self.segmented[t, ...] = 0
                    self.segmented[t, ...] = np.logical_or(self.segmented[t - 1, ...], self.segmented[t, ...])
        self.mean_distance_per_frame = None
        self.surfarea = np.zeros(self.dims[0], dtype =np.uint64)
        self.surfarea[:self.start] = np.sum(self.binary[:self.start, :, :], (1, 2))
        self.gravity_field = inverted_distance_transform(self.binary[(self.start - 1), :, :],
                                           np.sqrt(np.sum(self.binary[(self.start - 1), :, :])))
        if self.vars['correct_errors_around_initial']:
            self.rays, self.sun = draw_me_a_sun(self.binary[(self.start - 1), :, :], ray_length_coef=1)  # plt.imshow(sun)
            self.holes = np.zeros(self.dims[1:], dtype=np.uint8)
            self.pixel_ring_depth += 2
            self.update_ring_width()

        if self.vars['prevent_fast_growth_near_periphery']:
            self.near_periphery = np.zeros(self.dims[1:])
            if self.vars['arena_shape'] == 'circle':
                periphery_width = self.vars['periphery_width'] * 2
                elliperiphery = create_ellipse(self.dims[1] - periphery_width, self.dims[2] - periphery_width, min_size=3)
                half_width = periphery_width // 2
                if periphery_width % 2 == 0:
                    self.near_periphery[half_width:-half_width, half_width:-half_width] = elliperiphery
                else:
                    self.near_periphery[half_width:-half_width - 1, half_width:-half_width - 1] = elliperiphery
                self.near_periphery = 1 - self.near_periphery
            else:
                self.near_periphery[:self.vars['periphery_width'], :] = 1
                self.near_periphery[-self.vars['periphery_width']:, :] = 1
                self.near_periphery[:, :self.vars['periphery_width']] = 1
                self.near_periphery[:, -self.vars['periphery_width']:] = 1
            self.near_periphery = np.nonzero(self.near_periphery)

    def update_shape(self, show_seg: bool):
        """
        Update the shape of detected objects in the current frame by analyzing
        segmentation potentials and applying morphological operations.

        Parameters
        ----------
        show_seg : bool
            Flag indicating whether to display segmentation results.

        Notes
        -----
        This function performs several operations to update the shape of detected objects:
        - Analyzes segmentation potentials from previous frames.
        - Applies morphological operations to refine the shape.
        - Updates internal state variables such as `binary` and `covering_intensity`.

        """
        # Get from gradients, a 2D matrix of potentially covered pixels
        # I/ dilate the shape made with covered pixels to assess for covering

        # I/ 1) Only keep pixels that have been detected at least two times in the three previous frames
        if self.dims[0] < 100:
            new_potentials = self.segmented[self.t, :, :]
        else:
            if self.t > 1:
                new_potentials = np.sum(self.segmented[(self.t - 2): (self.t + 1), :, :], 0, dtype=np.uint8)
            else:
                new_potentials = np.sum(self.segmented[: (self.t + 1), :, :], 0, dtype=np.uint8)
            new_potentials[new_potentials == 1] = 0
            new_potentials[new_potentials > 1] = 1

        # I/ 2) If an image displays more new potential pixels than 50% of image pixels,
        # one of these images is considered noisy and we try taking only one.
        frame_counter = -1
        maximal_size = 0.5 * new_potentials.size
        if (self.vars["do_threshold_segmentation"] or self.vars["frame_by_frame_segmentation"]) and self.t > np.max((self.start + self.step, 6)):
           maximal_size = np.min((np.max(self.binary[:self.t].sum((1, 2))) * (1 + self.vars['maximal_growth_factor']), self.borders.sum()))
        while np.logical_and(np.sum(new_potentials) > maximal_size,
                             frame_counter <= 5):  # np.logical_and(np.sum(new_potentials > 0) > 5 * np.sum(dila_ring), frame_counter <= 5):
            frame_counter += 1
            if frame_counter > self.t:
                break
            else:
                if frame_counter < 5:
                    new_potentials = self.segmented[self.t - frame_counter, :, :]
                else:
                # If taking only one image is not enough, use the inverse of the fadinged matrix as new_potentials
                # Given it haven't been processed by any slope calculation, it should be less noisy
                    new_potentials = np.sum(self.segmented[(self.t - 5): (self.t + 1), :, :], 0, dtype=np.uint8)
                    new_potentials[new_potentials < 6] = 0
                    new_potentials[new_potentials == 6] = 1


        new_shape = deepcopy(self.binary[self.t - 1, :, :])
        new_potentials = cv2.morphologyEx(new_potentials, cv2.MORPH_CLOSE, cross_33)
        new_potentials = cv2.morphologyEx(new_potentials, cv2.MORPH_OPEN, cross_33) * self.borders
        new_shape = np.logical_or(new_shape, new_potentials).astype(np.uint8)
        # Add distant shapes within a radius, score every added pixels according to their distance
        if not self.vars['several_blob_per_arena']:
            if new_shape.sum() == 0:
                new_shape = deepcopy(new_potentials)
            else:
                pads = ProgressivelyAddDistantShapes(new_potentials, new_shape, self.max_distance)
                r = weakref.ref(pads)
                # If max_distance is non nul look for distant shapes
                pads.consider_shapes_sizes(self.vars['min_size_for_connection'],
                                                     self.vars['max_size_for_connection'])
                pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=True)

                new_shape = deepcopy(pads.expanded_shape)
                new_shape[new_shape > 1] = 1
                if np.logical_and(self.t > self.step, self.t < self.dims[0]):
                    if np.any(pads.expanded_shape > 5):
                        # Add distant shapes back in time at the covering speed of neighbors
                        self.binary[self.t][np.nonzero(new_shape)] = 1
                        self.binary[(self.step):(self.t + 1), :, :] = \
                            pads.modify_past_analysis(self.binary[(self.step):(self.t + 1), :, :],
                                                      self.segmented[(self.step):(self.t + 1), :, :])
                        new_shape = deepcopy(self.binary[self.t, :, :])
                pads = None

            # Fill holes
            new_shape = cv2.morphologyEx(new_shape, cv2.MORPH_CLOSE, cross_33)

        if self.vars['do_fading'] and (self.t > self.step + self.lost_frames):
            # Shape Erosion
            # I/ After a substantial growth, erode the shape made with covered pixels to assess for fading
            # Use the newly covered pixels to calculate their mean covering intensity
            new_idx = np.nonzero(np.logical_xor(new_shape, self.binary[self.t - 1, :, :]))
            start_intensity_monitoring = self.t - self.lost_frames - self.step
            end_intensity_monitoring = self.t - self.lost_frames
            self.covering_intensity[new_idx[0], new_idx[1]] = np.median(self.converted_video[start_intensity_monitoring:end_intensity_monitoring, new_idx[0], new_idx[1]], axis=0)
            previous_binary = self.binary[self.t - 1, :, :]
            greyscale_image = self.converted_video[self.t - self.lost_frames, :, :]
            protect_from_fading = None
            if self.vars['origin_state'] == 'constant':
                protect_from_fading = self.origin
            new_shape, self.covering_intensity = cell_leaving_detection(new_shape, self.covering_intensity, previous_binary, greyscale_image, self.vars['fading'], self.vars['lighter_background'], self.vars['several_blob_per_arena'], self.erodila_disk, protect_from_fading)

        self.covering_intensity *= new_shape
        self.binary[self.t, :, :] = new_shape * self.borders
        self.surfarea[self.t] = np.sum(self.binary[self.t, :, :])

        # Calculate the mean distance covered per frame and correct for a ring of not really fading pixels
        if self.mean_distance_per_frame is None:
            if self.vars['correct_errors_around_initial'] and not self.vars['several_blob_per_arena']:
                if np.logical_and((self.t % 20) == 0,
                                  np.logical_and(self.surfarea[self.t] > self.substantial_growth,
                                                 self.surfarea[self.t] < self.substantial_growth * 2)):
                    shape = self.binary[self.t, :, :] * self.sun
                    back = (1 - self.binary[self.t, :, :]) * self.sun
                    for ray in self.rays:
                        # For each sun's ray, see how they cross the shape/back and
                        # store the gravity_field value of these pixels (distance to the original shape).
                        ray_through_shape = (shape == ray) * self.gravity_field
                        ray_through_back = (back == ray) * self.gravity_field
                        if np.any(ray_through_shape):
                            if np.any(ray_through_back):
                                # If at least one back pixel is nearer to the original shape than a shape pixel,
                                # there is a hole to fill.
                                if np.any(ray_through_back > np.min(ray_through_shape[ray_through_shape > 0])):
                                    # Check if the nearest pixels are shape, if so, supress them until the nearest pixel
                                    # becomes back
                                    while np.max(ray_through_back) <= np.max(ray_through_shape):
                                        ray_through_shape[ray_through_shape == np.max(ray_through_shape)] = 0
                                    # Now, all back pixels that are nearer than the closest shape pixel should get filled
                                    # To do so, replace back pixels further than the nearest shape pixel by 0
                                    ray_through_back[ray_through_back < np.max(ray_through_shape)] = 0
                                    self.holes[np.nonzero(ray_through_back)] = 1
                            else:
                                self.rays = np.concatenate((self.rays[:(ray - 2)], self.rays[(ray - 1):]))
                        ray_through_shape = None
                        ray_through_back = None
            if np.any(self.surfarea[:self.t] > self.substantial_growth * 2):

                if self.vars['correct_errors_around_initial'] and not self.vars['several_blob_per_arena']:
                    # Apply the hole correction
                    self.holes = cv2.morphologyEx(self.holes, cv2.MORPH_CLOSE, cross_33, iterations=10)
                    # If some holes are not covered by now
                    if np.any(self.holes * (1 - self.binary[self.t, :, :])):
                        self.binary[:(self.t + 1), :, :], holes_time_end, distance_against_time = \
                            dynamically_expand_to_fill_holes(self.binary[:(self.t + 1), :, :], self.holes)
                        if holes_time_end is not None:
                            self.binary[holes_time_end:(self.t + 1), :, :] += self.binary[holes_time_end, :, :]
                            self.binary[holes_time_end:(self.t + 1), :, :][
                                self.binary[holes_time_end:(self.t + 1), :, :] > 1] = 1
                            self.surfarea[:(self.t + 1)] = np.sum(self.binary[:(self.t + 1), :, :], (1, 2))

                    else:
                        distance_against_time = [1, 2]
                else:
                    distance_against_time = [1, 2]
                distance_against_time = np.diff(distance_against_time)
                if len(distance_against_time) > 0:
                    self.mean_distance_per_frame = np.mean(- distance_against_time)
                else:
                    self.mean_distance_per_frame = 1

        if self.vars['prevent_fast_growth_near_periphery']:
            # growth_near_periphery = np.diff(self.binary[self.t-1:self.t+1, :, :] * self.near_periphery, axis=0)
            growth_near_periphery = np.diff(self.binary[self.t-1:self.t+1, self.near_periphery[0], self.near_periphery[1]], axis=0)
            if (growth_near_periphery == 1).sum() > self.vars['max_periphery_growth']:
                # self.binary[self.t, self.near_periphery[0], self.near_periphery[1]] = self.binary[self.t - 1, self.near_periphery[0], self.near_periphery[1]]
                periphery_to_remove = np.zeros(self.dims[1:], dtype=np.uint8)
                periphery_to_remove[self.near_periphery[0], self.near_periphery[1]] = self.binary[self.t, self.near_periphery[0], self.near_periphery[1]]
                shapes, stats, centers = cc(periphery_to_remove)
                periphery_to_remove = np.nonzero(np.isin(shapes, np.nonzero(stats[:, 4] > self.vars['max_periphery_growth'])[0][1:]))
                self.binary[self.t, periphery_to_remove[0], periphery_to_remove[1]] = self.binary[self.t - 1, periphery_to_remove[0], periphery_to_remove[1]]
                if not self.vars['several_blob_per_arena']:
                    shapes, stats, centers = cc(self.binary[self.t, ...])
                    shapes[shapes != 1] = 0
                    self.binary[self.t, ...] = shapes

        # Display
        if show_seg:
            if self.visu is not None:
                im_to_display = deepcopy(self.visu[self.t, ...])
                contours = np.nonzero(cv2.morphologyEx(self.binary[self.t, :, :], cv2.MORPH_GRADIENT, cross_33))
                if self.vars['lighter_background']:
                    im_to_display[contours[0], contours[1]] = 0
                else:
                    im_to_display[contours[0], contours[1]] = 255
            else:
                im_to_display = self.binary[self.t, :, :] * 255
            imtoshow = cv2.resize(im_to_display, (540, 540))
            cv2.imshow("shape_motion", imtoshow)
            cv2.waitKey(1)
        self.t += 1

    def get_descriptors_from_binary(self, release_memory: bool=True):
        """

        Methods: get_descriptors_from_binary

        Summary
        -------
        Generates shape descriptors for binary images, computes these descriptors for each frame and handles colony
        tracking. This method can optionally release memory to reduce usage, apply scaling factors to descriptors
        in millimeters and computes solidity separately if requested.

        Parameters
        ----------
        release_memory : bool, optional
            Flag to determine whether memory should be released after computation. Default is True.

        Other Parameters
        ----------------
        **self.one_row_per_frame**
            DataFrame to store one row of descriptors per frame.
            - **'arena'**: Arena identifier, repeated for each frame.
            - **'time'**: Array of time values corresponding to frames.

        **self.binary**
            3D array representing binary images over time.
            - **t,x,y**: Time index, x-coordinate, and y-coordinate.

        **self.dims**
            Tuple containing image dimensions.
            - **0**: Number of time frames.
            - **1,2**: Image width and height respectively.

        **self.surfarea**
            Array containing surface areas for each frame.

        **self.time_interval**
            Time interval between frames, calculated only if provided timings are non-zero.

        Notes
        -----
        This method uses various helper methods and classes like `ShapeDescriptors` for computing shape descriptors,
        `PercentAndTimeTracker` for progress tracking, and other image processing techniques such as connected components analysis.

        """
        logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Computing and saving specimen(s) coordinates and required descriptors")
        if release_memory:
            self.substantial_image = None
            self.covering_intensity = None
            self.segmented = None
            self.gravity_field = None
            self.sun = None
            self.rays = None
            self.holes = None
            collect()
        self.surfarea = self.binary.sum((1, 2))
        timings = self.vars['exif']
        if len(timings) < self.dims[0]:
            timings = np.arange(self.dims[0])
        if np.any(timings > 0):
            self.time_interval = np.mean(np.diff(timings))
        else:
            self.time_interval = 1.
        timings = timings[:self.dims[0]]

        # Detect first motion
        self.one_descriptor_per_arena['first_move'] = detect_first_move(self.surfarea, self.vars['first_move_threshold'])

        self.compute_solidity_separately: bool = self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena'] and not self.vars['descriptors']['solidity']
        if self.compute_solidity_separately:
            self.solidity = np.zeros(self.dims[0], dtype=np.float64)
        if not self.vars['several_blob_per_arena']:
            # solidity must be added if detect growth transition is computed
            if self.compute_solidity_separately:
                for t in np.arange(self.dims[0]):
                    solidity = ShapeDescriptors(self.binary[t, :, :], ["solidity"])
                    self.solidity[t] = solidity.descriptors["solidity"]
            self.one_row_per_frame = compute_one_descriptor_per_frame(self.binary,
                                                                      self.one_descriptor_per_arena['arena'], timings,
                                                                      self.vars['descriptors'],
                                                                      self.vars['output_in_mm'],
                                                                      self.vars['average_pixel_size'],
                                                                      self.vars['do_fading'],
                                                                       self.vars['save_coord_specimen'])
        else:
            self.one_row_per_frame = compute_one_descriptor_per_colony(self.binary,
                                                                       self.one_descriptor_per_arena['arena'], timings,
                                                                       self.vars['descriptors'],
                                                                       self.vars['output_in_mm'],
                                                                       self.vars['average_pixel_size'],
                                                                       self.vars['do_fading'],
                                                                       self.vars['first_move_threshold'],
                                                                       self.vars['save_coord_specimen'])
        self.one_descriptor_per_arena["final_area"] = self.binary[-1, :, :].sum()
        if self.vars['output_in_mm']:
            self.one_descriptor_per_arena = scale_descriptors(self.one_descriptor_per_arena, self.vars['average_pixel_size'])

    def detect_growth_transitions(self):
        """
        Detect growth transitions in a biological image processing context.

        Analyzes the growth transitions of a shape within an arena, determining
        whether growth is isotropic and identifying any breaking points.

        Notes:
            This method modifies the `one_descriptor_per_arena` dictionary in place
            to include growth transition information.

        """
        if self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena']:
            self.one_descriptor_per_arena['iso_digi_transi'] = pd.NA
            if not pd.isna(self.one_descriptor_per_arena['first_move']):
                logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting growth transition analysis.")

                # II) Once a pseudopod is deployed, look for a disk/ around the original shape
                growth_begining = self.surfarea < ((self.surfarea[0] * 1.2) + ((self.dims[1] / 4) * (self.dims[2] / 4)))
                dilated_origin = cv2.dilate(self.binary[self.one_descriptor_per_arena['first_move'], :, :], kernel=cross_33, iterations=10, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                isisotropic = np.sum(self.binary[:, :, :] * dilated_origin, (1, 2))
                isisotropic *= growth_begining
                # Ask if the dilated origin area is 90% covered during the growth beginning
                isisotropic = isisotropic > 0.9 * dilated_origin.sum()
                if np.any(isisotropic):
                    self.one_descriptor_per_arena['is_growth_isotropic'] = 1
                    # Determine a solidity reference to look for a potential breaking of the isotropic growth
                    if self.compute_solidity_separately:
                        solidity_reference = np.mean(self.solidity[:self.one_descriptor_per_arena['first_move']])
                        different_solidity = self.solidity < (0.9 * solidity_reference)
                        del self.solidity
                    else:
                        solidity_reference = np.mean(
                            self.one_row_per_frame.iloc[:(self.one_descriptor_per_arena['first_move']), :]["solidity"])
                        different_solidity = self.one_row_per_frame["solidity"].values < (0.9 * solidity_reference)
                    # Make sure that isotropic breaking not occur before isotropic growth
                    if np.any(different_solidity):
                        self.one_descriptor_per_arena["iso_digi_transi"] = np.nonzero(different_solidity)[0][0] * self.time_interval
                else:
                    self.one_descriptor_per_arena['is_growth_isotropic'] = 0
            else:
                self.one_descriptor_per_arena['is_growth_isotropic'] = pd.NA
                

    def check_converted_video_type(self):
        """
        Check if the converted video type is uint8 and normalize it if necessary.
        """
        if self.converted_video.dtype != "uint8":
            self.converted_video = bracket_to_uint8_image_contrast(self.converted_video)

    def networks_analysis(self, show_seg: bool=False):
        """
        Perform network detection within a given arena.

        This function carries out the task of detecting networks in an arena
        based on several parameters and variables. It involves checking video
        type, performing network detection over time, potentially detecting
        pseudopods, and smoothing segmentation. The results can be visualized or saved.
Extract and analyze graphs from a binary representation of network dynamics, producing vertex
        and edge tables that represent the graph structure over time.

        Args:
            None

        Attributes:
            vars (dict): Dictionary of variables that control the graph extraction process.
                - 'save_graph': Boolean indicating if graph extraction should be performed.
                - 'save_coord_network': Boolean indicating if the coordinate network should be saved.

            one_descriptor_per_arena (dict): Dictionary containing descriptors for each arena.

            dims (tuple): Tuple containing dimension information.
                - [0]: Integer representing the number of time steps.
                - [1]: Integer representing the y-dimension size.
                - [2]: Integer representing the x-dimension size.

            origin (np.ndarray): Binary image representing the origin of the network.

            binary (np.ndarray): Binary representation of network dynamics over time.
                Shape: (time_steps, y_dimension, x_dimension).

            converted_video (np.ndarray): Converted video data.
                Shape: (y_dimension, x_dimension, time_steps).

            network_dynamics (np.ndarray): Network dynamics representation.
                Shape: (time_steps, y_dimension, x_dimension).

        Notes:
            - This method performs graph extraction and saves the vertex and edge tables to CSV files.
            - The CSV files are named according to the arena, time steps, and dimensions.

        Args:
            show_seg: bool = False
                A flag that determines whether to display the segmentation visually.
        """
        coord_pseudopods = None
        if not self.vars['several_blob_per_arena'] and self.vars['save_coord_network']:
            self.check_converted_video_type()

            if self.vars['origin_state'] == "constant":
                self.coord_network, coord_pseudopods = detect_network_dynamics(self.converted_video, self.binary,
                                                           self.one_descriptor_per_arena['arena'], 0,
                                                           self.visu, self.origin, True, True,
                                                           self.vars['save_coord_network'], show_seg)
            else:
                self.coord_network, coord_pseudopods = detect_network_dynamics(self.converted_video, self.binary,
                                                           self.one_descriptor_per_arena['arena'], 0,
                                                           self.visu, None, True, True,
                                                           self.vars['save_coord_network'], show_seg)

        if not self.vars['several_blob_per_arena'] and self.vars['save_graph']:
            if self.coord_network is None:
                self.coord_network = np.array(np.nonzero(self.binary))
            if self.vars['origin_state'] == "constant":
                extract_graph_dynamics(self.converted_video, self.coord_network, self.one_descriptor_per_arena['arena'],
                                       0, self.origin, coord_pseudopods)
            else:
                extract_graph_dynamics(self.converted_video, self.coord_network, self.one_descriptor_per_arena['arena'],
                                       0, None, coord_pseudopods)

    def study_cytoscillations(self, show_seg: bool=False):
        """

            Study the cytoskeletal oscillations within a video frame by frame.

            This method performs an analysis of cytoskeletal oscillations in the video,
            identifying regions of influx and efflux based on pixel connectivity.
            It also handles memory allocation for the oscillations video, computes
            connected components, and optionally displays the segmented regions.

            Args:
                show_seg (bool): If True, display the segmentation results.
        """
        if self.vars['save_coord_thickening_slimming'] or self.vars['oscilacyto_analysis']:
            oscillations_video = detect_oscillations_dynamics(self.converted_video, self.binary,
                                                              self.one_descriptor_per_arena['arena'], self.start,
                                                              self.vars['expected_oscillation_period'],
                                                              self.time_interval,
                                                              self.vars['minimal_oscillating_cluster_size'],
                                                              self.vars['min_ram_free'],
                                                              self.vars['lose_accuracy_to_save_memory'],
                                                              self.vars['save_coord_thickening_slimming'])
            del oscillations_video


    def fractal_descriptions(self):
        """

        Method for analyzing fractal patterns in binary data.

        Fractal analysis is performed on the binary representation of the data,
        optionally considering network dynamics if specified. The results
        include fractal dimensions, R-values, and box counts for the data.

        If network analysis is enabled, additional fractal dimensions,
        R-values, and box counts are calculated for the inner network.
        If 'output_in_mm' is True, then values in mm can be obtained.

        """
        if self.vars['fractal_analysis']:
            logging.info(f"Arena n°{self.one_descriptor_per_arena['arena']}. Starting fractal analysis.")

            if self.vars['save_coord_network']:
                box_counting_dimensions = np.zeros((self.dims[0], 7), dtype=np.float64)
            else:
                box_counting_dimensions = np.zeros((self.dims[0], 3), dtype=np.float64)

            for t in np.arange(self.dims[0]):
                if self.vars['save_coord_network']:
                    current_network = np.zeros(self.dims[1:], dtype=np.uint8)
                    net_t = self.coord_network[1:, self.coord_network[0, :] == t]
                    current_network[net_t[0], net_t[1]] = 1
                    box_counting_dimensions[t, 0] = current_network.sum()
                    zoomed_binary, side_lengths = prepare_box_counting(self.binary[t, ...], min_mesh_side=self.vars[
                        'fractal_box_side_threshold'], zoom_step=self.vars['fractal_zoom_step'], contours=True)
                    box_counting_dimensions[t, 1], box_counting_dimensions[t, 2], box_counting_dimensions[
                        t, 3] = box_counting_dimension(zoomed_binary, side_lengths)
                    zoomed_binary, side_lengths = prepare_box_counting(current_network,
                                                                       min_mesh_side=self.vars[
                                                                           'fractal_box_side_threshold'],
                                                                       zoom_step=self.vars['fractal_zoom_step'],
                                                                       contours=False)
                    box_counting_dimensions[t, 4], box_counting_dimensions[t, 5], box_counting_dimensions[
                        t, 6] = box_counting_dimension(zoomed_binary, side_lengths)
                else:
                    zoomed_binary, side_lengths = prepare_box_counting(self.binary[t, ...],
                                                                       min_mesh_side=self.vars['fractal_box_side_threshold'],
                                                                       zoom_step=self.vars['fractal_zoom_step'], contours=True)
                    box_counting_dimensions[t, :] = box_counting_dimension(zoomed_binary, side_lengths)

            if self.vars['save_coord_network']:
                self.one_row_per_frame["inner_network_size"] = box_counting_dimensions[:, 0]
                self.one_row_per_frame["fractal_dimension"] = box_counting_dimensions[:, 1]
                self.one_row_per_frame["fractal_r_value"] = box_counting_dimensions[:, 2]
                self.one_row_per_frame["fractal_box_nb"] = box_counting_dimensions[:, 3]
                self.one_row_per_frame["inner_network_fractal_dimension"] = box_counting_dimensions[:, 4]
                self.one_row_per_frame["inner_network_fractal_r_value"] = box_counting_dimensions[:, 5]
                self.one_row_per_frame["inner_network_fractal_box_nb"] = box_counting_dimensions[:, 6]
                if self.vars['output_in_mm']:
                    self.one_row_per_frame["inner_network_size"] *= self.vars['average_pixel_size']
            else:
                self.one_row_per_frame["fractal_dimension"] = box_counting_dimensions[:, 0]
                self.one_row_per_frame["fractal_box_nb"] = box_counting_dimensions[:, 1]
                self.one_row_per_frame["fractal_r_value"] = box_counting_dimensions[:, 2]

            if self.vars['save_coord_network']:
                del self.coord_network

    def save_efficiency_tests(self):
        """
        Provide images allowing to assess the analysis efficiency

        This method generates two test images used for assessing
        the efficiency of the analysis. It performs various operations on
        video frames to create these images, including copying and manipulating
        frames from the video, detecting contours on binary images,
        and drawing the arena label on the left of the frames.
        """
        # Provide images allowing to assess the analysis efficiency
        if self.dims[0] > 1:
            after_one_tenth_of_time = np.ceil(self.dims[0] / 10).astype(np.uint64)
        else:
            after_one_tenth_of_time = 0

        last_good_detection = self.dims[0] - 1
        if self.dims[0] > self.lost_frames:
            if self.vars['do_threshold_segmentation']:
                last_good_detection -= self.lost_frames
        else:
            last_good_detection = 0
        if self.visu is None:
            if len(self.converted_video.shape) == 3:
                self.converted_video = np.stack((self.converted_video, self.converted_video, self.converted_video),
                                             axis=3)
            self.efficiency_test_1 = deepcopy(self.converted_video[after_one_tenth_of_time, ...])
            self.efficiency_test_2 = deepcopy(self.converted_video[last_good_detection, ...])
        else:
            self.efficiency_test_1 = deepcopy(self.visu[after_one_tenth_of_time, :, :, :])
            self.efficiency_test_2 = deepcopy(self.visu[last_good_detection, :, :, :])

        position = (25, self.dims[1] // 2)
        text = str(self.one_descriptor_per_arena['arena'])
        contours = np.nonzero(get_contours(self.binary[after_one_tenth_of_time, :, :]))
        self.efficiency_test_1[contours[0], contours[1], :] = self.vars['contour_color']
        self.efficiency_test_1 = cv2.putText(self.efficiency_test_1, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (self.vars["contour_color"], self.vars["contour_color"],
                                         self.vars["contour_color"], 255), 3)

        eroded_binary = cv2.erode(self.binary[last_good_detection, :, :], cross_33)
        contours = np.nonzero(self.binary[last_good_detection, :, :] - eroded_binary)
        self.efficiency_test_2[contours[0], contours[1], :] = self.vars['contour_color']
        self.efficiency_test_2 = cv2.putText(self.efficiency_test_2, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                         (self.vars["contour_color"], self.vars["contour_color"],
                                          self.vars["contour_color"], 255), 3)

    def save_video(self):
        """
        Save processed video with contours and other annotations.

        This method processes the binary image to extract contours, overlay them
        on a video, and save the resulting video file.

        Notes:
            - This method uses OpenCV for image processing and contour extraction.
            - The processed video includes contours colored according to the
              `contour_color` specified in the variables.
            - Additional annotations such as time in minutes are added to each
              frame if applicable.

        """
        if self.vars['save_processed_videos']:
            self.check_converted_video_type()
            if len(self.converted_video.shape) == 3:
                self.converted_video = np.stack((self.converted_video, self.converted_video, self.converted_video),
                                                axis=3)
            for t in np.arange(self.dims[0]):

                eroded_binary = cv2.erode(self.binary[t, :, :], cross_33)
                contours = np.nonzero(self.binary[t, :, :] - eroded_binary)
                self.converted_video[t, contours[0], contours[1], :] = self.vars['contour_color']
                if "iso_digi_transi" in self.one_descriptor_per_arena.keys():
                    if self.vars['iso_digi_analysis']  and not self.vars['several_blob_per_arena'] and not pd.isna(self.one_descriptor_per_arena["iso_digi_transi"]):
                        if self.one_descriptor_per_arena['is_growth_isotropic'] == 1:
                            if t < self.one_descriptor_per_arena["iso_digi_transi"]:
                                self.converted_video[t, contours[0], contours[1], :] = 0, 0, 255
            del self.binary
            del self.surfarea
            del self.borders
            del self.origin
            del self.origin_idx
            del self.mean_intensity_per_frame
            del self.erodila_disk
            collect()
            if self.visu is None:
                true_frame_width = self.dims[2]
                if len(self.vars['background_list']) == 0:
                    self.background = None
                else:
                    self.background = self.vars['background_list'][self.one_descriptor_per_arena['arena'] - 1]
                if os.path.isfile(f"ind_{self.one_descriptor_per_arena['arena']}.npy"):
                    self.visu = video2numpy(f"ind_{self.one_descriptor_per_arena['arena']}.npy",
                              None, true_frame_width=true_frame_width)
                else:
                    self.visu = self.converted_video
                if len(self.visu.shape) == 3:
                    self.visu = np.stack((self.visu, self.visu, self.visu), axis=3)
            self.converted_video = np.concatenate((self.visu, self.converted_video), axis=2)

            if np.any(self.one_row_per_frame['time'] > 0):
                position = (5, self.dims[1] - 5)
                if self.vars['time_step_is_arbitrary']:
                    time_unit = ""
                else:
                    time_unit = " min"
                for t in np.arange(self.dims[0]):
                    image = self.converted_video[t, ...]
                    text = str(self.one_row_per_frame['time'][t]) + time_unit
                    image = cv2.putText(image,  # numpy array on which text is written
                                    text,  # text
                                    position,  # position at which writing has to start
                                    cv2.FONT_HERSHEY_SIMPLEX,  # font family
                                    1,  # font size
                                    (self.vars["contour_color"], self.vars["contour_color"], self.vars["contour_color"], 255),  #(209, 80, 0, 255),  
                                    2)  # font stroke
                    self.converted_video[t, ...] = image
            vid_name = f"ind_{self.one_descriptor_per_arena['arena']}{self.vars['videos_extension']}"
            write_video(self.converted_video, vid_name, is_color=True, fps=self.vars['video_fps'])

    def save_results(self):
        """
        Save the results of testing and video processing.

        This method handles the saving of efficiency tests, video files,
        and CSV data related to test results. It checks for existing files before writing new data.
        Additionally, it cleans up temporary files if configured to do so.
        """
        self.save_efficiency_tests()
        self.save_video()
        if self.vars['several_blob_per_arena']:
            try:
                with open(f"one_row_per_frame_arena{self.one_descriptor_per_arena['arena']}.csv", 'w') as file:
                    self.one_row_per_frame.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error(f"Never let one_row_per_frame_arena{self.one_descriptor_per_arena['arena']}.csv open when Cellects runs")

            create_new_csv: bool = False
            if os.path.isfile("one_row_per_arena.csv"):
                try:
                    with open(f"one_row_per_arena.csv", 'r') as file:
                        stats = pd.read_csv(file, header=0, sep=";")
                except PermissionError:
                    logging.error("Never let one_row_per_arena.csv open when Cellects runs")

                if len(self.one_descriptor_per_arena) == len(stats.columns) - 1:
                    try:
                        with open(f"one_row_per_arena.csv", 'w') as file:
                            stats.iloc[(self.one_descriptor_per_arena['arena'] - 1), 1:] = self.one_descriptor_per_arena.values()
                            stats.to_csv(file, sep=';', index=False, lineterminator='\n')
                    except PermissionError:
                        logging.error("Never let one_row_per_arena.csv open when Cellects runs")
                else:
                    create_new_csv = True
            else:
                create_new_csv = True
            if create_new_csv:
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']), len(self.one_descriptor_per_arena))),
                               columns=list(self.one_descriptor_per_arena.keys()))
                    stats.iloc[(self.one_descriptor_per_arena['arena'] - 1), :] = self.one_descriptor_per_arena.values()
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
        if not self.vars['keep_unaltered_videos'] and os.path.isfile(f"ind_{self.one_descriptor_per_arena['arena']}.npy"):
            os.remove(f"ind_{self.one_descriptor_per_arena['arena']}.npy")

    def change_results_of_one_arena(self, save_video: bool = True):
        """
        Manages the saving and updating of CSV files based on data extracted from analyzed
        one arena. Specifically handles three CSV files: "one_row_per_arena.csv",
        "one_row_per_frame.csv".
        Each file is updated or created based on the presence of existing data.
        The method ensures that each CSV file contains the relevant information for
        the given arena, frame, and oscillator cluster data.
        """
        if save_video:
            self.save_video()
        # I/ Update/Create one_row_per_arena.csv
        create_new_csv: bool = False
        if os.path.isfile("one_row_per_arena.csv"):
            try:
                with open(f"one_row_per_arena.csv", 'r') as file:
                    stats = pd.read_csv(file, header=0, sep=";")
                for stat_name, stat_value in self.one_descriptor_per_arena.items():
                    if stat_name in stats.columns:
                        stats.loc[(self.one_descriptor_per_arena['arena'] - 1), stat_name] = np.uint32(self.one_descriptor_per_arena[stat_name])
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")
            except Exception as e:
                logging.error(f"{e}")
                create_new_csv = True
        else:
            create_new_csv = True
        if create_new_csv:
            logging.info("Create a new one_row_per_arena.csv file")
            try:
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']), len(self.one_descriptor_per_arena))),
                               columns=list(self.one_descriptor_per_arena.keys()))
                    stats.iloc[(self.one_descriptor_per_arena['arena'] - 1), :] = self.one_descriptor_per_arena.values() #  np.array(list(self.one_descriptor_per_arena.values()), dtype=np.uint32)
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")

        # II/ Update/Create one_row_per_frame.csv
        create_new_csv = False
        if os.path.isfile("one_row_per_frame.csv"):
            try:
                with open(f"one_row_per_frame.csv", 'r') as file:
                    descriptors = pd.read_csv(file, header=0, sep=";")
                for stat_name, stat_value in self.one_row_per_frame.items():
                    if stat_name in descriptors.columns:
                        descriptors.loc[((self.one_descriptor_per_arena['arena'] - 1) * self.dims[0]):((self.one_descriptor_per_arena['arena']) * self.dims[0] - 1), stat_name] = self.one_row_per_frame.loc[:, stat_name].values[:]
                with open(f"one_row_per_frame.csv", 'w') as file:
                    descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_frame.csv open when Cellects runs")
            except Exception as e:
                logging.error(f"{e}")
                create_new_csv = True
        else:
            create_new_csv = True
        if create_new_csv:
            logging.info("Create a new one_row_per_frame.csv file")
            try:
                with open(f"one_row_per_frame.csv", 'w') as file:
                    descriptors = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']) * self.dims[0], len(self.one_row_per_frame.columns))),
                               columns=list(self.one_row_per_frame.keys()))
                    descriptors.iloc[((self.one_descriptor_per_arena['arena'] - 1) * self.dims[0]):((self.one_descriptor_per_arena['arena']) * self.dims[0]), :] = self.one_row_per_frame
                    descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_frame.csv open when Cellects runs")

