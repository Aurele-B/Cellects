#!/usr/bin/env python3
"""
This script contains the MotionAnalysis class
It loads a video, check origin shape, get the average covering duration of a pixel,
 perform a rough segmentation (detection), improves it through post-processing, compute all wanted descriptors
 and save them with validation images and videos.
"""

import logging
import os
from gc import collect
# from scipy.stats import linregress
# from scipy import signal as si
from timeit import default_timer
from time import sleep
from copy import deepcopy as dcopy
from cv2 import (
    connectedComponents, connectedComponentsWithStats, MORPH_CROSS,
    getStructuringElement, CV_16U, erode, dilate, morphologyEx, MORPH_OPEN,
    MORPH_CLOSE, MORPH_GRADIENT, BORDER_CONSTANT, resize, imshow, waitKey, destroyAllWindows,
    FONT_HERSHEY_SIMPLEX, putText)
from numba.typed import Dict as TDict
from numpy import (
    c_, char, floor, pad, append, round, ceil, uint64, float32, absolute, sum,
    mean, median, quantile, ptp, diff, square, sqrt, convolve, gradient, zeros,
    ones, empty, array, arange, nonzero, newaxis, argmin, argmax, unique,
    isin, repeat, tile, stack, concatenate, logical_and, logical_or,vstack,
    logical_xor, float16, less, greater, save, sign, uint8, int8, logical_not, load,
    uint32, float64, expand_dims, min, max, any, row_stack, column_stack, full)
from numpy.ma.core import ones_like, logical_not
from pandas import DataFrame as df
from pandas import read_csv, concat, NA, isna
from psutil import virtual_memory
from cellects.image_analysis.cell_leaving_detection import cell_leaving_detection
from cellects.image_analysis.network_detection import NetworkDetection
from cellects.image_analysis.fractal_analysis import FractalAnalysis, box_counting
from cellects.image_analysis.shape_descriptors import ShapeDescriptors, from_shape_descriptors_class
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.morphological_operations import (cross_33, find_major_incline, Ellipse,
    CompareNeighborsWithValue, image_borders, draw_me_a_sun, make_gravity_field, cc, expand_to_fill_holes)
from cellects.utils.formulas import (bracket_to_uint8_image_contrast, eudist, moving_average)
from cellects.image_analysis.image_segmentation import segment_with_lum_value
from cellects.image_analysis.cluster_flux_study import ClusterFluxStudy
from cellects.utils.utilitarian import PercentAndTimeTracker, smallest_memory_array
from cellects.utils.load_display_save import write_video, video2numpy


class MotionAnalysis:

    def __init__(self, l):

        """
        :param video_name: The name of the video to read
        :param convert_for_motion: The dict specifying the linear combination
                                   of color channels (rgb_hsv_lab) to use
        """
        self.statistics = {}
        self.statistics['arena'] = l[1]
        vars = l[2]
        detect_shape = l[3]
        analyse_shape = l[4]
        show_seg = l[5]
        videos_already_in_ram = l[6]
        self.visu = None
        self.binary = None
        self.origin_idx = None
        self.smoothing_flag: bool = False
        logging.info(f"Start the motion analysis of the arena n°{self.statistics['arena']}")

        self.vars = vars
        # self.origin = self.vars['first_image'][self.vars['top'][l[0]]:(
        #    self.vars['bot'][l[0]] + 1),
        #               self.vars['left'][l[0]]:(self.vars['right'][l[0]] + 1)]
        self.load_images_and_videos(videos_already_in_ram, l[0])

        self.dims = self.converted_video.shape
        self.segmentation = zeros(self.dims, dtype=uint8)

        self.covering_intensity = zeros(self.dims[1:], dtype=float64)
        self.mean_intensity_per_frame = mean(self.converted_video, (1, 2))
        if self.vars['arena_shape'] == "circle":
            self.borders = Ellipse(self.dims[1:]).create()
            img_contours = image_borders(self.dims[1:])
            self.borders = self.borders * img_contours
        else:
            self.borders = image_borders(self.dims[1:])
        self.pixel_ring_depth = 9
        self.step = 10
        self.lost_frames = 10
        self.update_ring_width()

        self.start = None
        if detect_shape:
            #self=self.motion
            #self.drift_correction()
            self.start = None
            # Here to conditional layers allow to detect if an expansion/exploration occured
            self.get_origin_shape()
            # The first, user-defined is the 'first_move_threshold' and the second is the detection of the
            # substantial image: if any of them is not detected, the program considers there is not exp.
            if self.dims[0] >= 40:
                step = self.dims[0] // 20
            else:
                step = 1
            if self.start >= (self.dims[0] - step - 1):
                self.start = None
            else:
                self.get_covering_duration(step)
                if self.start is not None:
                    # self.vars['fading'] = -0.5
                    # self.vars['do_threshold_segmentation']: bool = False
                    # self.vars['do_slope_segmentation'] = True
                    # self.vars['true_if_use_light_AND_slope_else_OR']: bool = False
                    self.detection()
                    self.initialize_post_processing()
                    ###
                    if os.path.isfile(f"coord_specimen{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy"):
                        binary_coord = load(f"coord_specimen{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy")
                        self.binary = zeros((self.dims[0], self.dims[1], self.dims[2]), dtype=uint8)
                        self.binary[binary_coord[0, :], binary_coord[1, :], binary_coord[2, :]] = 1
                    else:
                    ###
                        self.t = self.start
                        # print_progress = ForLoopCounter(self.start)
                        # self.binary = self.segmentation # HERE
                        while self.t < self.binary.shape[0]:  #200:  #
                            self.update_shape(show_seg)
                #
            if self.start is None:
                self.binary = repeat(expand_dims(self.origin, 0), self.converted_video.shape[0], axis=0)

            """
            height = screen_height * 2 // 3
            width = int(height * (self.dims[2] / self.dims[1]))
            video_list = [self.converted_video, self.converted_video]
            dimensions = [height, width, 0, (self.binary.shape[0])]
            contours = self.binary
            viewing(video_list, dimensions, contours)
            """
            if analyse_shape:
                self.get_descriptors_from_binary()
                self.detect_growth_transitions()
                self.networks_detection(show_seg)
                self.study_cytoscillations(show_seg)
                self.fractal_descriptions()
                self.get_descriptors_summary()
                if videos_already_in_ram is None:
                    self.save_results()

    def load_images_and_videos(self, videos_already_in_ram, i):
        logging.info(f"Arena n°{self.statistics['arena']}. Load images and videos")
        # pickle_rick = PickleRick()
        # data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
        # try:
        #     with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
        #         data_to_run_cellects_quickly = pickle.load(fileopen)
        # except pickle.UnpicklingError:
        #     try:
        #         with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
        #             data_to_run_cellects_quickly = pickle.load(fileopen)
        #     except pickle.UnpicklingError:
        #         logging.info(f"Data to run Cellects quickly.pkl has not been saved properly in {os.getcwd()}")

        # if data_to_run_cellects_quickly is None or not 'background_and_origin_list' in data_to_run_cellects_quickly:
        #     logging.info(f"Arena n°{self.statistics['arena']}. Data to run Cellects quickly.pkl has not been saved properly in {os.getcwd()}")
        # self.origin = data_to_run_cellects_quickly['background_and_origin_list'][0][i]# self.vars['origins_list'][i]
        self.origin = self.vars['origin_list'][i]# self.vars['origins_list'][i]
        if videos_already_in_ram is None:
            true_frame_width = self.origin.shape[1]
            vid_name = f"ind_{self.statistics['arena']}.npy"
            if len(self.vars['background_list']) == 0:
                self.background = None
            else:
                self.background = self.vars['background_list'][i]
            if len(self.vars['background_list2']) == 0:
                self.background2 = None
            else:
                self.background2 = self.vars['background_list2'][i]
            # self.background2 = None
            # if len(data_to_run_cellects_quickly['background_and_origin_list'][1]) == 0: # len(self.vars['background_list']) == 0:
            #     self.background = None
            # else:
            #     self.background = data_to_run_cellects_quickly['background_and_origin_list'][1][i]# self.vars['background_list'][i]
            #     if self.vars['convert_for_motion']['logical'] != 'None':
            #         self.background2 = data_to_run_cellects_quickly['background_and_origin_list'][2][i]# self.vars['background_list2'][i]
            if self.vars['already_greyscale']:
                self.converted_video = video2numpy(
                    vid_name, None, self.background, true_frame_width)
                if len(self.converted_video.shape) == 4:
                    self.converted_video = self.converted_video[:, :, :, 0]
            else:
                self.visu = video2numpy(
                    vid_name, None, self.background, true_frame_width)
                self.get_converted_video()
        else:
            if self.vars['already_greyscale']:
                self.converted_video = videos_already_in_ram
            else:
                if self.vars['convert_for_motion']['logical'] == 'None':
                    self.visu, self.converted_video = videos_already_in_ram
                else:
                    (self.visu,
                        self.converted_video,
                        self.converted_video2) = videos_already_in_ram

    def get_converted_video(self):
        if not self.vars['already_greyscale']:
            logging.info(f"Arena n°{self.statistics['arena']}. Convert the RGB visu video into a greyscale image using the color space combination: {self.vars['convert_for_motion']}")
            first_dict = TDict()
            second_dict = TDict()
            c_spaces = []
            for k, v in self.vars['convert_for_motion'].items():
                if k != 'logical' and v.sum() > 0:
                    if k[-1] != '2':
                        first_dict[k] = v
                        c_spaces.append(k)
                    else:
                        second_dict[k[:-1]] = v
                        c_spaces.append(k[:-1])
            if self.vars['lose_accuracy_to_save_memory']:
                self.converted_video = zeros(self.visu.shape[:3], dtype=uint8)
            else:
                self.converted_video = zeros(self.visu.shape[:3], dtype=float64)
            # self.converted_video = zeros(self.dims, dtype=uint8)
            if self.vars['convert_for_motion']['logical'] != 'None':
                if self.vars['lose_accuracy_to_save_memory']:
                    self.converted_video2 = zeros(self.visu.shape[:3], dtype=uint8)
                else:
                    self.converted_video2 = zeros(self.visu.shape[:3], dtype=float64)
                # self.converted_video2 = zeros(self.dims, dtype=uint8)

            # Trying to subtract the first image to the first image is a nonsense so,
            # when doing background subtraction, the first and the second image are equal
            for counter in arange(self.visu.shape[0]):
                if self.vars['subtract_background'] and counter == 0:
                    csc = self.visu[1, ...]
                else:
                    csc = self.visu[counter, ...]
                csc = OneImageAnalysis(csc)
                if self.vars['subtract_background']:
                    csc.generate_color_space_combination(c_spaces, first_dict, second_dict, self.background, self.background2)
                else:
                    csc.generate_color_space_combination(c_spaces, first_dict, second_dict, None, None)
                if self.vars['lose_accuracy_to_save_memory']:
                    self.converted_video[counter, ...] = bracket_to_uint8_image_contrast(csc.image)
                else:
                    self.converted_video[counter, ...] = csc.image
                if self.vars['convert_for_motion']['logical'] != 'None':
                    if self.vars['lose_accuracy_to_save_memory']:
                        self.converted_video2[counter, ...] = bracket_to_uint8_image_contrast(csc.image2)
                    else:
                        self.converted_video2[counter, ...] = csc.image2

    def drift_correction(self):
        """
        DO NOT WORK
            Sample the video with a large interval to detect when there is a drift.
            To do so, segment a set of frames, and subtract one with another.
            If there are a (high and) comparable number of 1 and -1 : there is a drift.
            When a drift is detected, progressively reduce the interval to find if it occurs
            between only two frames.
            Otherwise, make a vector of speed telling the number of drifting pixels per unit of time
            Roll frames accordingly
        """
        step = 100
        sampled_frames = arange(0, self.dims[0], step)
        drift_matrix = zeros((self.dims[0], self.dims[1], self.dims[2]), uint8)
        frame_i = OneImageAnalysis(self.visu[0, ...])
        frame_i.conversion(self.vars['convert_for_drift'])
        frame_i.thresholding()
        drift_matrix[0, :, :] = frame_i.binary_image
        for frame in sampled_frames:
            frame_i = OneImageAnalysis(self.visu[frame, ...])
            frame_i.conversion(self.vars['convert_for_drift'])
            luminosity_thresh = median(frame_i.image)
            init_lum_thresh = luminosity_thresh
            frame_i.thresholding(luminosity_thresh, self.vars['lighter_background'])
            while logical_and(frame_i.binary_image.sum() > 1.05 * drift_matrix[0, :, :].sum(),
                                 luminosity_thresh > 0.6 * init_lum_thresh):
                luminosity_thresh -= 5
                frame_i.thresholding(luminosity_thresh, self.vars['lighter_background'])
            if luminosity_thresh <= 0.6 * init_lum_thresh:
                luminosity_thresh = init_lum_thresh
                while logical_and(frame_i.binary_image.sum() < 0.95 * drift_matrix[0, :, :].sum(),
                                     luminosity_thresh < 1.4 * init_lum_thresh):
                    luminosity_thresh += 5
                    frame_i.thresholding(luminosity_thresh, self.vars['lighter_background'])
                if luminosity_thresh >= 1.4 * init_lum_thresh:
                    logging.error("Drift correction failed when studying the frame " + str(frame))
                else:
                    drift_matrix[frame, :, :] = frame_i.binary_image
            else:
                drift_matrix[frame, :, :] = frame_i.binary_image

    def get_origin_shape(self):
        logging.info(f"Arena n°{self.statistics['arena']}. Make sure of origin shape")
        if self.vars['origin_state'] == "constant":
            self.start = 1
            self.origin_idx = nonzero(self.origin)
            if self.vars['lighter_background']:
                # Initialize the covering_intensity matrix as a reference for pixel fading
                self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = 200
            self.substantial_growth = 1.2 * self.origin.sum()
        else:
            self.start = 0
            analysisi = OneImageAnalysis(self.converted_video[0, :, :])
            analysisi.binary_image = 0
            if self.vars['drift_already_corrected']:
                mask_coord = zeros((self.dims[0], 4), dtype=uint32)
                for frame_i in arange(self.dims[0]):  # 100):#
                    true_pixels = nonzero(self.converted_video[frame_i, ...])
                    mask_coord[frame_i, :] = min(true_pixels[0]), max(true_pixels[0]), min(true_pixels[1]), max(
                        true_pixels[1])
            else:
                mask_coord = None
            while logical_and(sum(analysisi.binary_image) < self.vars['first_move_threshold'], self.start < self.dims[0]):
                analysisi = self.frame_by_frame_segmentation(self.start, mask_coord)
                self.start += 1

                # frame_i = OneImageAnalysis(self.converted_video[self.start, :, :])
                # frame_i.thresholding(self.vars['luminosity_threshold'], self.vars['lighter_background'])
                # frame_i.thresholding(self.vars['luminosity_threshold'], self.vars['lighter_background'])
                # self.start += 1

            # Use connected components to find which shape is the nearest from the image center.
            if self.vars['several_blob_per_arena']:
                self.origin = analysisi.binary_image
            else:
                nb_components, output, stats, centroids = connectedComponentsWithStats(analysisi.binary_image,
                                                                                           connectivity=8)
                if self.vars['first_detection_method'] == 'most_central':
                    center = array((self.dims[2] // 2, self.dims[1] // 2))
                    stats = zeros(nb_components - 1)
                    for shape_i in arange(1, nb_components):
                        stats[shape_i - 1] = eudist(center, centroids[shape_i, :])
                    # The shape having the minimal euclidean distance from the center will be the original shape
                    self.origin = zeros((self.dims[1], self.dims[2]), dtype=uint8)
                    self.origin[output == (argmin(stats) + 1)] = 1
                elif self.vars['first_detection_method'] == 'largest':
                    self.origin = zeros((self.dims[1], self.dims[2]), dtype=uint8)
                    self.origin[output == argmax(stats[1:, 4])] = 1
            self.origin_idx = nonzero(self.origin)
            self.substantial_growth = self.origin.sum() + 250
        ##

    def get_covering_duration(self, step):
        logging.info(f"Arena n°{self.statistics['arena']}. Find a frame with a significant growth/motion and determine the number of frames necessary for a pixel to get covered")
        ## Find the time at which growth reached a substantial growth.
        self.substantial_time = self.start
        # To avoid noisy images to have deleterious effects, make sure that area area reaches the threshold thrice.
        occurrence = 0
        if self.vars['drift_already_corrected']:
            mask_coord = zeros((self.dims[0], 4), dtype=uint32)
            for frame_i in arange(self.dims[0]):  # 100):#
                true_pixels = nonzero(self.converted_video[frame_i, ...])
                mask_coord[frame_i, :] = min(true_pixels[0]), max(true_pixels[0]), min(true_pixels[1]), max(
                    true_pixels[1])
        else:
            mask_coord = None
        while logical_and(occurrence < 3, self.substantial_time < (self.dims[0] - step - 1)):
            self.substantial_time += step
            growth_vision = self.frame_by_frame_segmentation(self.substantial_time, mask_coord)

            # growth_vision = OneImageAnalysis(self.converted_video[self.substantial_time, :, :])
            # # growth_vision.thresholding()
            # if self.vars['convert_for_motion']['logical'] != 'None':
            #     growth_vision.image2 = self.converted_video2[self.substantial_time, ...]
            #
            # growth_vision.segmentation(self.vars['convert_for_motion']['logical'], self.vars['color_number'],
            #                            bio_label=self.vars["bio_label"], bio_label2=self.vars["bio_label2"],
            #                            grid_segmentation=self.vars['grid_segmentation'],
            #                            lighter_background=self.vars['lighter_background'])

            surfarea = sum(growth_vision.binary_image * self.borders)
            if surfarea > self.substantial_growth:
                occurrence += 1
        # get a rough idea of the area covered during this time
        if (self.substantial_time - self.start) > 20:
            if self.vars['lighter_background']:
                growth = (sum(self.converted_video[self.start:(self.start + 10), :, :], 0) / 10) - (sum(self.converted_video[(self.substantial_time - 10):self.substantial_time, :, :], 0) / 10)
            else:
                growth = (sum(self.converted_video[(self.substantial_time - 10):self.substantial_time, :, :], 0) / 10) - (
                            sum(self.converted_video[self.start:(self.start + 10), :, :], 0) / 10)
        else:
            if self.vars['lighter_background']:
                growth = self.converted_video[self.start, ...] - self.converted_video[self.substantial_time, ...]
            else:
                growth = self.converted_video[self.substantial_time, ...] - self.converted_video[self.start, ...]
        intensity_extent = ptp(self.converted_video[self.start:self.substantial_time, :, :], axis=0)
        growth[logical_or(growth < 0, intensity_extent < median(intensity_extent))] = 0
        growth = bracket_to_uint8_image_contrast(growth)
        growth *= self.borders
        growth_vision = OneImageAnalysis(growth)
        growth_vision.thresholding()
        self.substantial_image = erode(growth_vision.binary_image, cross_33, iterations=2)

        # New stuff: (removed because was too stringent and impeded a lot of analyses)
        # already_in_origin = self.substantial_image * self.origin
        # not_only_change_from_origin = logical_xor(self.substantial_image, already_in_origin).sum() > 0.1 * self.origin.sum()


        # if any(self.substantial_image) and not_only_change_from_origin:
        if any(self.substantial_image):
            natural_noise = nonzero(intensity_extent == min(intensity_extent))
            natural_noise = self.converted_video[self.start:self.substantial_time, natural_noise[0][0], natural_noise[1][0]]
            natural_noise = moving_average(natural_noise, 5)
            natural_noise = ptp(natural_noise)
            subst_idx = nonzero(self.substantial_image)
            cover_lengths = zeros(len(subst_idx[0]), dtype=uint32)
            for index in arange(len(subst_idx[0])):
                vector = self.converted_video[self.start:self.substantial_time, subst_idx[0][index], subst_idx[1][index]]
                left, right = find_major_incline(vector, natural_noise)
                # If find_major_incline did find a major incline: (otherwise it put 0 to left and 1 to right)
                if not logical_and(left == 0, right == 1):
                    cover_lengths[index] = len(vector[left:-right])
            # If this analysis fails put a deterministic step
            if len(cover_lengths[cover_lengths > 0]) > 0:
                self.step = (round(mean(cover_lengths[cover_lengths > 0])).astype(uint32) // 2) + 1
                logging.info(f"Arena n°{self.statistics['arena']}. Pre-processing detection: the time for a pixel to get covered is set to {self.step}")
            else:
                logging.info(f"Arena n°{self.statistics['arena']}. Pre-processing detection: could not automatically find the time for a pixel to get covered. Default value is 1 for video length < 40 and 10 otherwise")

            # Make sure to avoid a step overestimation
            if self.step > self.dims[0] // 20:
                self.step = self.dims[0] // 20

            if self.step == 0:
                self.step = 1
        # When the first_move_threshold is not stringent enough the program may detect a movement due to noise
        # In that case, the substantial_image is empty and there is no reason to proceed further
        else:
            self.start = None
        ##

    def detection(self, compute_all_possibilities=False):
        # self.lost_frames = (self.step - 1) * self.vars['iterate_smoothing'] # relevant when smoothing did not use padding.
        self.lost_frames = self.step
        # I/ Image by image segmentation algorithms
        # If images contain a drift correction (zeros at borders of the image,
        # Replace these 0 by normal background values before segmenting
        if self.vars['frame_by_frame_segmentation'] or compute_all_possibilities:
            logging.info(f"Arena n°{self.statistics['arena']}. Detect cell motion and growth using the frame by frame segmentation algorithm")
            self.segmentation = zeros(self.dims, dtype=uint8)
            if self.vars['drift_already_corrected']:
                logging.info(f"Arena n°{self.statistics['arena']}. Adjust images to drift correction and segment them")
                # 1. Get the mask valid for a number of images around it (step).
                mask_coord = zeros((self.dims[0], 4), dtype=uint32)
                for frame_i in arange(self.dims[0]):#100):#
                    true_pixels = nonzero(self.converted_video[frame_i, ...])
                    mask_coord[frame_i, :] = min(true_pixels[0]), max(true_pixels[0]), min(true_pixels[1]), max(true_pixels[1])
            else:
                mask_coord = None

            for t in arange(self.dims[0]):#20):#
                analysisi = self.frame_by_frame_segmentation(t, mask_coord)
                self.segmentation[t, ...] = analysisi.binary_image

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
            if self.vars['do_slope_segmentation'] or compute_all_possibilities:
                gradient_segmentation = self.lum_slope_segmentation(self.converted_video)
                gradient_segmentation[-self.lost_frames:, ...] = repeat(gradient_segmentation[-self.lost_frames, :, :][newaxis, :, :], self.lost_frames, axis=0)
            if self.vars['convert_for_motion']['logical'] != 'None':
                if self.vars['do_threshold_segmentation'] or compute_all_possibilities:
                    luminosity_segmentation2, l_threshold_over_time2 = self.lum_value_segmentation(self.converted_video2, do_threshold_segmentation=True)
                    if self.vars['convert_for_motion']['logical'] == 'Or':
                        luminosity_segmentation = logical_or(luminosity_segmentation, luminosity_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'And':
                        luminosity_segmentation = logical_and(luminosity_segmentation, luminosity_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'Xor':
                        luminosity_segmentation = logical_xor(luminosity_segmentation, luminosity_segmentation2)
                self.converted_video2 = self.smooth_pixel_slopes(self.converted_video2)
                if self.vars['do_slope_segmentation'] or compute_all_possibilities:
                    gradient_segmentation2 = self.lum_slope_segmentation(self.converted_video2)
                    gradient_segmentation2[-self.lost_frames:, ...] = repeat(gradient_segmentation2[-self.lost_frames, :, :][newaxis, :, :], self.lost_frames, axis=0)
                    if self.vars['convert_for_motion']['logical'] == 'Or':
                        gradient_segmentation = logical_or(gradient_segmentation, gradient_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'And':
                        gradient_segmentation = logical_and(gradient_segmentation, gradient_segmentation2)
                    elif self.vars['convert_for_motion']['logical'] == 'Xor':
                        gradient_segmentation = logical_xor(gradient_segmentation, gradient_segmentation2)

            if compute_all_possibilities:
                logging.info(f"Arena n°{self.statistics['arena']}. Compute all options to detect cell motion and growth. Maximal growth per frame: {self.vars['max_growth_per_frame']}")
                self.luminosity_segmentation = nonzero(luminosity_segmentation)
                self.gradient_segmentation = nonzero(gradient_segmentation)
                self.logical_and = nonzero(logical_and(luminosity_segmentation, gradient_segmentation))
                self.logical_or = nonzero(logical_or(luminosity_segmentation, gradient_segmentation))
            elif not self.vars['frame_by_frame_segmentation']:
                if self.vars['do_threshold_segmentation'] and not self.vars['do_slope_segmentation']:
                    logging.info(f"Arena n°{self.statistics['arena']}. Detect with luminosity threshold segmentation algorithm")
                    self.segmentation = luminosity_segmentation
                if self.vars['do_slope_segmentation']:# and not self.vars['do_threshold_segmentation']: NEW
                    logging.info(f"Arena n°{self.statistics['arena']}. Detect with luminosity slope segmentation algorithm")
                    # gradient_segmentation[:(self.lost_frames + 1), ...] = luminosity_segmentation[:(self.lost_frames + 1), ...]
                    if not self.vars['do_threshold_segmentation']:# NEW
                        self.segmentation = gradient_segmentation
                if logical_and(self.vars['do_threshold_segmentation'], self.vars['do_slope_segmentation']):
                    if self.vars['true_if_use_light_AND_slope_else_OR']:
                        logging.info(f"Arena n°{self.statistics['arena']}. Detection resuts from threshold AND slope segmentation algorithms")
                        self.segmentation = logical_and(luminosity_segmentation, gradient_segmentation)
                    else:
                        logging.info(f"Arena n°{self.statistics['arena']}. Detection resuts from threshold OR slope segmentation algorithms")
                        self.segmentation = logical_or(luminosity_segmentation, gradient_segmentation)
                self.segmentation = self.segmentation.astype(uint8)
        self.converted_video2 = None


    def frame_by_frame_segmentation(self, t, mask_coord=None):

        contrasted_im = bracket_to_uint8_image_contrast(self.converted_video[t, :, :])
        if self.vars['convert_for_motion']['logical'] != 'None':
            contrasted_im2 = bracket_to_uint8_image_contrast(self.converted_video2[t, :, :])
        # 1. Get the mask valid for a number of images around it (step).
        if self.vars['drift_already_corrected']:
            if t < self.step // 2:
                t_start = 0
                t_end = self.step
            elif t > (self.dims[0] - self.step // 2):
                t_start = self.dims[0] - self.step
                t_end = self.dims[0]
            else:
                t_start = t - (self.step // 2)
                t_end = t + (self.step // 2)
            min_y, max_y = max(mask_coord[t_start:t_end, 0]), min(mask_coord[t_start:t_end, 1])
            min_x, max_x = max(mask_coord[t_start:t_end, 2]), min(mask_coord[t_start:t_end, 3])
            # 3. Bracket the focal image
            image_i = contrasted_im[min_y:(max_y + 1), min_x:(max_x + 1)].astype(float64)
            image_i /= mean(image_i)
            image_i = OneImageAnalysis(image_i)
            if self.vars['convert_for_motion']['logical'] != 'None':
                image_i2 = contrasted_im2[min_y:(max_y + 1), min_x:(max_x + 1)]
                image_i2 /= mean(image_i2)
                image_i.image2 = image_i2
            mask = (self.converted_video[t, ...] > 0).astype(uint8)
        else:
            mask = None
        # 3. Bracket the focal image
        if self.vars['grid_segmentation']:
            int_variation_thresh = 100 - (ptp(contrasted_im) * 90 / 255)
        else:
            int_variation_thresh = None
        analysisi = OneImageAnalysis(bracket_to_uint8_image_contrast(contrasted_im / mean(contrasted_im)))
        if self.vars['convert_for_motion']['logical'] != 'None':
            analysisi.image2 = bracket_to_uint8_image_contrast(contrasted_im2 / mean(contrasted_im2))

        if t == 0:
            analysisi.previous_binary_image = self.origin
        else:
            analysisi.previous_binary_image = self.segmentation[t - 1, ...].copy()

        analysisi.segmentation(self.vars['convert_for_motion']['logical'], self.vars['color_number'],
                               bio_label=self.vars["bio_label"], bio_label2=self.vars["bio_label2"],
                               grid_segmentation=self.vars['grid_segmentation'],
                               lighter_background=self.vars['lighter_background'],
                               side_length=20, step=5, int_variation_thresh=int_variation_thresh, mask=mask)

        # if self.vars['drift_already_corrected']:
        #     im = zeros(self.dims[1:], dtype=uint8)
        #     im[min_y:(max_y + 1), min_x:(max_x + 1)] = analysisi.image
        #     analysisi.image = im.copy()
        #     if self.vars['convert_for_motion']['logical'] != 'None':
        #         im2 = zeros(self.dims[1:], dtype=uint8)
        #         im2[min_y:(max_y + 1), min_x:(max_x + 1)] = analysisi.image2
        #         analysisi.image2 = im2.copy()
        #     bin_im = zeros(self.dims[1:], dtype=uint8)
        #     bin_im[min_y:(max_y + 1), min_x:(max_x + 1)] = analysisi.binary_image
        #     analysisi.binary_image = bin_im.copy()
        return analysisi

        # 1. Get the mask valid for a number of images around it (step).


    def lum_value_segmentation(self, converted_video, do_threshold_segmentation):
        shape_motion_failed: bool = False
        if self.vars['lighter_background']:
            covering_l_values = min(converted_video[:self.substantial_time, :, :],
                                             0) * self.substantial_image
        else:
            covering_l_values = max(converted_video[:self.substantial_time, :, :],
                                             0) * self.substantial_image
        # Avoid errors by checking whether the covering values are nonzero
        covering_l_values = covering_l_values[covering_l_values != 0]
        if len(covering_l_values) == 0:
            shape_motion_failed = True
        if not shape_motion_failed:
            # do_threshold_segmentation = 0.8
            # i = arange(17)
            # l_values_thresholds = power(-1, i) * (0.8 - 0.1 * (i // 2))
            value_segmentation_thresholds = arange(0.8, -0.7, -0.1)
            validated_thresholds = zeros(value_segmentation_thresholds.shape, dtype=bool)
            counter = 0
            while_condition = True
            max_motion_per_frame = (self.dims[1] * self.dims[2]) * self.vars['max_growth_per_frame'] * 2
            if self.vars['lighter_background']:
                basic_bckgrnd_values = quantile(converted_video[:(self.lost_frames + 1), ...], 0.9, axis=(1, 2))
            else:
                basic_bckgrnd_values = quantile(converted_video[:(self.lost_frames + 1), ...], 0.1, axis=(1, 2))
            # Try different values of do_threshold_segmentation and keep the one that does not
            # segment more than x percent of the image
            while counter <= 14:
                value_threshold = value_segmentation_thresholds[counter]
                if self.vars['lighter_background']:
                    l_threshold = (1 + value_threshold) * max(covering_l_values)
                else:
                    l_threshold = (1 - value_threshold) * min(covering_l_values)
                starting_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video[:(self.lost_frames + 1), ...],
                                                               basic_bckgrnd_values, l_threshold,
                                                               self.vars['lighter_background'])

                changing_pixel_number = sum(absolute(diff(starting_segmentation.astype(int8), 1, 0)), (1, 2))
                validation = max(sum(starting_segmentation, (1, 2))) < max_motion_per_frame and (
                        max(changing_pixel_number) < max_motion_per_frame)
                validated_thresholds[counter] = validation
                if any(validated_thresholds):
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
                    uint8(floor(mean(nonzero(validated_thresholds)[0][index_to_keep])))]
            else:
                value_threshold = 0

            if self.vars['lighter_background']:
                l_threshold = (1 + value_threshold) * max(covering_l_values)
            else:
                l_threshold = (1 - value_threshold) * min(covering_l_values)
            if do_threshold_segmentation:
                if self.vars['lighter_background']:
                    basic_bckgrnd_values = quantile(converted_video, 0.9, axis=(1, 2))
                else:
                    basic_bckgrnd_values = quantile(converted_video, 0.1, axis=(1, 2))
                luminosity_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video, basic_bckgrnd_values,
                                                                 l_threshold, self.vars['lighter_background'])
            else:
                luminosity_segmentation, l_threshold_over_time = segment_with_lum_value(converted_video[:(self.lost_frames + 1), ...],
                                                               basic_bckgrnd_values, l_threshold,
                                                               self.vars['lighter_background'])
        else:
            luminosity_segmentation = None

        return luminosity_segmentation, l_threshold_over_time

    def smooth_pixel_slopes(self, converted_video):
        # smoothed_video = zeros(
        #     (self.dims[0] - self.lost_frames, self.dims[1], self.dims[2]),
        #     dtype=float64)
        try:
            if self.vars['lose_accuracy_to_save_memory']:
                smoothed_video = zeros(self.dims, dtype=float16)
                smooth_kernel = ones(self.step) / self.step
                for i in arange(converted_video.shape[1]):
                    for j in arange(converted_video.shape[2]):
                        padded = pad(converted_video[:, i, j] / self.mean_intensity_per_frame,
                                     (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                        moving_average = convolve(padded, smooth_kernel, mode='valid')
                        if self.vars['iterate_smoothing'] > 1:
                            for it in arange(1, self.vars['iterate_smoothing']):
                                padded = pad(moving_average,
                                             (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                                moving_average = convolve(padded, smooth_kernel, mode='valid')
                                # moving_average = convolve(moving_average, smooth_kernel, mode='valid')
                        smoothed_video[:, i, j] = moving_average.astype(float16)
                # smoothed_video -= min(smoothed_video)
                # smoothed_video = round(255 * (smoothed_video / max(smoothed_video))).astype(uint8)
            else:
                smoothed_video = zeros(self.dims, dtype=float64)
                smooth_kernel = ones(self.step) / self.step
                for i in arange(converted_video.shape[1]):
                    for j in arange(converted_video.shape[2]):
                        padded = pad(converted_video[:, i, j] / self.mean_intensity_per_frame,
                                     (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                        moving_average = convolve(padded, smooth_kernel, mode='valid')
                        if self.vars['iterate_smoothing'] > 1:
                            for it in arange(1, self.vars['iterate_smoothing']):
                                padded = pad(moving_average,
                                             (self.step // 2, self.step - 1 - self.step // 2), mode='edge')
                                moving_average = convolve(padded, smooth_kernel, mode='valid')
                                # moving_average = convolve(moving_average, smooth_kernel, mode='valid')
                        smoothed_video[:, i, j] = moving_average
            return smoothed_video

        except MemoryError:
            logging.error("Not enough RAM available to smooth pixel curves. Detection may fail.")
            smoothed_video = converted_video
            return smoothed_video

    def lum_slope_segmentation(self, converted_video):
        shape_motion_failed : bool = False
        gradient_segmentation = zeros(self.dims, uint8)
        # 2) Contrast increase
        oridx = nonzero(self.origin)
        notoridx = nonzero(1 - self.origin)
        do_increase_contrast = mean(converted_video[0, oridx[0], oridx[1]]) * 10 > mean(
                converted_video[0, notoridx[0], notoridx[1]])
        necessary_memory = self.dims[0] * self.dims[1] * self.dims[2] * 64 * 2 * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        if self.vars['lose_accuracy_to_save_memory']:
            derive = converted_video.astype(float16)
        else:
            derive = converted_video.astype(float64)
        if necessary_memory > available_memory:
            converted_video = None

        if do_increase_contrast:
            derive = square(derive)

        # 3) Get the gradient
        necessary_memory = derive.size * 64 * 4 * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        if necessary_memory > available_memory:
            for cy in arange(self.dims[1]):
                for cx in arange(self.dims[2]):
                    if self.vars['lose_accuracy_to_save_memory']:
                        derive[:, cy, cx] = gradient(derive[:, cy, cx], self.step).astype(float16)
                    else:
                        derive[:, cy, cx] = gradient(derive[:, cy, cx], self.step)
        else:
            if self.vars['lose_accuracy_to_save_memory']:
                derive = gradient(derive, self.step, axis=0).astype(float16)
            else:
                derive = gradient(derive, self.step, axis=0)

        # 4) Segment
        if self.vars['lighter_background']:
            covering_slopes = min(derive[:self.substantial_time, :, :], 0) * self.substantial_image
        else:
            covering_slopes = max(derive[:self.substantial_time, :, :], 0) * self.substantial_image
        covering_slopes = covering_slopes[covering_slopes != 0]
        if len(covering_slopes) == 0:
            shape_motion_failed = True

        if not shape_motion_failed:
            ####
            # ease_slope_segmentation = 0.8
            value_segmentation_thresholds = arange(0.8, -0.7, -0.1)
            validated_thresholds = zeros(value_segmentation_thresholds.shape, dtype=bool)
            counter = 0
            while_condition = True
            max_motion_per_frame = (self.dims[1] * self.dims[2]) * self.vars['max_growth_per_frame']
            # Try different values of do_slope_segmentation and keep the one that does not
            # segment more than x percent of the image
            while counter <= 14:
                ease_slope_segmentation = value_segmentation_thresholds[counter]
                if self.vars['lighter_background']:
                    gradient_threshold = (1 + ease_slope_segmentation) * max(covering_slopes)
                    sample = less(derive[:self.substantial_time], gradient_threshold)
                else:
                    gradient_threshold = (1 - ease_slope_segmentation) * min(covering_slopes)
                    sample = greater(derive[:self.substantial_time], gradient_threshold)
                changing_pixel_number = sum(absolute(diff(sample.astype(int8), 1, 0)), (1, 2))
                validation = max(sum(sample, (1, 2))) < max_motion_per_frame and (
                        max(changing_pixel_number) < max_motion_per_frame)
                validated_thresholds[counter] = validation
                if any(validated_thresholds):
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
                    uint8(floor(mean(nonzero(validated_thresholds)[0][index_to_keep])))]
            else:
                ease_slope_segmentation = 0

            if self.vars['lighter_background']:
                gradient_threshold = (1 - ease_slope_segmentation) * max(covering_slopes)
                # gradient_segmentation = less(derive, gradient_threshold)#New
                gradient_segmentation[:-self.lost_frames, :, :] = less(derive, gradient_threshold)[self.lost_frames:, :, :]
            else:
                gradient_threshold = (1 - ease_slope_segmentation) * min(covering_slopes)
                # gradient_segmentation = greater(derive, gradient_threshold)#New
                gradient_segmentation[:-self.lost_frames, :, :] = greater(derive, gradient_threshold)[self.lost_frames:, :, :]
        else:
            gradient_segmentation = None
        return gradient_segmentation

    def update_ring_width(self):
        # Make sure that self.pixels_depths are odd and greater than 3
        if self.pixel_ring_depth <= 3:
            self.pixel_ring_depth = 3
        if self.pixel_ring_depth % 2 == 0:
            self.pixel_ring_depth = self.pixel_ring_depth + 1
        self.erodila_disk = Ellipse((self.pixel_ring_depth, self.pixel_ring_depth)).create().astype(uint8)
        self.max_distance = self.pixel_ring_depth * self.vars['ease_connect_distant_shape']

    def initialize_post_processing(self):
        ## Initialization
        logging.info(f"Arena n°{self.statistics['arena']}. Starting Post_processing. Fading detection: {self.vars['do_fading']}: {self.vars['fading']}, Subtract background: {self.vars['subtract_background']}, Correct errors around initial shape: {self.vars['ring_correction']}, Connect distant shapes: {self.vars['ease_connect_distant_shape'] > 0}, How to select appearing cell(s): {self.vars['first_detection_method']}")

        self.binary = zeros(self.dims[:3], dtype=uint8)
        if self.origin.shape[0] != self.binary[self.start - 1, :, :].shape[0] or self.origin.shape[1] != self.binary[self.start - 1, :, :].shape[1]:
            logging.error("Unaltered videos deprecated, they have been created with different settings.\nDelete .npy videos and Data to run Cellects quickly.pkl and re-run")

        if self.vars['origin_state'] == "invisible":
            self.binary[self.start - 1, :, :] = self.origin.copy()
            self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = self.converted_video[self.start, self.origin_idx[0], self.origin_idx[1]]
        else:
            if self.vars['origin_state'] == "fluctuating":
                self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = median(self.converted_video[:self.start, self.origin_idx[0], self.origin_idx[1]], axis=0)

            self.binary[:self.start, :, :] = repeat(expand_dims(self.origin, 0), self.start, axis=0)
            if self.start < self.step:
                frames_to_assess = self.step
                self.segmentation[self.start - 1, ...] = self.binary[self.start - 1, :, :]
                for t in arange(self.start, self.lost_frames):
                    # Only keep pixels that are always detected
                    always_found = sum(self.segmentation[t:(t + frames_to_assess), ...], 0)
                    always_found = always_found == frames_to_assess
                    # Remove too small shapes
                    without_small, stats, centro = cc(always_found.astype(uint8))
                    large_enough = nonzero(stats[1:, 4] > ((self.vars['first_move_threshold'] + 1) // 2))[0]
                    if len(large_enough) > 0:
                        always_found *= isin(always_found, large_enough + 1)
                        always_found = logical_or(always_found, self.segmentation[t - 1, ...])
                        self.segmentation[t, ...] *= always_found
                    else:
                        self.segmentation[t, ...] = 0
                    self.segmentation[t, ...] = logical_or(self.segmentation[t - 1, ...], self.segmentation[t, ...])
        self.mean_distance_per_frame = None
        self.surfarea = zeros(self.dims[0], dtype=uint64)
        self.surfarea[:self.start] = sum(self.binary[:self.start, :, :], (1, 2))
        self.gravity_field = make_gravity_field(self.binary[(self.start - 1), :, :],
                                           sqrt(sum(self.binary[(self.start - 1), :, :])))
        if self.vars['ring_correction']:
            self.rays, self.sun = draw_me_a_sun(self.binary[(self.start - 1), :, :], cross_33, ray_length_coef=1.25)  # plt.imshow(sun)
            self.holes = zeros(self.dims[1:], dtype=uint8)
            # cannot_fading = self.holes.copy()
            self.pixel_ring_depth += 2
            self.update_ring_width()

        if self.vars['prevent_fast_growth_near_periphery']:
            self.near_periphery = zeros(self.dims[1:])
            if self.vars['arena_shape'] == 'circle':
                periphery_width = self.vars['periphery_width'] * 2
                elliperiphery = Ellipse((self.dims[1] - periphery_width, self.dims[2] - periphery_width)).create()
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
            self.near_periphery = nonzero(self.near_periphery)
            # near_periphery = zeros(self.dims[1:])
            # near_periphery[self.near_periphery] = 1

    def update_shape(self, show_seg):

        # Get from gradients, a 2D matrix of potentially covered pixels
        # I/ dilate the shape made with covered pixels to assess for covering

        # I/ 1) Only keep pixels that have been detected at least two times in the three previous frames
        if self.dims[0] < 100:
            new_potentials = self.segmentation[self.t, :, :]
        else:
            if self.t > 1:
                new_potentials = sum(self.segmentation[(self.t - 2): (self.t + 1), :, :], 0, dtype=uint8)
            else:
                new_potentials = sum(self.segmentation[: (self.t + 1), :, :], 0, dtype=uint8)
            new_potentials[new_potentials == 1] = 0
            new_potentials[new_potentials > 1] = 1

        # I/ 2) If an image displays more new potential pixels than 50% of image pixels,
        # one of these images is considered noisy and we try taking only one.
        frame_counter = -1
        maximal_size = 0.5 * new_potentials.size
        if (self.vars["do_threshold_segmentation"] or self.vars["frame_by_frame_segmentation"]) and self.t > max((self.start + self.step, 6)):
           maximal_size = min((max(self.binary[:self.t].sum((1, 2))) * (1 + self.vars['max_growth_per_frame']), self.borders.sum()))
        while logical_and(sum(new_potentials) > maximal_size,
                             frame_counter <= 5):  # logical_and(sum(new_potentials > 0) > 5 * sum(dila_ring), frame_counter <= 5):
            frame_counter += 1
            if frame_counter > self.t:
                break
            else:
                if frame_counter < 5:
                    new_potentials = self.segmentation[self.t - frame_counter, :, :]
                else:
                # If taking only one image is not enough, use the inverse of the fadinged matrix as new_potentials
                # Given it haven't been processed by any slope calculation, it should be less noisy
                    new_potentials = sum(self.segmentation[(self.t - 5): (self.t + 1), :, :], 0, dtype=uint8)
                    new_potentials[new_potentials < 6] = 0
                    new_potentials[new_potentials == 6] = 1


        new_shape = self.binary[self.t - 1, :, :].copy()
        new_potentials = morphologyEx(new_potentials, MORPH_CLOSE, cross_33) # TO REMOVE!!!
        new_potentials = morphologyEx(new_potentials, MORPH_OPEN, cross_33) * self.borders # TO REMOVE!!!
        new_shape = logical_or(new_shape, new_potentials).astype(uint8)
        # Add distant shapes within a radius, score every added pixels according to their distance
        if not self.vars['several_blob_per_arena']:
            # new_potentials *= self.borders
            # if self.vars['origin_state'] == "constant":
            #     new_shape = logical_or(new_potentials, self.origin).astype(uint8)
            # else:
            #     new_shape = new_potentials.copy()
        # else:
            # Remove noise by opening and by keeping the shape within borders
            # new_potentials = morphologyEx(new_potentials, MORPH_OPEN, cross_33) * self.borders # TO REMOVE!!!
            ## Build the new shape state from the t-1 one
            if new_shape.sum() == 0:
                new_shape = new_potentials.copy()
            else:
                pads = ProgressivelyAddDistantShapes(new_potentials, new_shape, self.max_distance)
                # If max_distance is non nul look for distant shapes
                pads.consider_shapes_sizes(self.vars['min_distant_shape_size'],
                                                     self.vars['max_distant_shape_size'])
                pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=True)

                new_shape = pads.expanded_shape.copy()
                new_shape[new_shape > 1] = 1
                if logical_and(self.t > self.step, self.t < self.dims[0]):
                    if any(pads.expanded_shape > 5):
                        # Add distant shapes back in time at the covering speed of neighbors
                        self.binary[self.t][nonzero(new_shape)] = 1
                        self.binary[(self.step):(self.t + 1), :, :] = \
                            pads.modify_past_analysis(self.binary[(self.step):(self.t + 1), :, :],
                                                      self.segmentation[(self.step):(self.t + 1), :, :])
                        new_shape = self.binary[self.t, :, :].copy()
                pads = None

            # Fill holes
            new_shape = morphologyEx(new_shape, MORPH_CLOSE, cross_33)

        if self.vars['do_fading'] and (self.t > self.step + self.lost_frames):
            # Shape Erosion
            # I/ After a substantial growth, erode the shape made with covered pixels to assess for fading
            # Use the newly covered pixels to calculate their mean covering intensity
            new_idx = nonzero(logical_xor(new_shape, self.binary[self.t - 1, :, :]))
            start_intensity_monitoring = self.t - self.lost_frames - self.step
            end_intensity_monitoring = self.t - self.lost_frames
            self.covering_intensity[new_idx[0], new_idx[1]] = median(self.converted_video[start_intensity_monitoring:end_intensity_monitoring, new_idx[0], new_idx[1]], axis=0)
            previous_binary = self.binary[self.t - 1, :, :]
            greyscale_image = self.converted_video[self.t - self.lost_frames, :, :]
            protect_from_fading = None
            if self.vars['origin_state'] == 'constant':
                protect_from_fading = self.origin
            new_shape, self.covering_intensity = cell_leaving_detection(new_shape, self.covering_intensity, previous_binary, greyscale_image, self.vars['fading'], self.vars['lighter_background'], self.vars['several_blob_per_arena'], self.erodila_disk, protect_from_fading)

        self.covering_intensity *= new_shape
        self.binary[self.t, :, :] = new_shape * self.borders
        self.surfarea[self.t] = sum(self.binary[self.t, :, :])

        # Calculate the mean distance covered per frame and correct for a ring of not really fading pixels
        if self.mean_distance_per_frame is None:
            if self.vars['ring_correction'] and not self.vars['several_blob_per_arena']:
                if logical_and((self.t % 20) == 0,
                                  logical_and(self.surfarea[self.t] > self.substantial_growth,
                                                 self.surfarea[self.t] < self.substantial_growth * 2)):
                    shape = self.binary[self.t, :, :] * self.sun
                    back = (1 - self.binary[self.t, :, :]) * self.sun
                    for ray in self.rays:
                        # For each sun's ray, see how they cross the shape/back and
                        # store the gravity_field value of these pixels (distance to the original shape).
                        ray_through_shape = (shape == ray) * self.gravity_field
                        ray_through_back = (back == ray) * self.gravity_field
                        if any(ray_through_shape):
                            if any(ray_through_back):
                                # If at least one back pixel is nearer to the original shape than a shape pixel,
                                # there is a hole to fill.
                                if any(ray_through_back > min(ray_through_shape[ray_through_shape > 0])):
                                    # Check if the nearest pixels are shape, if so, supress them until the nearest pixel
                                    # becomes back
                                    while max(ray_through_back) <= max(ray_through_shape):
                                        ray_through_shape[ray_through_shape == max(ray_through_shape)] = 0
                                    # Now, all back pixels that are nearer than the closest shape pixel should get filled
                                    # To do so, replace back pixels further than the nearest shape pixel by 0
                                    ray_through_back[ray_through_back < max(ray_through_shape)] = 0
                                    self.holes[nonzero(ray_through_back)] = 1
                            else:
                                self.rays = concatenate((self.rays[:(ray - 2)], self.rays[(ray - 1):]))
                        ray_through_shape = None
                        ray_through_back = None
            if any(self.surfarea[:self.t] > self.substantial_growth * 2):

                if self.vars['ring_correction'] and not self.vars['several_blob_per_arena']:
                    # Apply the hole correction
                    self.holes = morphologyEx(self.holes, MORPH_CLOSE, cross_33, iterations=10)
                    # If some holes are not covered by now
                    if any(self.holes * (1 - self.binary[self.t, :, :])):  # if any(self.holes > 0):
                        self.binary[:(self.t + 1), :, :], holes_time_end, distance_against_time = \
                            expand_to_fill_holes(self.binary[:(self.t + 1), :, :], self.holes)
                        if holes_time_end is not None:
                            self.binary[holes_time_end:(self.t + 1), :, :] += self.binary[holes_time_end, :, :]
                            self.binary[holes_time_end:(self.t + 1), :, :][
                                self.binary[holes_time_end:(self.t + 1), :, :] > 1] = 1
                            self.surfarea[:(self.t + 1)] = sum(self.binary[:(self.t + 1), :, :], (1, 2))

                    else:
                        distance_against_time = [1, 2]
                else:
                    distance_against_time = [1, 2]
                distance_against_time = diff(distance_against_time)
                if len(distance_against_time) > 0:
                    self.mean_distance_per_frame = mean(- distance_against_time)
                else:
                    self.mean_distance_per_frame = 1

        if self.vars['prevent_fast_growth_near_periphery']:
            # growth_near_periphery = diff(self.binary[self.t-1:self.t+1, :, :] * self.near_periphery, axis=0)
            growth_near_periphery = diff(self.binary[self.t-1:self.t+1, self.near_periphery[0], self.near_periphery[1]], axis=0)
            if (growth_near_periphery == 1).sum() > self.vars['max_periphery_growth']:
                # self.binary[self.t, self.near_periphery[0], self.near_periphery[1]] = self.binary[self.t - 1, self.near_periphery[0], self.near_periphery[1]]
                periphery_to_remove = zeros(self.dims[1:], dtype=uint8)
                periphery_to_remove[self.near_periphery[0], self.near_periphery[1]] = self.binary[self.t, self.near_periphery[0], self.near_periphery[1]]
                shapes, stats, centers = cc(periphery_to_remove)
                periphery_to_remove = nonzero(isin(shapes, nonzero(stats[:, 4] > self.vars['max_periphery_growth'])[0][1:]))
                self.binary[self.t, periphery_to_remove[0], periphery_to_remove[1]] = self.binary[self.t - 1, periphery_to_remove[0], periphery_to_remove[1]]
                if not self.vars['several_blob_per_arena']:
                    shapes, stats, centers = cc(self.binary[self.t, ...])
                    shapes[shapes != 1] = 0
                    self.binary[self.t, ...] = shapes

        # Display

        if show_seg:
            if self.visu is not None:
                im_to_display = self.visu[self.t, ...].copy()
                contours = nonzero(morphologyEx(self.binary[self.t, :, :], MORPH_GRADIENT, cross_33))
                if self.vars['lighter_background']:
                    im_to_display[contours[0], contours[1]] = 0
                else:
                    im_to_display[contours[0], contours[1]] = 255
            else:
                im_to_display = self.binary[self.t, :, :] * 255
            imtoshow = resize(im_to_display, (540, 540))
            imshow("shape_motion", imtoshow)
            waitKey(1)
        self.t += 1


    def get_descriptors_from_binary(self, release_memory=True):
        ##
        if self.vars['save_binary_masks']:
            save(f"coord_specimen{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy",
                 smallest_memory_array(nonzero(self.binary), "uint"))
        if release_memory:
            self.substantial_image = None
            self.covering_intensity = None
            self.segmentation = None
            self.gravity_field = None
            self.sun = None
            self.rays = None
            self.holes = None
            collect()
        if self.vars['do_fading']:
            self.newly_explored_area = zeros(self.dims[0], dtype=uint64)
            self.already_explored_area = self.origin.copy()
            for self.t in range(self.dims[0]):
                self.newly_explored_area[self.t] = ((self.binary[self.t, :, :] - self.already_explored_area) == 1).sum()
                self.already_explored_area = logical_or(self.already_explored_area, self.binary[self.t, :, :])

        self.surfarea = self.binary.sum((1, 2))
        timings = self.vars['exif']
        if len(timings) < self.dims[0]:
            timings = arange(self.dims[0])
        if any(timings > 0):
            self.time_interval = mean(diff(timings))
        timings = timings[:self.dims[0]]
        available_descriptors_in_sd = list(from_shape_descriptors_class.keys())
        # ["area", "perimeter", "circularity", "rectangularity", "total_hole_area", "solidity",
        #                          "convexity", "eccentricity", "euler_number", "standard_deviation_y",
        #                          "standard_deviation_x", "skewness_y", "skewness_x", "kurtosis_y", "kurtosis_x",
        #                          "major_axis_len", "minor_axis_len", "axes_orientation"]
        all_descriptors = []
        to_compute_from_sd = []
        for name, do_compute in self.vars['descriptors'].items():
            if do_compute:# and
                all_descriptors.append(name)
                if isin(name, available_descriptors_in_sd):
                    to_compute_from_sd.append(name)
        self.compute_solidity_separately: bool = self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena'] and not isin("solidity", to_compute_from_sd)
        if self.compute_solidity_separately:
            self.solidity = zeros(self.dims[0], dtype=float64)
        if not self.vars['several_blob_per_arena']:
            self.whole_shape_descriptors = df(zeros((self.dims[0], 2 + len(all_descriptors))),
                                              columns=['arena', 'time'] + all_descriptors)
            self.whole_shape_descriptors['arena'] = [self.statistics['arena']] * self.dims[0]
            self.whole_shape_descriptors['time'] = timings
            # solidity must be added if detect growth transition is computed
            origin = self.binary[0, :, :]
            self.statistics["first_move"] = NA

            for t in arange(self.dims[0]):
                SD = ShapeDescriptors(self.binary[t, :, :], to_compute_from_sd)


                # NEW
                for descriptor in to_compute_from_sd:
                    self.whole_shape_descriptors.loc[t, descriptor] = SD.descriptors[descriptor]
                # Old
                # self.whole_shape_descriptors.iloc[t, 2: 2 + len(descriptors)] = SD.descriptors.values()


                if self.compute_solidity_separately:
                    solidity = ShapeDescriptors(self.binary[t, :, :], ["solidity"])
                    self.solidity[t] = solidity.descriptors["solidity"]
                    # self.solidity[t] = list(solidity.descriptors.values())[0]
                # I) Find a first pseudopod [aim: time]
                if isna(self.statistics["first_move"]):
                    if self.surfarea[t] >= (origin.sum() + self.vars['first_move_threshold']):
                        self.statistics["first_move"] = t

            # Apply the scale to the variables
            if self.vars['output_in_mm']:
                if isin('area', to_compute_from_sd):
                    self.whole_shape_descriptors['area'] *= self.vars['average_pixel_size']
                if isin('total_hole_area', to_compute_from_sd):
                    self.whole_shape_descriptors['total_hole_area'] *= self.vars['average_pixel_size']
                if isin('perimeter', to_compute_from_sd):
                    self.whole_shape_descriptors['perimeter'] *= sqrt(self.vars['average_pixel_size'])
                if isin('major_axis_len', to_compute_from_sd):
                    self.whole_shape_descriptors['major_axis_len'] *= sqrt(self.vars['average_pixel_size'])
                if isin('minor_axis_len', to_compute_from_sd):
                    self.whole_shape_descriptors['minor_axis_len'] *= sqrt(self.vars['average_pixel_size'])
        else:
            # Objective: create a matrix with 4 columns (time, y, x, colony) containing the coordinates of all colonies
            # against time
            self.statistics["first_move"] = 1
            max_colonies = 0
            for t in arange(self.dims[0]):
                nb, shapes = connectedComponents(self.binary[t, :, :])
                max_colonies = max((max_colonies, nb))

            time_descriptor_colony = zeros((self.dims[0], len(to_compute_from_sd) * max_colonies * self.dims[0]),
                                              dtype=float32)  # Adjust max_colonies
            colony_number = 0
            colony_id_matrix = zeros(self.dims[1:], dtype=uint64)
            coord_colonies = []
            centroids = []

            pat_tracker = PercentAndTimeTracker(self.dims[0], compute_with_elements_number=True)
            for t in arange(self.dims[0]):  #21):#
                # t=0
                # t+=1
                # We rank colonies in increasing order to make sure that the larger colony issued from a colony division
                # keeps the previous colony name.
                shapes, stats, centers = cc(self.binary[t, :, :])

                # Consider that shapes bellow 3 pixels are noise. The loop will stop at nb and not compute them
                nb = stats[stats[:, 4] >= 4].shape[0]

                # nb = stats.shape[0]
                current_percentage, eta = pat_tracker.get_progress(t, element_number=nb)
                logging.info(f"Arena n°{self.statistics['arena']}, Colony descriptors computation: {current_percentage}%{eta}")

                updated_colony_names = zeros(1, dtype=uint32)
                for colony in (arange(nb - 1) + 1):  # 120)):# #92
                    # colony = 1
                    # colony+=1
                    # logging.info(f'Colony number {colony}')
                    current_colony_img = (shapes == colony).astype(uint8)

                    # I/ Find out which names the current colony had at t-1
                    colony_previous_names = unique(current_colony_img * colony_id_matrix)
                    colony_previous_names = colony_previous_names[colony_previous_names != 0]
                    # II/ Find out if the current colony name had already been analyzed at t
                    # If there no match with the saved colony_id_matrix, assign colony ID
                    if t == 0 or len(colony_previous_names) == 0:
                        # logging.info("New colony")
                        colony_number += 1
                        colony_names = [colony_number]
                    # If there is at least 1 match with the saved colony_id_matrix, we keep the colony_previous_name(s)
                    else:
                        colony_names = colony_previous_names.tolist()
                    # Handle colony division if necessary
                    if any(isin(updated_colony_names, colony_names)):
                        colony_number += 1
                        colony_names = [colony_number]

                    # Update colony ID matrix for the current frame
                    coords = nonzero(current_colony_img)
                    colony_id_matrix[coords[0], coords[1]] = colony_names[0]

                    # Add coordinates to coord_colonies
                    time_column = full(coords[0].shape, t, dtype=uint32)
                    colony_column = full(coords[0].shape, colony_names[0], dtype=uint32)
                    coord_colonies.append(column_stack((time_column, colony_column, coords[0], coords[1])))

                    # Calculate centroid and add to centroids list
                    centroid_x, centroid_y = centers[colony, :]
                    centroids.append((t, colony_names[0], centroid_y, centroid_x))

                    # Compute shape descriptors
                    SD = ShapeDescriptors(current_colony_img, to_compute_from_sd)
                    descriptors = list(SD.descriptors.values())
                    # Adjust descriptors if output_in_mm is specified
                    if self.vars['output_in_mm']:
                        if 'area' in to_compute_from_sd:
                            descriptors['area'] *= self.vars['average_pixel_size']
                        if 'total_hole_area' in to_compute_from_sd:
                            descriptors['total_hole_area'] *= self.vars['average_pixel_size']
                        if 'perimeter' in to_compute_from_sd:
                            descriptors['perimeter'] *= sqrt(self.vars['average_pixel_size'])
                        if 'major_axis_len' in to_compute_from_sd:
                            descriptors['major_axis_len'] *= sqrt(self.vars['average_pixel_size'])
                        if 'minor_axis_len' in to_compute_from_sd:
                            descriptors['minor_axis_len'] *= sqrt(self.vars['average_pixel_size'])

                    # Store descriptors in time_descriptor_colony
                    descriptor_index = (colony_names[0] - 1) * len(to_compute_from_sd)
                    time_descriptor_colony[t, descriptor_index:(descriptor_index + len(descriptors))] = descriptors

                    updated_colony_names = append(updated_colony_names, colony_names)

                # Reset colony_id_matrix for the next frame
                colony_id_matrix *= self.binary[t, :, :]

            coord_colonies = vstack(coord_colonies)
            centroids = array(centroids, dtype=float32)
            time_descriptor_colony = time_descriptor_colony[:, :(colony_number*len(to_compute_from_sd))]

            if self.vars['save_binary_masks']:
                coord_colonies = df(coord_colonies, columns=["time", "colony", "y", "x"])
                coord_colonies.to_csv(f"coord_colonies{self.statistics['arena']}_t{self.dims[0]}_col{colony_number}_y{self.dims[1]}_x{self.dims[2]}.csv", sep=';', index=False, lineterminator='\n')
                # save(f"coord_colonies{self.statistics['arena']}_t{self.dims[0]}_col{colony_number}_y{self.dims[1]}_x{self.dims[2]}.npy", coord_colonies)
            # save(f"colony_centroids{self.statistics['arena']}_t{self.dims[0]}_col{colony_number}_y{self.dims[1]}_x{self.dims[2]}.npy", centroids)
            centroids = df(centroids, columns=["time", "colony", "y", "x"])
            centroids.to_csv(f"colony_centroids{self.statistics['arena']}_t{self.dims[0]}_col{colony_number}_y{self.dims[1]}_x{self.dims[2]}.csv", sep=';', index=False, lineterminator='\n')

            # Format the final dataframe to have one row per time frame, and one column per descriptor_colony_name
            self.whole_shape_descriptors = df({'arena': self.statistics['arena'], 'time': timings, 'area_total': self.surfarea.astype(float64)})
            if self.vars['output_in_mm']:
                self.whole_shape_descriptors['area_total'] *= self.vars['average_pixel_size']
            column_names = char.add(repeat(to_compute_from_sd, colony_number),
                                    tile((arange(colony_number) + 1).astype(str), len(to_compute_from_sd)))
            time_descriptor_colony = df(time_descriptor_colony, columns=column_names)
            self.whole_shape_descriptors = concat([self.whole_shape_descriptors, time_descriptor_colony], axis=1)

            """
                    #
                    #     # This colony already existed: The colony_previous_names becomes the current colony_names
                    #     colony_names = colony_previous_names.copy()
                    #     colony_coord = array(nonzero(current_colony_img), dtype=uint32)
                    #     # If any of the current colony names already are in updated_colony_names, it means that a
                    #     # consistent colony from t-1 split into several (at least 2) colonies at t.
                    #     result_from_colony_division: bool = False
                    #     for name in colony_names:
                    #         if isin(name, updated_colony_names):
                    #             result_from_colony_division = True
                    #         if result_from_colony_division:
                    #             break
                    #     if result_from_colony_division:
                    #         # logging.info("This colony emerged from the division of a parent colony")
                    #         # We must add it as a new colony
                    #         colony_number += 1
                    #         colony_names = [colony_number]
                    #
                    # pixel_nb = colony_coord.shape[1]
                    # colony_id_matrix[colony_coord[0, :], colony_coord[1, :]] = colony_names[0]
                    # coord_colonies = row_stack((coord_colonies, column_stack((repeat(t, pixel_nb).astype(uint32),
                    #     colony_coord[0, :], colony_coord[1, :], repeat(colony_names[0], pixel_nb).astype(uint32)))))
                    # updated_colony_names = append(updated_colony_names, colony_names)


            

            coord_colonies2 = coord_colonies[coord_colonies[:, 3] < 500, :]
            save(f"coord_colonies{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}_col{max(coord_colonies2[:, 3])}.npy", coord_colonies2)

            colist = []
            for colony in unique(coord_colonies2[:, 3]):
                colist.append(ptp(coord_colonies2[coord_colonies2[:, 3] == colony, 2]))
            coord_colonies2[coord_colonies2[:, 3] == 36]
            nonzero(array(colist)==270)
            max(array(colist))

            # shapes, stats, centers = cc((colony_id_matrix==36).astype(uint8))

            from numpy import array_equal, zeros_like
            a = zeros_like(self.binary[t, ...])
            coord_t = coord_colonies[coord_colonies[:, 3] == 5, :]
            a[coord_t[:, 1], coord_t[:,2]] = 1
            See(a)
            array_equal(self.binary[t, ...] > 0, a)
            b = zeros((31, self.dims[1], self.dims[2]), uint8)
            for ti in range(31):
                b[ti, coord_colonies[logical_and(coord_colonies[:, 0] == ti, coord_colonies[:, 3] == 5), 1], coord_colonies[logical_and(coord_colonies[:, 0] == ti, coord_colonies[:, 3] == 5), 2]] = 1

            movie(b)
            See(b[0,...])
            ####

            self.statistics["first_move"] = 1
            colony_id_matrix = zeros(self.dims[1:], dtype=uint64)
            # updated_colony_id_matrix = zeros(self.dims[1:], dtype=uint32)
            # colony_dictionary = {}
            # colony_dictionary['arena'] = [self.statistics['arena']] * self.dims[0]
            # colony_dictionary['time'] = timings
            # colony_dictionary['area_total'] = self.surfarea.astype(float64)
            time_descriptor_colony = zeros((self.dims[0], len(to_compute_from_sd), 100), dtype=float64)
            # if self.vars['output_in_mm']:
            #     colony_dictionary['area_total'] *= self.vars['average_pixel_size']
            colony_number = 0
            # time_descriptor_colony[27:(t+1), 0, :5]
            pat_tracker = PercentAndTimeTracker(self.dims[0], compute_with_elements_number=True)
            previously_here = [0]
            for t in arange(self.dims[0]):#31):#
                # t+=1
                previous_colony_id_matrix = colony_id_matrix.copy()
                # nb, shapes = connectedComponents(self.binary[t, :, :])
                # We rank colonies in increasing order to make sure that the larger colony issued from a colony division
                # keeps the previous colony name.
                shapes, stats, centers = cc(self.binary[t, :, :])
                # Consider that shapes bellow 3 pixels are noise. Given they are ranked by decreasing order,
                # the loop will stop at nb and not compute them
                nb = stats[stats[:, 4] >= self.vars["first_move_threshold"]].shape[0]
                current_percentage, eta = pat_tracker.get_progress(t, element_number=nb)
                logging.info(f"Arena n°{self.statistics['arena']}, Colony descriptors computation: {current_percentage}%{eta}")
                # logging.info(nb)
                updated_colony_names = zeros(1, dtype=uint64)
                for colony in (arange(nb - 1) + 1):#120)):# #92
                    # colony+=1
                    # logging.info(f'Colony number {colony}')
                    current_colony_img = (shapes == colony).astype(uint8)
                    # if current_colony_img.sum() > 3:
                    # nb, shapes, stats, centro = connectedComponentsWithStats(self.binary[t, :, :])
                    # logging.info(nb)
                    # if nb > 2:
                    #     break
                    # I/ Find out which names the current colony had at t-1
                    colony_previous_names = unique(current_colony_img * colony_id_matrix)
                    colony_previous_names = colony_previous_names[colony_previous_names != 0]
                    # II/ Find out if the current colony names had already been analyzed at t
                    # If there no match with the saved colony_id_matrix, its a new colony, lets add it
                    if len(colony_previous_names) == 0:
                        # logging.info("here is a new colony")
                        colony_number += 1
                        colony_names = array(colony_number)
                        colony_id_matrix[nonzero(current_colony_img)] = colony_names
                        # for descriptor in descriptors:
                        #     colony_dictionary[descriptor + "_colony" + str(colony_number)] = zeros(self.dims[0], dtype=float64)
                        if colony_number > time_descriptor_colony.shape[2]:
                            time_descriptor_colony = concatenate((time_descriptor_colony, zeros((self.dims[0], len(to_compute_from_sd), 100), dtype=float64)), axis=2)
                        self.statistics[f"appear_t_colony" + str(colony_number)] = t
                        nb, small_shape, centroids, stats = connectedComponentsWithStats(current_colony_img)
                        self.statistics[f"appear_x_colony" + str(colony_number)] = str(centroids[1, 0])
                        self.statistics[f"appear_y_colony" + str(colony_number)] = str(centroids[1, 1])

                    # If there is at least 1 match with the saved colony_id_matrix, we keep the colony_previous_names
                    elif len(colony_previous_names) > 0:
                        # logging.info("This colony already existed")
                        # The colony_previous_names becomes the current colony_names
                        colony_names = colony_previous_names
                        # logging.info(f'Do this colony was 3 or 4: {isin([3,4], colony_names)}')
                        # If any of the current colony names already are in updated_colony_names, it means that a
                        # consistent colony from t-1 split into several (at least 2) colonies at t.
                        result_from_colony_division: bool = False
                        for name in colony_names:
                            if isin(name, updated_colony_names):
                                result_from_colony_division = True
                            if result_from_colony_division:
                                break
                        if result_from_colony_division:
                            # logging.info("This colony emerged from the division of a parent colony")
                            # We must add it as a new colony
                            colony_number += 1
                            colony_names = array(colony_number)
                            colony_id_matrix[nonzero(current_colony_img)] = colony_names
                            # for descriptor in descriptors:
                            #     colony_dictionary[descriptor + "_colony" + str(colony_number)] = zeros(self.dims[0], dtype=float64)
                            if colony_number > time_descriptor_colony.shape[2]:
                                time_descriptor_colony = concatenate((time_descriptor_colony,
                                                                      zeros((self.dims[0], len(to_compute_from_sd), 100),
                                                                            dtype=float64)), axis=2)
                            self.statistics["division_t_colony" + str(colony_number)] = t
                            nb, small_shape, centroids, stats = connectedComponentsWithStats(current_colony_img)
                            self.statistics["division_x_colony" + str(colony_number)] = str(centroids[1, 0])
                            self.statistics["division_y_colony" + str(colony_number)] = str(centroids[1, 1])
                        else:
                            # for col_name in colony_names:
                            #     colony_id_matrix[colony_id_matrix == col_name] = 0
                            colony_id_matrix[nonzero(current_colony_img)] = colony_names[0]

                        # and we will update the descriptors of that(these) colony(ies) using that(these) name(s).
                        # If there is more than 1 match, all these saved colonies merged together
                        if len(colony_previous_names) > 1:
                            # logging.info("This colony emerged from the fusion of several parent colonies")
                            # If there is more than one match: two colonies fused.
                            # -> Update all these colonies and store when they fused
                            if colony_names.size == 1:
                                col_names = [colony_names]
                            else:
                                col_names = colony_names
                            for colony_name in col_names:
                                self.statistics["fusion_t_colony" + str(colony_name)] = t
                                nb, small_shape, centroids, stats = connectedComponentsWithStats(current_colony_img)
                                self.statistics["fusion_x_colony" + str(colony_number)] = str(centroids[1, 0])
                                self.statistics["fusion_y_colony" + str(colony_number)] = str(centroids[1, 1])

                    updated_colony_names = append(updated_colony_names, colony_names)
                    # Compute the current colony descriptors
                    SD = ShapeDescriptors(current_colony_img, to_compute_from_sd)
                    if self.vars['output_in_mm']:
                        if isin('area', to_compute_from_sd):
                            SD.descriptors['area'] *= self.vars['average_pixel_size']
                        if isin('total_hole_area', to_compute_from_sd):
                            SD.descriptors['total_hole_area'] *= self.vars['average_pixel_size']
                        if isin('perimeter', to_compute_from_sd):
                            SD.descriptors['perimeter'] *= sqrt(self.vars['average_pixel_size'])
                        if isin('major_axis_len', to_compute_from_sd):
                            SD.descriptors['major_axis_len'] *= sqrt(self.vars['average_pixel_size'])
                        if isin('minor_axis_len', to_compute_from_sd):
                            SD.descriptors['minor_axis_len'] *= sqrt(self.vars['average_pixel_size'])
                    # Save all colony descriptors in the time_descriptor_colony array:
                    # if the current_colony has several colony names, repeat descriptors in each colony column
                    time_descriptor_colony[t, :, (colony_names - 1)] = tile(list(SD.descriptors.values()), (colony_names.size, 1))

                # if any(colony_id_matrix == 3) | any(colony_id_matrix==4):
                #     logging.info(f'here t = {t} and c = {colony}')
                colony_id_matrix *= self.binary[t, :, :]
                currently_here = unique(colony_id_matrix)
                disappeared_colonies = ~(isin(previously_here, currently_here))
                if any(disappeared_colonies):
                    disappeared_colonies = previously_here[disappeared_colonies]
                    for disappeared_colony in disappeared_colonies:
                        self.statistics[f"disappear_t_colony" + str(colony_number)] = t - 1
                        nb, small_shape, centroids, stats = connectedComponentsWithStats(
                            (previous_colony_id_matrix == disappeared_colony).astype(uint8))
                        self.statistics[f"disappear_x_colony" + str(colony_number)] = str(centroids[1, 0])
                        self.statistics[f"disappear_y_colony" + str(colony_number)] = str(centroids[1, 1])
                previously_here = currently_here.copy()


            # Format the final dataframe to have one row per time frame, and one column per descriptor_colony_name
            self.whole_shape_descriptors = df()
            self.whole_shape_descriptors['arena'] = [self.statistics['arena']] * self.dims[0]
            self.whole_shape_descriptors['time'] = timings
            self.whole_shape_descriptors['area_total'] = self.surfarea.astype(float64)
            if self.vars['output_in_mm']:
                self.whole_shape_descriptors['area_total'] *= self.vars['average_pixel_size']
            time_descriptor_colony = time_descriptor_colony[:, :, :colony_number]
            time_descriptor_colony = concatenate(stack(time_descriptor_colony, axis=1), axis=1)
            column_names = char.add(repeat(to_compute_from_sd, colony_number), tile(arange(colony_number) + 1, len(to_compute_from_sd)).astype(str))
            time_descriptor_colony = df(time_descriptor_colony, columns=column_names)
            self.whole_shape_descriptors = self.whole_shape_descriptors.join(time_descriptor_colony)
            """


        if self.vars['do_fading']:
            self.whole_shape_descriptors['newly_explored_area'] = self.newly_explored_area
            if self.vars['output_in_mm']:
                self.whole_shape_descriptors['newly_explored_area'] *= self.vars['average_pixel_size']
                # for colony_name in colony_names:
            #     for descriptor in descriptors:
            #         colony_dictionary[descriptor + "_colony" + str(colony_name)][t] = SD.descriptors[descriptor]
            #         if self.vars['output_in_mm']:
            #             if descriptor == 'area':
            #                 colony_dictionary[descriptor + "_colony" + str(colony_name)][t] *= self.vars['average_pixel_size']
            #             elif isin(descriptor, ['perimeter', 'major_axis_len', 'minor_axis_len']):
            #                 colony_dictionary[descriptor + "_colony" + str(colony_name)][t] *= sqrt(self.vars['average_pixel_size'])
            """Always working algo
                If one colony become two or more : Create new colomns for each colony and store when they split
                IF two or more colonies become one : update all these colonies and store when they fused
            """

            """ I/ with_unexpected_motion_algorithm """
            """ II/ no_unexpected_motion_algorithm """
            """ If this colony don't match any previously identified colony: 
                                        save it as a new one
                                    """
            """ If this colony match only one previously identified colony: 
                Update the corresponding colony features
                pb: if one (previous) becomes two (the previous that moved and another that moved at the place of the previous) 
                because of brownian motion / image drift
                --> One colony information will be lost
            """
            """ If this colony match more than one previously identified colony: 
                Update each of these with the current colony features
                pb: if one (previous) becomes two (new) because of brownian motion / image drift
                --> This may identify merging of two colonies that never merged
            """
            """ Solution:
                Add a user implemented parameters: 
                if there is no unexpected motion : apply the no_unexpected_motion_algorithm
                else : apply the with_unexpected_motion_algorithm
            """

            # Fill the whole_shape_descriptors table
            # self.whole_shape_descriptors = df.from_dict(colony_dictionary)

    def detect_growth_transitions(self):
        ##
        if self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena']:
            self.statistics["iso_digi_transi"] = NA
            if not isna(self.statistics["first_move"]):
                logging.info(f"Arena n°{self.statistics['arena']}. Starting growth transition analysis.")

                # II) Once a pseudopod is deployed, look for a disk/ around the original shape
                growth_begining = self.surfarea < ((self.surfarea[0] * 1.2) + ((self.dims[1] / 4) * (self.dims[2] / 4)))
                dilated_origin = dilate(self.binary[self.statistics["first_move"], :, :], kernel=cross_33, iterations=10, borderType=BORDER_CONSTANT, borderValue=0)
                isisotropic = sum(self.binary[:, :, :] * dilated_origin, (1, 2))
                isisotropic *= growth_begining
                # Ask if the dilated origin area is 90% covered during the growth beginning
                isisotropic = isisotropic > 0.9 * dilated_origin.sum()
                if any(isisotropic):
                    self.statistics["is_growth_isotropic"] = 1
                    # Determine a solidity reference to look for a potential breaking of the isotropic growth
                    if self.compute_solidity_separately:
                        solidity_reference = mean(self.solidity[:self.statistics["first_move"]])
                        different_solidity = self.solidity < (0.9 * solidity_reference)
                    else:
                        solidity_reference = mean(
                            self.whole_shape_descriptors.iloc[:(self.statistics["first_move"]), :]["solidity"])
                        different_solidity = self.whole_shape_descriptors["solidity"].values < (0.9 * solidity_reference)
                    # Make sure that isotropic breaking not occur before isotropic growth
                    if any(different_solidity):
                        self.statistics["iso_digi_transi"] = nonzero(different_solidity)[0][0] * self.time_interval
                else:
                    self.statistics["is_growth_isotropic"] = 0
            else:
                self.statistics["is_growth_isotropic"] = NA

            """
            if logical_or(self.statistics["iso_digi_transi"] != 0, overlap.sum() > ):
                self.statistics["is_growth_isotropic"] = True
            # [aim: isotropic phase occurrence]

            growth_begining = self.surfarea < ((self.surfarea[0] * 1.2) + ((self.dims[1] / 4) * (self.dims[2] / 4)))


            # Find moments of the first growth phase that has a solidity similar to the one at the beginning
            solid_growth = logical_and(growth_begining, similar_solidity)
            # 
            if any(logical_not(solid_growth)):
                self.statistics["iso_digi_transi"] = sum(solid_growth)



            best_isotropic_candidate = self.binary[solid_growth, :, :][-1]
            # Use the largest side of the bounding box as a reference to calculate how we should iterate dilatation
            self.whole_shape_descriptors[solid_growth, :]
            dilated_origin = dilate(self.shape_evolution.binary, kernel=cross_33, iterations=10,
                                        borderType=BORDER_CONSTANT, borderValue=0)
            if sum(best_isotropic_candidate * dilated_origin) > 0.95 * dilated_origin.sum():
                self.isotropic_phase = True
                self.isotropic_start = nonzero(solid_growth)[0][-1]

            nonzero(self.whole_shape_descriptors[t, 4] > (0.6 * self.whole_shape_descriptors[0, 4]))[0]
            for t in arange(self.shape_evolution.start, self.shape_evolution.binary.shape[2]):

                new_shapes = self.shape_evolution.binary[:, :, t] - origin
                # if the newly added area is inferior to the original suface
                if sum(new_shapes) < sum(origin):
                    # If the solidity of the current shape is similar to the one at the beginning
                    if self.whole_shape_descriptors[t, 4] > (0.6 * self.whole_shape_descriptors[0, 4]):
                        SD = ShapeDescriptors(new_shapes, ["perimeter"])
                        # If the perimeter of the added shape is superior to the perimeter of the shape at the beginning
                        if SD.results[0] > self.whole_shape_descriptors[t, 0]:  # will require to be more specific
                            self.isotropic_phase = True
                            self.isotropic_start = t

            # III) Look for solidity changes showing the breaking of isotropic growth [aim: time]
            if self.isotropic_phase:
                self.isotropic_breaking = \
                nonzero(self.whole_shape_descriptors[:, 4] < (0.8 * self.whole_shape_descriptors[0, 4]))[0]
            # IV) Isolate secondary pseudopods and save their shape descriptors just before some of them change
            # drastically
            previous_t = 0
            for t in arange(self.shape_evolution.start, self.shape_evolution.binary.shape[2]):
                if self.whole_shape_descriptors[t, 4] < 0.9 * self.whole_shape_descriptors[previous_t, 4]:
                    added_material = self.shape_evolution.binary[:, :, t] - self.shape_evolution.binary[:, :,
                                                                            previous_t]
                    new_order, stats, centers = cc(added_material)
                    pseudopods = nonzero(
                        stats[:, 4][:-1] > origin_info.first_move_threshold)  # Will need to be more specific
                    for pseu in pseudopods:
                        only_pseu = zeros(self.shape_evolution.binary.shape[:2], dtype=uint8)
                        only_pseu[output == pseu] = 1
                        SD = ShapeDescriptors(only_pseu, ro.descriptors_list)
                        SD.results
            # I) Find a first pseudopod [aim: time]
            # II) If a pseudopod is deployed, look for a disk/ around the original shape
            # [aim: isotropic phase occurrence]
            # III) Look for solidity changes showing the breaking of isotropic growth [aim: time]
            # IV) Isolate secondary pseudopods and save their shape descriptors just before some of them change
            # drastically
        # https://pythonexamples.org/python-opencv-write-text-on-image-puttext/
            """

    def check_converted_video_type(self):
        if self.converted_video.dtype != "uint8":
            self.converted_video -= min(self.converted_video)
            self.converted_video = round((255 * (self.converted_video / max(self.converted_video)))).astype(uint8)


    def networks_detection(self, show_seg=False):
        if not isna(self.statistics["first_move"]) and not self.vars['several_blob_per_arena'] and self.vars['network_detection']:
            logging.info(f"Arena n°{self.statistics['arena']}. Starting network detection.")
            # self.network_dynamics = load(f"coord_tubular_network{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy")
            # if self.vars['origin_state'] == 'constant':
            #     origin = (1 - self.origin)
            # else:
            #     origin = ones(self.dims[1:], dtype=uint8)
            self.network_dynamics = zeros(self.dims, dtype=uint8)
            self.graph = zeros(self.dims, dtype=uint8)
            # if len(self.converted_video.shape) == 3:
            #     self.converted_video = stack((self.converted_video, self.converted_video, self.converted_video), axis=3)
            self.check_converted_video_type()
            # self.converted_video = bracket_to_uint8_image_contrast(self.converted_video)
            self.covering_intensity = zeros(self.dims[1:], dtype=float64)
            if self.vars['origin_state'] == "fluctuating":
                self.covering_intensity = self.origin * self.converted_video[0, :, :]
            if self.vars['origin_state'] == "constant":
                if self.origin_idx is None:
                    self.origin_idx = nonzero(self.origin)
                if self.vars['lighter_background']:
                    self.covering_intensity[self.origin_idx[0], self.origin_idx[1]] = 200

            if self.vars['fractal_analysis']:
                box_counting_dimensions = zeros((self.dims[0], 7), dtype=float64)


            for t in arange(self.statistics["first_move"], self.dims[0]):  #20):#
                # t = 10
                # t = 200
                if any(self.binary[t, ...]):
                    nd = NetworkDetection(self.converted_video[t, ...], self.binary[t, ...], self.vars['lighter_background'])

                    nd.segment_locally(side_length=self.vars['network_mesh_side_length'],
                                       step=self.vars['network_mesh_step_length'],
                                       int_variation_thresh=self.vars['network_detection_threshold'])
                    # See(nd.binary_image)
                    # See(nd.network)
                    nd.selects_elongated_or_holed_shapes(hole_surface_ratio=0.1, eccentricity=0.65)
                    # nd.segment_globally()

                    if t > 0:
                        nd.add_pixels_to_the_network(nonzero(self.network_dynamics[t - 1, ...]))
                        if self.vars['do_fading'] and (t > self.step + self.lost_frames):
                            # Remove the faded areas
                            remove_faded_pixels = 1 - self.binary[t - 1, ...]
                            nd.remove_pixels_from_the_network(nonzero(remove_faded_pixels), remove_homogeneities=True)

                    # Add the origin as a part of the network:
                    if self.vars['origin_state'] == 'constant':
                        nd.add_pixels_to_the_network(self.origin_idx)

                    # Connect all network pieces together:
                    nd.connect_network(maximal_distance_connection=self.max_distance)  # int(4*max(self.dims[1:3])//5))

                    self.network_dynamics[t, ...] = nd.network

                    nd.skeletonize()
                    nd.get_graph()
                    self.graph[t, ...] = nd.graph.copy()

                    if self.vars['fractal_analysis']:
                        box_counting_dimensions[t, 0] = self.network_dynamics[t, ...].sum()
                        box_counting_dimensions[t, 1], box_counting_dimensions[t, 2], box_counting_dimensions[t, 3] = box_counting(self.binary[t, ...])
                        box_counting_dimensions[t, 4], box_counting_dimensions[t, 5], box_counting_dimensions[t, 6] = box_counting(self.network_dynamics[t, ...])

                    # #### Houssam ####
                    # segments = nd.find_segments(labeled_nodes, label_to_position)
                    # node_degrees = nd.extract_node_degrees(segments)
                    # # largeurs = nd.get_segment_width(self.binary[t, ...], segments, distance_map)
                    # #### Houssam ####

                    imtoshow = self.visu[t, ...].copy()
                    net_coord = nonzero(morphologyEx(self.network_dynamics[t, ...], MORPH_GRADIENT, cross_33))
                    imtoshow[net_coord[0], net_coord[1], :] = (34, 34, 158)

                    # Remember to uncomment this when done!!! For visualization.
                    if show_seg:
                        imshow("", resize(imtoshow, (1000, 1000)))
                        waitKey(1)
                    else:
                        self.visu[t, ...] = imtoshow.copy()
            if show_seg:
                destroyAllWindows()

            if self.vars['fractal_analysis']:
                box_counting_dimensions = box_counting_dimensions[1:, :]
                self.whole_shape_descriptors["inner_network_size"] = box_counting_dimensions[:, 0]
                self.whole_shape_descriptors["fractal_dimension"] = box_counting_dimensions[:, 1]
                self.whole_shape_descriptors["fractal_r_value"] = box_counting_dimensions[:, 2]
                self.whole_shape_descriptors["fractal_box_nb"] = box_counting_dimensions[:, 3]
                self.whole_shape_descriptors["inner_network_fractal_dimension"] = box_counting_dimensions[:, 4]
                self.whole_shape_descriptors["inner_network_fractal_r_value"] = box_counting_dimensions[:, 5]
                self.whole_shape_descriptors["inner_network_fractal_box_nb"] = box_counting_dimensions[:, 6]
                if self.vars['output_in_mm']:
                    self.whole_shape_descriptors["inner_network_size"] *= self.vars['average_pixel_size']
                # box_counting_dimensions = df(box_counting_dimensions, columns=["inner_network_size", "fractal_dimension", "fractal_r_value", "fractal_box_nb", "inner_network_fractal_dimension", "inner_network_fractal_r_value", "inner_network_fractal_box_nb"])
                # box_counting_dimensions.to_csv(f"box_counting_dimensions{self.statistics['arena']}.csv", sep=';', index=False, lineterminator='\n')

            self.network_dynamics = smallest_memory_array(nonzero(self.network_dynamics), "uint")
            edges = smallest_memory_array(nonzero(self.graph == 1), "uint")
            vertices = smallest_memory_array(nonzero(self.graph == 2), "uint")
            if self.vars['save_binary_masks']:
                save(f"coord_tubular_network{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy", self.network_dynamics)
                save(f"coord_network_edges{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy", edges)
                save(f"coord_network_vertices{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy", vertices)

            del self.graph

            # save(f"coord_network{self.statistics['arena']}.npy", self.network_dynamics)
            # del self.network_dynamics
            # save(f"coord_binary{self.statistics['arena']}.npy", smallest_memory_array(nonzero(self.binary), "uint"))

            # self.network_dynamics += self.binary
            # save(f"video_of_network{self.statistics['arena']}.npy", self.network_dynamics)
                # self.network_dynamics = nonzero(self.network_dynamics)
                # d = {}
                # d["time_coord"] = self.network_dynamics[0]
                # d["y_coord"] = self.network_dynamics[1]
                # d["x_coord"] = self.network_dynamics[2]
                # PickleRick().write_file(d, f"network_dynamics{self.statistics['arena']}.pkl")
                #
                # self.network_dynamics = nonzero(self.binary)
                # d = {}
                # d["time_coord"] = self.network_dynamics[0]
                # d["y_coord"] = self.network_dynamics[1]
                # d["x_coord"] = self.network_dynamics[2]
                # PickleRick().write_file(d, f"plasmodia_dynamics{self.statistics['arena']}.pkl")

    def memory_allocation_for_cytoscillations(self):
        try:
            period_in_frame_nb = int(self.vars['oscillation_period'] / self.time_interval)
            if period_in_frame_nb < 2:
                period_in_frame_nb = 2
            necessary_memory = self.converted_video.shape[0] * self.converted_video.shape[1] * \
                               self.converted_video.shape[2] * 64 * 4 * 1.16415e-10
            available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
            if len(self.converted_video.shape) == 4:
                self.converted_video = self.converted_video[:, :, :, 0]
            average_intensities = mean(self.converted_video, (1, 2))
            if self.vars['lose_accuracy_to_save_memory'] or (necessary_memory > available_memory):
                oscillations_video = zeros(self.converted_video.shape, dtype=float16)
                for cy in arange(self.converted_video.shape[1]):
                    for cx in arange(self.converted_video.shape[2]):
                        oscillations_video[:, cy, cx] = round(gradient(self.converted_video[:, cy, cx, ...]/average_intensities,
                                                                      period_in_frame_nb), 3).astype(float16)
                        # oscillations_video[:, cy, cx] = round(gradient(self.converted_video[:, cy, cx].astype(int16),
                        #                                               period_in_frame_nb), 3).astype(float16)
            else:
                oscillations_video = gradient(self.converted_video/average_intensities, period_in_frame_nb, axis=0)
                # oscillations_video = gradient(self.converted_video.astype(int16), period_in_frame_nb, axis=0)
            # check if conv change here
            # if not self.vars['lose_accuracy_to_save_memory']:
            self.check_converted_video_type()
            if len(self.converted_video.shape) == 3:
                self.converted_video = stack((self.converted_video, self.converted_video, self.converted_video), axis=3)
            # self.cytoscillations = zeros(self.dims, dtype=uint8)
            oscillations_video = sign(oscillations_video)
            return oscillations_video
        except Exception as exc:
            logging.error(f"{exc}. Retrying to allocate for 10 minutes before crashing. ")
            return None


    def study_cytoscillations(self, show_seg):
        if not isna(self.statistics["first_move"]) and self.vars['oscilacyto_analysis']:
            logging.info(f"Arena n°{self.statistics['arena']}. Starting oscillation analysis.")
            oscillations_video = None
            staring_time = default_timer()
            current_time = staring_time
            while oscillations_video is None and (current_time - staring_time) < 600:
                oscillations_video = self.memory_allocation_for_cytoscillations()
                if oscillations_video is None:
                    sleep(30)
                    current_time = default_timer()

            mean_cluster_area = zeros(oscillations_video.shape[0])
            cluster_number = zeros(oscillations_video.shape[0])
            named_cluster_number = 0
            dotted_image = ones(self.converted_video.shape[1:3], uint8)
            for cy in arange(dotted_image.shape[0]):
                if cy % 2 != 0:
                    dotted_image[cy, :] = 0
            for cx in arange(dotted_image.shape[1]):
                if cx % 2 != 0:
                    dotted_image[:, cx] = 0
            within_range = (1 - self.binary[0, :, :]) * self.borders

            # To get the median oscillatory period of each oscillating cluster,
            # we create a dict containing two lists (for influx and efflux)
            # Each list element correspond to a cluster and stores :
            # All pixel coordinates of that cluster, their corresponding lifespan, their time of disappearing
            # Row number will give the size. Euclidean distance between pix coord, the wave distance
            self.clusters_final_data = empty((0, 6), dtype=float32)# ["mean_pixel_period", "phase", "total_size", "edge_distance", cy, cx]
            # self.clusters_final_data = empty((0, 4), dtype=float32)# ["mean_pixel_period", "phase", "total_size", "edge_distance"]
            period_tracking = zeros(self.converted_video.shape[1:3], dtype=uint32)
            efflux_study = ClusterFluxStudy(self.converted_video.shape[:3])
            influx_study = ClusterFluxStudy(self.converted_video.shape[:3])
            if self.start is None:
                self.start = 0

            # New analysis to get the surface dynamic of every oscillatory cluster: Part 1 openning
            # max_clusters = 10000000
            if self.vars['fractal_analysis']:
                oscillating_clusters_temporal_dynamics = zeros(13,  dtype=float64)  # time, cluster_id, flow, centroid_y, centroid_x, area, inner_network_area, box_count_dim, inner_network_box_count_dim

            cluster_id_matrix = zeros(self.dims[1:], dtype=uint64)
            # New analysis to get the surface dynamic of every oscillatory cluster: Part 1 ending
            pat_tracker = PercentAndTimeTracker(self.dims[0], compute_with_elements_number=True)
            for t in arange(self.dims[0]):#arange(60): #
                contours = morphologyEx(self.binary[t, :, :], MORPH_GRADIENT, cross_33)
                contours_idx = nonzero(contours)
                imtoshow = self.converted_video[t, ...].copy()
                imtoshow[contours_idx[0], contours_idx[1], :] = self.vars['contour_color']
                if self.vars['iso_digi_analysis'] and not self.vars['several_blob_per_arena'] and not isna(self.statistics["iso_digi_transi"]):
                    if self.statistics["is_growth_isotropic"] == 1:
                        if t < self.statistics["iso_digi_transi"]:
                            imtoshow[contours_idx[0], contours_idx[1], 2] = 255
                oscillations_image = zeros(self.dims[1:], uint8)
                if t >= self.start:
                    # Add in or ef if a pixel has at least 4 neighbor in or ef
                    neigh_comp = CompareNeighborsWithValue(oscillations_video[t, :, :], connectivity=8, data_type=int8)
                    neigh_comp.is_inf(0, and_itself=False)
                    neigh_comp.is_sup(0, and_itself=False)
                    # Not verified if influx is really influx (resp efflux)
                    influx = neigh_comp.sup_neighbor_nb * self.binary[t, :, :] * within_range
                    efflux = neigh_comp.inf_neighbor_nb * self.binary[t, :, :] * within_range
                    # Only keep pixels having at least 4 positive (resp. negative) neighbors
                    influx[influx <= 4] = 0
                    efflux[efflux <= 4] = 0
                    influx[influx > 4] = 1
                    efflux[efflux > 4] = 1

                    influx, in_stats, in_centroids = cc(influx)
                    efflux, ef_stats, ef_centroids = cc(efflux)
                    # Only keep clusters larger than 'minimal_oscillating_cluster_size' pixels (smaller are considered as noise
                    in_smalls = nonzero(in_stats[:, 4] < self.vars['minimal_oscillating_cluster_size'])[0]
                    if len(in_smalls) > 0:
                        influx[isin(influx, in_smalls)] = 0
                        in_stats = in_stats[:in_smalls[0], :]
                        in_centroids = in_centroids[:in_smalls[0], :]
                    in_stats = in_stats[1:]
                    in_centroids = in_centroids[1:]
                    ef_smalls = nonzero(ef_stats[:, 4] < self.vars['minimal_oscillating_cluster_size'])[0]
                    if len(ef_smalls) > 0:
                        efflux[isin(efflux, ef_smalls)] = 0
                        ef_stats = ef_stats[:(ef_smalls[0]), :]
                        ef_centroids = ef_centroids[:(ef_smalls[0]), :]
                    ef_stats = ef_stats[1:]
                    ef_centroids = ef_centroids[1:]

                    in_idx = nonzero(influx)  # NEW
                    ef_idx = nonzero(efflux)  # NEW
                    oscillations_image[in_idx[0], in_idx[1]] = 1  # NEW
                    oscillations_image[ef_idx[0], ef_idx[1]] = 2  # NEW

                    if t > self.lost_frames:
                        # Sum the number of connected components minus the background to get the number of clusters
                        cluster_number[t] = in_stats.shape[0] + ef_stats.shape[0]
                        updated_cluster_names = [0]
                        if cluster_number[t] > 0:
                            current_percentage, eta = pat_tracker.get_progress(t, element_number=cluster_number[t])
                            logging.info(f"Arena n°{self.statistics['arena']}, Oscillatory cluster computation: {current_percentage}%{eta}")
                            if self.vars['fractal_analysis']:
                                # New analysis to get the surface dynamic of every oscillatory cluster: Part 2 openning:
                                network_at_t = zeros(self.dims[1:], dtype=uint8)
                                network_idx = self.network_dynamics[:, self.network_dynamics[0, :] == t]
                                network_at_t[network_idx[1, :], network_idx[2, :]] = 1
                                shapes = zeros(self.dims[1:], dtype=uint32)
                                shapes[in_idx[0], in_idx[1]] = influx[in_idx[0], in_idx[1]]
                                max_in = in_stats.shape[0]
                                shapes[ef_idx[0], ef_idx[1]] = max_in + efflux[ef_idx[0], ef_idx[1]]
                                centers = vstack((in_centroids, ef_centroids))
                                # shapes, stats, centers = cc(oscillations_image)
                                for cluster in (arange(cluster_number[t] - 1, dtype=uint32) + 1):  # 120)):# #92
                                    # cluster = 1
                                    # print(cluster)
                                    current_cluster_img = (shapes == cluster).astype(uint8)
                                    # I/ Find out which names the current cluster had at t-1
                                    cluster_previous_names = unique(current_cluster_img * cluster_id_matrix)
                                    cluster_previous_names = cluster_previous_names[cluster_previous_names != 0]
                                    # II/ Find out if the current cluster name had already been analyzed at t
                                    # If there no match with the saved cluster_id_matrix, assign cluster ID
                                    if t == 0 or len(cluster_previous_names) == 0:
                                        # logging.info("New cluster")
                                        named_cluster_number += 1
                                        cluster_names = [named_cluster_number]
                                    # If there is at least 1 match with the saved cluster_id_matrix, we keep the cluster_previous_name(s)
                                    else:
                                        cluster_names = cluster_previous_names.tolist()
                                    # Handle cluster division if necessary
                                    if any(isin(updated_cluster_names, cluster_names)):
                                        named_cluster_number += 1
                                        cluster_names = [named_cluster_number]

                                    # Get flow direction:
                                    if unique(oscillations_image * current_cluster_img)[1] == 1:
                                        flow = 1
                                    else:
                                        flow = - 1
                                    # Update cluster ID matrix for the current frame
                                    coords = nonzero(current_cluster_img)
                                    cluster_id_matrix[coords[0], coords[1]] = cluster_names[0]

                                    # Save the current cluster areas:
                                    inner_network = current_cluster_img * network_at_t
                                    inner_network_area = inner_network.sum()
                                    box_count_dim, r_value, box_nb = box_counting(current_cluster_img)
                                    if any(inner_network):
                                        inner_network_box_count_dim, inner_net_r_value, inner_net_box_nb = box_counting(inner_network)
                                    else:
                                        inner_network_box_count_dim, inner_net_r_value, inner_net_box_nb = 0, 0, 0
                                    # Calculate centroid and add to centroids list
                                    centroid_x, centroid_y = centers[cluster, :]
                                    curr_tempo_dyn = array((t, cluster_names[0], flow, centroid_y, centroid_x, current_cluster_img.sum(), inner_network_area, box_count_dim, r_value, box_nb, inner_network_box_count_dim, inner_net_r_value, inner_net_box_nb), dtype=float64)
                                    # time, cluster_id, flow, centroid_y, centroid_x, area, inner_network_area, box_counting_dimension

                                    oscillating_clusters_temporal_dynamics = vstack((oscillating_clusters_temporal_dynamics, curr_tempo_dyn))

                                    updated_cluster_names = append(updated_cluster_names, cluster_names)
                                # Reset cluster_id_matrix for the next frame
                                cluster_id_matrix *= self.binary[t, :, :]

                            # # Mettre dans return
                            # pixels_lost_from_flux = pixels_lost_from_efflux
                            # clusters_id = efflux_clusters_id
                            # alive_clusters_in_flux = alive_clusters_in_efflux
                            # cluster_total_number = cluster_total_number_efflux
                            period_tracking, self.clusters_final_data = efflux_study.update_flux(t, contours, efflux, period_tracking, self.clusters_final_data)
                            period_tracking, self.clusters_final_data = influx_study.update_flux(t, contours, influx, period_tracking, self.clusters_final_data)
                            # See(period_tracking.astype(uint8) * 50)
                            # See(influx.astype(uint8) * 50)

                            # Get standardized area
                            # area_standardization = (sum(self.binary[t, :, :]).astype(int64) - sum(self.binary[0, :, :]).astype(int64))
                            mean_cluster_area[t] = mean(concatenate((in_stats[:, 4], ef_stats[:, 4])))

                            # Prepare the image for display
                            in_idx = influx.copy()
                            ef_idx = efflux.copy()
                            in_idx *= dotted_image
                            ef_idx *= dotted_image
                            in_idx = nonzero(in_idx)
                            ef_idx = nonzero(ef_idx)
                            imtoshow[in_idx[0], in_idx[1], :2] = 153  # Green: influx, intensity increase
                            imtoshow[in_idx[0], in_idx[1], 2] = 0
                            imtoshow[ef_idx[0], ef_idx[1], 1:] = 0  # Blue: efflux, intensity decrease
                            imtoshow[ef_idx[0], ef_idx[1], 0] = 204

                oscillations_video[t, :, :] = dcopy(oscillations_image)
                self.converted_video[t, ...] = dcopy(imtoshow)
                if show_seg:
                    imtoshow = resize(imtoshow, (540, 540))
                    imshow("shape_motion", imtoshow)
                    waitKey(1)


            if self.vars['fractal_analysis']:
                # coord_cluster = vstack(coord_cluster)
                oscillating_clusters_temporal_dynamics = oscillating_clusters_temporal_dynamics[1:, :]
                if self.vars['output_in_mm']:
                    oscillating_clusters_temporal_dynamics[:, 0] *= self.time_interval # phase
                    oscillating_clusters_temporal_dynamics[:, 5] *= self.vars['average_pixel_size']  # size
                    oscillating_clusters_temporal_dynamics[:, 6] *= self.vars['average_pixel_size']  # size
                oscillating_clusters_temporal_dynamics = df(oscillating_clusters_temporal_dynamics, columns=["time", "cluster_id", "flow", "centroid_y", "centroid_x", "area", "inner_network_area", "box_counting_dimension", "r_value", "box_nb", "inner_network_box_counting_dimension", "inner_net_r_value", "inner_net_box_nb"])
                oscillating_clusters_temporal_dynamics.to_csv(f"oscillating_clusters_temporal_dynamics{self.statistics['arena']}.csv", sep=';', index=False, lineterminator='\n')


            if self.vars['output_in_mm']:
                self.clusters_final_data[:, 1] *= self.time_interval # phase
                self.clusters_final_data[:, 2] *= self.vars['average_pixel_size']  # size
                self.clusters_final_data[:, 3] *= sqrt(self.vars['average_pixel_size'])  # distance
                self.whole_shape_descriptors['mean_cluster_area'] = mean_cluster_area * self.vars['average_pixel_size']
            self.whole_shape_descriptors['cluster_number'] = named_cluster_number

            if self.vars['save_binary_masks']:
                save(f"coord_thickening{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy", smallest_memory_array(nonzero(oscillations_video == 1), "uint"))
                save(f"coord_slimming{self.statistics['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.npy", smallest_memory_array(nonzero(oscillations_video == 2), "uint"))
            # save(f"coord_influx{self.statistics['arena']}.npy", smallest_memory_array(nonzero(oscillations_video == 1), "uint"))  # NEW
            # save(f"coord_efflux{self.statistics['arena']}.npy", smallest_memory_array(nonzero(oscillations_video == 2), "uint"))  # NEW
            # del self.cytoscillations
            del oscillations_video

        else:
            if not self.vars['lose_accuracy_to_save_memory']:
                self.converted_video -= min(self.converted_video)
                self.converted_video = 255 * (self.converted_video / max(self.converted_video))
                self.converted_video = round(self.converted_video).astype(uint8)
        if isna(self.statistics["first_move"]):
            # self.magnitudes_and_frequencies = [0, 0]
            if self.vars['oscilacyto_analysis']:
                self.whole_shape_descriptors['mean_cluster_area'] = NA
                self.whole_shape_descriptors['cluster_number'] = NA
                # self.statistics["max_magnitude"] = 'NA'
                # self.statistics["frequency_of_max_magnitude"] = 'NA'

    def fractal_descriptions(self):
        if not isna(self.statistics["first_move"]) and self.vars['fractal_analysis']:
            logging.info(f"Arena n°{self.statistics['arena']}. Starting fractal analysis.")

            if not self.vars['network_detection']:
                box_counting_dimensions = zeros((self.dims[0], 3), dtype=float64)
                for t in arange(self.dims[0]):
                    box_counting_dimensions[t, :] = box_counting(self.binary[t, ...])
                box_counting_dimensions = box_counting_dimensions[1:]
                box_counting_dimensions = df(box_counting_dimensions, columns=["dimension"])
                box_counting_dimensions.to_csv(f"box_counting_dimensions{self.statistics['arena']}.csv", sep=';',
                                               index=False, lineterminator='\n')
                self.whole_shape_descriptors["fractal_dimension"] = box_counting_dimensions[:, 0]
                self.whole_shape_descriptors["fractal_box_nb"] = box_counting_dimensions[:, 1]
                self.whole_shape_descriptors["fractal_r_value"] = box_counting_dimensions[:, 2]


            """
            if self.visu is None:
                true_frame_width = self.origin.shape[1]
                if len(self.vars['background_list']) == 0:
                    self.background = None
                else:
                    self.background = self.vars['background_list'][self.statistics['arena'] - 1]
                self.visu = video2numpy(f"ind_{self.statistics['arena']}.npy", None, self.background, true_frame_width)
                if len(self.visu.shape) == 3:
                    self.visu = stack((self.visu, self.visu, self.visu), axis=3)

            self.fractal_boxes = empty((0, 4), dtype=float64) # distance
            # im_test = zeros(self.binary[0, ...].shape, dtype=uint8)
            # import cv2
            # cv2.imwrite("bin_im_test0.jpg", im_test); cv2.imwrite("bin_im_test1.jpg", self.binary[0, ...]);cv2.imwrite("bin_im_test2.jpg", self.binary[100, ...]);cv2.imwrite("bin_im_test3.jpg", self.binary[-1, ...])
            # cv2.imwrite("im_test0.jpg", self.visu[0, ...]); cv2.imwrite("im_test1.jpg", self.visu[0, ...]);cv2.imwrite("im_test2.jpg", self.visu[100, ...]);cv2.imwrite("im_test3.jpg", self.visu[-1, ...])

            for t in arange(self.dims[0]):
                # t = 100; self.binary_image = self.binary[t, ...]

                fractan = FractalAnalysis(self.binary[t, ...])
                fractan.detect_fractal(threshold=self.vars['fractal_threshold_detection'])
                fractan.extract_fractal(self.visu[t, ...])
                fractan.get_dimension()
                self.whole_shape_descriptors.loc[t, 'minkowski_dimension'] = fractan.minkowski_dimension
                # self.whole_shape_descriptors['minkowski_dimension'].iloc[t] = fractan.minkowski_dimension
                self.fractal_boxes = concatenate((self.fractal_boxes,
                                            c_[repeat(self.statistics['arena'], len(fractan.fractal_box_widths)),
                                            repeat(t, len(fractan.fractal_box_widths)), fractan.fractal_box_lengths,
                                            fractan.fractal_box_widths]), axis=0)

            if self.vars['output_in_mm']:
                self.fractal_boxes[:, 2:] *= sqrt(self.vars['average_pixel_size'])
            # Save an illustration of the fractals of the last image
            fractan.save_fractal_mesh(f"last_image_fractal_mesh{self.statistics['arena']}.tif")
            """

    def get_descriptors_summary(self):
        potential_descriptors = ["area", "perimeter", "circularity", "rectangularity", "total_hole_area", "solidity",
                                 "convexity", "eccentricity", "euler_number", "standard_deviation_y",
                                 "standard_deviation_x", "skewness_y", "skewness_x", "kurtosis_y", "kurtosis_x",
                                 "major_axis_len", "minor_axis_len", "axes_orientation"]

        self.statistics["final_area"] = self.binary[-1, :, :].sum()


        # for k, v in self.vars['descriptors'].items():
        #     if isin(k, potential_descriptors):
        #         # if machin isin potential_descriptors
        #         if v[1]: #Calculate mean and sd from first move
        #             if self.statistics["first_move"] != 'NA':
        #                 self.statistics[f'{k}_mean'] = \
        #                 self.whole_shape_descriptors.iloc[self.statistics["first_move"]:, :][k].mean()
        #                 self.statistics[f'{k}_std'] = \
        #                 self.whole_shape_descriptors.iloc[self.statistics["first_move"]:, :][k].std()
        #             else:
        #                 self.statistics[f'{k}_mean'] = 'NA'
        #                 self.statistics[f'{k}_std'] = 'NA'
        #
        #         if v[2]: # Calculate linear regression
        #             if self.statistics["first_move"] != 'NA':
        #                 # Calculate slope and intercept of descriptors' value against time
        #                 X = self.whole_shape_descriptors[k].values[self.statistics["first_move"]:]
        #                 natural_noise = 0.1 * ptp(X)
        #                 left, right = find_major_incline(X, natural_noise)
        #                 T = self.whole_shape_descriptors['time'].values[self.statistics["first_move"]:]
        #                 if right == 1:
        #                     X = X[left:]
        #                     T = T[left:]
        #                     right = 0
        #                 else:
        #                     X = X[left:-right]
        #                     T = T[left:-right]
        #                 reg = linregress(T, X)
        #                 self.statistics[f"{k}_reg_start"] = self.statistics["first_move"] + left
        #                 self.statistics[f"{k}_reg_end"] = self.dims[0] - right
        #                 self.statistics[f"{k}_slope"] = reg.slope
        #                 self.statistics[f"{k}_intercept"] = reg.intercept
        #             else:
        #                 self.statistics[f"{k}_reg_start"] = 'NA'
        #                 self.statistics[f"{k}_reg_end"] = 'NA'
        #                 self.statistics[f"{k}_slope"] = 'NA'
        #                 self.statistics[f"{k}_intercept"] = 'NA'


        """
        self.statistics["final_area"] = self.binary[-1, :, :].sum()
        hour_to_frame = int(self.vars['when_to_measure_area_after_first_move'] * 60 / self.time_interval)
        if self.statistics["first_move"] != 'NA':
            if self.vars['oscilacyto_analysis']:
                descriptors = self.vars['descriptors_list'] + ['cluster_number', 'mean_cluster_area']
            for descriptor in descriptors:
                # I/ Get the area after a given time after first move
                stat_name = (f"{descriptor}_{self.vars['when_to_measure_area_after_first_move']}h_after_first_move")
                if (self.statistics["first_move"] + hour_to_frame) < self.dims[0]:
                    self.statistics[stat_name] = self.whole_shape_descriptors[descriptor].values[self.statistics["first_move"] + hour_to_frame]
                else:
                    self.statistics[stat_name] = self.dims[0]

                if self.vars['descriptors_means']:
                     # Calculate the mean value of each descriptors after first move
                    self.statistics[f'{descriptor}_mean'] = self.whole_shape_descriptors.iloc[self.statistics["first_move"]:, :][
                        descriptor].mean()
                    self.statistics[f'{descriptor}_std'] = self.whole_shape_descriptors.iloc[self.statistics["first_move"]:, :][
                        descriptor].std()
                if self.vars['descriptors_regressions']:
                    # Calculate slope and intercept of descriptors' value against time
                    X = self.whole_shape_descriptors[descriptor].values[self.statistics["first_move"]:]
                    natural_noise = 0.1 * ptp(X)
                    left, right = find_major_incline(X, natural_noise)
                    T = self.whole_shape_descriptors['time'].values[self.statistics["first_move"]:]
                    if right == 1:
                        X = X[left:]
                        T = T[left:]
                        right = 0
                    else:
                        X = X[left:-right]
                        T = T[left:-right]

                    reg = linregress(T, X)
                    self.statistics[f"{descriptor}_reg_start"] = self.statistics["first_move"] + left
                    self.statistics[f"{descriptor}_reg_end"] = self.dims[0] - right
                    self.statistics[f"{descriptor}_slope"] = reg.slope
                    self.statistics[f"{descriptor}_intercept"] = reg.intercept
        """

    def save_efficiency_tests(self):
        # Provide images allowing to assess the analysis efficiency
        if self.dims[0] > 1:
            after_one_tenth_of_time = ceil(self.dims[0] / 10).astype(uint64)
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
                self.converted_video = stack((self.converted_video, self.converted_video, self.converted_video),
                                             axis=3)
            self.efficiency_test_1 = self.converted_video[after_one_tenth_of_time, ...].copy()
            self.efficiency_test_2 = self.converted_video[last_good_detection, ...].copy()
        else:
            self.efficiency_test_1 = self.visu[after_one_tenth_of_time, :, :, :].copy()
            self.efficiency_test_2 = self.visu[last_good_detection, :, :, :].copy()

        position = (25, self.dims[1] // 2)
        text = str(self.statistics['arena'])
        contours = nonzero(morphologyEx(self.binary[after_one_tenth_of_time, :, :], MORPH_GRADIENT, cross_33))
        self.efficiency_test_1[contours[0], contours[1], :] = self.vars['contour_color']
        self.efficiency_test_1 = putText(self.efficiency_test_1, text, position, FONT_HERSHEY_SIMPLEX, 1,
                                        (self.vars["contour_color"], self.vars["contour_color"],
                                         self.vars["contour_color"], 255), 3)

        contours = nonzero(morphologyEx(self.binary[last_good_detection, :, :], MORPH_GRADIENT, cross_33))
        self.efficiency_test_2[contours[0], contours[1], :] = self.vars['contour_color']
        self.efficiency_test_2 = putText(self.efficiency_test_2, text, position, FONT_HERSHEY_SIMPLEX, 1,
                                         (self.vars["contour_color"], self.vars["contour_color"],
                                          self.vars["contour_color"], 255), 3)

    def save_video(self):

        if self.vars['save_processed_videos']:
            self.check_converted_video_type()
            # if self.converted_video.dtype != "uint8":
            #     self.converted_video -= min(self.converted_video)
            #     self.converted_video = 255 * (self.converted_video / max(self.converted_video))
            #     self.converted_video = round(self.converted_video).astype(uint8)
            if len(self.converted_video.shape) == 3:
                self.converted_video = stack((self.converted_video, self.converted_video, self.converted_video),
                                                axis=3)
            for t in arange(self.dims[0]):
                contours = nonzero(morphologyEx(self.binary[t, :, :], MORPH_GRADIENT, cross_33))
                self.converted_video[t, contours[0], contours[1], :] = self.vars['contour_color']
                if "iso_digi_transi" in self.statistics.keys():
                    if self.vars['iso_digi_analysis']  and not self.vars['several_blob_per_arena'] and not isna(self.statistics["iso_digi_transi"]):
                        if self.statistics["is_growth_isotropic"] == 1:
                            if t < self.statistics["iso_digi_transi"]:
                                self.converted_video[t, contours[0], contours[1], :] = 0, 0, 255
                # if self.statistics["iso_digi_transi"] != 'NA':
                #     if self.statistics["is_growth_isotropic"] == 1:
                #         before_transition = contours[0] < self.statistics["iso_digi_transi"]
                #         self.converted_video[contours[0][before_transition], contours[1][before_transition], contours[2][before_transition], 2] = 255
            del self.binary
            del self.surfarea
            del self.borders
            del self.origin
            del self.origin_idx
            del self.covering_intensity
            collect()
            if self.visu is None:
                true_frame_width = self.dims[2]
                if len(self.vars['background_list']) == 0:
                    self.background = None
                else:
                    self.background = self.vars['background_list'][self.statistics['arena'] - 1]
                self.visu = video2numpy(f"ind_{self.statistics['arena']}.npy", None, self.background, true_frame_width)
                if len(self.visu.shape) == 3:
                    self.visu = stack((self.visu, self.visu, self.visu), axis=3)
            self.converted_video = concatenate((self.visu, self.converted_video), axis=2)
            # self.visu = None

            if any(self.whole_shape_descriptors['time'] > 0):
                position = (5, self.dims[1] - 5)
                for t in arange(self.dims[0]):
                    image = self.converted_video[t, ...]
                    text = str(self.whole_shape_descriptors['time'][t]) + " min"
                    image = putText(image,  # numpy array on which text is written
                                    text,  # text
                                    position,  # position at which writing has to start
                                    FONT_HERSHEY_SIMPLEX,  # font family
                                    1,  # font size
                                    (self.vars["contour_color"], self.vars["contour_color"], self.vars["contour_color"], 255),  #(209, 80, 0, 255),  # repeat(self.vars["contour_color"], 3),# font color
                                    2)  # font stroke
                    self.converted_video[t, ...] = image
            vid_name = f"ind_{self.statistics['arena']}{self.vars['videos_extension']}"
            write_video(self.converted_video, vid_name, is_color=True, fps=self.vars['video_fps'])
            # self.converted_video = None

    def save_results(self):
        self.save_efficiency_tests()
        self.save_video()
        if self.vars['several_blob_per_arena']:
            try:
                with open(f"one_row_per_frame_arena{self.statistics['arena']}.csv", 'w') as file:
                    self.whole_shape_descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error(f"Never let one_row_per_frame_arena{self.statistics['arena']}.csv open when Cellects runs")

            create_new_csv: bool = False
            if os.path.isfile("one_row_per_arena.csv"):
                try:
                    with open(f"one_row_per_arena.csv", 'r') as file:
                        stats = read_csv(file, header=0, sep=";")
                except PermissionError:
                    logging.error("Never let one_row_per_arena.csv open when Cellects runs")

                if len(self.statistics) == len(stats.columns) - 1:
                    try:
                        with open(f"one_row_per_arena.csv", 'w') as file:
                            stats.iloc[(self.statistics['arena'] - 1), 1:] = self.statistics.values()
                            # if len(self.vars['analyzed_individuals']) == 1:
                            #     stats = df(self.statistics, index=[0])
                            # else:
                            #     stats = df.from_dict(self.statistics)
                        # stats.to_csv("stats.csv", sep=';', index=False, lineterminator='\n')
                            stats.to_csv(file, sep=';', index=False, lineterminator='\n')
                    except PermissionError:
                        logging.error("Never let one_row_per_arena.csv open when Cellects runs")
                else:
                    create_new_csv = True
            else:
                create_new_csv = True
            if create_new_csv:
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats = df(zeros((len(self.vars['analyzed_individuals']), len(self.statistics))),
                               columns=list(self.statistics.keys()))
                    stats.iloc[(self.statistics['arena'] - 1), :] = array(list(self.statistics.values()), dtype=uint32)
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')

            # if self.statistics["first_move"] != 'NA' and self.vars['oscilacyto_analysis']:
            #     oscil_i = df(
            #         c_[repeat(self.statistics['arena'], self.clusters_final_data.shape[0]), self.clusters_final_data],
            #         columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance'])
            #     self.one_row_per_oscillating_cluster = concat((self.one_row_per_oscillating_cluster, oscil_i))
        # if self.statistics["first_move"] != 'NA' and self.vars['oscilacyto_analysis']:
        #     savetxt(f"oscillation_cluster{self.statistics['arena']}.csv", self.clusters_final_data,
        #                     fmt='%1.9f', delimiter=',')

        # savetxt(f"magnitudes_and_frequencies_{self.statistics['arena']}.csv", self.magnitudes_and_frequencies,
        #         fmt='%1.9f', delimiter=',')


        # save(f"stats_{self.statistics['arena']}.npy", self.statistics)
        # if self.vars['descriptors_in_long_format']:
        #     self.whole_shape_descriptors.to_csv(f"shape_descriptors_{self.statistics['arena']}.csv", sep=";", index=False, lineterminator='\n')

        if not self.vars['keep_unaltered_videos'] and os.path.isfile(f"ind_{self.statistics['arena']}.npy"):
            os.remove(f"ind_{self.statistics['arena']}.npy")

    def change_results_of_one_arena(self):
        self.save_video()
        # with open(f"magnitudes_and_frequencies_{self.statistics['arena']}.csv", 'w') as file:
        #     savetxt(file, self.magnitudes_and_frequencies, fmt='%1.9f', delimiter=';')

        # I/ Update/Create one_row_per_arena.csv
        create_new_csv: bool = False
        if os.path.isfile("one_row_per_arena.csv"):
            try:
                with open(f"one_row_per_arena.csv", 'r') as file:
                    stats = read_csv(file, header=0, sep=";")
                for stat_name, stat_value in self.statistics.items():
                    if stat_name in stats.columns:
                        stats.loc[(self.statistics['arena'] - 1), stat_name] = uint32(self.statistics[stat_name])
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")
            except Exception as e:
                logging.error(f"{e}")
                create_new_csv = True
            # if len(self.statistics) == len(stats.columns):
            #     try:
            #         with open(f"one_row_per_arena.csv", 'w') as file:
            #             stats.iloc[(self.statistics['arena'] - 1), :] = self.statistics.values()
            #             # stats.to_csv("stats.csv", sep=';', index=False, lineterminator='\n')
            #             stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            #     except PermissionError:
            #         logging.error("Never let one_row_per_arena.csv open when Cellects runs")
            # else:
            #     create_new_csv = True
        else:
            create_new_csv = True
        if create_new_csv:
            logging.info("Create a new one_row_per_arena.csv file")
            try:
                with open(f"one_row_per_arena.csv", 'w') as file:
                    stats = df(zeros((len(self.vars['analyzed_individuals']), len(self.statistics))),
                               columns=list(self.statistics.keys()))
                    stats.iloc[(self.statistics['arena'] - 1), :] = array(list(self.statistics.values()), dtype=uint32)
                    stats.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")

        # II/ Update/Create one_row_per_frame.csv
        create_new_csv = False
        if os.path.isfile("one_row_per_frame.csv"):
            try:
                with open(f"one_row_per_frame.csv", 'r') as file:
                    descriptors = read_csv(file, header=0, sep=";")
                for stat_name, stat_value in self.whole_shape_descriptors.items():
                    if stat_name in descriptors.columns:
                        descriptors.loc[((self.statistics['arena'] - 1) * self.dims[0]):((self.statistics['arena']) * self.dims[0] - 1), stat_name] = self.whole_shape_descriptors.loc[:, stat_name].values[:]
                with open(f"one_row_per_frame.csv", 'w') as file:
                    descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
                # with open(f"one_row_per_frame.csv", 'w') as file:
                #     for descriptor in descriptors.keys():
                #         descriptors.loc[
                #         ((self.statistics['arena'] - 1) * self.dims[0]):((self.statistics['arena']) * self.dims[0]),
                #         descriptor] = self.whole_shape_descriptors[descriptor]
                #     descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')



                # if len(self.whole_shape_descriptors.columns) == len(descriptors.columns):
                #     with open(f"one_row_per_frame.csv", 'w') as file:
                #         # NEW
                #         for descriptor in descriptors.keys():
                #             descriptors.loc[((self.statistics['arena'] - 1) * self.dims[0]):((self.statistics['arena']) * self.dims[0]), descriptor] = self.whole_shape_descriptors[descriptor]
                #         # Old
                #         # descriptors.iloc[((self.statistics['arena'] - 1) * self.dims[0]):((self.statistics['arena']) * self.dims[0]), :] = self.whole_shape_descriptors
                #         descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
                # else:
                #     create_new_csv = True
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
                    descriptors = df(zeros((len(self.vars['analyzed_individuals']) * self.dims[0], len(self.whole_shape_descriptors.columns))),
                               columns=list(self.whole_shape_descriptors.keys()))
                    descriptors.iloc[((self.statistics['arena'] - 1) * self.dims[0]):((self.statistics['arena']) * self.dims[0]), :] = self.whole_shape_descriptors
                    descriptors.to_csv(file, sep=';', index=False, lineterminator='\n')
            except PermissionError:
                logging.error("Never let one_row_per_frame.csv open when Cellects runs")

        # III/ Update/Create one_row_per_oscillating_cluster.csv
        if not isna(self.statistics["first_move"]) and self.vars['oscilacyto_analysis']:
            oscil_i = df(
                c_[repeat(self.statistics['arena'], self.clusters_final_data.shape[0]), self.clusters_final_data],
                columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance', 'coord_y', 'coord_x'])
            if os.path.isfile("one_row_per_oscillating_cluster.csv"):
                try:
                    with open(f"one_row_per_oscillating_cluster.csv", 'r') as file:
                        one_row_per_oscillating_cluster = read_csv(file, header=0, sep=";")
                    with open(f"one_row_per_oscillating_cluster.csv", 'w') as file:
                        one_row_per_oscillating_cluster_before = one_row_per_oscillating_cluster[one_row_per_oscillating_cluster['arena'] < self.statistics['arena']]
                        one_row_per_oscillating_cluster_after = one_row_per_oscillating_cluster[one_row_per_oscillating_cluster['arena'] > self.statistics['arena']]
                        one_row_per_oscillating_cluster = concat((one_row_per_oscillating_cluster_before, oscil_i, one_row_per_oscillating_cluster_after))
                        one_row_per_oscillating_cluster.to_csv(file, sep=';', index=False, lineterminator='\n')

                        # one_row_per_oscillating_cluster = one_row_per_oscillating_cluster[one_row_per_oscillating_cluster['arena'] != self.statistics['arena']]
                        # one_row_per_oscillating_cluster = concat((one_row_per_oscillating_cluster, oscil_i))
                        # one_row_per_oscillating_cluster.to_csv(file, sep=';', index=False, lineterminator='\n')
                except PermissionError:
                    logging.error("Never let one_row_per_oscillating_cluster.csv open when Cellects runs")
            else:
                try:
                    with open(f"one_row_per_oscillating_cluster.csv", 'w') as file:
                        oscil_i.to_csv(file, sep=';', index=False, lineterminator='\n')
                except PermissionError:
                    logging.error("Never let one_row_per_oscillating_cluster.csv open when Cellects runs")

