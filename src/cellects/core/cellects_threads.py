#!/usr/bin/env python3
"""
Cellects graphical user interface interacts with computational scripts through threads.
Especially, each thread calls one or several methods of the class named "program_organizer",
which regroup all available computation of the software.
These threads are started from a children of WindowType, run methods from program_organizer and send messages and
results to the corresponding children of WindowType, allowing, for instance, to display a result in the interface.
"""

import logging
import weakref
from multiprocessing import Queue, Process, Manager
import os
import time
from glob import glob
from timeit import default_timer
from copy import deepcopy
import cv2
from numba.typed import Dict as TDict
import numpy as np
import pandas as pd
from PySide6 import QtCore
from cellects.image_analysis.morphological_operations import cross_33, Ellipse
from cellects.image_analysis.image_segmentation import generate_color_space_combination, apply_filter
from cellects.utils.load_display_save import read_and_rotate
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from cellects.utils.utilitarian import PercentAndTimeTracker, reduce_path_len
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.utils.load_display_save import write_video
from cellects.core.motion_analysis import MotionAnalysis


class LoadDataToRunCellectsQuicklyThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(LoadDataToRunCellectsQuicklyThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.look_for_data()
        self.parent().po.load_data_to_run_cellects_quickly()
        if self.parent().po.first_exp_ready_to_run:
            self.message_from_thread.emit("Data found, Video tracking window and Run all directly are available")
        else:
            self.message_from_thread.emit("")


class LookForDataThreadInFirstW(QtCore.QThread):
    def __init__(self, parent=None):
        super(LookForDataThreadInFirstW, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.look_for_data()


class LoadFirstFolderIfSeveralThread(QtCore.QThread):
    message_when_thread_finished = QtCore.Signal(bool)
    def __init__(self, parent=None):
        super(LoadFirstFolderIfSeveralThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.load_data_to_run_cellects_quickly()
        if not self.parent().po.first_exp_ready_to_run:
            self.parent().po.get_first_image()
        self.message_when_thread_finished.emit(self.parent().po.first_exp_ready_to_run)


class GetFirstImThread(QtCore.QThread):
    message_when_thread_finished = QtCore.Signal(bool)
    def __init__(self, parent=None):
        """
        This class read the first image of the (first of the) selected analysis.
        According to the first_detection_frame value,it can be another image
        If this is the first time a first image is read, it also gather the following variables:
            - img_number
            - dims (video dimensions: time, y, x)
            - raw_images (whether images are in a raw format)
        If the selected analysis contains videos instead of images, it opens the first video
        and read the first_detection_frame th image.
        :param parent: An object containing all necessary variables.
        """
        super(GetFirstImThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.get_first_image()
        self.message_when_thread_finished.emit(True)


class GetLastImThread(QtCore.QThread):
    def __init__(self, parent=None):
        super(GetLastImThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.get_last_image()


class UpdateImageThread(QtCore.QThread):
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(UpdateImageThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        # I/ If this thread runs from user input, get the right coordinates
        # and convert them to fit the displayed image size
        user_input = len(self.parent().imageanalysiswindow.saved_coord) > 0 or len(self.parent().imageanalysiswindow.temporary_mask_coord) > 0
        if user_input:
            if len(self.parent().imageanalysiswindow.temporary_mask_coord) > 0:
                idx = self.parent().imageanalysiswindow.temporary_mask_coord
            else:
                idx = self.parent().imageanalysiswindow.saved_coord
            if len(idx) < 2:
                user_input = False
            else:
                # Convert coordinates:
                self.parent().imageanalysiswindow.display_image.update_image_scaling_factors()
                sf = self.parent().imageanalysiswindow.display_image.scaling_factors
                idx = np.array(((np.round(idx[0][0] * sf[0]), np.round(idx[0][1] * sf[1])), (np.round(idx[1][0] * sf[0]), np.round(idx[1][1] * sf[1]))), dtype=np.int64)
                min_y = np.min(idx[:, 0])
                max_y = np.max(idx[:, 0])
                min_x = np.min(idx[:, 1])
                max_x = np.max(idx[:, 1])
                if max_y > self.parent().imageanalysiswindow.drawn_image.shape[0]:
                    max_y = self.parent().imageanalysiswindow.drawn_image.shape[0] - 1
                if max_x > self.parent().imageanalysiswindow.drawn_image.shape[1]:
                    max_x = self.parent().imageanalysiswindow.drawn_image.shape[1] - 1
                if min_y < 0:
                    min_y = 0
                if min_x < 0:
                    min_x = 0

        if len(self.parent().imageanalysiswindow.temporary_mask_coord) == 0:
            # not_load
            # II/ If this thread aims at saving the last user input and displaying all user inputs:
            # Update the drawn_image according to every saved masks
            # 1) The segmentation mask
            # 2) The back_mask and bio_mask
            # 3) The automatically detected video contours
            # (re-)Initialize drawn image
            self.parent().imageanalysiswindow.drawn_image = deepcopy(self.parent().po.current_image)
            if self.parent().imageanalysiswindow.drawn_image.size < 1000000:
                contour_width = 3
            else:
                contour_width = 6
            # 1) The segmentation mask
            logging.info('Add the segmentation mask to the image')
            if self.parent().imageanalysiswindow.is_first_image_flag:
                im_combinations = self.parent().po.first_image.im_combinations
                im_mean = self.parent().po.first_image.image.mean()
            else:
                im_combinations = self.parent().po.last_image.im_combinations
                im_mean = self.parent().po.last_image.bgr.mean()
            # If there are image combinations, get the current corresponding binary image
            if im_combinations is not None and len(im_combinations) != 0:
                binary_idx = im_combinations[self.parent().po.current_combination_id]["binary_image"]
                # If it concerns the last image, only keep the contour coordinates

                cv2.eroded_binary = cv2.erode(binary_idx, cross_33)
                binary_idx = binary_idx - cv2.eroded_binary
                binary_idx = cv2.dilate(binary_idx, kernel=cross_33, iterations=contour_width)
                binary_idx = np.nonzero(binary_idx)
                # Color these coordinates in magenta on bright images, and in pink on dark images
                if im_mean > 126:
                    # logging.info('Color the segmentation mask in magenta')
                    self.parent().imageanalysiswindow.drawn_image[binary_idx[0], binary_idx[1], :] = np.array((20, 0, 150), dtype=np.uint8)
                else:
                    # logging.info('Color the segmentation mask in pink')
                    self.parent().imageanalysiswindow.drawn_image[binary_idx[0], binary_idx[1], :] = np.array((94, 0, 213), dtype=np.uint8)
            if user_input:# save
                mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                if self.parent().imageanalysiswindow.back1_bio2 == 0:
                    logging.info("Save the user drawn mask of the current arena")
                    if self.parent().po.vars['arena_shape'] == 'circle':
                        ellipse = Ellipse((max_y - min_y, max_x - min_x)).create().astype(np.uint8)
                        mask[min_y:max_y, min_x:max_x, ...] = ellipse
                    else:
                        mask[min_y:max_y, min_x:max_x] = 1
                else:
                    logging.info("Save the user drawn mask of Cell or Back")

                    if self.parent().imageanalysiswindow.back1_bio2 == 2:
                        if self.parent().po.all['starting_blob_shape'] == 'circle':
                            ellipse = Ellipse((max_y - min_y, max_x - min_x)).create().astype(np.uint8)
                            mask[min_y:max_y, min_x:max_x, ...] = ellipse
                        else:
                            mask[min_y:max_y, min_x:max_x] = 1
                    else:
                        mask[min_y:max_y, min_x:max_x] = 1
                mask = np.nonzero(mask)

                if self.parent().imageanalysiswindow.back1_bio2 == 1:
                    self.parent().imageanalysiswindow.back_masks_number += 1
                    self.parent().imageanalysiswindow.back_mask[mask[0], mask[1]] = self.parent().imageanalysiswindow.available_back_names[0]
                elif self.parent().imageanalysiswindow.back1_bio2 == 2:
                    self.parent().imageanalysiswindow.bio_masks_number += 1
                    self.parent().imageanalysiswindow.bio_mask[mask[0], mask[1]] = self.parent().imageanalysiswindow.available_bio_names[0]
                elif self.parent().imageanalysiswindow.manual_delineation_flag:
                    self.parent().imageanalysiswindow.arena_masks_number += 1
                    self.parent().imageanalysiswindow.arena_mask[mask[0], mask[1]] = self.parent().imageanalysiswindow.available_arena_names[0]
                # 2)a) Apply all these masks to the drawn image:

            back_coord = np.nonzero(self.parent().imageanalysiswindow.back_mask)

            bio_coord = np.nonzero(self.parent().imageanalysiswindow.bio_mask)

            if self.parent().imageanalysiswindow.arena_mask is not None:
                arena_coord = np.nonzero(self.parent().imageanalysiswindow.arena_mask)
                self.parent().imageanalysiswindow.drawn_image[arena_coord[0], arena_coord[1], :] = np.repeat(self.parent().po.vars['contour_color'], 3).astype(np.uint8)

            self.parent().imageanalysiswindow.drawn_image[back_coord[0], back_coord[1], :] = np.array((224, 160, 81), dtype=np.uint8)

            self.parent().imageanalysiswindow.drawn_image[bio_coord[0], bio_coord[1], :] = np.array((17, 160, 212), dtype=np.uint8)

            image = self.parent().imageanalysiswindow.drawn_image
            # 3) The automatically detected video contours
            if self.parent().imageanalysiswindow.delineation_done:  # add a mask of the video contour
                # logging.info("Draw the delineation mask of each arena")
                for contour_i in range(len(self.parent().po.top)):
                    mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                    min_cy = self.parent().po.top[contour_i]
                    max_cy = self.parent().po.bot[contour_i]
                    min_cx = self.parent().po.left[contour_i]
                    max_cx = self.parent().po.right[contour_i]
                    text = f"{contour_i + 1}"
                    position = (self.parent().po.left[contour_i] + 25, self.parent().po.top[contour_i] + (self.parent().po.bot[contour_i] - self.parent().po.top[contour_i]) // 2)
                    image = cv2.putText(image,  # numpy array on which text is written
                                    text,  # text
                                    position,  # position at which writing has to start
                                    cv2.FONT_HERSHEY_SIMPLEX,  # font family
                                    1,  # font size
                                    (138, 95, 18, 255),
                                    # (209, 80, 0, 255),  # font color
                                    2)  # font stroke
                    if (max_cy - min_cy) < 0 or (max_cx - min_cx) < 0:
                        self.parent().imageanalysiswindow.message.setText("Error: the shape number or the detection is wrong")
                    if self.parent().po.vars['arena_shape'] == 'circle':
                        ellipse = Ellipse((max_cy - min_cy, max_cx - min_cx)).create().astype(np.uint8)
                        ellipse = cv2.morphologyEx(ellipse, cv2.MORPH_GRADIENT, cross_33)
                        mask[min_cy:max_cy, min_cx:max_cx, ...] = ellipse
                    else:
                        mask[(min_cy, max_cy), min_cx:max_cx] = 1
                        mask[min_cy:max_cy, (min_cx, max_cx)] = 1
                    mask = cv2.dilate(mask, kernel=cross_33, iterations=contour_width)

                    mask = np.nonzero(mask)
                    image[mask[0], mask[1], :] = np.array((138, 95, 18), dtype=np.uint8)# self.parent().po.vars['contour_color']

        else: #load
            if user_input:
                # III/ If this thread runs from user input: update the drawn_image according to the current user input
                # Just add the mask to drawn_image as quick as possible
                # Add user defined masks
                # Take the drawn image and add the temporary mask to it
                image = deepcopy(self.parent().imageanalysiswindow.drawn_image)
                if self.parent().imageanalysiswindow.back1_bio2 == 0:
                    # logging.info("Dynamic drawing of the arena outline")
                    if self.parent().po.vars['arena_shape'] == 'circle':
                        ellipse = Ellipse((max_y - min_y, max_x - min_x)).create()
                        ellipse = np.stack((ellipse, ellipse, ellipse), axis=2).astype(np.uint8)
                        image[min_y:max_y, min_x:max_x, ...] *= (1 - ellipse)
                        image[min_y:max_y, min_x:max_x, ...] += ellipse
                    else:
                        mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                        mask[min_y:max_y, min_x:max_x] = 1
                        mask = np.nonzero(mask)
                        image[mask[0], mask[1], :] = np.array((0, 0, 0), dtype=np.uint8)
                else:
                    # logging.info("Dynamic drawing of Cell or Back")
                    if self.parent().imageanalysiswindow.back1_bio2 == 2:
                        if self.parent().po.all['starting_blob_shape'] == 'circle':
                            ellipse = Ellipse((max_y - min_y, max_x - min_x)).create()
                            ellipse = np.stack((ellipse, ellipse, ellipse), axis=2).astype(np.uint8)
                            image[min_y:max_y, min_x:max_x, ...] *= (1 - ellipse)
                            ellipse[:, :, :] *= np.array((17, 160, 212), dtype=np.uint8)
                            image[min_y:max_y, min_x:max_x, ...] += ellipse
                        else:
                            mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                            mask[min_y:max_y, min_x:max_x] = 1
                            mask = np.nonzero(mask)
                            image[mask[0], mask[1], :] = np.array((17, 160, 212), dtype=np.uint8)
                    else:
                        mask = np.zeros(self.parent().imageanalysiswindow.drawn_image.shape[:2], dtype=np.uint8)
                        mask[min_y:max_y, min_x:max_x] = 1
                        mask = np.nonzero(mask)
                        image[mask[0], mask[1], :] = np.array((224, 160, 81), dtype=np.uint8)

        self.parent().imageanalysiswindow.display_image.update_image(image)
        self.message_when_thread_finished.emit(True)


class FirstImageAnalysisThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(FirstImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        tic = default_timer()
        biomask = None
        backmask = None
        if self.parent().imageanalysiswindow.bio_masks_number != 0:
            shape_nb, ordered_image = cv2.connectedComponents((self.parent().imageanalysiswindow.bio_mask > 0).astype(np.uint8))
            shape_nb -= 1
            biomask = np.nonzero(self.parent().imageanalysiswindow.bio_mask)
        else:
            shape_nb = 0
        if self.parent().imageanalysiswindow.back_masks_number != 0:
            backmask = np.nonzero(self.parent().imageanalysiswindow.back_mask)
        if self.parent().po.visualize or len(self.parent().po.first_im.shape) == 2 or shape_nb == self.parent().po.sample_number:
            self.message_from_thread.emit("Image segmentation, wait 30 seconds at most")
            if not self.parent().imageanalysiswindow.asking_first_im_parameters_flag and self.parent().po.all['scale_with_image_or_cells'] == 0 and self.parent().po.all["set_spot_size"]:
                self.parent().po.get_average_pixel_size()
                spot_size = self.parent().po.starting_blob_hsize_in_pixels
            else:
                spot_size = None
            self.parent().po.all["bio_mask"] = biomask
            self.parent().po.all["back_mask"] = backmask
            self.parent().po.fast_image_segmentation(is_first_image=True, biomask=biomask, backmask=backmask, spot_size=spot_size)
            if shape_nb == self.parent().po.sample_number and self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number'] != self.parent().po.sample_number:
                self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number'] = shape_nb
                self.parent().po.first_image.shape_number = shape_nb
                self.parent().po.first_image.validated_shapes = (self.parent().imageanalysiswindow.bio_mask > 0).astype(np.uint8)
                self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['binary_image'] = self.parent().po.first_image.validated_shapes
        else:
            self.message_from_thread.emit("Generating analysis options, wait...")
            if self.parent().po.vars["color_number"] > 2:
                kmeans_clust_nb = self.parent().po.vars["color_number"]
                if self.parent().po.carefully:
                    self.message_from_thread.emit("Generating analysis options, wait less than 30 minutes")
                else:
                    self.message_from_thread.emit("Generating analysis options, a few minutes")
            else:
                kmeans_clust_nb = None
                if self.parent().po.carefully:
                    self.message_from_thread.emit("Generating analysis options, wait a few minutes")
                else:
                    self.message_from_thread.emit("Generating analysis options, around 1 minute")
            if self.parent().imageanalysiswindow.asking_first_im_parameters_flag:
                self.parent().po.first_image.find_first_im_csc(sample_number=self.parent().po.sample_number,
                                                               several_blob_per_arena=None,
                                                               spot_shape=None, spot_size=None,
                                                               kmeans_clust_nb=kmeans_clust_nb,
                                                               biomask=self.parent().po.all["bio_mask"],
                                                               backmask=self.parent().po.all["back_mask"],
                                                               color_space_dictionaries=None,
                                                               carefully=self.parent().po.carefully)
            else:
                if self.parent().po.all['scale_with_image_or_cells'] == 0:
                    self.parent().po.get_average_pixel_size()
                else:
                    self.parent().po.starting_blob_hsize_in_pixels = None
                self.parent().po.first_image.find_first_im_csc(sample_number=self.parent().po.sample_number,
                                                                                   several_blob_per_arena=self.parent().po.vars['several_blob_per_arena'],
                                                                                   spot_shape=self.parent().po.all['starting_blob_shape'],
                                                               spot_size=self.parent().po.starting_blob_hsize_in_pixels,
                                                                                   kmeans_clust_nb=kmeans_clust_nb,
                                                                                   biomask=self.parent().po.all["bio_mask"],
                                                                                   backmask=self.parent().po.all["back_mask"],
                                                                                   color_space_dictionaries=None,
                                                                                   carefully=self.parent().po.carefully)

        logging.info(f" image analysis lasted {default_timer() - tic} secondes")
        logging.info(f" image analysis lasted {np.round((default_timer() - tic) / 60)} minutes")
        self.message_when_thread_finished.emit(True)


class LastImageAnalysisThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(LastImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.cropping(False)
        self.parent().po.get_background_to_subtract()
        biomask = None
        backmask = None
        if self.parent().imageanalysiswindow.bio_masks_number != 0:
            biomask = np.nonzero(self.parent().imageanalysiswindow.bio_mask)
        if self.parent().imageanalysiswindow.back_masks_number != 0:
            backmask = np.nonzero(self.parent().imageanalysiswindow.back_mask)
        if self.parent().po.visualize or len(self.parent().po.first_im.shape) == 2:
            self.message_from_thread.emit("Image segmentation, wait...")
            self.parent().po.fast_image_segmentation(is_first_image=False, biomask=biomask, backmask=backmask)
        else:
            self.message_from_thread.emit("Generating analysis options, wait...")
            if self.parent().po.vars['several_blob_per_arena']:
                concomp_nb = [self.parent().po.sample_number, self.parent().po.first_image.size // 50]
                max_shape_size = .75 * self.parent().po.first_image.size
                total_surfarea = .99 * self.parent().po.first_image.size
            else:
                concomp_nb = [self.parent().po.sample_number, self.parent().po.sample_number * 200]
                if self.parent().po.all['are_zigzag'] == "columns":
                    inter_dist = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.y_boundaries)))
                elif self.parent().po.all['are_zigzag'] == "rows":
                    inter_dist = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.x_boundaries)))
                else:
                    dist1 = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.y_boundaries)))
                    dist2 = np.mean(np.diff(np.nonzero(self.parent().po.videos.first_image.x_boundaries)))
                    inter_dist = np.max(dist1, dist2)
                if self.parent().po.all['starting_blob_shape'] == "circle":
                    max_shape_size = np.pi * np.square(inter_dist)
                else:
                    max_shape_size = np.square(2 * inter_dist)
                total_surfarea = max_shape_size * self.parent().po.sample_number
            out_of_arenas = None
            if self.parent().po.all['are_gravity_centers_moving'] != 1:
                out_of_arenas = np.ones_like(self.parent().po.videos.first_image.validated_shapes)
                for blob_i in np.arange(len(self.parent().po.vars['analyzed_individuals'])):
                    out_of_arenas[self.parent().po.top[blob_i]: (self.parent().po.bot[blob_i] + 1),
                    self.parent().po.left[blob_i]: (self.parent().po.right[blob_i] + 1)] = 0
            ref_image = self.parent().po.first_image.validated_shapes
            self.parent().po.first_image.generate_subtract_background(self.parent().po.vars['convert_for_motion'])
            kmeans_clust_nb = None
            self.parent().po.last_image.find_last_im_csc(concomp_nb, total_surfarea, max_shape_size, out_of_arenas,
                                                         ref_image, self.parent().po.first_image.subtract_background,
                                                         kmeans_clust_nb, biomask, backmask, color_space_dictionaries=None,
                                                         carefully=self.parent().po.carefully)
        self.message_when_thread_finished.emit(True)


class CropScaleSubtractDelineateThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(CropScaleSubtractDelineateThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        logging.info("Start cropping if required")
        self.parent().po.cropping(is_first_image=True)
        self.parent().po.cropping(is_first_image=False)
        self.parent().po.get_average_pixel_size()
        if os.path.isfile('Data to run Cellects quickly.pkl'):
            os.remove('Data to run Cellects quickly.pkl')
        logging.info("Save data to run Cellects quickly")
        self.parent().po.data_to_save['first_image'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.data_to_save['first_image'] = False
        if not self.parent().po.vars['several_blob_per_arena']:
            logging.info("Check whether the detected shape number is ok")
            nb, shapes, stats, centroids = cv2.connectedComponentsWithStats(self.parent().po.first_image.validated_shapes)
            y_lim = self.parent().po.first_image.y_boundaries
            if ((nb - 1) != self.parent().po.sample_number or np.any(stats[:, 4] == 1)):
                self.message_from_thread.emit("Image analysis failed to detect the right cell(s) number: restart the analysis.")
            elif len(np.nonzero(y_lim == - 1)) != len(np.nonzero(y_lim == 1)):
                self.message_from_thread.emit("Automatic arena delineation cannot work if one cell touches the image border.")
                self.parent().po.first_image.y_boundaries = None
            else:
                logging.info("Start automatic video delineation")
                analysis_status = self.parent().po.delineate_each_arena()
                self.message_when_thread_finished.emit(analysis_status["message"])
        else:
            logging.info("Start automatic video delineation")
            analysis_status = self.parent().po.delineate_each_arena()
            self.message_when_thread_finished.emit(analysis_status["message"])


class SaveManualDelineationThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(SaveManualDelineationThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.left = np.arange(self.parent().po.sample_number)
        self.parent().po.right = np.arange(self.parent().po.sample_number)
        self.parent().po.top = np.arange(self.parent().po.sample_number)
        self.parent().po.bot = np.arange(self.parent().po.sample_number)
        for arena in np.arange(1, self.parent().po.sample_number + 1):
            y, x = np.nonzero(self.parent().imageanalysiswindow.arena_mask == arena)
            self.parent().po.left[arena - 1] = np.min(x)
            self.parent().po.right[arena - 1] = np.max(x)
            self.parent().po.top[arena - 1] = np.min(y)
            self.parent().po.bot[arena - 1] = np.max(y)

        logging.info("Save data to run Cellects quickly")
        self.parent().po.data_to_save['coordinates'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.data_to_save['coordinates'] = False

        logging.info("Save manual video delineation")
        self.parent().po.vars['analyzed_individuals'] = np.arange(self.parent().po.sample_number) + 1
        self.parent().po.videos = OneVideoPerBlob(self.parent().po.first_image, self.parent().po.starting_blob_hsize_in_pixels, self.parent().po.all['raw_images'])
        self.parent().po.videos.left = self.parent().po.left
        self.parent().po.videos.right = self.parent().po.right
        self.parent().po.videos.top = self.parent().po.top
        self.parent().po.videos.bot = self.parent().po.bot


class GetExifDataThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(GetExifDataThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.extract_exif()


class FinalizeImageAnalysisThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(FinalizeImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.get_background_to_subtract()

        self.parent().po.get_origins_and_backgrounds_lists()

        if self.parent().po.last_image is None:
            self.parent().po.get_last_image()
            self.parent().po.fast_image_segmentation(False)
        self.parent().po.find_if_lighter_background()
        logging.info("The current (or the first) folder is ready to run")
        self.parent().po.first_exp_ready_to_run = True
        self.parent().po.data_to_save['coordinates'] = True
        self.parent().po.data_to_save['exif'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.data_to_save['coordinates'] = False
        self.parent().po.data_to_save['exif'] = False


class SaveAllVarsThread(QtCore.QThread):

    def __init__(self, parent=None):
        super(SaveAllVarsThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.save_variable_dict()

        #self.parent().po.all['global_pathway']
        #os.getcwd()

        self.set_current_folder()
        self.parent().po.save_data_to_run_cellects_quickly(new_one_if_does_not_exist=False)
        #if os.access(f"", os.R_OK):
        #    self.parent().po.save_data_to_run_cellects_quickly()
        #else:
        #    logging.error(f"No permission access to write in {os.getcwd()}")

    def set_current_folder(self):
        if self.parent().po.all['folder_number'] > 1: # len(self.parent().po.all['folder_list']) > 1:  # len(self.parent().po.all['folder_list']) > 0:
            logging.info(f"Use {self.parent().po.all['folder_list'][0]} folder")
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0],
                                              self.parent().po.all['folder_list'][0])
        else:
            curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {curr_path} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])


class OneArenaThread(QtCore.QThread):
    message_from_thread_starting = QtCore.Signal(str)
    image_from_thread = QtCore.Signal(dict)
    when_loading_finished = QtCore.Signal(bool)
    when_detection_finished = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(OneArenaThread, self).__init__(parent)
        self.setParent(parent)
        self._isRunning = False

    def run(self):
        continue_analysis = True
        self._isRunning = True
        self.message_from_thread_starting.emit("Video loading, wait...")

        self.set_current_folder()
        print(self.parent().po.vars['convert_for_motion'])
        if not self.parent().po.first_exp_ready_to_run:
            self.parent().po.load_data_to_run_cellects_quickly()
            if not self.parent().po.first_exp_ready_to_run:
                #Need a look for data when Data to run Cellects quickly.pkl and 1 folder selected amon several
                continue_analysis = self.pre_processing()
        if continue_analysis:
            print(self.parent().po.vars['convert_for_motion'])
            memory_diff = self.parent().po.update_available_core_nb()
            if self.parent().po.cores == 0:
                self.message_from_thread_starting.emit(f"Analyzing one arena requires {memory_diff}GB of additional RAM to run")
            else:
                if self.parent().po.motion is None or self.parent().po.load_quick_full == 0:
                    self.load_one_arena()
                if self.parent().po.load_quick_full > 0:
                    if self.parent().po.motion.start is not None:
                        logging.info("One arena detection has started")
                        self.detection()
                        if self.parent().po.load_quick_full > 1:
                            logging.info("One arena post-processing has started")
                            self.post_processing()
                        else:
                            self.when_detection_finished.emit("Detection done, read to see the result")
                    else:
                        self.message_from_thread_starting.emit(f"The current parameters failed to detect the cell(s) motion")

    def stop(self):
        self._isRunning = False

    def set_current_folder(self):
        if self.parent().po.all['folder_number'] > 1:
            logging.info(f"Use {self.parent().po.all['folder_list'][0]} folder")
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0],
                                              self.parent().po.all['folder_list'][0])
        else:
            curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {curr_path} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])

    def pre_processing(self):
        logging.info("Pre-processing has started")
        analysis_status = {"continue": True, "message": ""}

        self.parent().po.get_first_image()
        self.parent().po.fast_image_segmentation(is_first_image=True)
        if len(self.parent().po.vars['analyzed_individuals']) != self.parent().po.first_image.shape_number:
            self.message_from_thread_starting.emit(f"Wrong specimen number: (re)do the complete analysis.")
            analysis_status["continue"] = False
        else:
            self.parent().po.cropping(is_first_image=True)
            self.parent().po.get_average_pixel_size()
            analysis_status = self.parent().po.delineate_each_arena()
            if not analysis_status["continue"]:
                self.message_from_thread_starting.emit(analysis_status["message"])
                logging.error(analysis_status['message'])
            else:
                self.parent().po.data_to_save['exif'] = True
                self.parent().po.save_data_to_run_cellects_quickly()
                self.parent().po.data_to_save['exif'] = False
                self.parent().po.get_background_to_subtract()
                if len(self.parent().po.vars['analyzed_individuals']) != len(self.parent().po.top):
                    self.message_from_thread_starting.emit(f"Wrong specimen number: (re)do the complete analysis.")
                    analysis_status["continue"] = False
                else:
                    self.parent().po.get_origins_and_backgrounds_lists()
                    self.parent().po.get_last_image()
                    self.parent().po.fast_image_segmentation(False)
                    self.parent().po.find_if_lighter_backgnp.round()
                    logging.info("The current (or the first) folder is ready to run")
                    self.parent().po.first_exp_ready_to_run = True
        return analysis_status["continue"]

    def load_one_arena(self):
        arena = self.parent().po.all['arena']
        i = np.nonzero(self.parent().po.vars['analyzed_individuals'] == arena)[0][0]
        save_loaded_video: bool = False
        if not os.path.isfile(f'ind_{arena}.npy') or self.parent().po.all['overwrite_unaltered_videos']:
            logging.info(f"Starting to load arena n°{arena} from images")
            add_to_c = 1
            self.parent().po.one_arenate_done = True
            i = np.nonzero(self.parent().po.vars['analyzed_individuals'] == arena)[0][0]
            if self.parent().po.vars['lose_accuracy_to_save_memory']:
                self.parent().po.converted_video = np.zeros(
                    (len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c, self.parent().po.right[i] - self.parent().po.left[i] + add_to_c),
                    dtype=np.uint8)
            else:
                self.parent().po.converted_video = np.zeros(
                    (len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c, self.parent().po.right[i] - self.parent().po.left[i] + add_to_c),
                    dtype=float)
            if not self.parent().po.vars['already_greyscale']:
                self.parent().po.visu = np.zeros((len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c,
                                   self.parent().po.right[i] - self.parent().po.left[i] + add_to_c, 3), dtype=np.uint8)
                if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                    if self.parent().po.vars['lose_accuracy_to_save_memory']:
                        self.parent().po.converted_video2 = np.zeros((len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c,
                                                       self.parent().po.right[i] - self.parent().po.left[i] + add_to_c), dtype=np.uint8)
                    else:
                        self.parent().po.converted_video2 = np.zeros((len(self.parent().po.data_list), self.parent().po.bot[i] - self.parent().po.top[i] + add_to_c,
                                                       self.parent().po.right[i] - self.parent().po.left[i] + add_to_c), dtype=float)
                first_dict = TDict()
                second_dict = TDict()
                c_spaces = []
                for k, v in self.parent().po.vars['convert_for_motion'].items():
                     if k != 'logical' and v.sum() > 0:
                        if k[-1] != '2':
                            first_dict[k] = v
                            c_spaces.append(k)
                        else:
                            second_dict[k[:-1]] = v
                            c_spaces.append(k[:-1])
            prev_img = None
            background = None
            background2 = None
            pat_tracker = PercentAndTimeTracker(self.parent().po.vars['img_number'])
            for image_i, image_name in enumerate(self.parent().po.data_list):
                current_percentage, eta = pat_tracker.get_progress()
                is_landscape = self.parent().po.first_image.image.shape[0] < self.parent().po.first_image.image.shape[1]
                img = read_and_rotate(image_name, prev_img, self.parent().po.all['raw_images'], is_landscape)
                # img = self.parent().po.videos.read_and_rotate(image_name, prev_img)
                prev_img = deepcopy(img)
                if self.parent().po.first_image.cropped:
                    img = img[self.parent().po.first_image.crop_coord[0]:self.parent().po.first_image.crop_coord[1],
                          self.parent().po.first_image.crop_coord[2]:self.parent().po.first_image.crop_coord[3], :]
                img = img[self.parent().po.top[arena - 1]: (self.parent().po.bot[arena - 1] + add_to_c),
                      self.parent().po.left[arena - 1]: (self.parent().po.right[arena - 1] + add_to_c), :]

                self.image_from_thread.emit({"message": f"Video loading: {current_percentage}%{eta}", "current_image": img})
                if self.parent().po.vars['already_greyscale']:
                    if self.parent().po.reduce_image_dim:
                        self.parent().po.converted_video[image_i, ...] = img[:, :, 0]
                    else:
                        self.parent().po.converted_video[image_i, ...] = img
                else:
                    self.parent().po.visu[image_i, ...] = img

                    if self.parent().po.vars['subtract_background']:
                        background = self.parent().po.vars['background_list'][i]
                        if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                            background2 = self.parent().po.vars['background_list2'][i]
                    greyscale_image, greyscale_image2 = generate_color_space_combination(img, c_spaces,
                                                                                         first_dict,
                                                                                         second_dict,background,background2,
                                                                                         self.parent().po.vars[
                                                                                             'lose_accuracy_to_save_memory'])

                    if self.parent().po.vars['filter_spec'] is not None and self.parent().po.vars['filter_spec']['filter1_type'] != "":
                        greyscale_image = apply_filter(greyscale_image,
                                                       self.parent().po.vars['filter_spec']['filter1_type'],
                                                       self.parent().po.vars['filter_spec']['filter1_param'],
                                                       self.parent().po.vars['lose_accuracy_to_save_memory'])
                        if greyscale_image2 is not None and self.parent().po.vars['filter_spec']['filter2_type'] != "":
                            greyscale_image2 = apply_filter(greyscale_image2,
                                                            self.parent().po.vars['filter_spec']['filter2_type'],
                                                            self.parent().po.vars['filter_spec']['filter2_param'],
                                                            self.parent().po.vars['lose_accuracy_to_save_memory'])
                    self.parent().po.converted_video[image_i, ...] = greyscale_image
                    if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                        self.parent().po.converted_video2[image_i, ...] = greyscale_image2



                    # csc = OneImageAnalysis(img)
                    # if self.parent().po.vars['subtract_background']:
                    #     if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                    #         csc.generate_color_space_combination(c_spaces, first_dict, second_dict,
                    #                                              self.parent().po.vars['background_list'][i],
                    #                                              self.parent().po.vars['background_list2'][i])
                    #     else:
                    #         csc.generate_color_space_combination(c_spaces, first_dict, second_dict,
                    #                                              self.parent().po.vars['background_list'][i], None)
                    # else:
                    #     csc.generate_color_space_combination(c_spaces, first_dict, second_dict, None, None)
                    # # self.parent().po.converted_video[image_i, ...] = csc.image
                    # if self.parent().po.vars['lose_accuracy_to_save_memory']:
                    #     self.parent().po.converted_video[image_i, ...] = bracket_to_np.uint8_image_contrast(csc.image)
                    # else:
                    #     self.parent().po.converted_video[image_i, ...] = csc.image
                    # if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                    #     if self.parent().po.vars['lose_accuracy_to_save_memory']:
                    #         self.parent().po.converted_video2[image_i, ...] = bracket_to_np.uint8_image_contrast(csc.image2)
                    #     else:
                    #         self.parent().po.converted_video2[image_i, ...] = csc.image2



            # self.parent().po.load_one_arena(arena)
            save_loaded_video = True
            if self.parent().po.vars['already_greyscale']:
                self.videos_in_ram = self.parent().po.converted_video
            else:
                if self.parent().po.vars['convert_for_motion']['logical'] == 'None':
                    self.videos_in_ram = [self.parent().po.visu, deepcopy(self.parent().po.converted_video)]
                else:
                    self.videos_in_ram = [self.parent().po.visu, deepcopy(self.parent().po.converted_video), deepcopy(self.parent().po.converted_video2)]

            # videos = [self.parent().po.video.copy(), self.parent().po.converted_video.copy()]
        else:
            logging.info(f"Starting to load arena n°{arena} from .npy saved file")
            self.videos_in_ram = None
        l = [i, arena, self.parent().po.vars, False, False, False, self.videos_in_ram]
        self.parent().po.motion = MotionAnalysis(l)
        r = weakref.ref(self.parent().po.motion)

        if self.videos_in_ram is None:
            self.parent().po.converted_video = deepcopy(self.parent().po.motion.converted_video)
            if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                self.parent().po.converted_video2 = deepcopy(self.parent().po.motion.converted_video2)
        self.parent().po.motion.get_origin_shape()

        if self.parent().po.motion.dims[0] >= 40:
            step = self.parent().po.motion.dims[0] // 20
        else:
            step = 1
        if self.parent().po.motion.start >= (self.parent().po.motion.dims[0] - step - 1):
            self.parent().po.motion.start = None
        else:
            self.parent().po.motion.get_covering_duration(step)
        self.when_loading_finished.emit(save_loaded_video)

        if self.parent().po.motion.visu is None:
            visu = self.parent().po.motion.converted_video
            visu -= np.min(visu)
            visu = 255 * (visu / np.max(visu))
            visu = np.round(visu).astype(np.uint8)
            if len(visu.shape) == 3:
                visu = np.stack((visu, visu, visu), axis=3)
            self.parent().po.motion.visu = visu

    def detection(self):
        self.message_from_thread_starting.emit(f"Quick video segmentation")
        self.parent().po.motion.converted_video = deepcopy(self.parent().po.converted_video)
        if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
            self.parent().po.motion.converted_video2 = deepcopy(self.parent().po.converted_video2)
        # self.parent().po.motion.detection(compute_all_possibilities=True)
        self.parent().po.motion.detection(compute_all_possibilities=self.parent().po.all['compute_all_options'])
        if self.parent().po.all['compute_all_options']:
            self.parent().po.computed_video_options = np.ones(5, bool)
        else:
            self.parent().po.computed_video_options = np.zeros(5, bool)
            self.parent().po.computed_video_options[self.parent().po.all['video_option']] = True
        # if self.parent().po.vars['color_number'] > 2:

    def post_processing(self):
        self.parent().po.motion.smoothed_video = None
        # if self.parent().po.vars['already_greyscale']:
        #     if self.parent().po.vars['convert_for_motion']['logical'] == 'None':
        #         self.videos_in_ram = self.parent().po.converted_video
        #     else:
        #         self.videos_in_ram = self.parent().po.converted_video, self.parent().po.converted_video2
        # else:
        #     if self.parent().po.vars['convert_for_motion']['logical'] == 'None':
        #         videos_in_ram = self.parent().po.visu, self.parent().po.converted_video
        #     else:
        #         videos_in_ram = self.parent().po.visu, self.parent().po.converted_video, \
        #                         self.parent().po.converted_video2

        if self.parent().po.vars['color_number'] > 2:
            analyses_to_compute = [0]
        else:
            if self.parent().po.all['compute_all_options']:
                analyses_to_compute = np.arange(5)
            else:
                logging.info(f"option: {self.parent().po.all['video_option']}")
                analyses_to_compute = [self.parent().po.all['video_option']]
        time_parameters = [self.parent().po.motion.start, self.parent().po.motion.step,
                           self.parent().po.motion.lost_frames, self.parent().po.motion.substantial_growth]

        args = [self.parent().po.all['arena'] - 1, self.parent().po.all['arena'], self.parent().po.vars,
                False, False, False, self.videos_in_ram]
        if self.parent().po.vars['do_fading']:
            self.parent().po.newly_explored_area = np.zeros((self.parent().po.motion.dims[0], 5), np.unp.int64)
        for seg_i in analyses_to_compute:
            analysis_i = MotionAnalysis(args)
            r = weakref.ref(analysis_i)
            analysis_i.segmentation = np.zeros(analysis_i.converted_video.shape[:3], dtype=np.uint8)
            if self.parent().po.all['compute_all_options']:
                if seg_i == 0:
                    analysis_i.segmentation = self.parent().po.motion.segmentation
                else:
                    if seg_i == 1:
                        mask = self.parent().po.motion.luminosity_segmentation
                    elif seg_i == 2:
                        mask = self.parent().po.motion.gradient_segmentation
                    elif seg_i == 3:
                        mask = self.parent().po.motion.logical_and
                    elif seg_i == 4:
                        mask = self.parent().po.motion.logical_or
                    analysis_i.segmentation[mask[0], mask[1], mask[2]] = 1
            else:
                if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    analysis_i.segmentation = self.parent().po.motion.segmentation

            analysis_i.start = time_parameters[0]
            analysis_i.step = time_parameters[1]
            analysis_i.lost_frames = time_parameters[2]
            analysis_i.substantial_growth = time_parameters[3]
            analysis_i.origin_idx = self.parent().po.motion.origin_idx
            analysis_i.initialize_post_processing()
            analysis_i.t = analysis_i.start
            # print_progress = ForLoopCounter(self.start)

            while self._isRunning and analysis_i.t < analysis_i.binary.shape[0]:
                # analysis_i.update_shape(True)
                analysis_i.update_shape(False)
                contours = np.nonzero(
                    cv2.morphologyEx(analysis_i.binary[analysis_i.t - 1, :, :], cv2.MORPH_GRADIENT, cross_33))
                current_image = deepcopy(self.parent().po.motion.visu[analysis_i.t - 1, :, :, :])
                current_image[contours[0], contours[1], :] = self.parent().po.vars['contour_color']
                self.image_from_thread.emit(
                    {"message": f"Tracking option n°{seg_i + 1}. Image number: {analysis_i.t - 1}",
                     "current_image": current_image})
            if analysis_i.start is None:
                analysis_i.binary = np.repeat(np.expand_dims(analysis_i.origin, 0),
                                           analysis_i.converted_video.shape[0], axis=0)
                if self.parent().po.vars['color_number'] > 2:
                    self.message_from_thread_starting.emit(
                        f"Failed to detect motion. Redo image analysis (with only 2 colors?)")
                else:
                    self.message_from_thread_starting.emit(f"Tracking option n°{seg_i + 1} failed to detect motion")

            if self.parent().po.all['compute_all_options']:
                if seg_i == 0:
                    self.parent().po.motion.segmentation = analysis_i.binary
                elif seg_i == 1:
                    self.parent().po.motion.luminosity_segmentation = np.nonzero(analysis_i.binary)
                elif seg_i == 2:
                    self.parent().po.motion.gradient_segmentation = np.nonzero(analysis_i.binary)
                elif seg_i == 3:
                    self.parent().po.motion.logical_and = np.nonzero(analysis_i.binary)
                elif seg_i == 4:
                    self.parent().po.motion.logical_or = np.nonzero(analysis_i.binary)
            else:
                self.parent().po.motion.segmentation = analysis_i.binary

        # self.message_from_thread_starting.emit("If there are problems, change some parameters and try again")
        self.when_detection_finished.emit("Post processing done, read to see the result")



class VideoReaderThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super(VideoReaderThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        video_analysis = deepcopy(self.parent().po.motion.visu)
        self.message_from_thread.emit(
            {"current_image": video_analysis[0, ...], "message": f"Video preparation, wait..."})
        if self.parent().po.load_quick_full > 0:

            if self.parent().po.all['compute_all_options']:
                if self.parent().po.all['video_option'] == 0:
                    video_mask = self.parent().po.motion.segmentation
                else:
                    if self.parent().po.all['video_option'] == 1:
                        mask = self.parent().po.motion.luminosity_segmentation
                    elif self.parent().po.all['video_option'] == 2:
                        mask = self.parent().po.motion.gradient_segmentation
                    elif self.parent().po.all['video_option'] == 3:
                        mask = self.parent().po.motion.logical_and
                    elif self.parent().po.all['video_option'] == 4:
                        mask = self.parent().po.motion.logical_or
                    video_mask = np.zeros(self.parent().po.motion.dims[:3], dtype=np.uint8)
                    video_mask[mask[0], mask[1], mask[2]] = 1
            else:
                video_mask = np.zeros(self.parent().po.motion.dims[:3], dtype=np.uint8)
                if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    video_mask = self.parent().po.motion.segmentation

            if self.parent().po.load_quick_full == 1:
                video_mask = np.cumsum(video_mask.astype(np.uint32), axis=0)
                video_mask[video_mask > 0] = 1
                video_mask = video_mask.astype(np.uint8)
        logging.info(f"sum: {video_mask.sum()}")
        # timings = genfromtxt("timings.csv")
        for t in np.arange(self.parent().po.motion.dims[0]):
            mask = cv2.morphologyEx(video_mask[t, ...], cv2.MORPH_GRADIENT, cross_33)
            mask = np.stack((mask, mask, mask), axis=2)
            # current_image[current_image > 0] = self.parent().po.vars['contour_color']
            current_image = deepcopy(video_analysis[t, ...])
            current_image[mask > 0] = self.parent().po.vars['contour_color']
            self.message_from_thread.emit(
                {"current_image": current_image, "message": f"Reading in progress... Image number: {t}"}) #, "time": timings[t]
            time.sleep(1 / 50)
        self.message_from_thread.emit({"current_image": current_image, "message": ""})#, "time": timings[t]


class ChangeOneRepResultThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(ChangeOneRepResultThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.message_from_thread.emit(
            f"Arena n°{self.parent().po.all['arena']}: modifying its results...")
        # self.parent().po.motion2 = deepcopy(self.parent().po.motion)
        if self.parent().po.motion.start is None:
            self.parent().po.motion.binary = np.repeat(np.expand_dims(self.parent().po.motion.origin, 0),
                                                     self.parent().po.motion.converted_video.shape[0], axis=0).astype(np.uint8)
        else:
            if self.parent().po.all['compute_all_options']:
                if self.parent().po.all['video_option'] == 0:
                    self.parent().po.motion.binary = self.parent().po.motion.segmentation
                else:
                    if self.parent().po.all['video_option'] == 1:
                        mask = self.parent().po.motion.luminosity_segmentation
                    elif self.parent().po.all['video_option'] == 2:
                        mask = self.parent().po.motion.gradient_segmentation
                    elif self.parent().po.all['video_option'] == 3:
                        mask = self.parent().po.motion.logical_and
                    elif self.parent().po.all['video_option'] == 4:
                        mask = self.parent().po.motion.logical_or
                    self.parent().po.motion.binary = np.zeros(self.parent().po.motion.dims, dtype=np.uint8)
                    self.parent().po.motion.binary[mask[0], mask[1], mask[2]] = 1
            else:
                self.parent().po.motion.binary = np.zeros(self.parent().po.motion.dims[:3], dtype=np.uint8)
                if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    self.parent().po.motion.binary = self.parent().po.motion.segmentation

        if self.parent().po.vars['do_fading']:
            self.parent().po.motion.newly_explored_area = self.parent().po.newly_explored_area[:, self.parent().po.all['video_option']]
        self.parent().po.motion.max_distance = 9 * self.parent().po.vars['detection_range_factor']
        self.parent().po.motion.get_descriptors_from_binary(release_memory=False)
        self.parent().po.motion.detect_growth_transitions()
        self.parent().po.motion.networks_detection(False)
        self.parent().po.motion.study_cytoscillations(False)
        self.parent().po.motion.fractal_descriptions()
        self.parent().po.motion.get_descriptors_summary()
        self.parent().po.motion.change_results_of_one_arena()
        self.parent().po.motion = None
        # self.parent().po.motion = None
        self.message_from_thread.emit("")


class WriteVideoThread(QtCore.QThread):
    # message_from_thread_in_thread = QtCore.Signal(bool)
    def __init__(self, parent=None):
        super(WriteVideoThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        # self.message_from_thread_in_thread.emit({True})
        arena = self.parent().po.all['arena']
        if not self.parent().po.vars['already_greyscale']:
            write_video(self.parent().po.visu, f'ind_{arena}.npy')
        else:
            write_video(self.parent().po.converted_video, f'ind_{arena}.npy')


class RunAllThread(QtCore.QThread):
    message_from_thread = QtCore.Signal(str)
    image_from_thread = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super(RunAllThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        analysis_status = {"continue": True, "message": ""}
        message = self.set_current_folder(0)

        if self.parent().po.first_exp_ready_to_run:

            self.message_from_thread.emit(message + ": Write videos...")
            if not self.parent().po.vars['several_blob_per_arena'] and self.parent().po.sample_number != len(self.parent().po.bot):
                analysis_status["continue"] = False
                analysis_status["message"] = f"Wrong specimen number: redo the first image analysis."
                self.message_from_thread.emit(f"Wrong specimen number: restart Cellects and do another analysis.")
            else:
                analysis_status = self.run_video_writing(message)
                if analysis_status["continue"]:
                    self.message_from_thread.emit(message + ": Analyse all videos...")
                    analysis_status = self.run_motion_analysis(message)
                if analysis_status["continue"]:
                    if self.parent().po.all['folder_number'] > 1:
                        self.parent().po.all['folder_list'] = self.parent().po.all['folder_list'][1:]
                        self.parent().po.all['sample_number_per_folder'] = self.parent().po.all['sample_number_per_folder'][1:]
        else:
            self.parent().po.look_for_data()

        if analysis_status["continue"] and (not self.parent().po.first_exp_ready_to_run or self.parent().po.all['folder_number'] > 1):
            folder_number = np.max((len(self.parent().po.all['folder_list']), 1))

            for exp_i in np.arange(folder_number):
                if len(self.parent().po.all['folder_list']) > 0:
                    logging.info(self.parent().po.all['folder_list'][exp_i])
                self.parent().po.first_im = None
                self.parent().po.first_image = None
                self.parent().po.last_im = None
                self.parent().po.last_image = None
                self.parent().po.videos = None
                self.parent().po.top = None

                message = self.set_current_folder(exp_i)
                self.message_from_thread.emit(f'{message}, pre-processing...')
                self.parent().po.load_data_to_run_cellects_quickly()
                if not self.parent().po.first_exp_ready_to_run:
                    analysis_status = self.pre_processing()
                if analysis_status["continue"]:
                    self.message_from_thread.emit(message + ": Write videos from images before analysis...")
                    if not self.parent().po.vars['several_blob_per_arena'] and self.parent().po.sample_number != len(self.parent().po.bot):
                        self.message_from_thread.emit(f"Wrong specimen number: first image analysis is mandatory.")
                        analysis_status["continue"] = False
                        analysis_status["message"] = f"Wrong specimen number: first image analysis is mandatory."
                    else:
                        analysis_status = self.run_video_writing(message)
                        if analysis_status["continue"]:
                            self.message_from_thread.emit(message + ": Starting analysis...")
                            analysis_status = self.run_motion_analysis(message)

                if not analysis_status["continue"]:
                    # self.message_from_thread.emit(analysis_status["message"])
                    break
                # if not continue_analysis:
                #     self.message_from_thread.emit(f"Error: wrong folder or parameters")
                #     break
                # if not enough_memory:
                #     self.message_from_thread.emit(f"Error: not enough memory")
                #     break
                print(self.parent().po.vars['convert_for_motion'])
        if analysis_status["continue"]:
            if self.parent().po.all['folder_number'] > 1:
                self.message_from_thread.emit(f"Exp {self.parent().po.all['folder_list'][0]} to {self.parent().po.all['folder_list'][-1]} analyzed.")
            else:
                curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
                self.message_from_thread.emit(f'Exp {curr_path}, analyzed.')
        else:
            logging.error(message + " " + analysis_status["message"])
            self.message_from_thread.emit(message + " " + analysis_status["message"])

    def set_current_folder(self, exp_i):
        if self.parent().po.all['folder_number'] > 1:
            logging.info(f"Use {self.parent().po.all['folder_list'][exp_i]} folder")

            message = f"{str(self.parent().po.all['global_pathway'])[:6]} ... {self.parent().po.all['folder_list'][exp_i]}"
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][exp_i],
                                              self.parent().po.all['folder_list'][exp_i])
        else:
            message = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {message} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])
        return message

    def pre_processing(self):
        analysis_status = {"continue": True, "message": ""}
        logging.info("Pre-processing has started")
        if len(self.parent().po.data_list) > 0:
            self.parent().po.get_first_image()
            self.parent().po.fast_image_segmentation(True)
            self.parent().po.cropping(is_first_image=True)
            self.parent().po.get_average_pixel_size()
            try:
                analysis_status = self.parent().po.delineate_each_arena()
            except ValueError:
                analysis_status[
                    "message"] = f"Failed to detect the right cell(s) number: the first image analysis is mandatory."
                analysis_status["continue"] = False

            if analysis_status["continue"]:
                self.parent().po.data_to_save['exif'] = True
                self.parent().po.save_data_to_run_cellects_quickly()
                self.parent().po.data_to_save['exif'] = False
                # self.parent().po.extract_exif()
                self.parent().po.get_background_to_subtract()
                if len(self.parent().po.vars['analyzed_individuals']) != len(self.parent().po.top):
                    analysis_status["message"] = f"Failed to detect the right cell(s) number: the first image analysis is mandatory."
                    analysis_status["continue"] = False
                elif self.parent().po.top is None and self.parent().imageanalysiswindow.manual_delineation_flag:
                    analysis_status["message"] = f"Auto video delineation failed, use manual delineation tool"
                    analysis_status["continue"] = False
                else:
                    self.parent().po.get_origins_and_backgrounds_lists()
                    self.parent().po.get_last_image()
                    self.parent().po.fast_image_segmentation(is_first_image=False)
                    self.parent().po.find_if_lighter_backgnp.round()
            return analysis_status
        else:
            analysis_status["message"] = f"Wrong folder or parameters"
            analysis_status["continue"] = False
            return analysis_status

    def run_video_writing(self, message):
        analysis_status = {"continue": True, "message": ""}
        look_for_existing_videos = glob('ind_' + '*' + '.npy')
        there_already_are_videos = len(look_for_existing_videos) == len(self.parent().po.vars['analyzed_individuals'])
        logging.info(f"{len(look_for_existing_videos)} .npy video files found for {len(self.parent().po.vars['analyzed_individuals'])} arenas to analyze")
        do_write_videos = not there_already_are_videos or (
                there_already_are_videos and self.parent().po.all['overwrite_unaltered_videos'])
        if do_write_videos:
            logging.info(f"Starting video writing")
            # self.videos.write_videos_as_np_arrays(self.data_list, self.vars['convert_for_motion'], in_colors=self.vars['save_in_colors'])
            in_colors = not self.parent().po.vars['already_greyscale']
            self.parent().po.videos = OneVideoPerBlob(self.parent().po.first_image,
                                                      self.parent().po.starting_blob_hsize_in_pixels,
                                                      self.parent().po.all['raw_images'])
            self.parent().po.videos.left = self.parent().po.left
            self.parent().po.videos.right = self.parent().po.right
            self.parent().po.videos.top = self.parent().po.top
            self.parent().po.videos.bot = self.parent().po.bot
            self.parent().po.videos.first_image.shape_number = self.parent().po.sample_number
            bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining = self.parent().po.videos.prepare_video_writing(
                self.parent().po.data_list, self.parent().po.vars['min_ram_free'], in_colors)
            if analysis_status["continue"]:
                # Check that there is enough available RAM for one video par bunch and ROM for all videos
                if video_nb_per_bunch > 0 and rom_memory_required is None:
                    pat_tracker1 = PercentAndTimeTracker(bunch_nb * self.parent().po.vars['img_number'])
                    pat_tracker2 = PercentAndTimeTracker(len(self.parent().po.vars['analyzed_individuals']))
                    arena_percentage = 0
                    is_landscape = self.parent().po.first_image.image.shape[0] < self.parent().po.first_image.image.shape[1]
                    for bunch in np.arange(bunch_nb):
                        # Update the labels of arenas and the video_bunch to write
                        if bunch == (bunch_nb - 1) and remaining > 0:
                            arena = np.arange(bunch * video_nb_per_bunch, bunch * video_nb_per_bunch + remaining)
                        else:
                            arena = np.arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch)
                        if self.parent().po.videos.use_list_of_vid:
                            video_bunch = [np.zeros(sizes[i, :], dtype=np.uint8) for i in arena]
                        else:
                            video_bunch = np.zeros(np.append(sizes[0, :], len(arena)), dtype=np.uint8)
                        prev_img = None
                        images_done = bunch * self.parent().po.vars['img_number']
                        for image_i, image_name in enumerate(self.parent().po.data_list):
                            image_percentage, remaining_time = pat_tracker1.get_progress(image_i + images_done)
                            self.message_from_thread.emit(message + f" Step 1/2: Video writing ({np.round((image_percentage + arena_percentage) / 2, 2)}%)")
                            if not os.path.exists(image_name):
                                raise FileNotFoundError(image_name)
                            img = read_and_rotate(image_name, prev_img, self.parent().po.all['raw_images'], is_landscape, self.parent().po.first_image.crop_coord)
                            prev_img = deepcopy(img)
                            if self.parent().po.vars['already_greyscale'] and self.parent().po.reduce_image_dim:
                                img = img[:, :, 0]

                            for arena_i, arena_name in enumerate(arena):
                                try:
                                    sub_img = img[self.parent().po.top[arena_name]: (self.parent().po.bot[arena_name] + 1),
                                              self.parent().po.left[arena_name]: (self.parent().po.right[arena_name] + 1), ...]
                                    if self.parent().po.videos.use_list_of_vid:
                                        video_bunch[arena_i][image_i, ...] = sub_img
                                    else:
                                        if len(video_bunch.shape) == 5:
                                            video_bunch[image_i, :, :, :, arena_i] = sub_img
                                        else:
                                            video_bunch[image_i, :, :, arena_i] = sub_img
                                except ValueError:
                                    analysis_status["message"] = f"One (or more) image has a different size (restart)"
                                    analysis_status["continue"] = False
                                    logging.info(f"In the {message} folder: one (or more) image has a different size (restart)")
                                    break
                            if not analysis_status["continue"]:
                                break
                        if not analysis_status["continue"]:
                            break
                        if analysis_status["continue"]:
                            for arena_i, arena_name in enumerate(arena):
                                try:
                                    arena_percentage, eta = pat_tracker2.get_progress()
                                    self.message_from_thread.emit(message + f" Step 1/2: Video writing ({np.round((image_percentage + arena_percentage) / 2, 2)}%)")# , ETA {remaining_time}
                                    if self.parent().po.videos.use_list_of_vid:
                                        np.save(vid_names[arena_name], video_bunch[arena_i])
                                    else:
                                        if len(video_bunch.shape) == 5:
                                            np.save(vid_names[arena_name], video_bunch[:, :, :, :, arena_i])
                                        else:
                                            np.save(vid_names[arena_name], video_bunch[:, :, :, arena_i])
                                except OSError:
                                    self.message_from_thread.emit(message + f"full disk memory, clear space and retry")
                        logging.info(f"Bunch n°{bunch + 1} over {bunch_nb} saved.")
                    logging.info("When they exist, do not overwrite unaltered video")
                    self.parent().po.all['overwrite_unaltered_videos'] = False
                    self.parent().po.save_variable_dict()
                    self.parent().po.save_data_to_run_cellects_quickly()
                    analysis_status["message"] = f"Video writing complete."
                    if self.parent().po.videos is not None:
                        del self.parent().po.videos
                    return analysis_status
                else:
                    analysis_status["continue"] = False
                    if video_nb_per_bunch == 0:
                        memory_diff = self.parent().po.update_available_core_nb()
                        ram_message = f"{memory_diff}GB of additional RAM"
                    if rom_memory_required is not None:
                        rom_message = f"at least {rom_memory_required}GB of free ROM"

                    if video_nb_per_bunch == 0 and rom_memory_required is not None:
                        analysis_status["message"] = f"Requires {ram_message} and {rom_message} to run"
                        # self.message_from_thread.emit(f"Analyzing {message} requires {ram_message} and {rom_message} to run")
                    elif video_nb_per_bunch == 0:
                        analysis_status["message"] = f"Requires {ram_message} to run"
                        # self.message_from_thread.emit(f"Analyzing {message} requires {ram_message} to run")
                    elif rom_memory_required is not None:
                        analysis_status["message"] = f"Requires {rom_message} to run"
                        # self.message_from_thread.emit(f"Analyzing {message} requires {rom_message} to run")
                    logging.info(f"Cellects is not writing videos: insufficient memory")
                    return analysis_status
            else:
                return analysis_status


        else:
            logging.info(f"Cellects is not writing videos: unnecessary")
            analysis_status["message"] = f"Cellects is not writing videos: unnecessary"
            return analysis_status

    def run_motion_analysis(self, message):
        analysis_status = {"continue": True, "message": ""}
        logging.info(f"Starting motion analysis with the detection method n°{self.parent().po.all['video_option']}")
        self.parent().po.instantiate_tables()
        try:
            memory_diff = self.parent().po.update_available_core_nb()
            if self.parent().po.cores > 0: # i.e. enough memory
                if not self.parent().po.all['do_multiprocessing'] or self.parent().po.cores == 1:
                    self.message_from_thread.emit(f"{message} Step 2/2: Video analysis")
                    logging.info("fStarting sequential analysis")
                    tiii = default_timer()
                    pat_tracker = PercentAndTimeTracker(len(self.parent().po.vars['analyzed_individuals']))
                    for i, arena in enumerate(self.parent().po.vars['analyzed_individuals']):

                        l = [i, arena, self.parent().po.vars, True, True, False, None]
                        # l = [0, 1, self.parent().po.vars, True, False, False, None]
                        analysis_i = MotionAnalysis(l)
                        r = weakref.ref(analysis_i)
                        if not self.parent().po.vars['several_blob_per_arena']:
                            # Save basic statistics
                            self.parent().po.update_one_row_per_arena(i, analysis_i.one_descriptor_per_arena)


                            # Save descriptors in long_format
                            self.parent().po.update_one_row_per_frame(i * self.parent().po.vars['img_number'], arena * self.parent().po.vars['img_number'], analysis_i.one_row_per_frame)
                            
                            # Save cytosol_oscillations
                        if not pd.isna(analysis_i.one_descriptor_per_arena["first_move"]):
                            if self.parent().po.vars['oscilacyto_analysis']:
                                oscil_i = pd.DataFrame(
                                    np.c_[np.repeat(arena,
                                              analysis_i.clusters_final_data.shape[0]), analysis_i.clusters_final_data],
                                    columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance', 'coord_y', 'coord_x'])
                                if self.parent().po.one_row_per_oscillating_cluster is None:
                                    self.parent().po.one_row_per_oscillating_cluster = oscil_i
                                else:
                                    self.parent().po.one_row_per_oscillating_cluster = pd.concat((self.parent().po.one_row_per_oscillating_cluster, oscil_i))
                                
                        # Save efficiency visualization
                        self.parent().po.add_analysis_visualization_to_first_and_last_images(i, analysis_i.efficiency_test_1,
                                                                                 analysis_i.efficiency_test_2)
                        # Emit message to the interface
                        current_percentage, eta = pat_tracker.get_progress()
                        self.image_from_thread.emit({"current_image": self.parent().po.last_image.bgr,
                                                     "message": f"{message} Step 2/2: analyzed {arena} out of {len(self.parent().po.vars['analyzed_individuals'])} arenas ({current_percentage}%){eta}"})
                        del analysis_i
                    logging.info(f"Sequential analysis lasted {(default_timer() - tiii)/ 60} minutes")
                else:
                    self.message_from_thread.emit(
                        f"{message}, Step 2/2:  Analyse all videos using {self.parent().po.cores} cores...")

                    logging.info("fStarting analysis in parallel")

                    # new
                    tiii = default_timer()
                    arena_number = len(self.parent().po.vars['analyzed_individuals'])
                    self.advance = 0
                    self.pat_tracker = PercentAndTimeTracker(len(self.parent().po.vars['analyzed_individuals']),
                                                        core_number=self.parent().po.cores)

                    fair_core_workload = arena_number // self.parent().po.cores
                    cores_with_1_more = arena_number % self.parent().po.cores
                    EXTENTS_OF_SUBRANGES = []
                    bound = 0
                    parallel_organization = [fair_core_workload + 1 for _ in range(cores_with_1_more)] + [fair_core_workload for _ in range(self.parent().po.cores - cores_with_1_more)]
                    # Emit message to the interface
                    self.image_from_thread.emit({"current_image": self.parent().po.last_image.bgr,
                                                 "message": f"{message} Step 2/2: Analysis running on {self.parent().po.cores} CPU cores"})
                    for i, extent_size in enumerate(parallel_organization):
                        EXTENTS_OF_SUBRANGES.append((bound, bound := bound + extent_size))

                    try:
                        PROCESSES = []
                        subtotals = Manager().Queue()# Queue()
                        for extent in EXTENTS_OF_SUBRANGES:
                            # print(extent)
                            p = Process(target=motion_analysis_process, args=(extent[0], extent[1], self.parent().po.vars, subtotals))
                            p.start()
                            PROCESSES.append(p)

                        for p in PROCESSES:
                            p.join()

                        self.message_from_thread.emit(f"{message}, Step 2/2:  Saving all results...")
                        for i in range(subtotals.qsize()):
                            grouped_results = subtotals.get()
                            for j, results_i in enumerate(grouped_results):
                                if not self.parent().po.vars['several_blob_per_arena']:
                                    # Save basic statistics
                                    self.parent().po.update_one_row_per_arena(results_i['i'], results_i['one_row_per_arena'])
                                    # Save descriptors in long_format
                                    self.parent().po.update_one_row_per_frame(results_i['i'] * self.parent().po.vars['img_number'],
                                                                              results_i['arena'] * self.parent().po.vars['img_number'],
                                                                              results_i['one_row_per_frame'])
                                if not pd.isna(results_i['first_move']):
                                    # Save cytosol_oscillations
                                    if self.parent().po.vars['oscilacyto_analysis']:
                                        if self.parent().po.one_row_per_oscillating_cluster is None:
                                            self.parent().po.one_row_per_oscillating_cluster = results_i['one_row_per_oscillating_cluster']
                                        else:
                                            self.parent().po.one_row_per_oscillating_cluster = pd.concat((self.parent().po.one_row_per_oscillating_cluster, results_i['one_row_per_oscillating_cluster']))
                                        
                                # Save efficiency visualization
                                self.parent().po.add_analysis_visualization_to_first_and_last_images(results_i['i'], results_i['efficiency_test_1'],
                                                                                         results_i['efficiency_test_2'])
                        self.image_from_thread.emit(
                            {"current_image": self.parent().po.last_image.bgr,
                             "message": f"{message} Step 2/2: analyzed {len(self.parent().po.vars['analyzed_individuals'])} out of {len(self.parent().po.vars['analyzed_individuals'])} arenas ({100}%)"})

                        logging.info(f"Parallel analysis lasted {(default_timer() - tiii)/ 60} minutes")
                    except MemoryError:
                        analysis_status["continue"] = False
                        analysis_status["message"] = f"Not enough memory, reduce the core number for parallel analysis"
                        self.message_from_thread.emit(f"Analyzing {message} requires to reduce the core number for parallel analysis")
                        return analysis_status
                self.parent().po.save_tables()
                return analysis_status
            else:
                analysis_status["continue"] = False
                analysis_status["message"] = f"Requires an additional {memory_diff}GB of RAM to run"
                self.message_from_thread.emit(f"Analyzing {message} requires an additional {memory_diff}GB of RAM to run")
                return analysis_status
        except MemoryError:
            analysis_status["continue"] = False
            analysis_status["message"] = f"Requires additional memory to run"
            self.message_from_thread.emit(f"Analyzing {message} requires additional memory to run")
            return analysis_status


def motion_analysis_process(lower_bound: int, upper_bound: int, vars: dict, subtotals: Queue) -> None:
    grouped_results = []
    for i in range(lower_bound, upper_bound):
        analysis_i = MotionAnalysis([i, i + 1, vars, True, True, False, None])
        r = weakref.ref(analysis_i)
        results_i = dict()
        results_i['arena'] = analysis_i.one_descriptor_per_arena['arena']
        results_i['i'] = analysis_i.one_descriptor_per_arena['arena'] - 1
        arena = results_i['arena']
        i = arena - 1
        if not vars['several_blob_per_arena']:
            # Save basic statistics
            results_i['one_row_per_arena'] = analysis_i.one_descriptor_per_arena
            # Save descriptors in long_format
            results_i['one_row_per_frame'] = analysis_i.one_row_per_frame
            # Save cytosol_oscillations

        results_i['first_move'] = analysis_i.one_descriptor_per_arena["first_move"]
        if not pd.isna(analysis_i.one_descriptor_per_arena["first_move"]):
            if vars['oscilacyto_analysis']:
                results_i['clusters_final_data'] = analysis_i.clusters_final_data
                results_i['one_row_per_oscillating_cluster'] = pd.DataFrame(
                    np.c_[np.repeat(arena, analysis_i.clusters_final_data.shape[0]), analysis_i.clusters_final_data],
                    columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size', 'edge_distance', 'coord_y', 'coord_x'])
            if vars['fractal_analysis']:
                results_i['fractal_box_sizes'] = pd.DataFrame(analysis_i.fractal_boxes,
                               columns=['arena', 'time', 'fractal_box_lengths', 'fractal_box_widths'])

        # Save efficiency visualization
        results_i['efficiency_test_1'] = analysis_i.efficiency_test_1
        results_i['efficiency_test_2'] = analysis_i.efficiency_test_2
        grouped_results.append(results_i)

    subtotals.put(grouped_results)