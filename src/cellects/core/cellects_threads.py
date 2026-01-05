#!/usr/bin/env python3
"""
Cellects GUI module implementing threaded image/video analysis workflows.

This module provides a Qt-based interface for analyzing biological motion and growth through color space combinations,
segmentation strategies, arena delineation, and video processing. Uses QThreaded workers to maintain UI responsiveness
during computationally intensive tasks like segmentation, motion tracking, network detection, oscillation and fractal
analysis.

Main Components
LoadDataToRunCellectsQuicklyThread : Loads necessary data asynchronously for quick Cellects execution.
FirstImageAnalysisThread : Analyzes first image with automatic color space selection and segmentation.
LastImageAnalysisThread : Processes last frame analysis for optimized color space combinations.
CropScaleSubtractDelineateThread : Handles cropping, scaling, and arena boundary detection.
OneArenaThread : Performs complete motion analysis on a single arena with post-processing.
RunAllThread : Executes full batch analysis across multiple arenas/experiments.

Notes
Uses QThread for background operations to maintain UI responsiveness. Key workflows include automated color space
optimization, adaptive segmentation algorithms, multithreaded video processing, and arena delineation via geometric
analysis or manual drawing. Implements special post-processing for Physarum polycephalum network detection and oscillatory
activity tracking.
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
from numpy.typing import NDArray
import pandas as pd
from PySide6 import QtCore
from cellects.image_analysis.morphological_operations import cross_33, create_ellipse, create_mask, draw_img_with_mask, get_contours
from cellects.image_analysis.image_segmentation import convert_subtract_and_filter_video
from cellects.utils.formulas import scale_coordinates, bracket_to_uint8_image_contrast, get_contour_width_from_im_shape
from cellects.utils.load_display_save import (read_one_arena, read_and_rotate, read_rotate_crop_and_reduce_image,
                                              create_empty_videos, write_video)
from cellects.utils.utilitarian import PercentAndTimeTracker, reduce_path_len, split_dict
from cellects.core.motion_analysis import MotionAnalysis


class LoadDataToRunCellectsQuicklyThread(QtCore.QThread):
    """
    Load data to run Cellects quickly in a separate thread.

    This class is responsible for loading necessary data asynchronously
    in order to speed up the process of running Cellects.

    Signals
    -------
    message_when_thread_finished : Signal(str)
        Emitted when the thread finishes execution, indicating whether data loading was successful.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread = QtCore.Signal(str)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for quickly loading data to run Cellects.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(LoadDataToRunCellectsQuicklyThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Execute the data loading and preparation process for running cellects without setting all parameters in the GUI.

        This method triggers the parent object's methods to look for data and load it,
        then checks if the first experiment is ready. If so, it emits a message.
        """
        self.parent().po.look_for_data()
        self.parent().po.load_data_to_run_cellects_quickly()
        if self.parent().po.first_exp_ready_to_run:
            self.message_from_thread.emit("Data found, Video tracking window and Run all directly are available")
        else:
            self.message_from_thread.emit("")


class LookForDataThreadInFirstW(QtCore.QThread):
    """
    Find and process data in a separate thread.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    def __init__(self, parent=None):

        """
        Initialize the worker thread for finding data to run Cellects.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(LookForDataThreadInFirstW, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Run the data lookup process.
        """
        self.parent().po.look_for_data()


class LoadFirstFolderIfSeveralThread(QtCore.QThread):
    """
    Thread for loading data from the first folder if there are several folders.

    Signals
    -------
    message_when_thread_finished : Signal(bool)
        Emitted when the thread finishes execution, indicating whether data loading was successful.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_when_thread_finished = QtCore.Signal(bool)
    def __init__(self, parent=None):
        """
        Initialize the worker thread for loading data and parameters to run Cellects when analyzing several folders.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(LoadFirstFolderIfSeveralThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Run the data lookup process.
        """
        self.parent().po.load_data_to_run_cellects_quickly()
        if not self.parent().po.first_exp_ready_to_run:
            self.parent().po.get_first_image()
        self.message_when_thread_finished.emit(self.parent().po.first_exp_ready_to_run)


class GetFirstImThread(QtCore.QThread):
    """
    Thread for getting the first image.

    Signals
    -------
    message_when_thread_finished : Signal(bool)
        Emitted when the thread finishes execution, indicating whether data loading was successful.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_when_thread_finished = QtCore.Signal(np.ndarray)
    def __init__(self, parent=None):
        """
        Initialize the worker thread for loading the first image of one folder.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(GetFirstImThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Run the first image reading task in the parent process and emit a signal when it finishes.
        """
        self.parent().po.get_first_image()
        self.message_when_thread_finished.emit(self.parent().po.first_im)


class GetLastImThread(QtCore.QThread):
    """
    Thread for getting the last image.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    def __init__(self, parent=None):
        """
        Initialize the worker thread for loading the last image of one folder.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(GetLastImThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Run the last image reading task in the parent process.
        """
        self.parent().po.get_last_image()


class UpdateImageThread(QtCore.QThread):
    """
    Thread for updating GUI image.

    Signals
    -------
    message_when_thread_finished : Signal(bool)
        Emitted when the thread finishes execution, indicating whether image displaying was successful.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for updating the image displayed in GUI

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(UpdateImageThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Execute the image display process, including user input handling and mask application.

        This method performs several steps to analyze an image based on user input
        and saved mask coordinates. It updates the drawn image with segmentation masks,
        back masks, bio masks, and video contours.

        Other Parameters
        ----------------
        user_input : bool, optional
            Flag indicating whether user input is available.
        idx : list or numpy.ndarray, optional
            Coordinates of the user- defined region of interest.
        temp_mask_coord : list, optional
            Temporary mask coordinates.
        saved_coord : list, optional
            Saved mask coordinates.

        Notes
        -----
        - This function updates several attributes of `self.parent().imageanalysiswindow`.
        - Performance considerations include handling large images efficiently.
        - Important behavioral caveats: Ensure coordinates are within image bounds.
        """
        # I/ If this thread runs from user input, get the right coordinates
        # and convert them to fit the displayed image size
        user_input = len(self.parent().imageanalysiswindow.saved_coord) > 0 or len(self.parent().imageanalysiswindow.temporary_mask_coord) > 0
        dims = self.parent().imageanalysiswindow.drawn_image.shape
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
                idx, min_y, max_y, min_x, max_x = scale_coordinates(coord=idx, scale=sf, dims=dims)
                minmax = min_y, max_y, min_x, max_x

        if len(self.parent().imageanalysiswindow.temporary_mask_coord) == 0:
            # not_load
            # II/ If this thread aims at saving the last user input and displaying all user inputs:
            # Update the drawn_image according to every saved masks
            # 1) The segmentation mask
            # 2) The back_mask and bio_mask
            # 3) The automatically detected video contours
            # (re-)Initialize drawn image
            self.parent().imageanalysiswindow.drawn_image = deepcopy(self.parent().po.current_image)
            contour_width = get_contour_width_from_im_shape(dims)
            # 1) Add the segmentation mask to the image
            if self.parent().imageanalysiswindow.is_first_image_flag:
                im_combinations = self.parent().po.first_image.im_combinations
                im_mean = self.parent().po.first_image.image.mean()
            else:
                im_combinations = self.parent().po.last_image.im_combinations
                im_mean = self.parent().po.last_image.image.mean()
            # If there are image combinations, get the current corresponding binary image
            if im_combinations is not None and len(im_combinations) != 0:
                binary_idx = im_combinations[self.parent().po.current_combination_id]["binary_image"]
                # If it concerns the last image, only keep the contour coordinates
                binary_idx = cv2.dilate(get_contours(binary_idx), kernel=cross_33, iterations=contour_width)
                binary_idx = np.nonzero(binary_idx)
                # Color these coordinates in magenta on bright images, and in pink on dark images
                if im_mean > 126:
                    # Color the segmentation mask in magenta
                    self.parent().imageanalysiswindow.drawn_image[binary_idx[0], binary_idx[1], :] = np.array((20, 0, 150), dtype=np.uint8)
                else:
                    # Color the segmentation mask in pink
                    self.parent().imageanalysiswindow.drawn_image[binary_idx[0], binary_idx[1], :] = np.array((94, 0, 213), dtype=np.uint8)
            if user_input:# save
                if self.parent().imageanalysiswindow.back1_bio2 == 0:
                    mask_shape = self.parent().po.vars['arena_shape']
                elif self.parent().imageanalysiswindow.back1_bio2 == 1:
                    mask_shape = "rectangle"
                elif self.parent().imageanalysiswindow.back1_bio2 == 2:
                    mask_shape = self.parent().po.all['starting_blob_shape']
                    if mask_shape is None:
                        mask_shape = 'circle'
                # Save the user drawn mask
                mask = create_mask(dims, minmax, mask_shape)
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

            image = self.parent().imageanalysiswindow.drawn_image.copy()
            # 3) The automatically detected video contours
            if self.parent().imageanalysiswindow.delineation_done:  # add a mask of the video contour
                if self.parent().po.vars['contour_color'] == 255:
                    arena_contour_col = (240, 232, 202)
                else:
                    arena_contour_col = (138, 95, 18)
                # Draw the delineation mask of each arena
                for _i, (min_cy, max_cy, min_cx, max_cx) in enumerate(zip(self.parent().po.top, self.parent().po.bot, self.parent().po.left, self.parent().po.right)):
                    position = (min_cx + 25, min_cy + (max_cy - min_cy) // 2)
                    image = cv2.putText(image, f"{_i + 1}", position, cv2.FONT_HERSHEY_SIMPLEX, 1,  arena_contour_col + (255,),2)
                    if (max_cy - min_cy) < 0 or (max_cx - min_cx) < 0:
                        self.parent().imageanalysiswindow.message.setText("Error: the shape number or the detection is wrong")
                    image = draw_img_with_mask(image, dims, (min_cy, max_cy - 1, min_cx, max_cx - 1),
                                               self.parent().po.vars['arena_shape'], arena_contour_col, True, contour_width)
        else: #load
            if user_input:
                # III/ If this thread runs from user input: update the drawn_image according to the current user input
                # Just add the mask to drawn_image as quick as possible
                # Add user defined masks
                # Take the drawn image and add the temporary mask to it
                image = self.parent().imageanalysiswindow.drawn_image.copy()
                if self.parent().imageanalysiswindow.back1_bio2 == 2:
                    color = (17, 160, 212)
                    mask_shape = self.parent().po.all['starting_blob_shape']
                    if mask_shape is None:
                        mask_shape = 'circle'
                elif self.parent().imageanalysiswindow.back1_bio2 == 1:
                    color = (224, 160, 81)
                    mask_shape = "rectangle"
                else:
                    color = (0, 0, 0)
                    mask_shape = self.parent().po.vars['arena_shape']
                image = draw_img_with_mask(image, dims, minmax, mask_shape, color)
        self.parent().imageanalysiswindow.display_image.update_image(image)
        self.message_when_thread_finished.emit(True)


class FirstImageAnalysisThread(QtCore.QThread):
    """
    Thread for analyzing the first image of a given folder.

    Signals
    -------
    message_from_thread : Signal(str)
        Signal emitted when progress messages are available.
    message_when_thread_finished : Signal(bool)
        Signal emitted upon completion of the thread's task.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for analyzing the first image of a given folder

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(FirstImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Perform image analysis and segmentation based on the current state of the application.

        This function handles both bio-mask and background mask processing, emits status messages,
        computes average pixel size if necessary, and performs image segmentation or generates
        analysis options.

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Should have attributes:
            - parent: Reference to the parent object
            - message_from_thread.emit: Method to emit messages from the thread
            - message_when_thread_finished.emit: Method to signal thread completion

        Returns
        -------
        None
            This method does not return a value but emits messages and modifies the state of
            self.parent objects.
        Notes
        -----
        This method performs several complex operations involving image segmentation and
        analysis generation. It handles both bio-masks and background masks, computes average
        pixel sizes, and updates various state attributes on the parent object.
        """
        tic = default_timer()
        if self.parent().po.visualize or len(self.parent().po.first_im.shape) == 2:
            self.message_from_thread.emit("Image segmentation, wait...")
        else:
            self.message_from_thread.emit("Generating segmentation options, wait...")
        self.parent().po.full_first_image_segmentation(not self.parent().imageanalysiswindow.asking_first_im_parameters_flag,
                                                       self.parent().imageanalysiswindow.bio_mask, self.parent().imageanalysiswindow.back_mask)

        logging.info(f" image analysis lasted {np.floor((default_timer() - tic) / 60).astype(int)} minutes {np.round((default_timer() - tic) % 60).astype(int)} secondes")
        self.message_when_thread_finished.emit(True)


class LastImageAnalysisThread(QtCore.QThread):
    """
    Thread for analyzing the last image of a given folder.

    Signals
    -------
    message_from_thread : Signal(str)
        Signal emitted when progress messages are available.
    message_when_thread_finished : Signal(bool)
        Signal emitted upon completion of the thread's task.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for analyzing the last image of a given folder

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(LastImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Summary:
        Run the image processing and analysis pipeline based on current settings.

        Extended Description:
        This function initiates the workflow for image processing and analysis,
        including segmenting images, generating analysis options, and handling
        various masks and settings based on the current state of the parent object.

        Returns:
        --------
        None
            This method does not return a value. It emits signals to indicate the
            progress and completion of the processing tasks.

        Notes:
        ------
        This function uses various attributes from the parent class to determine
        how to process and analyze images. The specific behavior is heavily
        dependent on the state of these attributes.

        Attributes:
        -----------
        parent() : object
            The owner of this instance, containing necessary settings and methods.
        message_from_thread.emit(s : str) : signal
            Signal to indicate progress messages from the thread.
        message_when_thread_finished.emit(success : bool) : signal
            Signal to indicate the completion of the thread.
        """
        if self.parent().po.visualize or (len(self.parent().po.first_im.shape) == 2 and not self.parent().po.network_shaped):
            self.message_from_thread.emit("Image segmentation, wait...")
        else:
            self.message_from_thread.emit("Generating analysis options, wait...")
        self.parent().po.full_last_image_segmentation(self.parent().imageanalysiswindow.bio_mask, self.parent().imageanalysiswindow.back_mask)
        self.message_when_thread_finished.emit(True)


class CropScaleSubtractDelineateThread(QtCore.QThread):
    """
    Thread for detecting crop and arena coordinates.

    Signals
    -------
    message_from_thread : Signal(str)
        Signal emitted when progress messages are available.
    message_when_thread_finished : Signal(dict)
        Signal emitted upon completion of the thread's task.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread = QtCore.Signal(str)
    message_when_thread_finished = QtCore.Signal(dict)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for detecting crop and arena coordinates in the first image

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """

        super(CropScaleSubtractDelineateThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Start cropping if required, perform initial processing,
        and handle subsequent operations based on configuration.

        Extended Description
        --------------------
        This method initiates the cropping process if necessary,
        performs initial processing steps, and manages subsequent operations
        depending on whether multiple blobs are detected per arena.

        Notes
        -----
        This method uses several logging operations to track its progress.
        It interacts with various components of the parent object
        to perform necessary image processing tasks.
        """
        logging.info("Start cropping if required")
        analysis_status = {"continue": True, "message": ""}
        self.parent().po.cropping(is_first_image=True)
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
                analysis_status["message"] = "Image analysis failed to detect the right cell(s) number: restart the analysis."
                analysis_status['continue'] = False
            elif y_lim is None:
                analysis_status["message"] = "The shapes detected in the image did not allow automatic arena delineation."
                analysis_status['continue'] = False
            elif (y_lim == - 1).sum() != (y_lim == 1).sum():
                analysis_status["message"] = "Automatic arena delineation cannot work if one cell touches the image border."
                self.parent().po.first_image.y_boundaries = None
                analysis_status['continue'] = False
        if analysis_status['continue']:
            logging.info("Start automatic video delineation")
            analysis_status = self.parent().po.delineate_each_arena()
        else:
            self.parent().po.first_image.validated_shapes = np.zeros(self.parent().po.first_image.image.shape[:2], dtype=np.uint8)
            logging.info(analysis_status["message"])
        self.message_when_thread_finished.emit(analysis_status)


class SaveManualDelineationThread(QtCore.QThread):
    """
    Thread for saving user's defined arena delineation through the GUI.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    def __init__(self, parent=None):
        """
        Initialize the worker thread for saving the arena coordinates when the user draw them manually

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(SaveManualDelineationThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Do save the coordinates.
        """
        self.parent().po.left = np.zeros(self.parent().po.sample_number)
        self.parent().po.right = np.zeros(self.parent().po.sample_number)
        self.parent().po.top = np.zeros(self.parent().po.sample_number)
        self.parent().po.bot = np.zeros(self.parent().po.sample_number)
        for arena_i in np.arange(self.parent().po.sample_number):
            y, x = np.nonzero(self.parent().imageanalysiswindow.arena_mask == arena_i + 1)
            self.parent().po.left[arena_i] = np.min(x)
            self.parent().po.right[arena_i] = np.max(x)
            self.parent().po.top[arena_i] = np.min(y)
            self.parent().po.bot[arena_i] = np.max(y)
        self.parent().po.list_coordinates()
        self.parent().po.save_data_to_run_cellects_quickly()

        logging.info("Save manual video delineation")
        self.parent().po.vars['analyzed_individuals'] = np.arange(self.parent().po.sample_number) + 1


class GetExifDataThread(QtCore.QThread):
    """
    Thread for loading exif data from images.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """

    def __init__(self, parent=None):
        """
        Initialize the worker thread for looking for the exif data.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(GetExifDataThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Do extract exif data..
        """
        self.parent().po.extract_exif()


class CompleteImageAnalysisThread(QtCore.QThread):
    """
    Thread for completing the last image analysis.

    Signals
    -------
    message_when_thread_finished : Signal(bool)
        Signal emitted upon completion of the thread's task.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_when_thread_finished = QtCore.Signal(bool)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for completing the last image analysis

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(CompleteImageAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        self.parent().po.get_background_to_subtract()
        self.parent().po.get_origins_and_backgrounds_lists()
        self.parent().po.data_to_save['exif'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.all['bio_mask'] = None
        self.parent().po.all['back_mask'] = None
        if self.parent().imageanalysiswindow.bio_masks_number != 0:
            self.parent().po.all['bio_mask'] = np.nonzero(self.parent().imageanalysiswindow.bio_mask)
        if self.parent().imageanalysiswindow.back_masks_number != 0:
            self.parent().po.all['back_mask'] = np.nonzero(self.parent().imageanalysiswindow.back_mask)
        self.parent().po.complete_image_analysis()
        self.message_when_thread_finished.emit(True)


class PrepareVideoAnalysisThread(QtCore.QThread):
    """
    Thread for preparing video analysis.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """

    def __init__(self, parent=None):
        """
        Initialize the worker thread for ending up the last image analysis and preparing video analysis.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(PrepareVideoAnalysisThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Run the image processing pipeline for the last image of the current folder.

        This method handles background subtraction,
        image segmentation, and data saving.
        """
        self.parent().po.get_background_to_subtract()

        self.parent().po.get_origins_and_backgrounds_lists()

        if self.parent().po.last_image is None:
            self.parent().po.get_last_image()
            self.parent().po.fast_last_image_segmentation()
        self.parent().po.find_if_lighter_background()
        logging.info("The current (or the first) folder is ready to run")
        self.parent().po.first_exp_ready_to_run = True
        self.parent().po.data_to_save['exif'] = True
        self.parent().po.save_data_to_run_cellects_quickly()
        self.parent().po.data_to_save['exif'] = False


class SaveAllVarsThread(QtCore.QThread):
    """
    Thread for saving the GUI parameters and updating current folder.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """

    def __init__(self, parent=None):
        """
        Initialize the worker thread for saving the GUI parameters and updating current folder

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(SaveAllVarsThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Execute a sequence of operations to save data and update the current folder.

        This method performs several steps:
            1. Save variable dictionary.
            2. Set the current folder.
            3. Save data to run Cellects quickly without creating a new one if it doesn't exist.
        """
        self.parent().po.save_variable_dict()
        self._set_current_folder()
        self.parent().po.save_data_to_run_cellects_quickly(new_one_if_does_not_exist=False)

    def _set_current_folder(self):
        """
        Set the current folder based on conditions.

        Sets the current folder to the first one in the list if there are multiple
        folders, otherwise sets it to a reduced global pathway.
        """
        if self.parent().po.all['folder_number'] > 1: # len(self.parent().po.all['folder_list']) > 1:  # len(self.parent().po.all['folder_list']) > 0:
            logging.info(f"Use {self.parent().po.all['folder_list'][0]} folder")
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0],
                                              self.parent().po.all['folder_list'][0])
        else:
            curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {curr_path} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])


class OneArenaThread(QtCore.QThread):
    """
    Thread for completing the analysis of one particular arena in the current folder.

    Signals
    -------
    message_from_thread_starting : Signal(str)
        Signal emitted when the thread successfully starts.
    image_from_thread : Signal(dict)
        Signal emitted during the video reading or analysis to display images of the current status to the GUI.
    when_loading_finished : Signal(bool)
        Signal emitted when the video is completely loaded.
    when_detection_finished : Signal(str)
        Signal emitted when the video analysis is finished.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread_starting = QtCore.Signal(str)
    image_from_thread = QtCore.Signal(dict)
    when_loading_finished = QtCore.Signal(bool)
    when_detection_finished = QtCore.Signal(str)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for saving the analyzing one arena entirely

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(OneArenaThread, self).__init__(parent)
        self.setParent(parent)
        self._isRunning = False

    def run(self):
        """

        Run analysis on one arena.

        This method prepares and initiates the analysis process for a video by setting up required folders,
        loading necessary data, and performing pre-processing steps. It manages the state of running analysis and
        handles memory allocation for efficient processing.

        Notes
        -----
        - This method uses threading to handle long-running operations without blocking the main UI.
        - The memory allocation is dynamically adjusted based on available system resources.

        Attributes
        ----------
        self.parent().po.vars['convert_for_motion'] : dict
            Dictionary containing variables related to motion conversion.
        self.parent().po.first_exp_ready_to_run : bool
            Boolean indicating if the first experiment is ready to run.
        self.parent().po.cores : int
            Number of cores available for processing.
        self.parent().po.motion : object
            Object containing motion-related data and methods.
        self.parent().po.load_quick_full : int
            Number of arenas to load quickly for full detection.
        """
        continue_analysis = True
        self._isRunning = True
        self.message_from_thread_starting.emit("Video loading, wait...")

        self.set_current_folder()
        if not self.parent().po.first_exp_ready_to_run:
            self.parent().po.load_data_to_run_cellects_quickly()
            if not self.parent().po.first_exp_ready_to_run:
                #Need a look for data when Data to run Cellects quickly.pkl and 1 folder selected amon several
                continue_analysis = self.pre_processing()
        if continue_analysis:
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
        """
        Stops the running process.

        This method is used to safely halt the current process.
        """
        self._isRunning = False

    def set_current_folder(self):
        """

        Sets the current folder based on conditions.

        This method determines which folder to use and updates the current
        folder ID accordingly. If there are multiple folders, it uses the first folder
        from the list; otherwise, it uses a reduced global pathway as the current.
        """
        if self.parent().po.all['folder_number'] > 1:
            logging.info(f"Use {self.parent().po.all['folder_list'][0]} folder")
            self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0],
                                              self.parent().po.all['folder_list'][0])
        else:
            curr_path = reduce_path_len(self.parent().po.all['global_pathway'], 6, 10)
            logging.info(f"Use {curr_path} folder")
            self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'])

    def pre_processing(self):
        """
        Pre-processes the data for running Cellects on one arena.

        This function logs various stages of preprocessing, validates specimen numbers,
        performs necessary segmentations and data saving operations. It handles the
        initialization, image analysis, and background extraction processes to prepare
        the folder for further analysis.

        Returns
        -------
        bool
            Returns True if pre-processing completed successfully; False otherwise.
        """
        logging.info("Pre-processing has started")
        analysis_status = {"continue": True, "message": ""}

        self.parent().po.get_first_image()
        self.parent().po.fast_first_image_segmentation()
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
                    self.parent().po.fast_last_image_segmentation()
                    self.parent().po.find_if_lighter_backgnp.round()
                    logging.info("The current (or the first) folder is ready to run")
                    self.parent().po.first_exp_ready_to_run = True
        return analysis_status["continue"]

    def load_one_arena(self):
        """
        Load a single arena from images or video to perform motion analysis.
        """
        arena = self.parent().po.all['arena']
        i = np.nonzero(self.parent().po.vars['analyzed_individuals'] == arena)[0][0]
        true_frame_width = self.parent().po.right[i] - self.parent().po.left[i]# self.parent().po.vars['origin_list'][i].shape[1]
        if self.parent().po.all['overwrite_unaltered_videos'] and os.path.isfile(f'ind_{arena}.npy'):
            os.remove(f'ind_{arena}.npy')
        background = None
        background2 = None
        if self.parent().po.vars['subtract_background']:
            background = self.parent().po.vars['background_list'][i]
            if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
                background2 = self.parent().po.vars['background_list2'][i]
        vid_name = None
        if self.parent().po.vars['video_list'] is not None:
            vid_name = self.parent().po.vars['video_list'][i]
        visu, converted_video, converted_video2 = read_one_arena(self.parent().po.all['arena'],
            self.parent().po.vars['already_greyscale'], self.parent().po.vars['convert_for_motion'],
            None, true_frame_width, vid_name, background, background2)

        save_loaded_video: bool = False
        if visu is None or (self.parent().po.vars['already_greyscale'] and converted_video is None):
            cr = [self.parent().po.top[i], self.parent().po.bot[i],
                  self.parent().po.left[i], self.parent().po.right[i]]
            vids = create_empty_videos(self.parent().po.data_list, cr,
                self.parent().po.vars['lose_accuracy_to_save_memory'], self.parent().po.vars['already_greyscale'],
                self.parent().po.vars['convert_for_motion'])
            self.parent().po.visu, self.parent().po.converted_video, self.parent().po.converted_video2 = vids
            logging.info(f"Starting to load arena n°{arena} from images")

            prev_img = None
            pat_tracker = PercentAndTimeTracker(self.parent().po.vars['img_number'])
            is_landscape = self.parent().po.first_image.image.shape[0] < self.parent().po.first_image.image.shape[1]
            for image_i, image_name in enumerate(self.parent().po.data_list):
                current_percentage, eta = pat_tracker.get_progress()
                reduce_image_dim = self.parent().po.vars['already_greyscale'] and self.parent().po.reduce_image_dim
                img, prev_img = read_rotate_crop_and_reduce_image(image_name, prev_img,
                    self.parent().po.first_image.crop_coord, cr, self.parent().po.all['raw_images'], is_landscape,
                    reduce_image_dim)
                self.image_from_thread.emit(
                    {"message": f"Video loading: {current_percentage}%{eta}", "current_image": img})
                if self.parent().po.vars['already_greyscale']:
                    self.parent().po.converted_video[image_i, ...] = img
                else:
                    self.parent().po.visu[image_i, ...] = img

            if not self.parent().po.vars['already_greyscale']:
                msg = "Video conversion"
                if background is not None :
                    msg += ", background subtraction"
                if self.parent().po.vars['filter_spec'] is not None:
                    msg += ", filtering"
                msg += ", wait..."
                self.image_from_thread.emit({"message": msg, "current_image": img})
                converted_videos = convert_subtract_and_filter_video(self.parent().po.visu,
                                                                        self.parent().po.vars['convert_for_motion'],
                                                                        background, background2,
                                                                        self.parent().po.vars['lose_accuracy_to_save_memory'],
                                                                        self.parent().po.vars['filter_spec'])
                self.parent().po.converted_video, self.parent().po.converted_video2 = converted_videos

            save_loaded_video = True
            if self.parent().po.vars['already_greyscale']:
                self.videos_in_ram = self.parent().po.converted_video
            else:
                if self.parent().po.vars['convert_for_motion']['logical'] == 'None':
                    self.videos_in_ram = [self.parent().po.visu, deepcopy(self.parent().po.converted_video)]
                else:
                    self.videos_in_ram = [self.parent().po.visu, deepcopy(self.parent().po.converted_video),
                                          deepcopy(self.parent().po.converted_video2)]
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
        self.parent().po.motion.assess_motion_detection()
        self.when_loading_finished.emit(save_loaded_video)

        if self.parent().po.motion.visu is None:
            visu = bracket_to_uint8_image_contrast(self.parent().po.motion.converted_video)
            if len(visu.shape) == 3:
                visu = np.stack((visu, visu, visu), axis=3)
            self.parent().po.motion.visu = visu

    def detection(self):
        """
        Perform quick video segmentation and update motion detection parameters.

        This method is responsible for initiating a quick video segmentation process and updating the motion detection
        parameters accordingly. It handles duplicate video conversion based on certain logical conditions and computes
        video options.
        """
        self.message_from_thread_starting.emit(f"Quick video segmentation")
        self.parent().po.motion.converted_video = deepcopy(self.parent().po.converted_video)
        if self.parent().po.vars['convert_for_motion']['logical'] != 'None':
            self.parent().po.motion.converted_video2 = deepcopy(self.parent().po.converted_video2)
        self.parent().po.motion.detection(compute_all_possibilities=self.parent().po.all['compute_all_options'])
        if self.parent().po.all['compute_all_options']:
            self.parent().po.computed_video_options = np.ones(5, bool)
        else:
            self.parent().po.computed_video_options = np.zeros(5, bool)
            self.parent().po.computed_video_options[self.parent().po.all['video_option']] = True

    def post_processing(self):
        """
        Handle post-processing operations for motion analysis and video processing.

        Extended Description
        --------------------
        This method is responsible for managing various post-processing steps,
        including video segmentation, contour detection, and updating motion analysis
        parameters. It processes different video options based on the configuration
        settings and handles motion detection failures by emitting appropriate signals.

        Notes
        -----
        This method performs a series of operations that are computationally intensive.
        It leverages NumPy and OpenCV for image processing tasks. The method assumes
        that the parent object has been properly initialized with all required attributes
        and configurations.

        Attributes
        ----------
        self.parent().po.motion.smoothed_video : NoneType
            A placeholder for the smoothed video data.
        self.parent().po.vars['already_greyscale'] : bool
            Indicates if the video is already in greyscale format.
        self.parent().po.vars['convert_for_motion']['logical'] : str
            Indicates the logical conversion method for motion analysis.
        self.parent().po.converted_video : ndarray
            The converted video data for motion analysis.
        self.parent().po.converted_video2 : ndarray
            Another converted video data for motion analysis.
        self.parent().po.visu : ndarray
            The visual representation of the video data.
        self.videos_in_ram : list or tuple
            The videos currently in RAM, either a single video or multiple.
        self.parent().po.vars['color_number'] : int
            The number of colors in the video.
        self.parent().po.all['compute_all_options'] : bool
            Indicates if all options should be computed.
        self.parent().po.all['video_option'] : int
            The current video option to be processed.
        self.parent().po.newly_explored_area : ndarray
            The area newly explored during motion detection.
        self.parent().po.motion.start : int
            The start frame for motion analysis.
        self.parent().po.motion.step : int
            The step interval in frames for motion analysis.
        self.parent().po.motion.lost_frames : int
            The number of lost frames during motion analysis.
        self.parent().po.motion.substantial_growth : int
            The substantial growth threshold for motion detection.
        self.parent().po.all['arena'] : int
            The arena identifier used in motion analysis.
        self.parent().po.vars['do_fading'] : bool
            Indicates if fading effects should be applied.
        self.parent().po.motion.dims : tuple
            The dimensions of the motion data.
        analyses_to_compute : list or ndarray
            List of analysis options to compute based on configuration settings.
        args : list
            Arguments used for initializing the MotionAnalysis object.
        analysis_i : MotionAnalysis
            An instance of MotionAnalysis for each segment to be processed.
        mask : tuple or NoneType
            The mask used for different segmentation options.

        """
        self.parent().po.motion.smoothed_video = None
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
            self.parent().po.newly_explored_area = np.zeros((self.parent().po.motion.dims[0], 5), np.int64)
        for seg_i in analyses_to_compute:
            analysis_i = MotionAnalysis(args)
            r = weakref.ref(analysis_i)
            analysis_i.segmented = np.zeros(analysis_i.converted_video.shape[:3], dtype=np.uint8)
            if self.parent().po.all['compute_all_options']:
                if seg_i == 0:
                    analysis_i.segmented = self.parent().po.motion.segmented
                else:
                    if seg_i == 1:
                        mask = self.parent().po.motion.luminosity_segmentation
                    elif seg_i == 2:
                        mask = self.parent().po.motion.gradient_segmentation
                    elif seg_i == 3:
                        mask = self.parent().po.motion.logical_and
                    elif seg_i == 4:
                        mask = self.parent().po.motion.logical_or
                    analysis_i.segmented[mask[0], mask[1], mask[2]] = 1
            else:
                if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    analysis_i.segmented = self.parent().po.motion.segmented

            analysis_i.start = time_parameters[0]
            analysis_i.step = time_parameters[1]
            analysis_i.lost_frames = time_parameters[2]
            analysis_i.substantial_growth = time_parameters[3]
            analysis_i.origin_idx = self.parent().po.motion.origin_idx
            analysis_i.initialize_post_processing()
            analysis_i.t = analysis_i.start

            while self._isRunning and analysis_i.t < analysis_i.binary.shape[0]:
                analysis_i.update_shape(False)
                contours = np.nonzero(get_contours(analysis_i.binary[analysis_i.t - 1, :, :]))
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
                    self.parent().po.motion.segmented = analysis_i.binary
                elif seg_i == 1:
                    self.parent().po.motion.luminosity_segmentation = np.nonzero(analysis_i.binary)
                elif seg_i == 2:
                    self.parent().po.motion.gradient_segmentation = np.nonzero(analysis_i.binary)
                elif seg_i == 3:
                    self.parent().po.motion.logical_and = np.nonzero(analysis_i.binary)
                elif seg_i == 4:
                    self.parent().po.motion.logical_or = np.nonzero(analysis_i.binary)
            else:
                self.parent().po.motion.segmented = analysis_i.binary
        self.when_detection_finished.emit("Post processing done, read to see the result")


class VideoReaderThread(QtCore.QThread):
    """
    Thread for reading a video in the GUI.

    Signals
    --------
    message_from_thread : Signal(dict)
        Signal emitted during the video reading to display images to the GUI.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread = QtCore.Signal(dict)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for reading a video in the GUI

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(VideoReaderThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Summary
        -------
        Run the video analysis process, applying segmentation and contouring to each frame.

        Extended Description
        --------------------
        This method performs video analysis by segmenting frames based on selected options and overlaying contours.
        It also updates the UI with progress messages.

        Notes
        -----
        This method emits signals to update the UI with progress messages and current images.
        It uses OpenCV for morphological operations on video frames.
        """
        video_analysis = deepcopy(self.parent().po.motion.visu)
        self.message_from_thread.emit(
            {"current_image": video_analysis[0, ...], "message": f"Video preparation, wait..."})
        if self.parent().po.load_quick_full > 0:

            if self.parent().po.all['compute_all_options']:
                if self.parent().po.all['video_option'] == 0:
                    video_mask = self.parent().po.motion.segmented
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
                    video_mask = self.parent().po.motion.segmented

            if self.parent().po.load_quick_full == 1:
                video_mask = np.cumsum(video_mask.astype(np.uint32), axis=0)
                video_mask[video_mask > 0] = 1
                video_mask = video_mask.astype(np.uint8)
        frame_delay = (8 + np.log10(self.parent().po.motion.dims[0])) / self.parent().po.motion.dims[0]
        for t in np.arange(self.parent().po.motion.dims[0]):
            mask = cv2.morphologyEx(video_mask[t, ...], cv2.MORPH_GRADIENT, cross_33)
            mask = np.stack((mask, mask, mask), axis=2)
            current_image = deepcopy(video_analysis[t, ...])
            current_image[mask > 0] = self.parent().po.vars['contour_color']
            self.message_from_thread.emit(
                {"current_image": current_image, "message": f"Reading in progress... Image number: {t}"}) #, "time": timings[t]
            time.sleep(frame_delay)
        self.message_from_thread.emit({"current_image": current_image, "message": ""})#, "time": timings[t]


class ChangeOneRepResultThread(QtCore.QThread):
    """
    Thread for modifying the results of one arena.

    Signals
    --------
    message_from_thread : Signal(str)
        Signal emitted when the result is changed.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread = QtCore.Signal(str)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for changing the saved results in the current folder, for a particular arena

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(ChangeOneRepResultThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Modify the motion and results of an arena.

        Extended Description
        --------------------
        This method performs various operations on the motion data of an arena,
        including binary mask creation, descriptor computation, and transition
        detection. It also handles optional computations like fading effects and
        segmentation based on different video options.
        """
        self.message_from_thread.emit(
            f"Arena n°{self.parent().po.all['arena']}: modifying its results...")
        if self.parent().po.motion.start is None:
            self.parent().po.motion.binary = np.repeat(np.expand_dims(self.parent().po.motion.origin, 0),
                                                     self.parent().po.motion.converted_video.shape[0], axis=0).astype(np.uint8)
        else:
            if self.parent().po.all['compute_all_options']:
                if self.parent().po.all['video_option'] == 0:
                    self.parent().po.motion.binary = self.parent().po.motion.segmented
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
                    self.parent().po.motion.binary = self.parent().po.motion.segmented

        if self.parent().po.vars['do_fading']:
            self.parent().po.motion.newly_explored_area = self.parent().po.newly_explored_area[:, self.parent().po.all['video_option']]
        self.parent().po.motion.max_distance = 9 * self.parent().po.vars['detection_range_factor']
        self.parent().po.motion.get_descriptors_from_binary(release_memory=False)
        self.parent().po.motion.detect_growth_transitions()
        self.parent().po.motion.networks_analysis(False)
        self.parent().po.motion.study_cytoscillations(False)
        self.parent().po.motion.fractal_descriptions()
        self.parent().po.motion.change_results_of_one_arena()
        self.parent().po.motion = None
        # self.parent().po.motion = None
        self.message_from_thread.emit(f"Arena n°{self.parent().po.all['arena']}: analysis finished.")


class WriteVideoThread(QtCore.QThread):
    """
    Thread for writing one video per arena in the current folder.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    def __init__(self, parent=None):
        """
        Initialize the worker thread for writing the video corresponding to the current arena

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(WriteVideoThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Run the visualization or converted video for a specific arena and save it as an .npy file.

        Parameters
        ----------
        self : object
            The instance of the class containing this method.

        Other Parameters
        ----------------
        arena : str
            Name of the arena.

        already_greyscale : bool
            Flag indicating if the video is already in greyscale format.

        Raises
        ------
        FileNotFoundError
            When the path to write the video is not specified.
        """
        arena = self.parent().po.all['arena']
        if not self.parent().po.vars['already_greyscale']:
            write_video(self.parent().po.visu, f'ind_{arena}.npy')
        else:
            write_video(self.parent().po.converted_video, f'ind_{arena}.npy')


class RunAllThread(QtCore.QThread):
    """
    Thread for running the analysis on all arenas of the current folder.

    Signals
    --------
    image_from_thread : Signal(str)
        Signal emitted to send information to the user through the GUI
    message_from_thread : Signal(str)
        Signal emitted to send images showing the current status of the analysis to the GUI.

    Notes
    -----
    This class uses `QThread` to manage the process asynchronously.
    """
    message_from_thread = QtCore.Signal(str)
    image_from_thread = QtCore.Signal(dict)

    def __init__(self, parent=None):
        """
        Initialize the worker thread for running a complete analysis on one folder or a folder containing several
        folders.

        Parameters
        ----------
        parent : QObject, optional
            The parent object of this thread instance. In use, an instance of CellectsMainWidget class. Default is None.
        """
        super(RunAllThread, self).__init__(parent)
        self.setParent(parent)

    def run(self):
        """
        Run the analysis process for video writing and motion analysis.

        This method manages the overall flow of the analysis including setting up
        folders, loading data, writing videos from images, and performing motion
        analysis. It handles various conditions like checking if the specimen number
        matches expectations or if multiple experiments are ready to run.

        Returns
        -------
        dict
            A dictionary containing:
            - 'continue': bool indicating if the analysis should continue.
            - 'message': str with a relevant message about the current status.
        Notes
        -----
        This method uses several internal methods like `set_current_folder`,
        `run_video_writing`, and `run_motion_analysis` to perform the analysis steps.
        It also checks various conditions based on parent object attributes.
        """
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
                    break
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

    def pre_processing(self) -> dict:
        """
        Pre-processes the video data for further analysis.

        Extended Description
        ---------------------
        This method performs several preprocessing steps on the video data, including image segmentation,
        cropping, background subtraction, and origin detection. It also handles errors related to image analysis
        and manual delineation.

        Returns
        -------
        dict
            A dictionary containing `continue` (bool) and `message` (str). If analysis can continue, `continue`
            is True; otherwise, it's False and a descriptive message is provided.

        Raises
        ------
        **ValueError**
            When the correct number of cells cannot be detected in the first image.

        Notes
        -----
        * The method logs important preprocessing steps using `logging.info`.
        * Assumes that parent object (`self.parent().po`) has methods and attributes required for preprocessing.
        """
        analysis_status = {"continue": True, "message": ""}
        logging.info("Pre-processing has started")
        if len(self.parent().po.data_list) > 0:
            self.parent().po.get_first_image()
            self.parent().po.fast_first_image_segmentation()
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
                    self.parent().po.fast_last_image_segmentation()
                    self.parent().po.find_if_lighter_backgnp.round()
            return analysis_status
        else:
            analysis_status["message"] = f"Wrong folder or parameters"
            analysis_status["continue"] = False
            return analysis_status

    def run_video_writing(self, message: str) -> dict:
        """
        Initiate the process of writing videos from image data.

        Parameters
        ----------
        message : str
            A string to emit as a status update during video writing.

        Returns
        -------
        dict
            A dictionary containing the analysis status with keys:
            - "continue": bool indicating whether to continue video writing
            - "message": str providing a status or error message

        Raises
        ------
        FileNotFoundError
            If an image file specified in `data_list` does not exist.
        OSError
            If there is an issue writing to disk, such as when the disk is full.

        Notes
        -----
        This function manages video writing in batches, checking available memory
        and handling errors related to file sizes or missing images
        """
        analysis_status = {"continue": True, "message": ""}
        look_for_existing_videos = glob('ind_' + '*' + '.npy')
        there_already_are_videos = len(look_for_existing_videos) == len(self.parent().po.vars['analyzed_individuals'])
        logging.info(f"{len(look_for_existing_videos)} .npy video files found for {len(self.parent().po.vars['analyzed_individuals'])} arenas to analyze")
        do_write_videos = not self.parent().po.all['im_or_vid'] and (not there_already_are_videos or (there_already_are_videos and self.parent().po.all['overwrite_unaltered_videos']))
        if do_write_videos:
            logging.info(f"Starting video writing")
            # self.videos.write_videos_as_np_arrays(self.data_list, self.vars['convert_for_motion'], in_colors=self.vars['save_in_colors'])
            in_colors = not self.parent().po.vars['already_greyscale']
            self.parent().po.first_image.shape_number = self.parent().po.sample_number
            bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining, use_list_of_vid, is_landscape = self.parent().po.prepare_video_writing(
                self.parent().po.data_list, self.parent().po.vars['min_ram_free'], in_colors)
            if analysis_status["continue"]:
                # Check that there is enough available RAM for one video par bunch and ROM for all videos
                if video_nb_per_bunch > 0 and rom_memory_required is None:
                    pat_tracker1 = PercentAndTimeTracker(bunch_nb * self.parent().po.vars['img_number'])
                    pat_tracker2 = PercentAndTimeTracker(len(self.parent().po.vars['analyzed_individuals']))
                    arena_percentage = 0
                    for bunch in np.arange(bunch_nb):
                        # Update the labels of arenas and the video_bunch to write
                        if bunch == (bunch_nb - 1) and remaining > 0:
                            arena = np.arange(bunch * video_nb_per_bunch, bunch * video_nb_per_bunch + remaining)
                        else:
                            arena = np.arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch)
                        if use_list_of_vid:
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
                                    sub_img = img[self.parent().po.top[arena_name]: self.parent().po.bot[arena_name],
                                              self.parent().po.left[arena_name]: self.parent().po.right[arena_name], ...]
                                    if use_list_of_vid:
                                        video_bunch[arena_i][image_i, ...] = sub_img
                                    else:
                                        if len(video_bunch.shape) == 5:
                                            video_bunch[image_i, :, :, :, arena_i] = sub_img
                                        else:
                                            video_bunch[image_i, :, :, arena_i] = sub_img
                                except ValueError:
                                    analysis_status["message"] = f"Some images have incorrect size, reset all settings in advanced parameters"
                                    analysis_status["continue"] = False
                                    logging.info(f"Reset all settings in advanced parameters")
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
                                    if use_list_of_vid:
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
                    if analysis_status["continue"]:
                        analysis_status["message"] = f"Video writing complete."
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

    def run_motion_analysis(self, message: str) -> dict:
        """
        Run motion analysis on analyzed individuals with optional multiprocessing.

        This method processes video frames to analyze motion attributes of individuals.
        It can operate in either sequential or parallel mode based on available system
        resources and configuration settings. Analysis results are saved in multiple
        output formats.

        Parameters
        ----------
        message : str
            A status message to be displayed during the analysis process.

        Returns
        -------
        dict
            A dictionary containing the status of the motion analysis.

        Raises
        ------
        MemoryError
            If there is insufficient memory to perform the analysis in parallel.

        Notes
        -----
        Sequential mode is used when multiprocessing is disabled or only one core
        is available. Parallel mode utilizes multiple CPU cores for faster processing.
        """
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
                        analysis_i = MotionAnalysis(l)
                        r = weakref.ref(analysis_i)
                        if not self.parent().po.vars['several_blob_per_arena']:
                            # Save basic statistics
                            self.parent().po.update_one_row_per_arena(i, analysis_i.one_descriptor_per_arena)


                            # Save descriptors in long_format
                            self.parent().po.update_one_row_per_frame(i * self.parent().po.vars['img_number'], arena * self.parent().po.vars['img_number'], analysis_i.one_row_per_frame)

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
                                                                              (results_i['i'] + 1) * self.parent().po.vars['img_number'],
                                                                              results_i['one_row_per_frame'])

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
    """
    Motion Analysis Process for parallel computing

    Process a group of motion analysis results and store them in a queue.

    Parameters
    ----------
    lower_bound : int
        The lower bound index for the range of analysis.
    upper_bound : int
        The upper bound index (exclusive) for the range of analysis.
    vars : dict
        Dictionary containing variables and configurations for the motion analysis process.
    subtotals : Queue
        A queue to store intermediate results.
    Notes
    -----
    This function processes a range of motion analysis results based on the provided configuration variables and
    stores the intermediate results in a queue.
    """
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

        results_i['first_move'] = analysis_i.one_descriptor_per_arena["first_move"]
        # Save efficiency visualization
        results_i['efficiency_test_1'] = analysis_i.efficiency_test_1
        results_i['efficiency_test_2'] = analysis_i.efficiency_test_2
        grouped_results.append(results_i)

    subtotals.put(grouped_results)