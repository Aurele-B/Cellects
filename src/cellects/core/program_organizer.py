#!/usr/bin/env python3
"""This file contains the class constituting the link between the graphical interface and the computations.
 First, Cellects analyze one image in order to get a color space combination maximizing the contrast between the specimens
 and the background.
 Second, Cellects automatically delineate each arena.
 Third, Cellects write one video for each arena.
 Fourth, Cellects segments the video and apply post-processing algorithms to improve the segmentation.
 Fifth, Cellects extract variables and store them in .csv files.
"""

import pickle
import sys
import os
import logging
from copy import deepcopy
import psutil
import cv2
from numba.typed import Dict as TDict
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from psutil import virtual_memory
from pathlib import Path
import natsort
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from cellects.utils.load_display_save import extract_time
from cellects.image_analysis.network_functions import detect_network_dynamics, extract_graph_dynamics
from cellects.utils.load_display_save import PickleRick, readim, is_raw_image, read_h5_array, get_h5_keys
from cellects.utils.utilitarian import insensitive_glob, vectorized_len
from cellects.core.cellects_paths import CELLECTS_DIR, ALL_VARS_PKL_FILE
from cellects.config.all_vars_dict import DefaultDicts
from cellects.image_analysis.shape_descriptors import from_shape_descriptors_class, compute_one_descriptor_per_frame, compute_one_descriptor_per_colony
from cellects.image_analysis.morphological_operations import create_ellipse, rank_from_top_to_bottom_from_left_to_right, \
    get_quick_bounding_boxes, get_bb_with_moving_centers, get_contours, keep_one_connected_component, box_counting_dimension, prepare_box_counting
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.core.one_image_analysis import OneImageAnalysis, init_params
from cellects.utils.load_display_save import read_and_rotate, video2numpy
from cellects.image_analysis.morphological_operations import shape_selection, draw_img_with_mask


class ProgramOrganizer:
    """
    Organizes and manages variables, configuration settings, and processing workflows for motion analysis in a Cellects project.

    This class maintains global state and analysis-specific data structures, handles file operations,
    processes image/video inputs, and generates output tables. It provides methods to load/save configurations,
    segment images, track objects across frames, and export results with metadata.

    Attributes
    ----------
    one_arena_done : bool
        Flag indicating whether a single arena has been processed.
    reduce_image_dim : bool
        Whether image dimensions should be reduced (e.g., from color to grayscale).
    first_exp_ready_to_run : bool
        Indicates if the initial experiment setup is complete and ready for execution.
    data_to_save : dict of {str: bool}
        Specifies which data types (first image, coordinates, EXIF) require saving.
    videos : OneVideoPerBlob or None
        Video processing container instance.
    motion : MotionAnalysis or None
        Motion tracking and analysis module.
    all : dict
        Global configuration parameters for the entire workflow.
    vars : dict
        Analysis-specific variables used by `MotionAnalysis`.
    first_im, last_im : np.ndarray or None
        First and last images of the dataset for preprocessing.
    data_list : list of str
        List of video/image file paths in the working directory.
    computed_video_options : np.ndarray of bool
        Flags indicating which video processing options have been applied.
    one_row_per_arena, one_row_per_frame : pd.DataFrame or None
        Result tables for different levels of analysis (per arena, per frame, and oscillating clusters).

    Methods:
    --------
    save_variable_dict() : Save configuration dictionaries to file.
    load_variable_dict() : Load saved configuration or initialize defaults.
    look_for_data() : Discover video/image files in the working directory.
    update_folder_id(...) : Update folder-specific metadata based on file structure.
    ...

    """
    def __init__(self):
        """
            This class stores all variables required for analysis as well as
            methods to process it.
            Global variables (i.e. that does not concern the MotionAnalysis)
            are directly stored in self.
            Variables used in the MotionAnalysis class are stored in a dict
            called self.vars
        """
        if os.path.isfile('PickleRick.pkl'):
            os.remove('PickleRick.pkl')
        if os.path.isfile('PickleRick0.pkl'):
            os.remove('PickleRick0.pkl')
        self.one_arena_done: bool = False
        self.reduce_image_dim: bool = False
        self.first_exp_ready_to_run: bool = False
        self.data_to_save = {'first_image': False, 'exif': False, 'vars': False}
        self.sample_number = 1
        self.top = None
        self.motion = None
        self.analysis_instance = None
        self.computed_video_options = np.zeros(5, bool)
        self.vars = {}
        self.all = {}
        self.all['folder_list'] = []
        self.vars['first_detection_frame'] = 0
        self.first_im = None
        self.last_im = None
        self.vars['background_list'] = []
        self.starting_blob_hsize_in_pixels = None
        self.vars['first_move_threshold'] = None
        self.vars['convert_for_origin'] = None
        self.vars['convert_for_motion'] = None
        self.current_combination_id = 0
        self.data_list = []
        self.one_row_per_arena = None
        self.one_row_per_frame = None
        self.not_analyzed_individuals = None
        self.visualize: bool = True
        self.network_shaped: bool = False

    def update_variable_dict(self):
        """

        Update the `all` and `vars` dictionaries with new data from `DefaultDicts`.

        This method updates the `all` and `vars` dictionaries of the current object with
        data from a new instance of `DefaultDicts`. It checks if any keys or descriptors
        are missing and adds them accordingly.

        Examples
        --------
        >>> organizer = ProgramOrganizer()
        >>> organizer.update_variable_dict()
        """
        dd = DefaultDicts()
        all = len(dd.all) != len(self.all)
        vars = len(dd.vars) != len(self.vars)
        all_desc = not 'descriptors' in self.all or len(dd.all['descriptors']) != len(self.all['descriptors'])
        vars_desc = not 'descriptors' in self.vars or len(dd.vars['descriptors']) != len(self.vars['descriptors'])
        if all:
            for key, val in dd.all.items():
                if not key in self.all:
                    self.all[key] = val
        if vars:
            for key, val in dd.vars.items():
                if not key in self.vars:
                    self.vars[key] = val
        if all_desc:
            for key, val in dd.all['descriptors'].items():
                if not key in self.all['descriptors']:
                    self.all['descriptors'][key] = val
        if vars_desc:
            for key, val in dd.vars['descriptors'].items():
                if not key in self.vars['descriptors']:
                    self.vars['descriptors'][key] = val
        self._set_analyzed_individuals()

    def save_variable_dict(self):
        """
        Saves the configuration dictionaries (`self.all` and `self.vars`) to a pickle file.

        If bio_mask or back_mask are not required for all folders, they are excluded from the saved data.

        Notes
        -----
        This method is used to preserve state between Cellects sessions or restart scenarios.
        """
        logging.info("Save the parameters dictionaries in the Cellects folder")
        self.all['vars'] = self.vars
        all_vars = deepcopy(self.all)
        if not self.all['keep_cell_and_back_for_all_folders']:
            all_vars['bio_mask'] = None
            all_vars['back_mask'] = None
        pickle_rick = PickleRick(0)
        pickle_rick.write_file(all_vars, ALL_VARS_PKL_FILE)

    def load_variable_dict(self):
        """
        Loads configuration dictionaries from a pickle file if available, otherwise initializes defaults.

        Tries to load saved parameters. If the file doesn't exist or loading fails due to corruption,
        default values are used instead (logging relevant warnings).

        Raises
        ------
        FileNotFoundError
            If no valid configuration file is found and default initialization fails.

        Notes
        -----
        This method ensures robust operation by handling missing or corrupted configuration files gracefully.
        """
        if os.path.isfile(ALL_VARS_PKL_FILE):
            logging.info("Load the parameters from all_vars.pkl in the config of the Cellects folder")
            try:
                with open(ALL_VARS_PKL_FILE, 'rb') as fileopen:
                    self.all = pickle.load(fileopen)
                self.vars = self.all['vars']
                self.update_variable_dict()
                logging.info("Success to load the parameters dictionaries from the Cellects folder")
            except Exception as exc:
                logging.error(f"Initialize default parameters because error: {exc}")
                default_dicts = DefaultDicts()
                self.all = default_dicts.all
                self.vars = default_dicts.vars
        else:
            logging.info("Initialize default parameters")
            default_dicts = DefaultDicts()
            self.all = default_dicts.all
            self.vars = default_dicts.vars
        if self.all['cores'] == 1:
            self.all['cores'] = os.cpu_count() - 1

    def look_for_data(self):
        """
        Discovers all relevant video/image data in the working directory.

        Uses natural sorting to handle filenames with numeric suffixes. Validates file consistency and logs warnings
        if filename patterns are inconsistent across folders.

        Raises
        ------
        ValueError
            If no files match the specified naming convention.

        Notes
        -----
        This method assumes all data files follow a predictable pattern with numeric extensions. Use caution in
        unpredictable directory structures where this may fail silently or produce incorrect results.

        Examples
        --------
        >>> organizer.look_for_data()
        >>> print(organizer.data_list)
        ['/path/to/video1.avi', '/path/to/video2.avi']
        """
        os.chdir(Path(self.all['global_pathway']))
        logging.info(f"Dir: {self.all['global_pathway']}")
        self.data_list = insensitive_glob(self.all['radical'] + '*' + self.all['extension'])  # Provides a list ordered by last modification date
        self.all['folder_list'] = []
        self.all['folder_number'] = 1
        self.vars['first_detection_frame'] = 0
        if len(self.data_list) > 0:
            self._sort_data_list()
            self.sample_number = self.all['first_folder_sample_number']
        else:
            content = os.listdir()
            for obj in content:
                if not os.path.isfile(obj):
                    data_list = insensitive_glob(obj + "/" + self.all['radical'] + '*' + self.all['extension'])
                    if len(data_list) > 0:
                        self.all['folder_list'].append(obj)
                        self.all['folder_number'] += 1
            self.all['folder_list'] = np.sort(self.all['folder_list'])

            if isinstance(self.all['sample_number_per_folder'], int) or len(self.all['sample_number_per_folder']) == 1:
                self.all['sample_number_per_folder'] = np.repeat(self.all['sample_number_per_folder'],
                                                              self.all['folder_number'])

    def _sort_data_list(self):
        """
        Sorts the data list using natural sorting.

        Extended Description
        --------------------
        This function sorts the `data_list` attribute of an instance using the natsort library,
        which is useful when filenames have a mixture of numbers and letters.
        """
        if len(self.data_list) > 0:
            lengths = vectorized_len(self.data_list)
            if len(lengths) > 1 and np.max(np.diff(lengths)) > np.log10(len(self.data_list)):
                logging.error(f"File names present strong variations and cannot be correctly sorted.")
            wrong_images = np.nonzero(np.char.startswith(self.data_list, "Analysis efficiency, ", ))[0]
            for w_im in wrong_images[::-1]:
                self.data_list.pop(w_im)
            self.data_list = natsort.natsorted(self.data_list)
        if self.all['im_or_vid'] == 1:
            self.vars['video_list'] = self.data_list
        else:
            self.vars['video_list'] = None

    def update_folder_id(self, sample_number: int, folder_name: str=""):
        """
        Update the current working directory and data list based on the given sample number
        and optional folder name.

        Parameters
        ----------
        sample_number : int
            The number of samples to analyze.
        folder_name : str, optional
            The name of the folder to change to. Default is an empty string.

        Notes
        -----
        This function changes the current working directory to the specified folder name
        and updates the data list based on the file names in that directory. It also performs
        sorting of the data list and checks for strong variations in file names.

        """
        os.chdir(Path(self.all['global_pathway']) / folder_name)
        self.data_list = insensitive_glob(
            self.all['radical'] + '*' + self.all['extension'])  # Provides a list ordered by last modification date
        # Sorting is necessary when some modifications (like rotation) modified the last modification date
        self._sort_data_list()
        if self.all['im_or_vid'] == 1:
            self.sample_number = sample_number
        else:
            self.vars['img_number'] = len(self.data_list)
            self.sample_number = sample_number
        if not 'analyzed_individuals' in self.vars:
            self._set_analyzed_individuals()

    def _set_analyzed_individuals(self):
        """
        Set the analyzed individuals variable in the dataset.
        """
        self.vars['analyzed_individuals'] = np.arange(self.sample_number) + 1
        if self.not_analyzed_individuals is not None:
            self.vars['analyzed_individuals'] = np.delete(self.vars['analyzed_individuals'],
                                                       self.not_analyzed_individuals - 1)

    def load_data_to_run_cellects_quickly(self):
        """
        Load data from a pickle file and update the current state of the object.

        Summarizes, loads, and validates data needed to run Cellects,
        updating the object's state accordingly. If the necessary data
        are not present or valid, it ensures the experiment is marked as
        not ready to run.

        Parameters
        ----------
        self : CellectsObject
            The instance of the class (assumed to be a subclass of
            CellectsObject) that this method belongs to.

        Returns
        -------
        None

        Notes
        -----
        This function relies on the presence of a pickle file 'Data to run Cellects quickly.pkl'.
        It updates the state of various attributes based on the loaded data
        and logs appropriate messages.
        """
        self.analysis_instance = None
        self.first_im = None
        self.first_image = None
        self.last_image = None
        current_global_pathway = self.all['global_pathway']
        folder_number = self.all['folder_number']
        if folder_number > 1:
            folder_list = deepcopy(self.all['folder_list'])
            sample_number_per_folder = deepcopy(self.all['sample_number_per_folder'])

        if os.path.isfile('Data to run Cellects quickly.pkl'):
            pickle_rick = PickleRick()
            data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
            if data_to_run_cellects_quickly is None:
                data_to_run_cellects_quickly = {}

            if ('validated_shapes' in data_to_run_cellects_quickly) and ('all' in data_to_run_cellects_quickly) and ('bb_coord' in data_to_run_cellects_quickly['all']['vars']):
                logging.info("Success to load Data to run Cellects quickly.pkl from the user chosen directory")
                self.all = data_to_run_cellects_quickly['all']
                # If you want to add a new variable, first run an updated version of all_vars_dict,
                # then put a breakpoint here and run the following + self.save_data_to_run_cellects_quickly() :
                self.vars = self.all['vars']
                self.update_variable_dict()
                folder_changed = False
                if current_global_pathway != self.all['global_pathway']:
                    folder_changed = True
                    logging.info(
                        "Although the folder is ready, it is not at the same place as it was during creation, updating")
                    self.all['global_pathway'] = current_global_pathway
                if folder_number > 1:
                    self.all['global_pathway'] = current_global_pathway
                    self.all['folder_list'] = folder_list
                    self.all['folder_number'] = folder_number
                    self.all['sample_number_per_folder'] = sample_number_per_folder

                if len(self.data_list) == 0:
                    self.look_for_data()
                    if folder_changed and folder_number > 1 and len(self.all['folder_list']) > 0:
                        self.update_folder_id(self.all['sample_number_per_folder'][0], self.all['folder_list'][0])
                self.get_first_image()
                self.get_last_image()
                (ccy1, ccy2, ccx1, ccx2, self.top, self.bot, self.left, self.right) = data_to_run_cellects_quickly['all']['vars']['bb_coord']
                if self.all['automatically_crop']:
                    self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                    logging.info("Crop first image")
                    self.first_image.automatically_crop(self.first_image.crop_coord)
                    logging.info("Crop last image")
                    self.last_image.automatically_crop(self.first_image.crop_coord)
                else:
                    self.first_image.crop_coord = None
                self.first_image.validated_shapes = data_to_run_cellects_quickly['validated_shapes']
                self.first_image.im_combinations = []
                self.current_combination_id = 0
                self.first_image.im_combinations.append({})
                self.first_image.im_combinations[self.current_combination_id]['csc'] = self.vars['convert_for_origin']
                self.first_image.im_combinations[self.current_combination_id]['binary_image'] = self.first_image.validated_shapes
                self.first_image.im_combinations[self.current_combination_id]['shape_number'] = data_to_run_cellects_quickly['shape_number']
                
                self.first_exp_ready_to_run = True
                if self.vars['subtract_background'] and len(self.vars['background_list']) == 0:
                    self.first_exp_ready_to_run = False
            else:
                self.first_exp_ready_to_run = False
        else:
            self.first_exp_ready_to_run = False
        if self.first_exp_ready_to_run:
            logging.info("The current (or the first) folder is ready to run")
        else:
            logging.info("The current (or the first) folder is not ready to run")

    def save_data_to_run_cellects_quickly(self, new_one_if_does_not_exist: bool=True):
        """
        Save data to a pickled file if it does not exist or update existing data.

        Parameters
        ----------
        new_one_if_does_not_exist : bool, optional
            Whether to create a new data file if it does not already exist.
            Default is True.

        Notes
        -----
        This method logs various information about its operations and handles the writing of data to a pickled file.
        """
        data_to_run_cellects_quickly = None
        if os.path.isfile('Data to run Cellects quickly.pkl'):
            logging.info("Update -Data to run Cellects quickly.pkl- in the user chosen directory")
            pickle_rick = PickleRick()
            data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
            if data_to_run_cellects_quickly is None:
                os.remove('Data to run Cellects quickly.pkl')
                logging.error("Failed to load Data to run Cellects quickly.pkl before update. Remove pre existing.")
        else:
            if new_one_if_does_not_exist:
                logging.info("Create Data to run Cellects quickly.pkl in the user chosen directory")
                data_to_run_cellects_quickly = {}
        if data_to_run_cellects_quickly is not None:
            if self.data_to_save['first_image']:
                data_to_run_cellects_quickly['validated_shapes'] = self.first_image.im_combinations[self.current_combination_id]['binary_image']
                data_to_run_cellects_quickly['shape_number'] = self.first_image.im_combinations[self.current_combination_id]['shape_number']
            if self.data_to_save['exif']:
                self.vars['exif'] = self.extract_exif()
            self.all['vars'] = self.vars
            data_to_run_cellects_quickly['all'] = self.all
            pickle_rick = PickleRick()
            pickle_rick.write_file(data_to_run_cellects_quickly, 'Data to run Cellects quickly.pkl')

    def list_coordinates(self):
        """
        Summarize the coordinates of images and video.

        Combine the crop coordinates from the first image with additional
        coordinates for left, right, top, and bottom boundaries to form a list of
        video coordinates. If the crop coordinates are not already set, initialize
        them to cover the entire image.

        Returns
        -------
        list of int
            A list containing the coordinates [left, right, top, bottom] for video.

        """
        if self.first_image.crop_coord is None:
            self.first_image.crop_coord = [0, self.first_image.image.shape[0], 0, self.first_image.image.shape[1]]
        self.vars['bb_coord'] = self.first_image.crop_coord + [self.top, self.bot, self.left, self.right]
        self.all['overwrite_unaltered_videos'] = True

    def get_first_image(self, first_im: NDArray=None, sample_number: int=None):
        """
        Load and process the first image or frame from a video.

        This method handles loading the first image or the first frame of a video
        depending on whether the data is an image or a video. It performs necessary
        preprocessing and initializes relevant attributes for subsequent analysis.
        """
        if sample_number is not None:
            self.sample_number = sample_number
        self.reduce_image_dim = False
        if first_im is not None:
            self.first_im = first_im
        else:
            logging.info("Load first image")
            if self.all['im_or_vid'] == 1:
                if self.analysis_instance is None:
                    self.analysis_instance = video2numpy(self.data_list[0])
                    self.sample_number = len(self.data_list)
                    self.vars['img_number'] = self.analysis_instance.shape[0]
                    self.first_im = self.analysis_instance[0, ...]
                    self.vars['dims'] = self.analysis_instance.shape[:3]
                else:
                    self.first_im = self.analysis_instance[self.vars['first_detection_frame'], ...]

            else:
                self.vars['img_number'] = len(self.data_list)
                self.all['raw_images'] = is_raw_image(self.data_list[0])
                self.first_im = readim(self.data_list[self.vars['first_detection_frame']], self.all['raw_images'])
                self.vars['dims'] = [self.vars['img_number'], self.first_im.shape[0], self.first_im.shape[1]]

                if len(self.first_im.shape) == 3:
                    if np.all(np.equal(self.first_im[:, :, 0], self.first_im[:, :, 1])) and np.all(
                            np.equal(self.first_im[:, :, 1], self.first_im[:, :, 2])):
                        self.reduce_image_dim = True
                    if self.reduce_image_dim:
                        self.first_im = self.first_im[:, :, 0]

        self.first_image = OneImageAnalysis(self.first_im, self.sample_number)
        self.vars['already_greyscale'] = self.first_image.already_greyscale
        if self.vars['already_greyscale']:
            self.vars["convert_for_origin"] = {"bgr": np.array((1, 1, 1), dtype=np.uint8), "logical": "None"}
            self.vars["convert_for_motion"] = {"bgr": np.array((1, 1, 1), dtype=np.uint8), "logical": "None"}
        if np.mean((np.mean(self.first_image.image[2, :, ...]), np.mean(self.first_image.image[-3, :, ...]), np.mean(self.first_image.image[:, 2, ...]), np.mean(self.first_image.image[:, -3, ...]))) > 127:
            self.vars['contour_color']: np.uint8 = 0
        else:
            self.vars['contour_color']: np.uint8 = 255
        if self.vars['first_detection_frame'] > 0:
            self.vars['origin_state'] = 'invisible'

    def get_last_image(self, last_im: NDArray=None):
        """

        Load the last image from a video or image list and process it based on given parameters.

        Parameters
        ----------
        last_im : NDArray, optional
            The last image to be loaded. If not provided, the last image will be loaded from the data list.
        """
        logging.info("Load last image")
        if last_im is not None:
            self.last_im = last_im
        else:
            if self.all['im_or_vid'] == 1:
                self.last_im = self.analysis_instance[-1, ...]
            else:
                is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
                self.last_im = read_and_rotate(self.data_list[-1], self.first_im, self.all['raw_images'], is_landscape)
                if self.reduce_image_dim:
                    self.last_im = self.last_im[:, :, 0]
        self.last_image = OneImageAnalysis(self.last_im)

    def extract_exif(self):
        """
        Extract EXIF data from image or video files.

        Notes
        -----
        If `extract_time_interval` is True and unsuccessful, arbitrary time steps will be used.
        Timings are normalized to minutes for consistency across different files.
        """
        self.vars['time_step_is_arbitrary'] = True
        if self.all['im_or_vid'] == 1:
            if not 'dims' in self.vars:
                self.vars['dims'] = self.analysis_instance.shape[:3]
            timings = np.arange(self.vars['dims'][0])
        else:
            timings = np.arange(len(self.data_list))
            if sys.platform.startswith('win'):
                pathway = os.getcwd() + '\\'
            else:
                pathway = os.getcwd() + '/'
            if not 'extract_time_interval' in self.all:
                self.all['extract_time_interval'] = True
            if self.all['extract_time_interval']:
                self.vars['time_step'] = 1
                try:
                    timings = extract_time(self.data_list, pathway, self.all['raw_images'])
                    timings = timings - timings[0]
                    timings = timings / 60
                    time_step = np.diff(timings)
                    if len(time_step) > 0:
                        time_step = np.mean(time_step)
                        digit_nb = 0
                        for i in str(time_step):
                            if i in {'.'}:
                                pass
                            elif i in {'0'}:
                                digit_nb += 1
                            else:
                                break
                        self.vars['time_step'] = np.round(time_step, digit_nb + 1)
                        self.vars['time_step_is_arbitrary'] = False
                except:
                    pass
            else:
                timings = np.arange(0, len(self.data_list) * self.vars['time_step'], self.vars['time_step'])
                self.vars['time_step_is_arbitrary'] = False
        return timings

    def fast_first_image_segmentation(self):
        """
        Segment the first or subsequent image in a series for biological and background masks.

        Notes
        -----
        This function processes the first or subsequent image in a sequence, applying biological and background masks,
        segmenting the image, and updating internal data structures accordingly. The function is specific to handling
        image sequences for biological analysis

        """
        if not "color_number" in self.vars:
            self.update_variable_dict()
        if self.vars['convert_for_origin'] is None:
            self.vars['convert_for_origin'] = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
        self.first_image.convert_and_segment(self.vars['convert_for_origin'], self.vars["color_number"],
                                             self.all["bio_mask"], self.all["back_mask"], subtract_background=None,
                                             subtract_background2=None,
                                             rolling_window_segmentation=self.vars["rolling_window_segmentation"],
                                             filter_spec=self.vars["filter_spec"])
        if not self.first_image.drift_correction_already_adjusted:
            self.vars['drift_already_corrected'] = self.first_image.check_if_image_border_attest_drift_correction()
            if self.vars['drift_already_corrected']:
                logging.info("Cellects detected that the images have already been corrected for drift")
                self.first_image.convert_and_segment(self.vars['convert_for_origin'], self.vars["color_number"],
                                                     self.all["bio_mask"], self.all["back_mask"],
                                                     subtract_background=None, subtract_background2=None,
                                                     rolling_window_segmentation=self.vars["rolling_window_segmentation"],
                                                     filter_spec=self.vars["filter_spec"],
                                                     allowed_window=self.first_image.drift_mask_coord)

        shapes_features = shape_selection(self.first_image.binary_image, true_shape_number=self.sample_number,
                                          horizontal_size=self.starting_blob_hsize_in_pixels,
                                          spot_shape=self.all['starting_blob_shape'],
                                          several_blob_per_arena=self.vars['several_blob_per_arena'],
                                          bio_mask=self.all["bio_mask"], back_mask=self.all["back_mask"])
        self.first_image.validated_shapes, shape_number, stats, centroids = shapes_features
        self.first_image.shape_number = shape_number
        if self.first_image.im_combinations is None:
            self.first_image.im_combinations = []
        if len(self.first_image.im_combinations) == 0:
            self.first_image.im_combinations.append({})
        self.current_combination_id = np.min((self.current_combination_id, len(self.first_image.im_combinations) - 1))
        self.first_image.im_combinations[self.current_combination_id]['csc'] = self.vars['convert_for_origin']
        self.first_image.im_combinations[self.current_combination_id]['binary_image'] = self.first_image.validated_shapes
        if self.first_image.greyscale is not None:
            greyscale = self.first_image.greyscale
        else:
            greyscale = self.first_image.image
        self.first_image.im_combinations[self.current_combination_id]['converted_image'] = bracket_to_uint8_image_contrast(greyscale)
        self.first_image.im_combinations[self.current_combination_id]['shape_number'] = shape_number

    def fast_last_image_segmentation(self, bio_mask: NDArray[np.uint8] = None, back_mask: NDArray[np.uint8] = None):
        """
        Segment the first or subsequent image in a series for biological and background masks.

        Parameters
        ----------
        bio_mask : NDArray[np.uint8], optional
            The biological mask to be applied to the image.
        back_mask : NDArray[np.uint8], optional
            The background mask to be applied to the image.

        Returns
        -------
        None

        Notes
        -----
        This function processes the first or subsequent image in a sequence, applying biological and background masks,
        segmenting the image, and updating internal data structures accordingly. The function is specific to handling
        image sequences for biological analysis

        """
        if self.vars['convert_for_motion'] is None:
            self.vars['convert_for_motion'] = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
        self.cropping(is_first_image=False)
        self.last_image.convert_and_segment(self.vars['convert_for_motion'], self.vars["color_number"],
                                            bio_mask, back_mask, self.first_image.subtract_background,
                                            self.first_image.subtract_background2,
                                            rolling_window_segmentation=self.vars["rolling_window_segmentation"],
                                            filter_spec=self.vars["filter_spec"])
        if self.vars['drift_already_corrected'] and not self.last_image.drift_correction_already_adjusted and not self.vars["rolling_window_segmentation"]['do']:
            self.last_image.check_if_image_border_attest_drift_correction()
            self.last_image.convert_and_segment(self.vars['convert_for_motion'], self.vars["color_number"],
                                                bio_mask, back_mask, self.first_image.subtract_background,
                                                self.first_image.subtract_background2,
                                                allowed_window=self.last_image.drift_mask_coord,
                                                filter_spec=self.vars["filter_spec"])
        
        if self.last_image.im_combinations is None:
            self.last_image.im_combinations = []
        if len(self.last_image.im_combinations) == 0:
            self.last_image.im_combinations.append({})
        self.current_combination_id = np.min((self.current_combination_id, len(self.last_image.im_combinations) - 1))
        self.last_image.im_combinations[self.current_combination_id]['csc'] = self.vars['convert_for_motion']
        self.last_image.im_combinations[self.current_combination_id]['binary_image'] = self.last_image.binary_image
        if self.last_image.greyscale is not None:
            greyscale = self.last_image.greyscale
        else:
            greyscale = self.last_image.image
        self.last_image.im_combinations[self.current_combination_id]['converted_image'] = bracket_to_uint8_image_contrast(greyscale)

    def save_user_masks(self, bio_mask=None, back_mask=None):
        self.all["bio_mask"] = None
        self.all["back_mask"] = None
        if self.all['keep_cell_and_back_for_all_folders']:
            self.all["bio_mask"] = bio_mask
            self.all["back_mask"] = back_mask

    def full_first_image_segmentation(self, first_param_known: bool, bio_mask: NDArray[np.uint8] = None, back_mask: NDArray[np.uint8] = None):
        if bio_mask.any():
            shape_nb, ordered_image = cv2.connectedComponents((bio_mask > 0).astype(np.uint8))
            shape_nb -= 1
            bio_mask = np.nonzero(bio_mask)
        else:
            shape_nb = 0
            bio_mask = None
        if back_mask.any():
            back_mask = np.nonzero(back_mask)
        else:
            back_mask = None
        self.save_user_masks(bio_mask=bio_mask, back_mask=back_mask)
        if self.visualize or len(self.first_im.shape) == 2:
            if not first_param_known and self.all['scale_with_image_or_cells'] == 0 and self.all["set_spot_size"]:
                self.get_average_pixel_size()
            else:
                self.starting_blob_hsize_in_pixels = None
            self.fast_first_image_segmentation()
            if not self.vars['several_blob_per_arena'] and bio_mask is not None and shape_nb == self.sample_number and self.first_image.im_combinations[self.current_combination_id]['shape_number'] != self.sample_number:
                self.first_image.im_combinations[self.current_combination_id]['shape_number'] = shape_nb
                self.first_image.shape_number = shape_nb
                self.first_image.validated_shapes = (ordered_image > 0).astype(np.uint8)
                self.first_image.im_combinations[self.current_combination_id]['binary_image'] = self.first_image.validated_shapes
        else:

            params = init_params()
            params['is_first_image'] = True
            params['blob_nb'] = self.sample_number
            if self.vars["color_number"] > 2:
                params['kmeans_clust_nb'] = self.vars["color_number"]
            params['bio_mask'] = self.all["bio_mask"]
            params['back_mask'] = self.all["back_mask"]
            params['filter_spec'] = self.vars["filter_spec"]

            if first_param_known:
                if self.all['scale_with_image_or_cells'] == 0:
                    self.get_average_pixel_size()
                else:
                    self.starting_blob_hsize_in_pixels = None
                params['several_blob_per_arena'] = self.vars['several_blob_per_arena']
                params['blob_shape'] = self.all['starting_blob_shape']
                params['blob_size'] = self.starting_blob_hsize_in_pixels

            self.first_image.find_color_space_combinations(params)

    def full_last_image_segmentation(self, bio_mask: NDArray[np.uint8] = None, back_mask: NDArray[np.uint8] = None):
        if bio_mask.any():
            bio_mask = np.nonzero(bio_mask)
        else:
            bio_mask = None
        if back_mask.any():
            back_mask = np.nonzero(back_mask)
        else:
            back_mask = None
        if self.last_im is None:
            self.get_last_image()
        self.cropping(False)
        self.get_background_to_subtract()
        if self.visualize or (len(self.first_im.shape) == 2 and not self.network_shaped):
            self.fast_last_image_segmentation(bio_mask=bio_mask, back_mask=back_mask)
        else:
            arenas_mask = None
            if self.all['are_gravity_centers_moving'] != 1:
                cr = [self.top, self.bot, self.left, self.right]
                arenas_mask = np.zeros_like(self.first_image.validated_shapes)
                for _i in np.arange(len(self.vars['analyzed_individuals'])):
                    if self.vars['arena_shape'] == 'circle':
                        ellipse = create_ellipse(cr[1][_i] - cr[0][_i], cr[3][_i] - cr[2][_i])
                        arenas_mask[cr[0][_i]: cr[1][_i], cr[2][_i]:cr[3][_i]] = ellipse
                    else:
                        arenas_mask[cr[0][_i]: cr[1][_i], cr[2][_i]:cr[3][_i]] = 1
            if self.network_shaped:
                self.last_image.network_detection(arenas_mask, csc_dict=self.vars["convert_for_motion"], lighter_background=None, bio_mask=bio_mask, back_mask=back_mask)
            else:
                ref_image = self.first_image.validated_shapes
                params = init_params()
                params['is_first_image'] = False
                params['several_blob_per_arena'] = self.vars['several_blob_per_arena']
                params['blob_nb'] = self.sample_number
                params['arenas_mask'] = arenas_mask
                params['ref_image'] = ref_image
                params['subtract_background'] = self.first_image.subtract_background
                params['bio_mask'] = bio_mask
                params['back_mask'] = back_mask
                params['filter_spec'] = self.vars["filter_spec"]

                self.last_image.find_color_space_combinations(params)

    def cropping(self, is_first_image: bool):
        """
        Crops the image based on specified conditions and settings.

        This method checks if drift correction has already been applied.
        If the image is the first one and hasn't been cropped yet, it will attempt
        to use pre-stored coordinates or compute new crop coordinates. If automatic
        cropping is enabled, it will apply the cropping process.

        Parameters
        ----------
        is_first_image : bool
            Indicates whether the image being processed is the first one in the sequence.
        """
        if not self.vars['drift_already_corrected']:
            if is_first_image:
                if not self.first_image.cropped:
                    if (not self.all['overwrite_unaltered_videos'] and os.path.isfile('Data to run Cellects quickly.pkl')):
                        self.first_image.get_crop_coordinates()
                        pickle_rick = PickleRick()
                        data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
                        if data_to_run_cellects_quickly is not None and 'bb_coord' in data_to_run_cellects_quickly['all']['vars']:
                            logging.info("Get crop coordinates from Data to run Cellects quickly.pkl")
                            (ccy1, ccy2, ccx1, ccx2, self.top, self.bot, self.left, self.right) = \
                                data_to_run_cellects_quickly['all']['vars']['bb_coord']
                            self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                    else:
                        self.first_image.get_crop_coordinates()
                    if self.all['automatically_crop']:
                        self.first_image.automatically_crop(self.first_image.crop_coord)
                    else:
                        self.first_image.crop_coord = None
            else:
                if not self.last_image.cropped and self.all['automatically_crop']:
                    self.last_image.automatically_crop(self.first_image.crop_coord)

    def get_average_pixel_size(self):
        """
        Calculate the average pixel size and related variables.

        Logs information about calculation steps, computes the average
        pixel size based on image or cell scaling settings,
        and sets initial thresholds for object detection.

        Notes
        -----
        - The average pixel size is determined by either image dimensions or blob sizes.
        - Thresholds for automatic detection are set based on configuration settings.

        """
        logging.info("Getting average pixel size")
        (self.first_image.shape_number,
            self.first_image.shapes,
            self.first_image.stats,
            centroids) = cv2.connectedComponentsWithStats(
                self.first_image.validated_shapes,
                connectivity=8)
        self.first_image.shape_number -= 1
        if self.all['scale_with_image_or_cells'] == 0:
            self.vars['average_pixel_size'] = np.square(self.all['image_horizontal_size_in_mm'] /
                                                        self.first_im.shape[1])
        else:
            if len(self.first_image.stats[1:, 2]) > 0:
                self.vars['average_pixel_size'] = np.square(self.all['starting_blob_hsize_in_mm'] /
                                                            np.mean(self.first_image.stats[1:, 2]))
            else:
                self.vars['average_pixel_size'] = 1.
                self.vars['output_in_mm'] = False

        if self.all['set_spot_size']:
            self.starting_blob_hsize_in_pixels = (self.all['starting_blob_hsize_in_mm'] /
                                                  np.sqrt(self.vars['average_pixel_size']))
        else:
            self.starting_blob_hsize_in_pixels = None

        if self.all['automatic_size_thresholding']:
            self.vars['first_move_threshold'] = 10
        else:
            self.vars['first_move_threshold'] = np.round(self.all['first_move_threshold_in_mm²'] /
                                                         self.vars['average_pixel_size']).astype(np.uint8)
        logging.info(f"The average pixel size is: {self.vars['average_pixel_size']} mm²")

    def get_background_to_subtract(self):
        """
        Determine if background subtraction should be applied to the image.

        Extended Description
        --------------------
        This function checks whether background subtraction should be applied.
        It utilizes the 'subtract_background' flag and potentially converts
        the image for motion estimation.

        Parameters
        ----------
        self : object
            The instance of the class containing this method.
            Must have attributes `vars` and `first_image`.
        """
        if self.vars['subtract_background']:
            self.first_image.generate_subtract_background(self.vars['convert_for_motion'], self.vars['drift_already_corrected'])

    def find_if_lighter_background(self):
        """
        Determines whether the background is lighter or darker than the cells.

        This function analyzes images to determine if their backgrounds are lighter
        or darker relative to the cells, updating attributes accordingly for analysis and display purposes.


        Notes
        -----
        This function modifies instance variables and does not return any value.
        The analysis involves comparing mean pixel values in specific areas of the image.
        """
        logging.info("Find if the background is lighter or darker than the cells")
        self.vars['lighter_background']: bool = True
        self.vars['contour_color']: np.uint8 = 0
        are_dicts_equal: bool = True
        if self.vars['convert_for_origin'] is not None and self.vars['convert_for_origin'] is not None:
            for key in self.vars['convert_for_origin'].keys():
                are_dicts_equal = are_dicts_equal and np.all(key in self.vars['convert_for_motion'] and self.vars['convert_for_origin'][key] == self.vars['convert_for_motion'][key])

            for key in self.vars['convert_for_motion'].keys():
                are_dicts_equal = are_dicts_equal and np.all(key in self.vars['convert_for_origin'] and self.vars['convert_for_motion'][key] == self.vars['convert_for_origin'][key])
        else:
            self.vars['convert_for_origin'] = {"logical": 'None', "PCA": np.ones(3, dtype=np.uint8)}
            are_dicts_equal = True
        if are_dicts_equal:
            if self.first_im is None:
                self.get_first_image()
                self.fast_first_image_segmentation()
                self.cropping(is_first_image=True)
            among = np.nonzero(self.first_image.validated_shapes)
            not_among = np.nonzero(1 - self.first_image.validated_shapes)
            # Use the converted image to tell if the background is lighter, for analysis purposes
            if self.first_image.image[among[0], among[1]].mean() > self.first_image.image[not_among[0], not_among[1]].mean():
                self.vars['lighter_background'] = False
            # Use the original image to tell if the background is lighter, for display purposes
            if self.first_image.bgr[among[0], among[1], ...].mean() > self.first_image.bgr[not_among[0], not_among[1], ...].mean():
                self.vars['contour_color'] = 255
        else:
            if self.last_im is None:
                self.get_last_image()
                # self.cropping(is_first_image=False)
                self.fast_last_image_segmentation()
            if self.last_image.binary_image.sum() == 0:
                self.fast_last_image_segmentation()
            among = np.nonzero(self.last_image.binary_image)
            not_among = np.nonzero(1 - self.last_image.binary_image)
            # Use the converted image to tell if the background is lighter, for analysis purposes
            if self.last_image.image[among[0], among[1]].mean() > self.last_image.image[not_among[0], not_among[1]].mean():
                self.vars['lighter_background'] = False
            # Use the original image to tell if the background is lighter, for display purposes
            if self.last_image.bgr[among[0], among[1], ...].mean() > self.last_image.bgr[not_among[0], not_among[1], ...].mean():
                self.vars['contour_color'] = 255
        if self.vars['origin_state'] == "invisible":
            binary_image = deepcopy(self.first_image.binary_image)
            self.first_image.convert_and_segment(self.vars['convert_for_motion'], self.vars["color_number"],
                                                 None, None, subtract_background=None,
                                                 subtract_background2=None,
                                                 rolling_window_segmentation=self.vars['rolling_window_segmentation'],
                                                 filter_spec=self.vars["filter_spec"])
            covered_values = self.first_image.image[np.nonzero(binary_image)]
            self.vars['luminosity_threshold'] = 127
            if len(covered_values) > 0:
                if self.vars['lighter_background']:
                    if np.max(covered_values) < 255:
                        self.vars['luminosity_threshold'] = np.max(covered_values) + 1
                else:
                    if np.min(covered_values) > 0:
                        self.vars['luminosity_threshold'] = np.min(covered_values) - 1

    def delineate_each_arena(self):
        """
        Determine the coordinates of each arena for video analysis.

        The function processes video frames to identify bounding boxes around
        specimens and determines valid arenas for analysis. In case of existing data,
        it uses previously computed coordinates if available and valid.

        Returns
        -------
        analysis_status : dict
            A dictionary containing flags and messages indicating the status of
            the analysis.
            - 'continue' (bool): Whether to continue processing.
            - 'message' (str): Informational or error message.

        Notes
        -----
        This function relies on the existence of certain attributes and variables
        defined in the class instance.
        """
        analysis_status = {"continue": True, "message": ""}
        if not self.vars['several_blob_per_arena'] and (self.sample_number > 1):
            compute_get_bb: bool = True
            if (not self.all['overwrite_unaltered_videos'] and os.path.isfile('Data to run Cellects quickly.pkl')):

                pickle_rick = PickleRick()
                data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
                if data_to_run_cellects_quickly is not None:
                    if 'bb_coord' in data_to_run_cellects_quickly['all']['vars']:
                        (ccy1, ccy2, ccx1, ccx2, self.top, self.bot, self.left, self.right) = \
                            data_to_run_cellects_quickly['all']['vars']['bb_coord']
                        self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                        if (self.first_image.image.shape[0] == (ccy2 - ccy1)) and (
                                self.first_image.image.shape[1] == (ccx2 - ccx1)):  # maybe useless now
                            logging.info("Get the coordinates of all arenas from Data to run Cellects quickly.pkl")
                            compute_get_bb = False

            if compute_get_bb:
                motion_list = None
                if self.all['are_gravity_centers_moving']:
                    motion_list = self._segment_blob_motion(sample_size=5)
                self.get_bounding_boxes(are_gravity_centers_moving=self.all['are_gravity_centers_moving'] == 1,
                    motion_list=motion_list, all_specimens_have_same_direction=self.all['all_specimens_have_same_direction'])

                if np.any(self.ordered_stats[:, 4] > 100 * np.median(self.ordered_stats[:, 4])):
                    analysis_status['message'] = "A specimen is at least 100 times larger: click previous and retry by specifying 'back' areas."
                    analysis_status['continue'] = False
                if np.any(self.ordered_stats[:, 4] < 0.01 * np.median(self.ordered_stats[:, 4])):
                    analysis_status['message'] = "A specimen is at least 100 times smaller: click previous and retry by specifying 'back' areas."
                    analysis_status['continue'] = False
                del self.ordered_stats
                logging.info(
                    str(self.not_analyzed_individuals) + " individuals are out of picture scope and cannot be analyzed")

        else:
            self._whole_image_bounding_boxes()
            self.sample_number = 1
        self._set_analyzed_individuals()
        self.vars['arena_coord'] = []
        self.list_coordinates()
        return analysis_status

    def _segment_blob_motion(self, sample_size: int) -> list:
        """
        Segment blob motion from the data list at specified sample sizes.

        Parameters
        ----------
        sample_size : int
            Number of samples to take from the data list.

        Returns
        -------
        list
            List containing segmented binary images at sampled frames.

        Notes
        -----
        This function uses numpy for handling array operations and assumes the presence of certain attributes in the object, namely `data_list`, `first_image`, and `vars`.

        Examples
        --------
        >>> motion_samples = _segment_blob_motion(10)
        >>> print(len(motion_samples))  # Expected output: 10
        """
        motion_list = list()
        if isinstance(self.data_list, list):
            frame_number = len(self.data_list)
        else:
            frame_number = self.data_list.shape[0]
        sample_numbers = np.floor(np.linspace(0, frame_number, sample_size)).astype(int)
        if not 'lighter_background' in self.vars.keys():
            self.find_if_lighter_background()
        for frame_idx in np.arange(sample_size):
            if frame_idx == 0:
                motion_list.insert(frame_idx, self.first_image.validated_shapes)
            else:
                if isinstance(self.data_list[0], str):
                    image = self.data_list[sample_numbers[frame_idx] - 1]
                else:
                    image = self.data_list[sample_numbers[frame_idx] - 1]
                if isinstance(image, str):
                    is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
                    image = read_and_rotate(image, self.first_image.bgr, self.all['raw_images'],
                                            is_landscape, self.first_image.crop_coord)
                    # image = readim(image)
                In = OneImageAnalysis(image)
                if self.vars['drift_already_corrected']:
                    In.check_if_image_border_attest_drift_correction()
                    # In.adjust_to_drift_correction(self.vars['convert_for_motion']['logical'])
                In.convert_and_segment(self.vars['convert_for_motion'], self.vars['color_number'], None, None,
                                       self.first_image.subtract_background, self.first_image.subtract_background2,
                                       self.vars['rolling_window_segmentation'], self.vars['lighter_background'],
                                       allowed_window=In.drift_mask_coord, filter_spec=self.vars['filter_spec'])
                motion_list.insert(frame_idx, In.binary_image)
        return motion_list


    def get_bounding_boxes(self, are_gravity_centers_moving: bool, motion_list: list=(), all_specimens_have_same_direction: bool=True, original_shape_hsize: int=None):
        """Get the coordinates of arenas using bounding boxes.

        Parameters
        ----------
        are_gravity_centers_moving : bool
            Flag indicating whether gravity centers are moving or not.
        motion_list : list
            List of motion information for the specimens.
        all_specimens_have_same_direction : bool, optional
            Flag indicating whether all specimens have the same direction,
            by default True.
        Notes
        -----
        This method uses various internal methods and variables to determine the bounding boxes.
        """
        # 7) Create required empty arrays: especially the bounding box coordinates of each video
        self.ordered_first_image = None
        self.shapes_to_remove = None
        if self.first_image.crop_coord is None:
            self.first_image.get_crop_coordinates()

        logging.info("Get the coordinates of all arenas using the get_bounding_boxes method of the VideoMaker class")
        if self.first_image.validated_shapes.any() and self.first_image.shape_number > 0:
            self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                self.first_image.validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
            self.unchanged_ordered_fimg = deepcopy(self.ordered_first_image)
            self.modif_validated_shapes = deepcopy(self.first_image.validated_shapes)
            self.standard = - 1
            counter = 0
            while np.any(np.less(self.standard, 0)) and counter < 20:
                counter += 1
                self.left = np.zeros(self.first_image.shape_number, dtype=np.int64)
                self.right = np.repeat(self.modif_validated_shapes.shape[1], self.first_image.shape_number)
                self.top = np.zeros(self.first_image.shape_number, dtype=np.int64)
                self.bot = np.repeat(self.modif_validated_shapes.shape[0], self.first_image.shape_number)
                if are_gravity_centers_moving:
                    self.top, self.bot, self.left, self.right, self.ordered_first_image = get_bb_with_moving_centers(motion_list, all_specimens_have_same_direction,
                                                     original_shape_hsize, self.first_image.validated_shapes,
                                                     self.first_image.y_boundaries)
                    new_ordered_first_image = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)

                    for i in np.arange(1, self.first_image.shape_number + 1):
                        previous_shape = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
                        previous_shape[np.nonzero(self.unchanged_ordered_fimg == i)] = 1
                        new_potentials = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
                        new_potentials[np.nonzero(self.ordered_first_image == i)] = 1
                        new_potentials[np.nonzero(self.unchanged_ordered_fimg == i)] = 0

                        pads = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance=2)
                        pads.consider_shapes_sizes(min_shape_size=10)
                        pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False)
                        new_ordered_first_image[np.nonzero(pads.expanded_shape)] = i
                    self.ordered_first_image = new_ordered_first_image
                    self.modif_validated_shapes = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
                    self.modif_validated_shapes[np.nonzero(self.ordered_first_image)] = 1
                    self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                        self.modif_validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
                    self.top, self.bot, self.left, self.right = get_quick_bounding_boxes(self.modif_validated_shapes, self.ordered_first_image, self.ordered_stats)
                else:
                    self.top, self.bot, self.left, self.right = get_quick_bounding_boxes(self.modif_validated_shapes, self.ordered_first_image, self.ordered_stats)
                self._standardize_video_sizes()
            if counter == 20:
                self.top[self.top < 0] = 1
                self.bot[self.bot >= self.ordered_first_image.shape[0] - 1] = self.ordered_first_image.shape[0] - 2
                self.left[self.left < 0] = 1
                self.right[self.right >= self.ordered_first_image.shape[1] - 1] = self.ordered_first_image.shape[1] - 2
            del self.ordered_first_image
            del self.unchanged_ordered_fimg
            del self.modif_validated_shapes
            del self.standard
            del self.shapes_to_remove
            self.bot += 1
            self.right += 1
        else:
            self._whole_image_bounding_boxes()

    def _whole_image_bounding_boxes(self):
        self.top, self.bot, self.left, self.right = np.array([0]), np.array([self.first_image.image.shape[0]]), np.array([0]), np.array([self.first_image.image.shape[1]])

    def _standardize_video_sizes(self):
        """
        Standardize video sizes by adjusting bounding boxes.

        Extended Description
        --------------------
        This function adjusts the bounding boxes of detected shapes in a video frame.
        It ensures that all bounding boxes are within the frame's boundaries and
        standardizes their sizes to avoid issues with odd dimensions during video writing.

        Returns
        -------
        None
            The function modifies the following attributes of the class instance:

        Attributes Modified
        ------------------
        standard : numpy.ndarray
            Standardized bounding boxes.
        shapes_to_remove : numpy.ndarray
            Indices of shapes to be removed from the image.
        modif_validated_shapes : numpy.ndarray
            Modified validated shapes after removing out-of-picture areas.
        ordered_stats : list of float
            Updated order statistics for the shapes.
        ordered_centroids : numpy.ndarray
            Centroids of the ordered shapes.
        ordered_first_image : numpy.ndarray
            First image with updated order statistics and centroids.
        first_image.shape_number : int
            Updated number of shapes in the first image.
        not_analyzed_individuals : numpy.ndarray
            Indices of individuals not analyzed after modifications.

        """
        distance_threshold_to_consider_an_arena_out_of_the_picture = None# in pixels, worked nicely with - 50

        # The modifications allowing to not make videos of setups out of view, do not work for moving centers
        y_diffs = self.bot - self.top
        x_diffs = self.right - self.left
        add_to_y = ((np.max(y_diffs) - y_diffs) / 2)
        add_to_x = ((np.max(x_diffs) - x_diffs) / 2)
        self.standard = np.zeros((len(self.top), 4), dtype=np.int64)
        self.standard[:, 0] = self.top - np.uint8(np.floor(add_to_y))
        self.standard[:, 1] = self.bot + np.uint8(np.ceil(add_to_y))
        self.standard[:, 2] = self.left - np.uint8(np.floor(add_to_x))
        self.standard[:, 3] = self.right + np.uint8(np.ceil(add_to_x))

        # Monitor if one bounding box gets out of picture shape
        out_of_pic = deepcopy(self.standard)
        out_of_pic[:, 1] = self.ordered_first_image.shape[0] - out_of_pic[:, 1] - 1
        out_of_pic[:, 3] = self.ordered_first_image.shape[1] - out_of_pic[:, 3] - 1

        if distance_threshold_to_consider_an_arena_out_of_the_picture is None:
            distance_threshold_to_consider_an_arena_out_of_the_picture = np.min(out_of_pic) - 1

        # If it occurs at least one time, apply a correction, otherwise, continue and write videos
        # If the overflow is strong, remove the corresponding individuals and remake bounding_box finding
        if np.any(np.less(out_of_pic, distance_threshold_to_consider_an_arena_out_of_the_picture)):
            # Remove shapes
            self.standard = - 1
            self.shapes_to_remove = np.nonzero(np.less(out_of_pic, - 20))[0]
            for shape_i in self.shapes_to_remove:
                self.ordered_first_image[self.ordered_first_image == (shape_i + 1)] = 0
            self.modif_validated_shapes = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
            self.modif_validated_shapes[np.nonzero(self.ordered_first_image)] = 1
            self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                self.modif_validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)

            self.first_image.shape_number = self.first_image.shape_number - len(self.shapes_to_remove)
            self.not_analyzed_individuals = np.unique(self.unchanged_ordered_fimg -
                                                      (self.unchanged_ordered_fimg * self.modif_validated_shapes))[1:]

        else:
            # Reduce all box sizes if necessary and proceed
            if np.any(np.less(out_of_pic, 0)):
                # When the overflow is weak, remake standardization with lower "add_to_y" and "add_to_x"
                overflow = np.nonzero(np.logical_and(np.less(out_of_pic, 0), np.greater_equal(out_of_pic, distance_threshold_to_consider_an_arena_out_of_the_picture)))[0]
                # Look if overflow occurs on the y axis
                if np.any(np.less(out_of_pic[overflow, :2], 0)):
                    add_to_top_and_bot = np.min(out_of_pic[overflow, :2])
                    self.standard[:, 0] = self.standard[:, 0] - add_to_top_and_bot
                    self.standard[:, 1] = self.standard[:, 1] + add_to_top_and_bot
                # Look if overflow occurs on the x axis
                if np.any(np.less(out_of_pic[overflow, 2:], 0)):
                    add_to_left_and_right = np.min(out_of_pic[overflow, 2:])
                    self.standard[:, 2] = self.standard[:, 2] - add_to_left_and_right
                    self.standard[:, 3] = self.standard[:, 3] + add_to_left_and_right
            # If x or y sizes are odd, make them even :
            # Don't know why, but opencv remove 1 to odd shapes when writing videos
            if (self.standard[0, 1] - self.standard[0, 0]) % 2 != 0:
                self.standard[:, 1] -= 1
            if (self.standard[0, 3] - self.standard[0, 2]) % 2 != 0:
                self.standard[:, 3] -= 1
            self.top = self.standard[:, 0]
            self.bot = self.standard[:, 1]
            self.left = self.standard[:, 2]
            self.right = self.standard[:, 3]

    def get_origins_and_backgrounds_lists(self):
        """
        Create origins and background lists for image processing.

        Extended Description
        --------------------
        This method generates the origin and background lists by slicing the first image
        and its background subtraction based on predefined boundaries. It handles cases where
        the top, bottom, left, and right boundaries are not yet initialized.

        Notes
        -----
        This method directly modifies the input image data. The `self.vars` dictionary is populated
        with lists of sliced arrays from the first image and its background.

        Attributes
        ----------
        self.vars : dict
            Dictionary to store processed data.
        self.first_image : ImageObject
            The first image object containing validated shapes and background subtraction arrays.
        """
        logging.info("Create origins and background lists")
        if self.top is None:
            self._whole_image_bounding_boxes()

        if not self.first_image.validated_shapes.any():
            if self.vars['convert_for_motion'] is not None:
                self.vars['convert_for_origin'] = self.vars['convert_for_motion']
            self.fast_first_image_segmentation()
        first_im = self.first_image.validated_shapes
        self.vars['origin_list'] = []
        self.vars['background_list'] = []
        self.vars['background_list2'] = []
        for rep in np.arange(len(self.vars['analyzed_individuals'])):
            origin_coord = np.nonzero(first_im[self.top[rep]:self.bot[rep], self.left[rep]:self.right[rep]])
            self.vars['origin_list'].append(origin_coord)
        if self.vars['subtract_background']:
            for rep in np.arange(len(self.vars['analyzed_individuals'])):
                self.vars['background_list'].append(
                    self.first_image.subtract_background[self.top[rep]:self.bot[rep], self.left[rep]:self.right[rep]])
                if self.vars['convert_for_motion']['logical'] != 'None':
                    self.vars['background_list2'].append(self.first_image.subtract_background2[self.top[rep]:
                                                         self.bot[rep], self.left[rep]:self.right[rep]])

    def complete_image_analysis(self):
        if not self.visualize and len(self.last_image.im_combinations) > 0:
            self.last_image.binary_image = self.last_image.im_combinations[self.current_combination_id]['binary_image']
            self.last_image.image = self.last_image.im_combinations[self.current_combination_id]['converted_image']
        self.instantiate_tables()
        if len(self.vars['exif']) > 1:
            self.vars['exif'] = self.vars['exif'][0]
        if len(self.last_image.all_c_spaces) == 0:
            self.last_image.all_c_spaces['bgr'] = self.last_image.bgr.copy()
        if self.all['bio_mask'] is not None:
            self.last_image.binary_image[self.all['bio_mask']] = 1
        if self.all['back_mask'] is not None:
            self.last_image.binary_image[self.all['back_mask']] = 0
        for i, arena in enumerate(self.vars['analyzed_individuals']):
            binary = self.last_image.binary_image[self.top[i]:self.bot[i], self.left[i]:self.right[i]]
            efficiency_test = self.last_image.all_c_spaces['bgr'][self.top[i]:self.bot[i], self.left[i]:self.right[i], :]
            if not self.vars['several_blob_per_arena']:
                binary = keep_one_connected_component(binary)
                one_row_per_frame = compute_one_descriptor_per_frame(binary[None, :, :],
                                                                     arena,
                                                                     self.vars['exif'],
                                                                     self.vars['descriptors'],
                                                                     self.vars['output_in_mm'],
                                                                     self.vars['average_pixel_size'],
                                                                     self.vars['do_fading'],
                                                                     self.vars['save_coord_specimen'])
                coord_network = None
                coord_pseudopods = None
                if self.vars['save_graph']:
                    if coord_network is None:
                        coord_network = np.array(np.nonzero(binary))
                    extract_graph_dynamics(self.last_image.image[None, :, :], coord_network, arena,
                                           0, None, coord_pseudopods)

            else:
                one_row_per_frame = compute_one_descriptor_per_colony(binary[None, :, :],
                                                                      arena,
                                                                      self.vars['exif'],
                                                                      self.vars['descriptors'],
                                                                      self.vars['output_in_mm'],
                                                                      self.vars['average_pixel_size'],
                                                                      self.vars['do_fading'],
                                                                      self.vars['first_move_threshold'],
                                                                      self.vars['save_coord_specimen'])
            if self.vars['fractal_analysis']:
                zoomed_binary, side_lengths = prepare_box_counting(binary,
                                                                   min_mesh_side=self.vars[
                                                                       'fractal_box_side_threshold'],
                                                                   zoom_step=self.vars['fractal_zoom_step'],
                                                                   contours=True)
                box_counting_dimensions = box_counting_dimension(zoomed_binary, side_lengths)
                one_row_per_frame["fractal_dimension"] = box_counting_dimensions[0]
                one_row_per_frame["fractal_box_nb"] = box_counting_dimensions[1]
                one_row_per_frame["fractal_r_value"] = box_counting_dimensions[2]

            one_descriptor_per_arena = {}
            one_descriptor_per_arena["arena"] = arena
            one_descriptor_per_arena["first_move"] = pd.NA
            one_descriptor_per_arena["final_area"] = binary.sum()
            one_descriptor_per_arena["iso_digi_transi"] = pd.NA
            one_descriptor_per_arena["is_growth_isotropic"] = pd.NA
            self.update_one_row_per_arena(i, one_descriptor_per_arena)
            self.update_one_row_per_frame(i * 1, (i + 1) * 1, one_row_per_frame)
            contours = np.nonzero(get_contours(binary))
            efficiency_test[contours[0], contours[1], :] = np.array((94, 0, 213), dtype=np.uint8)
            self.add_analysis_visualization_to_first_and_last_images(i, efficiency_test, None)
        self.save_tables(with_last_image=False)

    def prepare_video_writing(self, img_list: list, min_ram_free: float, in_colors: bool=False, pathway: str=""):
        """

        Prepare the raw video (.npy) writing process for Cellects.

        Parameters
        ----------
        img_list : list
            List of images to be processed.
        min_ram_free : float
            Minimum amount of RAM in GB that should remain free.
        in_colors : bool, optional
            Whether the images are in color. Default is False.
        pathway : str, optional
            Path to save the video files. Default is an empty string.

        Returns
        -------
        tuple
            A tuple containing:
            - bunch_nb: int, number of bunches needed for video writing.
            - video_nb_per_bunch: int, number of videos per bunch.
            - sizes: ndarray, dimensions of each video.
            - video_bunch: list or ndarray, initialized video arrays.
            - vid_names: list, names of the video files.
            - rom_memory_required: None or float, required ROM memory.
            - analysis_status: dict, status and message of the analysis process.
            - remaining: int, remainder videos that do not fit in a complete bunch.

        Notes
        -----
        - The function calculates necessary memory and ensures 10% extra to avoid issues.
        - It checks for available RAM and adjusts the number of bunches accordingly.
        - If using color images, memory requirements are tripled.

        expected output depends on the provided images and RAM availability
        """
        # 1) Create a list of video names
        if self.not_analyzed_individuals is not None:
            number_to_add = len(self.not_analyzed_individuals)
        else:
            number_to_add = 0
        vid_names = list()
        ind_i = 0
        counter = 0
        while ind_i < (self.first_image.shape_number + number_to_add):
            ind_i += 1
            while np.any(np.isin(self.not_analyzed_individuals, ind_i)):
                ind_i += 1
            vid_names.append(pathway + "ind_" + str(ind_i) + ".npy")
            counter += 1
        img_nb = len(img_list)

        # 2) Create a table of the dimensions of each video
        # Add 10% to the necessary memory to avoid problems
        necessary_memory = img_nb * np.multiply((self.bot - self.top).astype(np.uint64), (self.right - self.left).astype(np.uint64)).sum() * 8 * 1.16415e-10
        if in_colors:
            sizes = np.column_stack(
                (np.repeat(img_nb, self.first_image.shape_number), self.bot - self.top, self.right - self.left,
                 np.repeat(3, self.first_image.shape_number)))
            necessary_memory *= 3
        else:
            sizes = np.column_stack(
                (np.repeat(img_nb, self.first_image.shape_number), self.bot - self.top, self.right - self.left))
        use_list_of_vid = True
        if np.all(sizes[0, :] == sizes):
            use_list_of_vid = False
        available_memory = (psutil.virtual_memory().available >> 30) - min_ram_free
        if available_memory == 0:
            analysis_status = {"continue": False, "message": "There are not enough RAM available"}
            bunch_nb = 1
        else:
            bunch_nb = int(np.ceil(necessary_memory / available_memory))
            if bunch_nb > 1:
                # The program will need twice the memory to create the second bunch.
                bunch_nb = int(np.ceil(2 * necessary_memory / available_memory))

        video_nb_per_bunch = np.floor(self.first_image.shape_number / bunch_nb).astype(np.uint8)
        analysis_status = {"continue": True, "message": ""}
        video_bunch = None
        try:
            if use_list_of_vid:
                video_bunch = [np.zeros(sizes[i, :], dtype=np.uint8) for i in range(video_nb_per_bunch)]
            else:
                video_bunch = np.zeros(np.append(sizes[0, :], video_nb_per_bunch), dtype=np.uint8)
        except ValueError as v_err:
            analysis_status = {"continue": False, "message": "Probably failed to detect the right cell(s) number, do the first image analysis manually."}
            logging.error(f"{analysis_status['message']} error is: {v_err}")
        # Check for available ROM memory
        if (psutil.disk_usage('/')[2] >> 30) < (necessary_memory + 2):
            rom_memory_required = necessary_memory + 2
        else:
            rom_memory_required = None
        remaining = self.first_image.shape_number % bunch_nb
        if remaining > 0:
            bunch_nb += 1
        is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
        logging.info(f"Cellects will start writing {self.first_image.shape_number} videos. Given available memory, it will do it in {bunch_nb} time(s)")
        return bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining, use_list_of_vid, is_landscape



    def update_output_list(self):
        """
        Update the output list with various descriptors from the analysis results.

        This method processes different types of descriptors and assigns them to
        the `self.vars['descriptors']` dictionary. It handles special cases for
        descriptors related to 'xy' dimensions and ensures that all relevant metrics
        are stored in the output list.
        """
        self.vars['descriptors'] = {}
        for descriptor in self.all['descriptors'].keys():
            if descriptor == 'standard_deviation_xy':
                self.vars['descriptors']['standard_deviation_x'] = self.all['descriptors'][descriptor]
                self.vars['descriptors']['standard_deviation_y'] = self.all['descriptors'][descriptor]
            elif descriptor == 'skewness_xy':
                self.vars['descriptors']['skewness_x'] = self.all['descriptors'][descriptor]
                self.vars['descriptors']['skewness_y'] = self.all['descriptors'][descriptor]
            elif descriptor == 'kurtosis_xy':
                self.vars['descriptors']['kurtosis_x'] = self.all['descriptors'][descriptor]
                self.vars['descriptors']['kurtosis_y'] = self.all['descriptors'][descriptor]
            elif descriptor == 'major_axes_len_and_angle':
                self.vars['descriptors']['major_axis_len'] = self.all['descriptors'][descriptor]
                self.vars['descriptors']['minor_axis_len'] = self.all['descriptors'][descriptor]
                self.vars['descriptors']['axes_orientation'] = self.all['descriptors'][descriptor]
            else:
                if np.isin(descriptor, list(from_shape_descriptors_class.keys())):
                
                    self.vars['descriptors'][descriptor] = self.all['descriptors'][descriptor]
        self.vars['descriptors']['newly_explored_area'] = self.vars['do_fading']

    def update_available_core_nb(self, image_bit_number=256, video_bit_number=140):# video_bit_number=176
        """
        Update available computation resources based on memory and processing constraints.

        Parameters
        ----------
        image_bit_number : int, optional
            Number of bits per image pixel (default is 256).
        video_bit_number : int, optional
            Number of bits per video frame pixel (default is 140).

        Other Parameters
        ----------------
        lose_accuracy_to_save_memory : bool
            Flag to reduce accuracy for memory savings.
        convert_for_motion : dict
            Conversion settings for motion analysis.
        already_greyscale : bool
            Flag indicating if the image is already greyscale.
        save_coord_thickening_slimming : bool
            Flag to save coordinates for thickening and slimming.
        oscilacyto_analysis : bool
            Flag indicating if oscilacyto analysis is enabled.
        save_coord_network : bool
            Flag to save coordinates for network analysis.

        Returns
        -------
        float
            Rounded absolute difference between available memory and necessary memory in GB.

        Notes
        -----
        Performance considerations and limitations should be noted here if applicable.

        """
        if self.vars['lose_accuracy_to_save_memory']:
            video_bit_number -= 56
        if self.vars['convert_for_motion']['logical'] != 'None':
            video_bit_number += 64
            if self.vars['lose_accuracy_to_save_memory']:
                video_bit_number -= 56
        if self.vars['already_greyscale']:
            video_bit_number -= 64
        if self.vars['save_coord_thickening_slimming'] or self.vars['oscilacyto_analysis']:
            video_bit_number += 16
            image_bit_number += 128
        if self.vars['save_coord_network']:
            video_bit_number += 8
            image_bit_number += 64

        if isinstance(self.bot, list):
            one_image_memory = np.multiply((self.bot[0] - self.top[0]),
                                        (self.right[0] - self.left[0])).max().astype(np.uint64)
        else:
            one_image_memory = np.multiply((self.bot - self.top).astype(np.uint64),
                                        (self.right - self.left).astype(np.uint64)).max()
        one_video_memory = self.vars['img_number'] * one_image_memory
        necessary_memory = (one_image_memory * image_bit_number + one_video_memory * video_bit_number) * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        max_repeat_in_memory = (available_memory // necessary_memory).astype(np.uint16)
        if max_repeat_in_memory > 1:
            max_repeat_in_memory = np.max(((available_memory // (2 * necessary_memory)).astype(np.uint16), 1))


        self.cores = np.min((self.all['cores'], max_repeat_in_memory))
        if self.cores > self.sample_number:
            self.cores = self.sample_number
        return np.round(np.absolute(available_memory - necessary_memory), 3)


    def update_one_row_per_arena(self, i: int, table_to_add):
        """
        Update one row of the dataframe per arena.

        Add a row to a DataFrame for each arena, based on the provided table_to_add. If no previous rows exist,
        initialize a new DataFrame with zeros.

        Parameters
        ----------
        i : int
            Index of the arena to update.
        table_to_add : dict
            Dictionary containing values to add. Keys are column names, values are the data.

        """
        if not self.vars['several_blob_per_arena']:
            if self.one_row_per_arena is None:
                self.one_row_per_arena = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']), len(table_to_add)), dtype=float),
                                            columns=table_to_add.keys())
            self.one_row_per_arena.iloc[i, :] = table_to_add.values()


    def update_one_row_per_frame(self, i: int, j: int, table_to_add):
        """
        Update a range of rows in `self.one_row_per_frame` DataFrame with values from
        `table_to_add`.

        Parameters
        ----------
        i : int
            The starting row index to update in `self.one_row_per_frame`.
        j : int
            The ending row index (exclusive) to update in `self.one_row_per_frame`.
        table_to_add : dict
            A dictionary where keys are column labels and values are lists or arrays of
            data to insert into `self.one_row_per_frame`.
        Notes
        -----
        Ensures that one row per arena is being updated. If `self.one_row_per_frame` is
        None, it initializes a DataFrame to hold the data.
        """
        if not self.vars['several_blob_per_arena']:
            if self.one_row_per_frame is None:
                self.one_row_per_frame = pd.DataFrame(index=range(len(self.vars['analyzed_individuals']) *
                                                        self.vars['img_number']),
                                            columns=table_to_add.keys())

            self.one_row_per_frame.iloc[i:j, :] = table_to_add


    def instantiate_tables(self):
        """
        Update output list and prepare results tables and validation images.

        Extended Description
        --------------------
        This method performs necessary preparations for processing image sequences,
        including updating the output list and initializing key attributes required
        for subsequent operations.

        """
        self.update_output_list()
        logging.info("Instantiate results tables and validation images")
        self.fractal_box_sizes = None
        self.one_row_per_arena = None
        self.one_row_per_frame = None
        if self.vars['already_greyscale']:
            if len(self.first_image.bgr.shape) == 2:
                self.first_image.bgr = np.stack((self.first_image.bgr, self.first_image.bgr, self.first_image.bgr), axis=2).astype(np.uint8)
            if len(self.last_image.bgr.shape) == 2:
                self.last_image.bgr = np.stack((self.last_image.bgr, self.last_image.bgr, self.last_image.bgr), axis=2).astype(np.uint8)
            self.vars["convert_for_motion"] = {"bgr": np.array((1, 1, 1), dtype=np.uint8), "logical": "None"}

    def add_analysis_visualization_to_first_and_last_images(self, i: int, first_visualization: NDArray, last_visualization: NDArray=None):
        """
        Adds analysis visualizations to the first and last images of a sequence.

        Parameters
        ----------
        i : int
            Index of the image in the sequence.
        first_visualization : NDArray[np.uint8]
            The visualization to add to the first image.
        last_visualization : NDArray[np.uint8]
            The visualization to add to the last image.

        Other Parameters
        ----------------
        vars : dict
            Dictionary containing various parameters.
        arena_shape : str, optional
            The shape of the arena. Either 'circle' or other shapes.

        Notes
        -----
        If `arena_shape` is 'circle', the visualization will be masked by an ellipse.

        """
        minmax = (self.top[i], self.bot[i], self.left[i], self.right[i])
        self.first_image.bgr = draw_img_with_mask(self.first_image.bgr, self.first_image.bgr.shape[:2], minmax,
                                                  self.vars['arena_shape'], first_visualization)
        if last_visualization is not None:
            self.last_image.bgr = draw_img_with_mask(self.last_image.bgr, self.last_image.bgr.shape[:2], minmax,
                                                      self.vars['arena_shape'], last_visualization)


    def save_tables(self, with_last_image: bool=True):
        """
        Exports analysis results to CSV files and saves visualization outputs.

        Generates the following output:
        - one_row_per_arena.csv, one_row_per_frame.csv : Tracking data per arena/frame.
        - software_settings.csv : Full configuration settings for reproducibility.

        Raises
        ------
        PermissionError
            If any output file is already open in an external program (logged and re-raised).

        Notes
        -----
        Ensure no exported CSV files are open while running this method to avoid permission errors. This
        function will fail gracefully if the files cannot be overwritten.

        """
        logging.info("Save results tables and validation images")
        if not self.vars['several_blob_per_arena']:
            try:
                self.one_row_per_arena.to_csv("one_row_per_arena.csv", sep=";", index=False, lineterminator='\n')
                del self.one_row_per_arena
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")
            try:
                self.one_row_per_frame.to_csv("one_row_per_frame.csv", sep=";", index=False, lineterminator='\n')
                del self.one_row_per_frame
            except PermissionError:
                logging.error("Never let one_row_per_frame.csv open when Cellects runs")
        if self.all['extension'] == '.JPG':
            extension = '.PNG'
        else:
            extension = '.JPG'
        if with_last_image:
            cv2.imwrite(f"Analysis efficiency, last image{extension}", self.last_image.bgr)
        cv2.imwrite(
            f"Analysis efficiency, {np.ceil(self.vars['img_number'] / 10).astype(np.uint64)}th image{extension}",
            self.first_image.bgr)
        software_settings = deepcopy(self.vars)
        for key in ['descriptors', 'analyzed_individuals', 'exif', 'dims', 'origin_list', 'background_list', 'background_list2', 'descriptors', 'folder_list', 'sample_number_per_folder']:
            software_settings.pop(key, None)
        global_settings = deepcopy(self.all)
        for key in ['analyzed_individuals', 'night_mode', 'expert_mode', 'is_auto', 'arena', 'video_option', 'compute_all_options', 'vars', 'dims', 'origin_list', 'background_list', 'background_list2', 'descriptors', 'folder_list', 'sample_number_per_folder']:
            global_settings.pop(key, None)
        software_settings.update(global_settings)
        software_settings.pop('video_list', None)
        software_settings = pd.DataFrame.from_dict(software_settings, columns=["Setting"], orient='index')
        try:
            software_settings.to_csv("software_settings.csv", sep=";")
        except PermissionError:
            logging.error("Never let software_settings.csv open when Cellects runs")


