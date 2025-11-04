#!/usr/bin/env python3
"""This file contains the class constituting the link between the graphical interface and the computations
 First, Cellects analyze one image in order to get a color space combination maximizing the contrast between the specimens
 and the background.
 Second, Cellects automatically delineate each arena.
 Third, Cellects write one video for each arena.
 Fourth, Cellects segments the video and apply post-processing algorithms to improve the segmentation.
 Fifth, Cellects extract variables and store them in .csv files.
"""

import logging
import os
import pickle
import sys
from copy import deepcopy
import cv2
from numba.typed import Dict as TDict
import pandas as pd
import numpy as np
from psutil import virtual_memory
from pathlib import Path
import natsort
from cellects.image_analysis.image_segmentation import generate_color_space_combination
from cellects.utils.load_display_save import extract_time  # named exif
from cellects.image_analysis.one_image_analysis_threads import ProcessFirstImage
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.utils.load_display_save import PickleRick, read_and_rotate, readim, is_raw_image, read_h5_array, get_h5_keys
from cellects.utils.utilitarian import insensitive_glob, vectorized_len
from cellects.image_analysis.morphological_operations import Ellipse, cross_33
from cellects.core.cellects_paths import CELLECTS_DIR, ALL_VARS_PKL_FILE
from cellects.core.motion_analysis import MotionAnalysis
from cellects.core.one_video_per_blob import OneVideoPerBlob
from cellects.config.all_vars_dict import DefaultDicts
from cellects.image_analysis.shape_descriptors import from_shape_descriptors_class


class ProgramOrganizer:
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
        if os.path.isfile(Path(CELLECTS_DIR.parent / 'PickleRick.pkl')):
            os.remove(Path(CELLECTS_DIR.parent / 'PickleRick.pkl'))
        if os.path.isfile(Path(CELLECTS_DIR.parent / 'PickleRick0.pkl')):
            os.remove(Path(CELLECTS_DIR.parent / 'PickleRick0.pkl'))
        # self.delineation_number = 0
        self.one_arena_done: bool = False
        self.reduce_image_dim: bool = False
        self.first_exp_ready_to_run: bool = False
        self.data_to_save = {'first_image': False, 'coordinates': False, 'exif': False, 'vars': False}
        self.videos = None
        self.motion = None
        self.analysis_instance = None
        self.computed_video_options = np.zeros(5, bool)
        self.vars = {}
        self.all = {}
        self.all['folder_list'] = []
        self.all['first_detection_frame'] = 1
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
        self.one_row_per_oscillating_cluster = None
        # self.fractal_box_sizes = None

    def save_variable_dict(self):
        logging.info("Save the parameters dictionaries in the Cellects folder")
        self.all['vars'] = self.vars
        all_vars = deepcopy(self.all)
        if not self.all['keep_cell_and_back_for_all_folders']:
            all_vars['bio_mask'] = None
            all_vars['back_mask'] = None
        pickle_rick = PickleRick(0)
        pickle_rick.write_file(all_vars, ALL_VARS_PKL_FILE)

    def load_variable_dict(self):
        # loading_succeed: bool = False
        # if os.path.isfile('Data to run Cellects quickly.pkl'):
        #     try:
        #         with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
        #             data_to_run_cellects_quickly = pickle.load(fileopen)
        #         if 'vars' in data_to_run_cellects_quickly:
        #             self.vars = data_to_run_cellects_quickly['vars']
        #             loading_succeed = True
        #             logging.info("Success to load vars from the data folder")
        #     except EOFError:
        #         logging.error("Pickle error: will try to load vars from the Cellects folder")

        if os.path.isfile(ALL_VARS_PKL_FILE):
            logging.info("Load the parameters from all_vars.pkl in the config of the Cellects folder")
            try:  # NEW
                with open(ALL_VARS_PKL_FILE, 'rb') as fileopen:  # NEW
                    self.all = pickle.load(fileopen)  # NEW
                self.vars = self.all['vars']
                self.update_data()
                logging.info("Success to load the parameters dictionaries from the Cellects folder")
                logging.info(os.getcwd())
            except Exception as exc:  # NEW
                logging.error(f"Initialize default parameters because error: {exc}")  # NEW
                default_dicts = DefaultDicts()  # NEW
                self.all = default_dicts.all  # NEW
                self.vars = default_dicts.vars  # NEW
        else:
            logging.info("Initialize default parameters")
            default_dicts = DefaultDicts()
            self.all = default_dicts.all
            self.vars = default_dicts.vars
        if self.all['cores'] == 1:
            self.all['cores'] = os.cpu_count() - 1

    def analyze_without_gui(self):
        # Eventually load "all" dir before calling this function
        # self = po
        # if len(self.all['folder_list']) == 0:
        #     folder_list = "/"
        # else:
        #     folder_list = self.all['folder_list']
        # for exp_i, folder_name in enumerate(folder_list):
        #     # exp_i = 0 ; folder_name = folder_list

        self=ProgramOrganizer()
        self.load_variable_dict()
        # dd = DefaultDicts()
        # self.all = dd.all
        # self.vars = dd.vars
        self.all['global_pathway'] = "/Users/Directory/Data/dossier1"
        self.all['first_folder_sample_number'] = 6
        # self.all['global_pathway'] = "D:\Directory\Data\Audrey\dosier1"
        # self.all['first_folder_sample_number'] = 6
        # self.all['radical'] = "IMG"
        # self.all['extension'] = ".jpg"
        # self.all['im_or_vid'] = 0
        self.look_for_data()
        self.load_data_to_run_cellects_quickly()
        if not self.first_exp_ready_to_run:
            self.get_first_image()
            self.fast_image_segmentation(True)
            # self.first_image.find_first_im_csc(sample_number=self.sample_number,
            #                                    several_blob_per_arena=None,
            #                                    spot_shape=None, spot_size=None,
            #                                    kmeans_clust_nb=2,
            #                                    biomask=None, backmask=None,
            #                                    color_space_dictionaries=None,
            #                                    carefully=True)
            self.cropping(is_first_image=True)
            self.get_average_pixel_size()
            self.delineate_each_arena()
            self.get_background_to_subtract()
            self.get_origins_and_backgrounds_lists()
            self.get_last_image()
            self.fast_image_segmentation(is_first_image=False)
            self.find_if_lighter_background()
            self.extract_exif()
        self.update_output_list()
        look_for_existing_videos = insensitive_glob('ind_' + '*' + '.npy')
        there_already_are_videos = len(look_for_existing_videos) == len(self.vars['analyzed_individuals'])
        logging.info(
            f"{len(look_for_existing_videos)} .npy video files found for {len(self.vars['analyzed_individuals'])} arenas to analyze")
        do_write_videos = not there_already_are_videos or (
                there_already_are_videos and self.all['overwrite_unaltered_videos'])
        if do_write_videos:
            self.videos = OneVideoPerBlob(self.first_image, self.starting_blob_hsize_in_pixels, self.all['raw_images'])
            self.videos.left = self.left
            self.videos.right = self.right
            self.videos.top = self.top
            self.videos.bot = self.bot
            self.videos.first_image.shape_number = self.sample_number
            self.videos.write_videos_as_np_arrays(
                self.data_list, self.vars['min_ram_free'], not self.vars['already_greyscale'], self.reduce_image_dim)
        self.instantiate_tables()

        i=1
        show_seg=True

        if os.path.isfile(f"coord_specimen{i + 1}_t720_y1475_x1477.npy"):
            binary_coord = np.load(f"coord_specimen{i + 1}_t720_y1475_x1477.npy")
            l = [i, i + 1, self.vars, False, False, show_seg, None]
            sav = self
            self = MotionAnalysis(l)
            self.binary = np.zeros((720, 1475, 1477), dtype=np.uint8)
            self.binary[binary_coord[0, :], binary_coord[1, :], binary_coord[2, :]] = 1
        else:
            l = [i, i + 1, self.vars, True, False, show_seg, None]
            sav = self
            self = MotionAnalysis(l)
        self.get_descriptors_from_binary()
        self.detect_growth_transitions()
        # self.networks_detection(show_seg)
        self.study_cytoscillations(show_seg)

        # for i, arena in enumerate(self.vars['analyzed_individuals']):
        #     l = [i, i + 1, self.vars, True, False, False, None]
        #     analysis_i = MotionAnalysis(l)
        #     self.add_analysis_visualization_to_first_and_last_images(i, analysis_i.efficiency_test_1,
        #                                                              analysis_i.efficiency_test_2)
        # self.save_tables()
        #
        # self = MotionAnalysis(l)
        # l = [5, 6, self.vars, True, False, False, None]
        # sav=self
        # self.get_descriptors_from_binary()
        # self.detect_growth_transitions()

    def look_for_data(self):
        # global_pathway = 'I:\Directory\Tracking_data\generalization_and_potentiation\drop_nak1'
        os.chdir(Path(self.all['global_pathway']))
        logging.info(f"Dir: {self.all['global_pathway']}")
        self.data_list = insensitive_glob(
            self.all['radical'] + '*' + self.all['extension'])  # Provides a list ordered by last modification date
        self.data_list = insensitive_glob(self.all['radical'] + '*' + self.all['extension'])  # Provides a list ordered by last modification date
        self.all['folder_list'] = []
        self.all['folder_number'] = 1
        if len(self.data_list) > 0:
            lengths = vectorized_len(self.data_list)
            if np.max(np.diff(lengths)) > np.log10(len(self.data_list)):
                logging.error(f"File names present strong variations and cannot be correctly sorted.")
            self.data_list = natsort.natsorted(self.data_list)
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

    def update_folder_id(self, sample_number, folder_name=""):
        os.chdir(Path(self.all['global_pathway']) / folder_name)
        self.data_list = insensitive_glob(
            self.all['radical'] + '*' + self.all['extension'])  # Provides a list ordered by last modification date
        # Sorting is necessary when some modifications (like rotation) modified the last modification date
        lengths = vectorized_len(self.data_list)
        if np.max(np.diff(lengths)) > np.log10(len(self.data_list)):
            logging.error(f"File names present strong variations and cannot be correctly sorted.")
        self.data_list = natsort.natsorted(self.data_list)
        if self.all['im_or_vid'] == 1:
            self.sample_number = len(self.data_list)
        else:
            self.vars['img_number'] = len(self.data_list)
            self.sample_number = sample_number
        if len(self.vars['analyzed_individuals']) != sample_number:
            self.vars['analyzed_individuals'] = np.arange(sample_number) + 1

    def load_data_to_run_cellects_quickly(self):
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

            # try:
            #     with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
            #         data_to_run_cellects_quickly = pickle.load(fileopen)
            # except pickle.UnpicklingError:
            #     logging.error("Pickle error")
            #     data_to_run_cellects_quickly = {}
            if ('validated_shapes' in data_to_run_cellects_quickly) and ('coordinates' in data_to_run_cellects_quickly) and ('all' in data_to_run_cellects_quickly):
                logging.info("Success to load Data to run Cellects quickly.pkl from the user chosen directory")
                self.all = data_to_run_cellects_quickly['all']
                # If you want to add a new variable, first run an updated version of all_vars_dict,
                # then put a breakpoint here and run the following + self.save_data_to_run_cellects_quickly() :
                # self.all['vars']['lose_accuracy_to_save_memory'] = False
                self.vars = self.all['vars']
                self.update_data()
                print(self.vars['convert_for_motion'])
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
                (ccy1, ccy2, ccx1, ccx2, self.left, self.right, self.top, self.bot) = data_to_run_cellects_quickly[
                    'coordinates']
                if self.all['automatically_crop']:
                    self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                    logging.info("Crop first image")
                    self.first_image.automatically_crop(self.first_image.crop_coord)
                    logging.info("Crop last image")
                    self.last_image.automatically_crop(self.first_image.crop_coord)
                else:
                    self.first_image.crop_coord = None
                # self.cropping(True)
                # self.cropping(False)
                self.first_image.validated_shapes = data_to_run_cellects_quickly['validated_shapes']
                self.first_image.im_combinations = []
                self.current_combination_id = 0
                self.first_image.im_combinations.append({})
                self.first_image.im_combinations[self.current_combination_id]['csc'] = self.vars['convert_for_origin']
                self.first_image.im_combinations[self.current_combination_id]['binary_image'] = self.first_image.validated_shapes
                self.first_image.im_combinations[self.current_combination_id]['shape_number'] = data_to_run_cellects_quickly['shape_number']
                
                self.first_exp_ready_to_run = True
                print(f"Overwrite is {self.all['overwrite_unaltered_videos']}")
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

    def update_data(self):
        dd = DefaultDicts()
        all = len(dd.all) != len(self.all)
        vars = len(dd.vars) != len(self.vars)
        all_desc = len(dd.all['descriptors']) != len(self.all['descriptors'])
        vars_desc = len(dd.vars['descriptors']) != len(self.vars['descriptors'])
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

    def save_data_to_run_cellects_quickly(self, new_one_if_does_not_exist=True):
        data_to_run_cellects_quickly = None
        if os.path.isfile('Data to run Cellects quickly.pkl'):
            logging.info("Update -Data to run Cellects quickly.pkl- in the user chosen directory")
            pickle_rick = PickleRick()
            data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
            if data_to_run_cellects_quickly is None:
                logging.error("Failed to load Data to run Cellects quickly.pkl before update. Abort saving.")

            # try:
            #     with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
            #         data_to_run_cellects_quickly = pickle.load(fileopen)
            # except pickle.UnpicklingError:
            #     logging.error("Pickle error")
            #     data_to_run_cellects_quickly = {}
        else:
            if new_one_if_does_not_exist:
                logging.info("Create Data to run Cellects quickly.pkl in the user chosen directory")
                data_to_run_cellects_quickly = {}
        if data_to_run_cellects_quickly is not None:
            if self.data_to_save['first_image']:
                data_to_run_cellects_quickly['validated_shapes'] = self.first_image.im_combinations[self.current_combination_id]['binary_image']
                data_to_run_cellects_quickly['shape_number'] = self.first_image.im_combinations[self.current_combination_id]['shape_number']
                    # data_to_run_cellects_quickly['converted_image'] = self.first_image.im_combinations[self.current_combination_id]['converted_image']
            if self.data_to_save['coordinates']:
                data_to_run_cellects_quickly['coordinates'] = self.list_coordinates()
                logging.info("When they exist, do overwrite unaltered video")
                self.all['overwrite_unaltered_videos'] = True
            if self.data_to_save['exif']:
                self.vars['exif'] = self.extract_exif()
                # data_to_run_cellects_quickly['exif'] = self.extract_exif()
            # if self.data_to_save['background_and_origin_list']:
            #     logging.info(f"Origin shape is {self.vars['origin_list'][0].shape}")
            #     data_to_run_cellects_quickly['background_and_origin_list'] = [self.vars['origin_list'], self.vars['background_list'], self.vars['background_list2']]
            self.all['vars'] = self.vars
            print(self.vars['convert_for_motion'])
            data_to_run_cellects_quickly['all'] = self.all
            # data_to_run_cellects_quickly['all']['vars']['origin_state'] = "fluctuating"
            pickle_rick = PickleRick()
            pickle_rick.write_file(data_to_run_cellects_quickly, 'Data to run Cellects quickly.pkl')

    def list_coordinates(self):
        if self.first_image.crop_coord is None:
            self.first_image.crop_coord = [0, self.first_image.image.shape[0], 0,
                                                       self.first_image.image.shape[1]]
        videos_coordinates = self.first_image.crop_coord + [self.left, self.right, self.top, self.bot]
        return videos_coordinates

    def extract_exif(self):
        if self.all['im_or_vid'] == 1:
            timings = np.arange(self.vars['dims'][0])
        else:
            if sys.platform.startswith('win'):
                pathway = os.getcwd() + '\\'
            else:
                pathway = os.getcwd() + '/'
            arbitrary_time_step: bool = True
            if self.all['extract_time_interval']:
                self.vars['time_step'] = 1
                try:
                    timings = extract_time(self.data_list, pathway, self.all['raw_images'])
                    timings = timings - timings[0]
                    timings = timings / 60
                    time_step = np.mean(np.diff(timings))
                    digit_nb = 0
                    for i in str(time_step):
                        if i in {'.'}:
                            pass
                        elif i in {'0'}:
                            digit_nb += 1
                        else:
                            break
                    self.vars['time_step'] = np.round(time_step, digit_nb + 1)
                    arbitrary_time_step = False
                except:
                    pass
            if arbitrary_time_step:
                timings = np.arange(0, self.vars['dims'][0] * self.vars['time_step'], self.vars['time_step'])
                timings = timings - timings[0]
                timings = timings / 60
        return timings

        #
        # if not os.path.isfile("timings.csv") or self.all['overwrite_cellects_data']:
        #     if self.vars['time_step'] == 0:
        #         if self.all['im_or_vid'] == 1:
        #             savetxt("timings.csv", np.arange(self.vars['dims'][0]), fmt='%10d', delimiter=',')
        #         else:
        #             if sys.platform.startswith('win'):
        #                 pathway = os.getcwd() + '\\'
        #             else:
        #                 pathway = os.getcwd() + '/'
        #             timings = extract_time(self.data_list, pathway, self.all['raw_images'])
        #             timings = timings - timings[0]
        #             timings = timings / 60
        #     else:
        #         timings = np.arange(0, self.vars['dims'][0] * self.vars['time_step'], self.vars['time_step'])
        #     savetxt("timings.csv", timings, fmt='%1.2f', delimiter=',')

    def get_first_image(self):
        logging.info("Load first image")
        just_read_image = self.first_im is not None
        self.reduce_image_dim = False
        # just_read_image = self.analysis_instance is not None
        if self.all['im_or_vid'] == 1:
            cap = cv2.VideoCapture(self.data_list[0])
            counter = 0
            if not just_read_image:
                self.sample_number = len(self.data_list)
                self.vars['img_number'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.analysis_instance = np.zeros(
                    [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3])
                while cap.isOpened() and counter < 1:
                    ret, frame = cap.read()
                    if counter == 0:
                        self.first_im = frame
                        self.analysis_instance[0, ...] = self.first_im
                        break
                cap.release()
            elif np.sum(self.analysis_instance[self.all['first_detection_frame'] - 1, ...] == 0):
                cap = cv2.VideoCapture(self.data_list[0])
                counter = 0
                while cap.isOpened() and (counter < self.all['first_detection_frame']):
                    ret, frame = cap.read()
                    # if self.reduce_image_dim:
                    #     frame = frame[:, :, 0]
                    self.analysis_instance[counter, ...] = frame
                    counter += 1

                cap.release()
                self.first_im = self.analysis_instance[
                    self.all['first_detection_frame'] - 1, ...]
            self.vars['dims'] = self.analysis_instance.shape[:3]

        else:
            self.vars['img_number'] = len(self.data_list)
            self.all['raw_images'] = is_raw_image(self.data_list[0])
            self.first_im = readim(self.data_list[self.all['first_detection_frame'] - 1], self.all['raw_images'])
            # if self.reduce_image_dim:
            #     self.first_im = self.first_im[:, :, 0]
            self.vars['dims'] = [self.vars['img_number'], self.first_im.shape[0], self.first_im.shape[1]]
            # self.first_im = readim(self.data_list[0], self.all['raw_images'])
        if len(self.first_im.shape) == 3:
            if np.all(np.equal(self.first_im[:, :, 0], self.first_im[:, :, 1])) and np.all(
                    np.equal(self.first_im[:, :, 1], self.first_im[:, :, 2])):
                self.reduce_image_dim = True
            if self.reduce_image_dim:
                self.first_im = self.first_im[:, :, 0]
                if self.all['im_or_vid'] == 1:
                    self.analysis_instance = self.analysis_instance[:, :, :, 0]
        self.first_image = OneImageAnalysis(self.first_im)
        self.vars['already_greyscale'] = self.first_image.already_greyscale
        if self.vars['already_greyscale']:
            self.vars["convert_for_origin"] = {"bgr": np.array((1, 1, 1), dtype=np.uint8), "logical": "None"}
            self.vars["convert_for_motion"] = {"bgr": np.array((1, 1, 1), dtype=np.uint8), "logical": "None"}
        if np.mean((np.mean(self.first_image.image[2, :, ...]), np.mean(self.first_image.image[-3, :, ...]), np.mean(self.first_image.image[:, 2, ...]), np.mean(self.first_image.image[:, -3, ...]))) > 127:
            self.vars['contour_color']: np.uint8 = 0
        else:
            self.vars['contour_color']: np.uint8 = 255
        if self.all['first_detection_frame'] > 1:
            self.vars['origin_state'] = 'invisible'

    def get_last_image(self):
        logging.info("Load last image")
        if self.all['im_or_vid'] == 1:
            cap = cv2.VideoCapture(self.data_list[0])
            counter = 0
            while cap.isOpened() and counter < self.vars['img_number']:
                ret, frame = cap.read()
                if self.reduce_image_dim:
                    frame = frame[:, :, 0]
                self.analysis_instance[-1, ...] = frame
                # if counter == self.vars['img_number'] - 1:
                #     if self.reduce_image_dim:
                #         frame = frame[:, :, 0]
                #     break
                counter += 1
            self.last_im = frame
            cap.release()
        else:
            # self.last_im = readim(self.data_list[-1], self.all['raw_images'])
            is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
            self.last_im = read_and_rotate(self.data_list[-1], self.first_im, self.all['raw_images'], is_landscape)
            if self.reduce_image_dim:
                self.last_im = self.last_im[:, :, 0]
        self.last_image = OneImageAnalysis(self.last_im)

    # self.message_when_thread_finished.emit("")
    def fast_image_segmentation(self, is_first_image, biomask=None, backmask=None, spot_size=None):
        if is_first_image:
            self.first_image.convert_and_segment(self.vars['convert_for_origin'], self.vars["color_number"],
                                                 self.all["bio_mask"], self.all["back_mask"], subtract_background=None,
                                                 subtract_background2=None, grid_segmentation=False,
                                                 filter_spec=self.vars["filter_spec"])
            if not self.first_image.drift_correction_already_adjusted:
                self.vars['drift_already_corrected'] = self.first_image.check_if_image_border_attest_drift_correction()
                if self.vars['drift_already_corrected']:
                    logging.info("Cellects detected that the images have already been corrected for drift")
                    self.first_image.adjust_to_drift_correction(self.vars['convert_for_origin']['logical'])
            if self.vars["grid_segmentation"]:
                self.first_image.convert_and_segment(self.vars['convert_for_origin'], self.vars["color_number"],
                                                     self.all["bio_mask"], self.all["back_mask"],
                                                     subtract_background=None, subtract_background2=None,
                                                     grid_segmentation=True,
                                                     filter_spec=self.vars["filter_spec"])

            self.first_image.set_spot_shapes_and_size_confint(self.all['starting_blob_shape'])
            logging.info(self.sample_number)
            process_i = ProcessFirstImage(
                [self.first_image, False, False, None, self.vars['several_blob_per_arena'],
                 self.sample_number, spot_size, self.vars["color_number"], self.all["bio_mask"], self.all["back_mask"], None])
            process_i.binary_image = self.first_image.binary_image
            process_i.process_binary_image(use_bio_and_back_masks=True)

            if self.all["back_mask"] is not None:
                if np.any(process_i.shapes[self.all["back_mask"]]):
                    process_i.shapes[np.isin(process_i.shapes, np.unique(process_i.shapes[self.all["back_mask"]]))] = 0
                    process_i.validated_shapes = (process_i.shapes > 0).astype(np.uint8)
            if self.all["bio_mask"] is not None:
                process_i.validated_shapes[self.all["bio_mask"]] = 1
            if self.all["back_mask"] is not None or self.all["bio_mask"] is not None:
                process_i.shape_number, process_i.shapes = cv2.connectedComponents(process_i.validated_shapes, connectivity=8)
                process_i.shape_number -= 1

            self.first_image.validated_shapes = process_i.validated_shapes
            self.first_image.shape_number = process_i.shape_number
            if self.first_image.im_combinations is None:
                self.first_image.im_combinations = []
                self.first_image.im_combinations.append({})
            self.first_image.im_combinations[self.current_combination_id]['csc'] = self.vars['convert_for_origin']
            self.first_image.im_combinations[self.current_combination_id]['binary_image'] = self.first_image.validated_shapes
            self.first_image.im_combinations[self.current_combination_id]['converted_image'] = np.round(self.first_image.image).astype(np.uint8)
            self.first_image.im_combinations[self.current_combination_id]['shape_number'] = process_i.shape_number
            # self.first_image.generate_color_space_combination(self.vars['convert_for_origin'], subtract_background)
        else:
            # self.last_image.segmentation(self.vars['convert_for_motion']['logical'], self.vars['color_number'])
            # if self.vars['drift_already_corrected']:
            #     drift_correction, drift_correction2 = self.last_image.adjust_to_drift_correction()
            #     self.last_image.segmentation(self.vars['convert_for_motion']['logical'], self.vars['color_number'])
            self.cropping(is_first_image=False)
            print(self.vars["filter_spec"])
            self.last_image.convert_and_segment(self.vars['convert_for_motion'], self.vars["color_number"],
                                                biomask, backmask, self.first_image.subtract_background,
                                                self.first_image.subtract_background2,
                                                grid_segmentation=self.vars["grid_segmentation"],
                                                filter_spec=self.vars["filter_spec"])
            if self.vars['drift_already_corrected'] and not self.last_image.drift_correction_already_adjusted and not self.vars["grid_segmentation"]:
                self.last_image.adjust_to_drift_correction(self.vars['convert_for_motion']['logical'])
            
            if self.last_image.im_combinations is None:
                self.last_image.im_combinations = []
                self.last_image.im_combinations.append({})
            self.last_image.im_combinations[self.current_combination_id]['csc'] = self.vars['convert_for_motion']
            self.last_image.im_combinations[self.current_combination_id]['binary_image'] = self.last_image.binary_image
            self.last_image.im_combinations[self.current_combination_id]['converted_image'] = np.round(self.last_image.image).astype(np.uint8)

            # self.last_image.generate_color_space_combination(self.vars['convert_for_motion'], subtract_background)
            # if self.all["more_than_two_colors"]:
            #    self.last_image.kmeans(self.vars["color_number"])
            # else:
            # self.last_image.thresholding()
            # if self.all['are_gravity_centers_moving'] != 1:
            #     self.delineate_each_arena()

    def cropping(self, is_first_image):
        if not self.vars['drift_already_corrected']:
            if is_first_image:
                if not self.first_image.cropped:
                    if (not self.all['overwrite_unaltered_videos'] and os.path.isfile('Data to run Cellects quickly.pkl')):
                        pickle_rick = PickleRick()
                        data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
                        if data_to_run_cellects_quickly is not None:
                            if 'coordinates' in data_to_run_cellects_quickly:
                                logging.info("Get crop coordinates from Data to run Cellects quickly.pkl")
                                (ccy1, ccy2, ccx1, ccx2, self.left, self.right, self.top, self.bot) = \
                                    data_to_run_cellects_quickly['coordinates']
                                self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                            else:
                                self.first_image.get_crop_coordinates()
                        else:
                            self.first_image.get_crop_coordinates()


                        # try:
                        #     with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
                        #         data_to_run_cellects_quickly = pickle.load(fileopen)
                        #         if 'coordinates' in data_to_run_cellects_quickly:
                        #             logging.info("Get crop coordinates from Data to run Cellects quickly.pkl")
                        #             (ccy1, ccy2, ccx1, ccx2, self.left, self.right, self.top, self.bot) = \
                        #                 data_to_run_cellects_quickly['coordinates']
                        #             self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                        #         else:
                        #             self.first_image.get_crop_coordinates()
                        # except pickle.UnpicklingError:
                        #     logging.error("Pickle error")
                        #     self.first_image.get_crop_coordinates()


                    # if (not self.all['overwrite_unaltered_videos'] and os.path.isfile('coordinates.pkl')):
                    #     with open('coordinates.pkl', 'rb') as fileopen:
                    #         (ccy1, ccy2, ccx1, ccx2, self.videos.left, self.videos.right, self.videos.top,
                    #          self.videos.bot) = pickle.load(fileopen)
                    else:
                        self.first_image.get_crop_coordinates()
                    if self.all['automatically_crop']:
                        self.first_image.automatically_crop(self.first_image.crop_coord)
                    else:
                        self.first_image.crop_coord = None
            else:
                if not self.last_image.cropped and self.all['automatically_crop']:
                    self.last_image.automatically_crop(self.first_image.crop_coord)
        # if self.all['automatically_crop'] and not self.vars['drift_already_corrected']:
        #     if is_first_image:
        #         self.first_image.get_crop_coordinates()
        #         self.first_image.automatically_crop(self.first_image.crop_coord)
        #     else:
        #         self.last_image.automatically_crop(self.first_image.crop_coord)

    def get_average_pixel_size(self):
        logging.info("Get average pixel size")
        (self.first_image.shape_number,
            self.first_image.shapes,
            self.first_image.stats,
            centroids) = cv2.connectedComponentsWithStats(
                self.first_image.validated_shapes,
                connectivity=8)
        self.first_image.shape_number -= 1
        if self.all['scale_with_image_or_cells'] == 0:
            self.vars['average_pixel_size'] = np.square(
                self.all['image_horizontal_size_in_mm'] /
                self.first_im.shape[1])
        else:
            self.vars['average_pixel_size'] = np.square(
                self.all['starting_blob_hsize_in_mm'] /
                np.mean(self.first_image.stats[1:, 2]))
        if self.all['set_spot_size']:
            self.starting_blob_hsize_in_pixels = (
                self.all['starting_blob_hsize_in_mm'] /
                np.sqrt(self.vars['average_pixel_size']))
        else:
            self.starting_blob_hsize_in_pixels = None

        if self.all['automatic_size_thresholding']:
            self.vars['first_move_threshold'] = 10
        else:
            # if self.vars['origin_state'] != "invisible":
            self.vars['first_move_threshold'] = np.round(
                self.all['first_move_threshold_in_mm²'] /
                self.vars['average_pixel_size']).astype(np.uint8)
        logging.info(f"The average pixel size is: {self.vars['average_pixel_size']} mm²")

    def delineate_each_arena(self):
        self.videos = OneVideoPerBlob(
            self.first_image,
            self.starting_blob_hsize_in_pixels,
            self.all['raw_images'])
        # self.delineation_number += 1
        # if self.delineation_number > 1:
        #     print('stophere')
        analysis_status = {"continue": True, "message": ""}
        if (self.sample_number > 1 and not self.vars['several_blob_per_arena']):
            compute_get_bb: bool = True
            if (not self.all['overwrite_unaltered_videos'] and os.path.isfile('Data to run Cellects quickly.pkl')):

                pickle_rick = PickleRick()
                data_to_run_cellects_quickly = pickle_rick.read_file('Data to run Cellects quickly.pkl')
                if data_to_run_cellects_quickly is not None:
                    if 'coordinates' in data_to_run_cellects_quickly:
                        (ccy1, ccy2, ccx1, ccx2, self.left, self.right, self.top, self.bot) = \
                            data_to_run_cellects_quickly['coordinates']
                        self.videos.left, self.videos.right, self.videos.top, self.videos.bot = self.left, self.right, self.top, self.bot
                        self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                        if (self.first_image.image.shape[0] == (ccy2 - ccy1)) and (
                                self.first_image.image.shape[1] == (ccx2 - ccx1)):  # maybe useless now
                            logging.info("Get the coordinates of all arenas from Data to run Cellects quickly.pkl")
                            compute_get_bb = False


                # try:
                #     with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
                #         data_to_run_cellects_quickly = pickle.load(fileopen)
                # except pickle.UnpicklingError:
                #     logging.error("Pickle error")
                #     data_to_run_cellects_quickly = {}
                # if 'coordinates' in data_to_run_cellects_quickly:
                #     (ccy1, ccy2, ccx1, ccx2, self.left, self.right, self.top, self.bot) = \
                #         data_to_run_cellects_quickly['coordinates']
                #     self.first_image.crop_coord = [ccy1, ccy2, ccx1, ccx2]
                #     if (self.first_image.image.shape[0] == (ccy2 - ccy1)) and (
                #             self.first_image.image.shape[1] == (ccx2 - ccx1)):  # maybe useless now
                #         logging.info("Get the coordinates of all arenas from Data to run Cellects quickly.pkl")
                #         compute_get_bb = False

            # if (not self.all['overwrite_unaltered_videos'] and os.path.isfile('coordinates.pkl')):
            #     with open('coordinates.pkl', 'rb') as fileopen:
            #         (ccy1, ccy2, ccx1, ccx2, self.videos.left, self.videos.right, self.videos.top, self.videos.bot) = pickle.load(fileopen)

            # if (not self.all['overwrite_unaltered_videos'] and
            #         os.path.isfile('coordinates.pkl')):
            #     with open('coordinates.pkl', 'rb') as fileopen:
            #         (vertical_shape, horizontal_shape, self.videos.left, self.videos.right, self.videos.top,
            #             self.videos.bot) = pickle.load(fileopen)

            if compute_get_bb:
                if self.all['im_or_vid'] == 1:
                    self.videos.get_bounding_boxes(
                        are_gravity_centers_moving=self.all['are_gravity_centers_moving'] == 1,
                        img_list=self.analysis_instance,
                        color_space_combination=self.vars['convert_for_origin'],#self.vars['convert_for_motion']
                        color_number=self.vars["color_number"],
                        sample_size=5,
                        all_specimens_have_same_direction=self.all['all_specimens_have_same_direction'],
                        filter_spec=self.vars['filter_spec'])
                else:
                    self.videos.get_bounding_boxes(
                        are_gravity_centers_moving=self.all['are_gravity_centers_moving'] == 1,
                        img_list=self.data_list,
                        color_space_combination=self.vars['convert_for_origin'],
                        color_number=self.vars["color_number"],
                        sample_size=5,
                        all_specimens_have_same_direction=self.all['all_specimens_have_same_direction'],
                        filter_type=self.vars['filter_spec'])
                if np.any(self.videos.ordered_stats[:, 4] > 100 * np.median(self.videos.ordered_stats[:, 4])):
                    analysis_status['message'] = "A specimen is at least 100 times larger: (re)do the first image analysis."
                    analysis_status['continue'] = False
                if np.any(self.videos.ordered_stats[:, 4] < 0.01 * np.median(self.videos.ordered_stats[:, 4])):
                    analysis_status['message'] = "A specimen is at least 100 times smaller: (re)do the first image analysis."
                    analysis_status['continue'] = False
                logging.info(
                    str(self.videos.not_analyzed_individuals) + " individuals are out of picture scope and cannot be analyzed")
            self.left, self.right, self.top, self.bot = self.videos.left, self.videos.right, self.videos.top, self.videos.bot

        else:
            self.left, self.right, self.top, self.bot = np.array([1]), np.array([self.first_image.image.shape[1] - 2]), np.array([1]), np.array([self.first_image.image.shape[0] - 2])
            self.videos.left, self.videos.right, self.videos.top, self.videos.bot = np.array([1]), np.array([self.first_image.image.shape[1] - 2]), np.array([1]), np.array([self.first_image.image.shape[0] - 2])

        self.vars['analyzed_individuals'] = np.arange(self.sample_number) + 1
        if self.videos.not_analyzed_individuals is not None:
            self.vars['analyzed_individuals'] = np.delete(self.vars['analyzed_individuals'],
                                                       self.videos.not_analyzed_individuals - 1)
        # logging.info(self.top)
        return analysis_status

    def get_background_to_subtract(self):
        """
            Repenser le moment où ça arrive et trouver pourquoi ça marche pas
        """
        # self.vars['subtract_background'] = False
        if self.vars['subtract_background']:
            self.first_image.generate_subtract_background(self.vars['convert_for_motion'])

    def get_origins_and_backgrounds_lists(self):
        logging.info("Create origins and background lists")
        if self.top is None:
            # self.top = [1]
            # self.bot = [self.first_im.shape[0] - 2]
            # self.left = [1]
            # self.right = [self.first_im.shape[1] - 2]
            self.top = np.array([1])
            self.bot = np.array([self.first_im.shape[0] - 2])
            self.left = np.array([1])
            self.right = np.array([self.first_im.shape[1] - 2])

        add_to_c = 1
        first_im = self.first_image.validated_shapes
        self.vars['origin_list'] = []
        self.vars['background_list'] = []
        self.vars['background_list2'] = []
        for rep in np.arange(len(self.vars['analyzed_individuals'])):
            self.vars['origin_list'].append(first_im[self.top[rep]:(self.bot[rep] + add_to_c),
                                             self.left[rep]:(self.right[rep] + add_to_c)])
            if self.vars['subtract_background']:
                self.vars['background_list'].append(
                    self.first_image.subtract_background[self.top[rep]:(self.bot[rep] + add_to_c),
                    self.left[rep]:(self.right[rep] + add_to_c)])
                if self.vars['convert_for_motion']['logical'] != 'None':
                    self.vars['background_list2'].append(
                        self.first_image.subtract_background2[self.top[rep]:(self.bot[rep] + add_to_c),
                        self.left[rep]:(self.right[rep] + add_to_c)])

    def get_origins_and_backgrounds_one_by_one(self):
        add_to_c = 1
        self.vars['origin_list'] = []
        self.vars['background_list'] = []
        self.vars['background_list2'] = []

        for arena in np.arange(len(self.vars['analyzed_individuals'])):
            bgr_image = self.first_image.bgr[self.top[arena]:(self.bot[arena] + add_to_c),
                                             self.left[arena]:(self.right[arena] + add_to_c), ...]
            image = OneImageAnalysis(bgr_image)
            if self.vars['subtract_background']:
                image.generate_subtract_background(self.vars['convert_for_motion'])
                self.vars['background_list'].append(image.image)
                if self.vars['convert_for_motion']['logical'] != 'None':
                    self.vars['background_list2'].append(image.image2)

            # self.vars['origins_list'].append(self.first_image.validated_shapes[self.top[arena]:(self.bot[arena]),
            #                                  self.left[arena]:(self.right[arena])])
            #
            if self.vars['several_blob_per_arena']:
                image.validated_shapes = image.binary_image
            else:
                image.get_largest_shape()

            self.vars['origin_list'].append(image.validated_shapes)

    def choose_color_space_combination(self):
        # self = po
        # 2) Represent the segmentation using a particular color space combination
        if self.all['are_gravity_centers_moving'] != 1:
            analysis_status = self.delineate_each_arena()
        # self.fi.automatically_crop(self.first_image.crop_coord)
        self.last_image = OneImageAnalysis(self.last_im)
        self.last_image.automatically_crop(self.videos.first_image.crop_coord)
        # csc = ColorSpaceCombination(self.last_image.image)

        concomp_nb = [self.sample_number, self.sample_number * 50]
        if self.all['are_zigzag'] == "columns":
            inter_dist = np.mean(np.diff(np.nonzero(self.videos.first_image.y_boundaries)))
        elif self.all['are_zigzag'] == "rows":
            inter_dist = np.mean(np.diff(np.nonzero(self.videos.first_image.x_boundaries)))
        else:
            dist1 = np.mean(np.diff(np.nonzero(self.videos.first_image.y_boundaries)))
            dist2 = np.mean(np.diff(np.nonzero(self.videos.first_image.x_boundaries)))
            inter_dist = np.max(dist1, dist2)
        if self.all['starting_blob_shape'] == "circle":
            max_shape_size = np.pi * np.square(inter_dist)
        else:
            max_shape_size = np.square(2 * inter_dist)
        total_surfarea = max_shape_size * self.sample_number
        if self.all['are_gravity_centers_moving'] != 1:
            out_of_arenas = np.ones_like(self.videos.first_image.validated_shapes)
            for blob_i in np.arange(len(self.vars['analyzed_individuals'])):
                out_of_arenas[self.top[blob_i]: (self.bot[blob_i] + 1),
                self.left[blob_i]: (self.right[blob_i] + 1)] = 0
        else:
            out_of_arenas = None
        ref_image = self.videos.first_image.validated_shapes
        self.last_image.find_potential_channels(concomp_nb, total_surfarea, max_shape_size, out_of_arenas, ref_image)
        # csc.find_potential_channels(concomp_nb, total_surfarea, max_shape_size, out_of_arenas, ref_image)
        # csc.find_potential_channels(concomp_nb, total_surfarea, max_shape_size, out_of_arenas, ref_image, self.first_image.subtract_background)
        self.vars['convert_for_motion'] = self.last_image.channel_combination

        self.fast_image_segmentation(False)
        # if self.vars['subtract_background']:
        #     csc = ColorSpaceCombination(self.last_image.image)
        #     csc.generate(self.vars['convert_for_motion'], self.first_image.subtract_background)
        # if self.all["more_than_two_colors"]:
        #     csc.kmeans(self.vars["color_number"])
        # else:
        #     csc.thresholding()
        # self.last_image.image = csc.image
        # self.last_image.binary_image = csc.binary_image

    def untype_csc_dict(self):
        new_convert_for_origin = {}
        for k, v in self.vars['convert_for_origin'].items():
            new_convert_for_origin[k] = v
        if self.vars['logical_between_csc_for_origin'] is not None:
            new_convert_for_origin['logical'] = self.vars['logical_between_csc_for_origin']
            for k, v in self.vars['convert_for_origin2'].items():
                new_convert_for_origin[k] = v
        self.vars['convert_for_origin'] = new_convert_for_origin
        self.vars['convert_for_origin2'] = {}

        new_convert_for_motion = {}
        for k, v in self.vars['convert_for_motion'].items():
            new_convert_for_motion[k] = v
        if self.vars['convert_for_motion']['logical']  != 'None':
            new_convert_for_motion['logical'] = self.vars['convert_for_motion']['logical']
            for k, v in self.vars['convert_for_motion2'].items():
                new_convert_for_motion[k] = v
        self.vars['convert_for_motion'] = new_convert_for_motion
        self.vars['convert_for_motion2'] = {}

    def type_csc_dict(self):
        # self.vars['convert_for_motion']['logical'] = 'And'
        # self.vars['convert_for_motion']['hsv'] = np.array((0, 0, 1))
        # self.vars['convert_for_motion']['logical'] = 'And'
        # self.vars['convert_for_motion']['lab2'] = np.array((0, 0, 1))

        new_convert_for_origin = TDict()
        self.vars['convert_for_origin2'] = TDict()
        self.vars['logical_between_csc_for_origin'] = None
        for k, v in self.vars['convert_for_origin'].items():
             if k != 'logical' and v.sum() > 0:
                 if k[-1] != '2':
                     new_convert_for_origin[k] = v
                 else:
                     self.vars['convert_for_origin2'][k[:-1]] = v
             else:
                 self.vars['logical_between_csc_for_origin'] = v
        self.vars['convert_for_origin'] = new_convert_for_origin

        new_convert_for_motion = TDict()
        self.vars['convert_for_motion2'] = TDict()
        self.vars['convert_for_motion']['logical'] = None
        for k, v in self.vars['convert_for_motion'].items():
            if k != 'logical' and v.sum() > 0:
                if k[-1] != '2':
                    new_convert_for_motion[k] = v
                else:
                    self.vars['convert_for_motion2'][k[:-1]] = v
            else:
                self.vars['convert_for_motion']['logical'] = v
        self.vars['convert_for_motion'] = new_convert_for_motion

        if self.vars['color_number'] > 2:
            self.vars['bio_label'] = None#self.first_image.bio_label
            if self.vars['convert_for_motion']['logical']  != 'None':
                self.vars['bio_label2'] = None

    def find_if_lighter_background(self):
        logging.info("Find if the background is lighter or darker than the cells")
        self.vars['lighter_background']: bool = True
        self.vars['contour_color']: np.uint8 = 0
        are_dicts_equal: bool = True
        for key in self.vars['convert_for_origin'].keys():
            are_dicts_equal = are_dicts_equal and np.all(key in self.vars['convert_for_motion'] and self.vars['convert_for_origin'][key] == self.vars['convert_for_motion'][key])
        for key in self.vars['convert_for_motion'].keys():
            are_dicts_equal = are_dicts_equal and np.all(key in self.vars['convert_for_origin'] and self.vars['convert_for_motion'][key] == self.vars['convert_for_origin'][key])

        if are_dicts_equal:

            if self.first_im is None:
                self.get_first_image()
                self.fast_image_segmentation(True)
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
                self.fast_image_segmentation(is_first_image=False)
            if self.last_image.binary_image.sum() == 0:
                self.fast_image_segmentation(is_first_image=False)
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
                                                 grid_segmentation=self.vars["grid_segmentation"],
                                                 filter_spec=self.vars["filter_spec"])
            covered_values = self.first_image.image[np.nonzero(binary_image)]
            if self.vars['lighter_background']:
                if np.max(covered_values) < 255:
                    self.vars['luminosity_threshold'] = np.max(covered_values) + 1
                else:
                    self.vars['luminosity_threshold'] = 127
            else:
                if np.min(covered_values) > 0:
                    self.vars['luminosity_threshold'] = np.min(covered_values) - 1
                else:
                    self.vars['luminosity_threshold'] = 127

    def load_one_arena(self, arena):
        #self.delineate_each_arena()
        #self.choose_color_space_combination()
        add_to_c = 1
        self.one_arena_done = True
        i = np.nonzero(self.vars['analyzed_individuals'] == arena)[0][0]
        self.converted_video = np.zeros(
            (len(self.data_list), self.bot[i] - self.top[i] + add_to_c, self.right[i] - self.left[i] + add_to_c),
            dtype=float)
        if not self.vars['already_greyscale']:
            self.visu = np.zeros((len(self.data_list), self.bot[i] - self.top[i] + add_to_c, self.right[i] - self.left[i] + add_to_c, 3), dtype=np.uint8)
            if self.vars['convert_for_motion']['logical'] != 'None':
                self.converted_video2 = np.zeros((len(self.data_list), self.bot[i] - self.top[i] + add_to_c, self.right[i] - self.left[i] + add_to_c), dtype=float)
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
        prev_img = None
        background = None
        background2 = None
        is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
        for image_i, image_name in enumerate(self.data_list):
            img = read_and_rotate(image_name, prev_img, self.all['raw_images'], is_landscape, self.first_image.crop_coord)
            prev_img = deepcopy(img)
            # if self.videos.first_image.crop_coord is not None:
            #     img = img[self.videos.first_image.crop_coord[0]:self.videos.first_image.crop_coord[1],
            #               self.videos.first_image.crop_coord[2]:self.videos.first_image.crop_coord[3], :]
            img = img[self.top[arena - 1]: (self.bot[arena - 1] + add_to_c),
                                                    self.left[arena - 1]: (self.right[arena - 1] + add_to_c), :]

            if self.vars['already_greyscale']:
                if self.reduce_image_dim:
                    self.converted_video[image_i, ...] = img[:, :, 0]
                else:
                    self.converted_video[image_i, ...] = img
            else:
                self.visu[image_i, ...] = img
                if self.vars['subtract_background']:
                    background = self.vars['background_list'][i]
                    if self.vars['convert_for_motion']['logical'] != 'None':
                        background2 = self.vars['background_list2'][i]
                greyscale_image, greyscale_image2 = generate_color_space_combination(img, c_spaces, first_dict,
                                                                                     second_dict, background, background2,
                                                                                     self.vars[
                                                                                         'lose_accuracy_to_save_memory'])
                self.converted_video[image_i, ...] = greyscale_image
                if self.vars['convert_for_motion']['logical'] != 'None':
                    self.converted_video2[image_i, ...] = greyscale_image2
                # csc = OneImageAnalysis(img)
                # else:
                #     csc.generate_color_space_combination(c_spaces, first_dict, second_dict, None, None)
                # # self.converted_video[image_i, ...] = csc.image
                # self.converted_video[image_i, ...] = csc.image
                # if self.vars['convert_for_motion']['logical'] != 'None':
                #     self.converted_video2[image_i, ...] = csc.image2

        # write_video(self.visu, f"ind_{arena}{self.vars['videos_extension']}", is_color=True, fps=1)

    def update_output_list(self):
        self.vars['descriptors'] = {}
        # self.vars['descriptors']['final_area'] = True  # [False, True, False]
        # if self.vars['first_move_threshold'] is not None:
        #     self.vars['descriptors']['first_move'] = True  # [False, True, False]

        # if self.vars['iso_digi_analysis']:
        #     self.vars['descriptors']['is_growth_isotropic'] = True  # [False, True, False]
        #     self.vars['descriptors']['iso_digi_transi'] = True  # [False, True, False]

        # if self.vars['oscilacyto_analysis']:
        # self.vars['descriptors']['max_magnitude'] = True#[False, True, False]
        # self.vars['descriptors']['frequency_of_max_magnitude'] = True#[False, True, False]
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
        self.vars['descriptors']['cluster_number'] = self.vars['oscilacyto_analysis']
        self.vars['descriptors']['mean_cluster_area'] = self.vars['oscilacyto_analysis']
        self.vars['descriptors']['vertices_number'] = self.vars['network_analysis']
        self.vars['descriptors']['edges_number'] = self.vars['network_analysis']
        self.vars['descriptors']['newly_explored_area'] = self.vars['do_fading']
        """                             if self.vars['descriptors_means']:
                    self.vars['output_list'] += [f'{descriptor}_mean']
                    self.vars['output_list'] += [f'{descriptor}_std']
                if self.vars['descriptors_regressions']:
                    self.vars['output_list'] += [f"{descriptor}_reg_start"]
                    self.vars['output_list'] += [f"{descriptor}_reg_end"]
                    self.vars['output_list'] += [f'{descriptor}_slope']
                    self.vars['output_list'] += [f'{descriptor}_intercept']
        """

    def update_available_core_nb(self, image_bit_number=256, video_bit_number=140):# video_bit_number=176
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
        if self.vars['save_coord_network'] or self.vars['network_analysis']:
            video_bit_number += 8
            image_bit_number += 64

        if isinstance(self.bot, list):
            one_image_memory = np.multiply((self.bot[0] - self.top[0] + 1),
                                        (self.right[0] - self.left[0] + 1)).max().astype(np.uint64)
        else:
            one_image_memory = np.multiply((self.bot - self.top + 1).astype(np.uint64),
                                        (self.right - self.left + 1).astype(np.uint64)).max()
        one_video_memory = self.vars['img_number'] * one_image_memory
        necessary_memory = (one_image_memory * image_bit_number + one_video_memory * video_bit_number) * 1.16415e-10
        available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        max_repeat_in_memory = (available_memory // necessary_memory).astype(np.uint16)
        if max_repeat_in_memory > 1:
            max_repeat_in_memory = np.max(((available_memory // (2 * necessary_memory)).astype(np.uint16), 1))
        # if sys.platform.startswith('win'):
        #     available_memory = (virtual_memory().available >> 30) - self.vars['min_ram_free']
        # else:

        self.cores = np.min((self.all['cores'], max_repeat_in_memory))
        if self.cores > self.sample_number:
            self.cores = self.sample_number
        return np.round(np.absolute(available_memory - necessary_memory), 3)


    def update_one_row_per_arena(self, i, table_to_add):
        if not self.vars['several_blob_per_arena']:
            if self.one_row_per_arena is None:
                self.one_row_per_arena = pd.DataFrame(np.zeros((len(self.vars['analyzed_individuals']), len(table_to_add)), dtype=float),
                                            columns=table_to_add.keys())
            self.one_row_per_arena.iloc[i, :] = table_to_add.values()


    def update_one_row_per_frame(self, i, j, table_to_add):
        if not self.vars['several_blob_per_arena']:
            if self.one_row_per_frame is None:
                self.one_row_per_frame = pd.DataFrame(index=range(len(self.vars['analyzed_individuals']) *
                                                        self.vars['img_number']),
                                            columns=table_to_add.keys())

            self.one_row_per_frame.iloc[i:j, :] = table_to_add


    def instantiate_tables(self):
        self.update_output_list()
        logging.info("Instantiate results tables and validation images")
        self.one_row_per_oscillating_cluster = None
        self.fractal_box_sizes = None
        # if self.vars['oscilacyto_analysis']:
        #     self.one_row_per_oscillating_cluster = pd.DataFrame(columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size',
        #                                                        'edge_distance'])
        # if self.vars['fractal_analysis']:
        #     self.fractal_box_sizes = pd.DataFrame(columns=['arena', 'time', 'fractal_box_lengths', 'fractal_box_widths'])

        if self.vars['already_greyscale']:
            if len(self.first_image.bgr.shape) == 2:
                self.first_image.bgr = np.stack((self.first_image.bgr, self.first_image.bgr, self.first_image.bgr), axis=2).astype(np.uint8)
            if len(self.last_image.bgr.shape) == 2:
                self.last_image.bgr = np.stack((self.last_image.bgr, self.last_image.bgr, self.last_image.bgr), axis=2).astype(np.uint8)
            self.vars["convert_for_motion"] = {"bgr": np.array((1, 1, 1), dtype=np.uint8), "logical": "None"}

    def add_analysis_visualization_to_first_and_last_images(self, i, first_visualization, last_visualization):
        cr = ((self.top[i], self.bot[i] + 1),
              (self.left[i], self.right[i] + 1))
        if self.vars['arena_shape'] == 'circle':
            ellipse = Ellipse((cr[0][1] - cr[0][0], cr[1][1] - cr[1][0])).create()
            ellipse = np.stack((ellipse, ellipse, ellipse), axis=2).astype(np.uint8)
            first_visualization *= ellipse
            self.first_image.bgr[cr[0][0]:cr[0][1], cr[1][0]:cr[1][1], ...] *= (1 - ellipse)
            self.first_image.bgr[cr[0][0]:cr[0][1], cr[1][0]:cr[1][1], ...] += first_visualization
            last_visualization *= ellipse
            self.last_image.bgr[cr[0][0]:cr[0][1], cr[1][0]:cr[1][1], ...] *= (1 - ellipse)
            self.last_image.bgr[cr[0][0]:cr[0][1], cr[1][0]:cr[1][1], ...] += last_visualization
        else:
            self.first_image.bgr[cr[0][0]:cr[0][1], cr[1][0]:cr[1][1], ...] = first_visualization
            self.last_image.bgr[cr[0][0]:cr[0][1], cr[1][0]:cr[1][1], ...] = last_visualization


    def save_tables(self):
        logging.info("Save results tables and validation images")
        if not self.vars['several_blob_per_arena']:
            try:
                self.one_row_per_arena.to_csv("one_row_per_arena.csv", sep=";", index=False, lineterminator='\n')
                del self.one_row_per_arena
            except PermissionError:
                logging.error("Never let one_row_per_arena.csv open when Cellects runs")
                self.message_from_thread.emit(f"Never let one_row_per_arena.csv open when Cellects runs")
            try:
                self.one_row_per_frame.to_csv("one_row_per_frame.csv", sep=";", index=False, lineterminator='\n')
                del self.one_row_per_frame
            except PermissionError:
                logging.error("Never let one_row_per_frame.csv open when Cellects runs")
                self.message_from_thread.emit(f"Never let one_row_per_frame.csv open when Cellects runs")
        if self.vars['oscilacyto_analysis']:
            try:
                if self.one_row_per_oscillating_cluster is None:
                    self.one_row_per_oscillating_cluster = pd.DataFrame(columns=['arena', 'mean_pixel_period', 'phase', 'cluster_size',
                                                                       'edge_distance'])
                self.one_row_per_oscillating_cluster.to_csv("one_row_per_oscillating_cluster.csv", sep=";", index=False,
                                                            lineterminator='\n')
                del self.one_row_per_oscillating_cluster
            except PermissionError:
                logging.error("Never let one_row_per_oscillating_cluster.csv open when Cellects runs")
                self.message_from_thread.emit(f"Never let one_row_per_oscillating_cluster.csv open when Cellects runs")

            if self.vars['fractal_analysis']:
                if os.path.isfile(f"oscillating_clusters_temporal_dynamics.h5"):
                    array_names = get_h5_keys(f"oscillating_clusters_temporal_dynamics.h5")
                    arena_fractal_dynamics = read_h5_array(f"oscillating_clusters_temporal_dynamics.h5", key=array_names[0])
                    arena_fractal_dynamics = np.hstack((np.repeat(np.uint32(array_names[0][-1]), arena_fractal_dynamics.shape[0]), arena_fractal_dynamics))
                    for array_name in array_names[1:]:
                        fractal_dynamics = read_h5_array(f"oscillating_clusters_temporal_dynamics.h5", key=array_name)
                        fractal_dynamics = np.hstack((np.repeat(np.uint32(array_name[-1]), fractal_dynamics.shape[0]), fractal_dynamics))
                        arena_fractal_dynamics = np.vstack((arena_fractal_dynamics, fractal_dynamics))
                    arena_fractal_dynamics = pd.DataFrame(arena_fractal_dynamics, columns=["arena", "time", "cluster_id", "flow", "centroid_y", "centroid_x", "area", "inner_network_area", "box_count_dim", "inner_network_box_count_dim"])
                    arena_fractal_dynamics.to_csv(f"oscillating_clusters_temporal_dynamics.csv", sep=";", index=False,
                                                                lineterminator='\n')
                    del arena_fractal_dynamics
                    os.remove(f"oscillating_clusters_temporal_dynamics.h5")
        if self.all['extension'] == '.JPG':
            extension = '.PNG'
        else:
            extension = '.JPG'
        cv2.imwrite(f"Analysis efficiency, last image{extension}", self.last_image.bgr)
        cv2.imwrite(
            f"Analysis efficiency, {np.ceil(self.vars['img_number'] / 10).astype(np.uint64)}th image{extension}",
            self.first_image.bgr)
        # self.save_analysis_parameters.to_csv("analysis_parameters.csv", sep=";")

        software_settings = deepcopy(self.vars)
        for key in ['descriptors', 'analyzed_individuals', 'exif', 'dims', 'origin_list', 'background_list', 'background_list2', 'descriptors', 'folder_list', 'sample_number_per_folder']:
            software_settings.pop(key, None)
        global_settings = deepcopy(self.all)
        for key in ['analyzed_individuals', 'night_mode', 'expert_mode', 'is_auto', 'arena', 'video_option', 'compute_all_options', 'vars', 'dims', 'origin_list', 'background_list', 'background_list2', 'descriptors', 'folder_list', 'sample_number_per_folder']:
            global_settings.pop(key, None)
        software_settings.update(global_settings)
        software_settings = pd.DataFrame.from_dict(software_settings, columns=["Setting"], orient='index')
        try:
            software_settings.to_csv("software_settings.csv", sep=";")
        except PermissionError:
            logging.error("Never let software_settings.csv open when Cellects runs")
            self.message_from_thread.emit(f"Never let software_settings.csv open when Cellects runs")



# if __name__ == "__main__":
#     po = ProgramOrganizer()
#     os.chdir(Path("D:\Directory\Data\Example\Example\Example"))
#     # po.all['global_pathway'] = Path("C:/Users/APELab/Documents/Aurèle/Cellects/install/Installer_and_example/Example")
#     po.load_variable_dict()
#     po.all['global_pathway']
#     po.load_data_to_run_cellects_quickly()
#     po.all['global_pathway']
#     po.save_data_to_run_cellects_quickly()
