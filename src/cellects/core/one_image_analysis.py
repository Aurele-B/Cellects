#!/usr/bin/env python3
"""
Module providing tools for single-image color space analysis and segmentation.

The OneImageAnalysis class offers comprehensive image processing capabilities including
color space conversion (RGB, HSV, LAB, LUV, HLS, YUV), filtering (Gaussian, median, bilateral),
segmentation (Otsu thresholding, k-means clustering), and shape-based validation. It supports
multi-step optimization of color channel combinations to maximize contrast between organisms
and background through automated selection workflows involving logical operations on segmented regions.

Classes
OneImageAnalysis : Analyze images using multiple color spaces for optimal segmentation

Notes
Uses QThread for background operations during combination processing.
"""

import logging
import os
from copy import deepcopy
import numpy as np
import cv2  # named opencv-python
import multiprocessing.pool as mp
from numba.typed import List as TList
from numba.typed import Dict as TDict
from numpy.typing import NDArray
from typing import Tuple
from skimage.measure import perimeter
from cellects.image_analysis.morphological_operations import cross_33, create_ellipse, spot_size_coefficients
from cellects.image_analysis.image_segmentation import generate_color_space_combination, get_color_spaces, extract_first_pc, combine_color_spaces, apply_filter, otsu_thresholding, get_otsu_threshold, kmeans, windowed_thresholding
from cellects.image_analysis.one_image_analysis_threads import SaveCombinationThread, ProcessFirstImage
from cellects.image_analysis.network_functions import NetworkDetection
from cellects.utils.formulas import bracket_to_uint8_image_contrast
from cellects.utils.utilitarian import split_dict, translate_dict


class OneImageAnalysis:
    """
        This class takes a 3D matrix (2 space and 1 color [BGR] dimensions),
        Its methods allow image
        - conversion to any bgr/hsv/lab channels
        - croping
        - rotating
        - filtering using some of the mainly used techniques:
            - Gaussian, Median, Bilateral, Laplacian, Mexican hat
        - segmenting using thresholds or kmeans
        - shape selection according to horizontal size or shape ('circle' vs 'quadrilateral')

        ps: A viewing method displays the image before and after the most advanced modification made in instance
    """
    def __init__(self, image, shape_number=0):
        self.image = image
        if len(self.image.shape) == 2:
            self.already_greyscale = True
        else:
            self.already_greyscale = False
        self.image2 = None
        self.binary_image2 = None
        self.drift_correction_already_adjusted: bool = False
        # Create empty variables to fill in the following functions
        self.binary_image = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.previous_binary_image = None
        self.validated_shapes = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.centroids = 0
        self.shape_number = shape_number
        self.concomp_stats = 0
        self.y_boundaries = None
        self.x_boundaries = None
        self.crop_coord = None
        self.cropped: bool = False
        self.subtract_background = None
        self.subtract_background2 = None
        self.im_combinations = None
        self.bgr = image
        self.colorspace_list = TList(("bgr", "lab", "hsv", "luv", "hls", "yuv"))
        self.spot_shapes = None
        self.all_c_spaces = TDict()
        self.hsv = None
        self.hls = None
        self.lab = None
        self.luv = None
        self.yuv = None
        self.greyscale = None
        self.greyscale2 = None
        self.first_pc_vector = None
        self.drift_mask_coord = None
        self.saved_csc_nb = 0

    def convert_and_segment(self, c_space_dict: dict, color_number=2, biomask: NDArray[np.uint8]=None,
                            backmask: NDArray[np.uint8]=None, subtract_background: NDArray=None,
                            subtract_background2: NDArray=None, rolling_window_segmentation: dict=None,
                            lighter_background: bool=None,
                            allowed_window: NDArray=None, filter_spec: dict=None):
        """
        Convert an image to grayscale and segment it based on specified parameters.

        This method converts the given color space dictionary into grayscale
        images, combines them with existing color spaces and performs segmentation.
        It has special handling for images that are already in grayscale.

        **Args:**

        - `c_space_dict` (dict): Dictionary containing color spaces.
        - `color_number` (int, optional): Number of colors to use in segmentation. Defaults to 2.
        - `biomask` (NDArray[np.uint8], optional): Biomask for segmentation. Defaults to None.
        - `backmask` (NDArray[np.uint8], optional): Backmask for segmentation. Defaults to None.
        - `subtract_background` (NDArray, optional): Background to subtract. Defaults to None.
        - `subtract_background2` (NDArray, optional): Second background to subtract. Defaults to None.
        - rolling_window_segmentation (dict, optional): Flag for grid segmentation. Defaults to None.
        - `lighter_background` (bool, optional): Flag for lighter background. Defaults to None.
        - `mask` (NDArray, optional): Additional mask for segmentation. Defaults to None.
        - `filter_spec` (dict, optional): Filter specifications. Defaults to None.

        **Attributes:**

        - `self.already_greyscale` (bool): Indicates whether the image is already greyscale.
        - `self.all_c_spaces` (list): List of color spaces.

        """
        if not self.already_greyscale:
            first_dict, second_dict, c_spaces = split_dict(c_space_dict)
            self.image, self.image2, all_c_spaces, self.first_pc_vector = generate_color_space_combination(self.bgr, c_spaces, first_dict, second_dict, subtract_background, subtract_background2)
            if len(all_c_spaces) > len(self.all_c_spaces):
                self.all_c_spaces = all_c_spaces

        self.segmentation(logical=c_space_dict['logical'], color_number=color_number, biomask=biomask,
                          backmask=backmask, rolling_window_segmentation=rolling_window_segmentation,
                          lighter_background=lighter_background, allowed_window=allowed_window, filter_spec=filter_spec)


    def segmentation(self, logical: str='None', color_number: int=2, biomask: NDArray[np.uint8]=None,
                     backmask: NDArray[np.uint8]=None, bio_label=None, bio_label2=None,
                     rolling_window_segmentation: dict=None, lighter_background: bool=None, allowed_window: Tuple=None,
                     filter_spec: dict=None):
        """
        Implement segmentation on the image using various methods and parameters.

        Args:
            logical (str): Logical operation to perform between two binary images.
                           Options are 'Or', 'And', 'Xor'. Default is 'None'.
            color_number (int): Number of colors to use in segmentation. Must be greater than 2
                                for kmeans clustering. Default is 2.
            biomask (NDArray[np.uint8]): Binary mask for biological areas. Default is None.
            backmask (NDArray[np.uint8]): Binary mask for background areas. Default is None.
            bio_label (Any): Label for biological features. Default is None.
            bio_label2 (Any): Secondary label for biological features. Default is None.
            rolling_window_segmentation (dict): Whether to perform grid segmentation. Default is None.
            lighter_background (bool): Indicates if the background is lighter than objects.
                                       Default is None.
            allowed_window (Tuple): Mask to apply during segmentation. Default is None.
            filter_spec (dict): Dictionary of filters to apply on the image before segmentation.

        """
        # 1. Check valid pixels for segmentation (e.g. when there is a drift correction)
        if allowed_window is None:
            min_y, max_y, min_x, max_x = 0, self.image.shape[0] + 1, 0, self.image.shape[1] + 1
        else:
            min_y, max_y, min_x, max_x = allowed_window
        greyscale = self.image[min_y:max_y, min_x:max_x].copy()
        # 2. Apply filter on the greyscale images
        if filter_spec is not None and filter_spec["filter1_type"] != "":
            greyscale = apply_filter(greyscale, filter_spec["filter1_type"], filter_spec["filter1_param"])

        greyscale2 = None
        if logical != 'None':
            greyscale2 = self.image2[min_y:max_y, min_x:max_x].copy()
            if filter_spec is not None and filter_spec["filter2_type"] != "":
                greyscale2 = apply_filter(greyscale2, filter_spec["filter2_type"], filter_spec["filter2_param"])

        # 3. Do one of the three segmentation algorithms: kmeans, otsu, windowed
        if color_number > 2:
            binary_image, binary_image2, self.bio_label, self.bio_label2  = kmeans(greyscale, greyscale2, color_number, biomask, backmask, logical, bio_label, bio_label2)
        elif rolling_window_segmentation is not None and rolling_window_segmentation['do']:
            binary_image = windowed_thresholding(greyscale, lighter_background, rolling_window_segmentation['side_len'],
            rolling_window_segmentation['step'], rolling_window_segmentation['min_int_var'])
        else:
            binary_image = otsu_thresholding(greyscale)
        if logical != 'None' and color_number == 2:
            if rolling_window_segmentation is not None and rolling_window_segmentation['do']:
                binary_image2 = windowed_thresholding(greyscale2, lighter_background, rolling_window_segmentation['side_len'],
                rolling_window_segmentation['step'], rolling_window_segmentation['min_int_var'])
            else:
                binary_image2 = otsu_thresholding(greyscale2)

        # 4. Use previous_binary_image to make sure that the specimens are labelled with ones and the background zeros
        if self.previous_binary_image is not None:
            previous_binary_image = self.previous_binary_image[min_y:max_y, min_x:max_x]
            if not (binary_image * previous_binary_image).any() or (binary_image[0, :].all() and binary_image[-1, :].all() and binary_image[:, 0].all() and binary_image[:, -1].all()):
                # if (binary_image * (1 - previous_binary_image)).sum() > (binary_image * previous_binary_image).sum() + perimeter(binary_image):
                # Ones of the binary image have more in common with the background than with the specimen
                binary_image = 1 - binary_image
            if logical != 'None':
                if (binary_image2 * (1 - previous_binary_image)).sum() > (binary_image2 * previous_binary_image).sum():
                    binary_image2 = 1 - binary_image2

        # 5. Give back the image their original size and combine binary images (optional)
        self.binary_image = np.zeros(self.image.shape, dtype=np.uint8)
        self.binary_image[min_y:max_y, min_x:max_x] = binary_image
        self.greyscale = np.zeros(self.image.shape, dtype=np.uint8)
        self.greyscale[min_y:max_y, min_x:max_x] = greyscale
        if logical != 'None':
            self.binary_image2 = np.zeros(self.image.shape, dtype=np.uint8)
            self.binary_image2[min_y:max_y, min_x:max_x] = binary_image2
            self.greyscale2 = np.zeros(self.image.shape, dtype=np.uint8)
            self.greyscale2[min_y:max_y, min_x:max_x] = greyscale2
        if logical != 'None':
            if logical == 'Or':
                self.binary_image = np.logical_or(self.binary_image, self.binary_image2)
            elif logical == 'And':
                self.binary_image = np.logical_and(self.binary_image, self.binary_image2)
            elif logical == 'Xor':
                self.binary_image = np.logical_xor(self.binary_image, self.binary_image2)
            self.binary_image = self.binary_image.astype(np.uint8)

    def _get_all_color_spaces(self):
        """Generate and store all supported color spaces for the image."""
        if len(self.all_c_spaces) < 6 and not self.already_greyscale:
            self.all_c_spaces = get_color_spaces(self.bgr)

    def generate_subtract_background(self, c_space_dict: dict, drift_corrected: bool=False):
        """
        Generate a background-subtracted image using specified color space dictionary.

        This method first checks if color spaces have already been generated or
        if the image is greyscale. If not, it generates color spaces from the BGR
        image. It then converts and segments the image using the provided color space
        dictionary without grid segmentation. A disk-shaped structuring element is
        created and used to perform a morphological opening operation on the image,
        resulting in a background-subtracted version. If there is a second image
        (see Also: image2), the same operation is performed on it.

        Args:
            c_space_dict (dict): Dictionary containing color space specifications
                for the segmentation process.

        Attributes:
            disk_size: Radius of the disk-shaped structuring element
                used for morphological operations, calculated based on image dimensions.
            subtract_background: Background-subtracted version of `image` obtained
                after morphological operations with the disk-shaped structuring element.
            subtract_background2: Background-subtracted version of `image2` obtained
                after morphological operations with the disk-shaped structuring element,
                if `image2` is present."""
        logging.info("Generate background using the generate_subtract_background method of OneImageAnalysis class")
        self._get_all_color_spaces()
        if drift_corrected:
            # self.adjust_to_drift_correction(c_space_dict['logical'])
            self.check_if_image_border_attest_drift_correction()
        self.convert_and_segment(c_space_dict, rolling_window_segmentation=None, allowed_window=self.drift_mask_coord)
        disk_size = np.max((3, int(np.floor(np.sqrt(np.min(self.bgr.shape[:2])) / 2))))
        disk = create_ellipse(disk_size, disk_size).astype(np.uint8)
        self.subtract_background = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, disk)
        if self.image2 is not None:
            self.subtract_background2 = cv2.morphologyEx(self.image2, cv2.MORPH_OPEN, disk)

    def check_if_image_border_attest_drift_correction(self) -> bool:
        """
        Check if the given binary image requires border attenuation and drift correction.

        In order to determine the need for border attenuation or drift correction, this function
        evaluates the borders of a binary image. If any two opposite borders are fully black,
        it assumes that there is an issue requiring correction.

        Returns:
            bool: True if border attenuation or drift correction is required, False otherwise.

        """
        t = np.all(self.binary_image[0, :])
        b = np.all(self.binary_image[-1, :])
        l = np.all(self.binary_image[:, 0])
        r = np.all(self.binary_image[:, -1])
        self.drift_mask_coord = None
        if (t and b) or (t and r) or (t and l) or (t and r) or (b and l) or (b and r) or (l and r):
            cc_nb, shapes = cv2.connectedComponents(self.binary_image)
            if cc_nb > 1:
                if cc_nb == 2:
                    drift_mask_coord = np.nonzero(1 - self.binary_image)
                else:
                    back = np.unique(np.concatenate((shapes[0, :], shapes[-1, :],  shapes[:, 0], shapes[:, -1]), axis=0))
                    drift_mask_coord = np.nonzero(np.logical_or(1 - self.binary_image, 1 - np.isin(shapes, back[back != 0])))
                drift_mask_coord = (np.min(drift_mask_coord[0]), np.max(drift_mask_coord[0]) + 1,
                                    np.min(drift_mask_coord[1]), np.max(drift_mask_coord[1]) + 1)
                self.drift_mask_coord = drift_mask_coord
                return True
            else:
                return False
        else:
            return False

    def adjust_to_drift_correction(self, logical: str):
        """
        Adjust the image and binary image to correct for drift.

        This method applies a drift correction by dilating the binary image, calculating
        the mean value of the drifted region and applying it back to the image. After this,
        it applies Otsu's thresholding method to determine a new binary image and adjusts
        the second image if present. The logical operation specified is then applied to the
        binary images.

        Args:
            logical (str): Logical operation ('Or', 'And', 'Xor') to apply to the binary
                images."""
        if not self.drift_correction_already_adjusted:
            self.drift_correction_already_adjusted = True

            mask = cv2.dilate(self.binary_image, kernel=cross_33)
            mask -= self.binary_image
            mask = np.nonzero(mask)
            drift_correction = np.mean(self.image[mask[0], mask[1]])
            self.image[np.nonzero(self.binary_image)] = drift_correction
            threshold = get_otsu_threshold(self.image)
            binary = (self.image > threshold)
            self.binary_image = binary.astype(np.uint8)

            if self.image2 is not None:
                drift_correction2 = np.mean(self.image2[mask[0], mask[1]])
                self.image2[np.nonzero(self.binary_image)] = drift_correction2
                threshold = get_otsu_threshold(self.image2)
                binary1 = (self.image2 > threshold)
                binary2 = np.logical_not(binary1)
                if binary1.sum() < binary2.sum():
                    binary = binary1
                else:
                    binary = binary2
                while np.any(binary * self.binary_image2) and threshold > 1:
                    threshold -= 1
                    binary1 = (self.image2 > threshold)
                    binary2 = np.logical_not(binary1)
                    if binary1.sum() < binary2.sum():
                        binary = binary1
                    else:
                        binary = binary2
                self.binary_image2 = binary.astype(np.uint8)
                if logical == 'Or':
                    self.binary_image = np.logical_or(self.binary_image, self.binary_image2)
                elif logical == 'And':
                    self.binary_image = np.logical_and(self.binary_image, self.binary_image2)
                elif logical == 'Xor':
                    self.binary_image = np.logical_xor(self.binary_image, self.binary_image2)
                self.binary_image = self.binary_image.astype(np.uint8)

    def find_first_im_csc(self, sample_number: int=None, several_blob_per_arena:bool=True,  spot_shape: str=None,
                          spot_size=None, kmeans_clust_nb: int=None, biomask: NDArray[np.uint8]=None,
                          backmask: NDArray[np.uint8]=None, color_space_dictionaries: TList=None, basic: bool=True):
        """
        Prepare color space lists, dictionaries and matrices.

        Args:
            sample_number: An integer representing the sample number. Defaults to None.
            several_blob_per_arena: A boolean indicating whether there are several blobs per arena. Defaults to True.
            spot_shape: A string representing the shape of the spot. Defaults to None.
            spot_size: An integer representing the size of the spot. Defaults to None.
            kmeans_clust_nb: An integer representing the number of clusters for K-means. Defaults to None.
            biomask: A 2D numpy array of type np.uint8 representing the bio mask. Defaults to None.
            backmask: A 2D numpy array of type np.uint8 representing the background mask. Defaults to None.
            color_space_dictionaries: A list of dictionaries containing color space information. Defaults to None.
            basic: A boolean indicating whether to process the data basic. Defaults to True.

        Note:
            This method processes the input data to find the first image that matches certain criteria, using various color spaces and masks.

        """
        logging.info(f"Start automatic detection of the first image")
        self.im_combinations = []
        self.saved_images_list = TList()
        self.converted_images_list = TList()
        self.saved_color_space_list = list()
        self.saved_csc_nb = 0

        if self.image.any():
            self._get_all_color_spaces()
            if color_space_dictionaries is None:
                if basic:
                    colorspace_list = ["bgr", "lab", "hsv", "luv", "hls", "yuv"]
                else:
                    colorspace_list = ["bgr"]
                color_space_dictionaries = TList()
                for i, c_space in enumerate(colorspace_list):
                    for i in np.arange(3):
                        channels = np.array((0, 0, 0), dtype=np.int8)
                        channels[i] = 1
                        csc_dict = TDict()
                        csc_dict[c_space] = channels
                        color_space_dictionaries.append(csc_dict)

            self.combination_features = np.zeros((len(color_space_dictionaries) + 50, 11), dtype=np.uint32)
            unaltered_cc_nb, cc_nb, area, width_std, height_std, area_std, biosum, backsum = 3, 4, 5, 6, 7, 8, 9, 10
            self.save_combination_thread = SaveCombinationThread(self)
            get_one_channel_result = True
            combine_channels = False
            logging.info(f"Try detection with each available color space channel, one by one.")
            for csc_dict in color_space_dictionaries:
                list_args = [self, get_one_channel_result, combine_channels, csc_dict, several_blob_per_arena,
                             sample_number, spot_size, spot_shape, kmeans_clust_nb, biomask, backmask, None]
                ProcessFirstImage(list_args)

            if sample_number is not None and basic:
                # Try to add csc together
                possibilities = []
                if self.saved_csc_nb > 6:
                    different_color_spaces = np.unique(self.saved_color_space_list)
                    for color_space in different_color_spaces:
                        csc_idx = np.nonzero(np.isin(self.saved_color_space_list, color_space))[0]
                        possibilities.append(csc_idx[0] + np.argmin(self.combination_features[csc_idx, area_std]))
                    if len(possibilities) <= 6:
                        remaining_possibilities = np.arange(len(self.saved_color_space_list))
                        remaining_possibilities = remaining_possibilities[np.logical_not(np.isin(remaining_possibilities, possibilities))]
                        while len(possibilities) <= 6:
                            new_possibility = np.argmin(self.combination_features[remaining_possibilities, area_std])
                            possibilities.append(new_possibility)
                            remaining_possibilities = remaining_possibilities[remaining_possibilities != new_possibility]


                pool = mp.ThreadPool(processes=os.cpu_count() - 1)
                get_one_channel_result = False
                combine_channels = True
                list_args = [[self, get_one_channel_result, combine_channels, i, several_blob_per_arena, sample_number,
                              spot_size, spot_shape, kmeans_clust_nb, biomask, backmask, possibilities] for i in possibilities]
                for process_i in pool.imap_unordered(ProcessFirstImage, list_args):
                    pass

            # Get the most and the least covered images and the 2 best biomask and backmask scores
            # To try combinations of those
            if self.saved_csc_nb <= 1:
                csc_dict = {'bgr': np.array((1, 1, 1))}
                list_args = [self, False, False, csc_dict, several_blob_per_arena,
                             sample_number, spot_size, spot_shape, kmeans_clust_nb, biomask, backmask, None]
                process_i = ProcessFirstImage(list_args)
                process_i.image = self.bgr.mean(axis=-1)
                process_i.binary_image = otsu_thresholding(process_i.image)
                process_i.csc_dict = csc_dict
                process_i.total_area = process_i.binary_image.sum()
                process_i.process_binary_image()
                process_i.unaltered_concomp_nb, shapes = cv2.connectedComponents(process_i.validated_shapes)
                self.save_combination_features(process_i)
                self.combination_features = self.combination_features[:self.saved_csc_nb, :]
                fit = np.array([True])
            else:
                coverage = np.argsort(self.combination_features[:self.saved_csc_nb, area])
                most1 = coverage[-1]; most2 = coverage[-2]
                least1 = coverage[0]; least2 = coverage[1]
                if biomask is not None:
                    bio_sort = np.argsort(self.combination_features[:self.saved_csc_nb, biosum])
                    bio1 = bio_sort[-1]; bio2 = bio_sort[-2]
                if backmask is not None:
                    back_sort = np.argsort(self.combination_features[:self.saved_csc_nb, backsum])
                    back1 = back_sort[-1]; back2 = back_sort[-2]

                # Try a logical And between the most covered images
                # Should only need one instanciation
                process_i = ProcessFirstImage(
                    [self, False, False, None, several_blob_per_arena, sample_number, spot_size, spot_shape, kmeans_clust_nb, biomask, backmask, None])
                process_i.binary_image = np.logical_and(self.saved_images_list[most1], self.saved_images_list[most2]).astype(np.uint8)
                process_i.image = self.converted_images_list[most1]
                process_i.process_binary_image()
                process_i.csc_dict = {list(self.saved_color_space_list[most1].keys())[0]: self.combination_features[most1, :3],
                            "logical": "And",
                            list(self.saved_color_space_list[most2].keys())[0] + "2": self.combination_features[most2, :3]}
                process_i.unaltered_concomp_nb = np.min(self.combination_features[(most1, most2), unaltered_cc_nb])
                process_i.total_area = process_i.binary_image.sum()
                self.save_combination_features(process_i)
                process_i.image = self.converted_images_list[least1]
                process_i.binary_image = np.logical_or(self.saved_images_list[least1], self.saved_images_list[least2]).astype(np.uint8)
                process_i.process_binary_image()
                process_i.csc_dict = {list(self.saved_color_space_list[least1].keys())[0]: self.combination_features[least1, :3],
                            "logical": "Or",
                            list(self.saved_color_space_list[least2].keys())[0] + "2": self.combination_features[least2, :3]}
                process_i.unaltered_concomp_nb = np.max(self.combination_features[(least1, least2), unaltered_cc_nb])
                process_i.total_area = process_i.binary_image.sum()
                self.save_combination_features(process_i)

                # self.save_combination_features(csc_dict, unaltered_concomp_nb, self.binary_image.sum(), biomask, backmask)

                # If most images are very low in biosum or backsum, try to mix them together to improve that score
                # Do a logical And between the two best biomasks
                if biomask is not None:
                    if not np.all(np.isin((bio1, bio2), (most1, most2))):
                        process_i.image = self.converted_images_list[bio1]
                        process_i.binary_image = np.logical_and(self.saved_images_list[bio1], self.saved_images_list[bio2]).astype(
                        np.uint8)
                        process_i.process_binary_image()
                        process_i.csc_dict = {list(self.saved_color_space_list[bio1].keys())[0]: self.combination_features[bio1, :3],
                                    "logical": "And",
                                    list(self.saved_color_space_list[bio2].keys())[0] + "2": self.combination_features[bio2,:3]}
                        process_i.unaltered_concomp_nb = np.min(self.combination_features[(bio1, bio2), unaltered_cc_nb])
                        process_i.total_area = process_i.binary_image.sum()

                        self.save_combination_features(process_i)

                # Do a logical And between the two best backmask
                if backmask is not None:
                    if not np.all(np.isin((back1, back2), (most1, most2))):
                        process_i.image = self.converted_images_list[back1]
                        process_i.binary_image = np.logical_and(self.saved_images_list[back1], self.saved_images_list[back2]).astype(
                        np.uint8)
                        process_i.process_binary_image()
                        process_i.csc_dict = {list(self.saved_color_space_list[back1].keys())[0]: self.combination_features[back1, :3],
                                    "logical": "And",
                                    list(self.saved_color_space_list[back2].keys())[0] + "2": self.combination_features[back2,:3]}
                        process_i.unaltered_concomp_nb = np.min(self.combination_features[(back1, back2), unaltered_cc_nb])
                        process_i.total_area = process_i.binary_image.sum()
                        self.save_combination_features(process_i)
                # Do a logical Or between the best biomask and the best backmask
                if biomask is not None and backmask is not None:
                    if not np.all(np.isin((bio1, back1), (least1, least2))):
                        process_i.image = self.converted_images_list[bio1]
                        process_i.binary_image = np.logical_and(self.saved_images_list[bio1], self.saved_images_list[back1]).astype(
                            np.uint8)
                        process_i.process_binary_image()
                        process_i.csc_dict = {list(self.saved_color_space_list[bio1].keys())[0]: self.combination_features[bio1, :3],
                                    "logical": "Or",
                                    list(self.saved_color_space_list[back1].keys())[0] + "2": self.combination_features[back1, :3]}
                        process_i.unaltered_concomp_nb = np.max(self.combination_features[(bio1, back1), unaltered_cc_nb])
                        # self.save_combination_features(csc_dict, unaltered_concomp_nb, self.binary_image.sum(), biomask,
                        #                                backmask)
                        process_i.total_area = self.binary_image.sum()
                        self.save_combination_features(process_i)

                if self.save_combination_thread.is_alive():
                    self.save_combination_thread.join()
                self.combination_features = self.combination_features[:self.saved_csc_nb, :]
                # Only keep the row that filled conditions
                # Save all combinations if they fulfill the following conditions:
                #   - Their conncomp number is lower than 3 times the smaller conncomp number.
                #   - OR The minimal area variations
                #   - OR The minimal width variations
                #   - OR The minimal height variations
                #   - AND/OR their segmentation fits with biomask and backmask
                width_std_fit = self.combination_features[:, width_std] == np.min(self.combination_features[:, width_std])
                height_std_fit = self.combination_features[:, height_std] == np.min(self.combination_features[:, height_std])
                area_std_fit = self.combination_features[:, area_std] < np.min(self.combination_features[:, area_std]) * 10
                fit = np.logical_or(np.logical_or(width_std_fit, height_std_fit), area_std_fit)
                biomask_fit = np.ones(self.saved_csc_nb, dtype=bool)
                backmask_fit = np.ones(self.saved_csc_nb, dtype=bool)
                if biomask is not None or backmask is not None:
                    if biomask is not None:
                        biomask_fit = self.combination_features[:, biosum] > 0.9 * len(biomask[0])
                    if backmask is not None:
                        backmask_fit = self.combination_features[:, backsum] > 0.9 * len(backmask[0])
                    # First test a logical OR between the precedent options and the mask fits.
                    fit = np.logical_or(fit, np.logical_and(biomask_fit, backmask_fit))
                    # If this is not stringent enough, use a logical AND and increase progressively the proportion of pixels that
                    # must match the biomask and the backmask
                    if np.sum(fit) > 5:
                        to_add = 0
                        while np.sum(fit) > 5 and to_add <= 0.25:
                            if biomask is not None:
                                biomask_fit = self.combination_features[:, biosum] > (0.75 + to_add) * len(biomask[0])
                            if backmask is not None:
                                backmask_fit = self.combination_features[:, backsum] > (0.75 + to_add) * len(backmask[0])
                            test_fit = np.logical_and(fit, np.logical_and(biomask_fit, backmask_fit))
                            if np.sum(test_fit) != 0:
                                fit = test_fit
                            to_add += 0.05
            # If saved_csc_nb is too low, try bool operators to mix them together to fill holes for instance
            # Order the table according to the number of shapes that have been removed by filters
            # cc_efficiency_order = np.argsort(self.combination_features[:, unaltered_cc_nb] - self.combination_features[:, cc_nb])
            cc_efficiency_order = np.argsort(self.combination_features[:, area_std])
            # Save and return a dictionnary containing the selected color space combinations
            # and their corresponding binary images

            for saved_csc in cc_efficiency_order:
                if fit[saved_csc]:
                    self.im_combinations.append({})
                    # self.im_combinations.append({})
                    # self.im_combinations[len(self.im_combinations) - 1]["csc"] = self.saved_color_space_list[saved_csc]
                    self.im_combinations[len(self.im_combinations) - 1]["csc"] = {}
                    self.im_combinations[len(self.im_combinations) - 1]["csc"]['logical'] = 'None'
                    for k, v in self.saved_color_space_list[saved_csc].items():
                        self.im_combinations[len(self.im_combinations) - 1]["csc"][k] = v
                    if backmask is not None:
                        shape_number, shapes = cv2.connectedComponents(self.saved_images_list[saved_csc], connectivity=8)
                        if np.any(shapes[backmask]):
                            shapes[np.isin(shapes, np.unique(shapes[backmask]))] = 0
                            self.saved_images_list[saved_csc] = (shapes > 0).astype(np.uint8)
                    if biomask is not None:
                        self.saved_images_list[saved_csc][biomask] = 1
                    if backmask is not None or biomask is not None:
                        self.combination_features[saved_csc, cc_nb], shapes = cv2.connectedComponents(self.saved_images_list[saved_csc], connectivity=8)
                        self.combination_features[saved_csc, cc_nb] -= 1
                    self.im_combinations[len(self.im_combinations) - 1]["binary_image"] = self.saved_images_list[saved_csc]
                    self.im_combinations[len(self.im_combinations) - 1]["shape_number"] = self.combination_features[saved_csc, cc_nb]
                    self.im_combinations[len(self.im_combinations) - 1]["converted_image"] = self.converted_images_list[saved_csc]

            self.saved_color_space_list = []
            self.saved_images_list = None
            self.converted_images_list = None
            self.combination_features = None

    def save_combination_features(self, process_i: object):
        """
        Saves the combination features of a given processed image.

        Args:
            process_i (object): The processed image object containing various attributes
                such as validated_shapes, image, csc_dict, unaltered_concomp_nb,
                shape_number, total_area, stats, biomask, and backmask.

            Attributes:
                processed image object
                    validated_shapes (array-like): The validated shapes of the processed image.
                    image (array-like): The image data.
                    csc_dict (dict): Color space conversion dictionary
        """
        if process_i.validated_shapes.any():
            self.saved_images_list.append(process_i.validated_shapes)
            self.converted_images_list.append(np.round(process_i.image).astype(np.uint8))
            self.saved_color_space_list.append(process_i.csc_dict)
            self.combination_features[self.saved_csc_nb, :3] = list(process_i.csc_dict.values())[0]
            self.combination_features[
                self.saved_csc_nb, 3] = process_i.unaltered_concomp_nb - 1  # unaltered_cc_nb
            self.combination_features[self.saved_csc_nb, 4] = process_i.shape_number  # cc_nb
            self.combination_features[self.saved_csc_nb, 5] = process_i.total_area  # area
            self.combination_features[self.saved_csc_nb, 6] = np.std(process_i.stats[1:, 2])  # width_std
            self.combination_features[self.saved_csc_nb, 7] = np.std(process_i.stats[1:, 3])  # height_std
            self.combination_features[self.saved_csc_nb, 8] = np.std(process_i.stats[1:, 4])  # area_std
            if process_i.biomask is not None:
                self.combination_features[self.saved_csc_nb, 9] = np.sum(
                    process_i.validated_shapes[process_i.biomask[0], process_i.biomask[1]])
            if process_i.backmask is not None:
                self.combination_features[self.saved_csc_nb, 10] = np.sum(
                    (1 - process_i.validated_shapes)[process_i.backmask[0], process_i.backmask[1]])
            self.saved_csc_nb += 1

    def update_current_images(self, current_combination_id: int):
        """
        Update the current images based on a given combination ID.

        This method updates two attributes of the instance: `image` and
        `validated_shapes`. The `image` attribute is set to the value of the key
        "converted_image" from a dictionary in `im_combinations` which is
        indexed by the provided `current_combination_id`. Similarly, the
        `validated_shapes` attribute is set to the value of the key "binary_image"
        from the same dictionary.

        Args:
            current_combination_id (int): The ID of the combination whose
                images should be set as the current ones.

        """
        self.image = self.im_combinations[current_combination_id]["converted_image"]
        self.validated_shapes = self.im_combinations[current_combination_id]["binary_image"]

    def find_last_im_csc(self, concomp_nb: int, total_surfarea: int, max_shape_size: int, arenas_mask: NDArray=None,
                         ref_image: NDArray=None, subtract_background: NDArray=None, kmeans_clust_nb: int=None,
                         biomask: NDArray[np.uint8]=None, backmask: NDArray[np.uint8]=None,
                         color_space_dictionaries: dict=None, basic: bool=True):
        """
        Find the last image color space configurations that meets given criteria.

        Args:
            concomp_nb (int): A tuple of two integers representing the minimum and maximum number of connected components.
            total_surfarea (int): The total surface area required for the image.
            max_shape_size (int): The maximum shape size allowed in the image.
            arenas_mask (NDArray, optional): A numpy array representing areas inside the field of interest.
            ref_image (NDArray, optional): A reference image for comparison.
            subtract_background (NDArray, optional): A numpy array representing the background to be subtracted.
            kmeans_clust_nb (int, optional): The number of clusters for k-means clustering.
            biomask (NDArray[np.uint8], optional): A binary mask for biological structures.
            backmask (NDArray[np.uint8], optional): A binary mask for background areas.
            color_space_dictionaries (dict, optional): Dictionaries of color space configurations.
            basic (bool, optional): A flag indicating whether to process colorspaces basic.

        """
        logging.info(f"Start automatic detection of the last image")
        self.im_combinations = []
        self.saved_images_list = TList()
        self.converted_images_list = TList()
        self.saved_color_space_list = list()
        self.saved_csc_nb = 0

        if self.image.any():
            if arenas_mask is None:
                arenas_mask = np.ones_like(self.binary_image)
            out_of_arenas = 1 - arenas_mask
            self._get_all_color_spaces()
            if color_space_dictionaries is None:
                if basic:
                    colorspace_list = TList(("bgr", "lab", "hsv", "luv", "hls", "yuv"))
                else:
                    colorspace_list = TList(("lab", "hsv"))
                color_space_dictionaries = TList()
                channels = np.array((1, 1, 1), dtype=np.int8)
                csc_dict = TDict()
                csc_dict["bgr"] = channels
                color_space_dictionaries.append(csc_dict)
                for i, c_space in enumerate(colorspace_list):
                    for i in np.arange(3):
                        channels = np.array((0, 0, 0), dtype=np.int8)
                        channels[i] = 1
                        csc_dict = TDict()
                        csc_dict[c_space] = channels
                        color_space_dictionaries.append(csc_dict)
            if ref_image is not None:
                ref_image = cv2.dilate(ref_image, cross_33)
            else:
                ref_image = np.ones(self.bgr.shape[:2], dtype=np.uint8)
            out_of_arenas_threshold = 0.01 * out_of_arenas.sum()
            self.combination_features = np.zeros((len(color_space_dictionaries) + 50, 10), dtype=np.uint32)
            cc_nb_idx, area_idx, out_of_arenas_idx, in_arena_idx, surf_in_common_idx, biosum_idx, backsum_idx = 3, 4, 5, 6, 7, 8, 9
            self.save_combination_thread = SaveCombinationThread(self)

            # Start with a PCA:
            pca_dict = TDict()
            pca_dict['PCA'] = np.array([1, 1, 1], dtype=np.int8)
            self.image, explained_variance_ratio, first_pc_vector = extract_first_pc(self.bgr)
            self.binary_image = otsu_thresholding(self.image)
            nb, shapes = cv2.connectedComponents(self.binary_image)
            nb -= 1
            surf = self.binary_image.sum()
            outside_pixels = np.sum(self.binary_image * out_of_arenas)
            inside_pixels = np.sum(self.binary_image * arenas_mask)
            in_common = np.sum(ref_image * self.binary_image)
            self.converted_images_list.append(self.image)
            self.saved_images_list.append(self.binary_image)
            self.saved_color_space_list.append(pca_dict)
            self.combination_features[self.saved_csc_nb, :3] = list(pca_dict.values())[0]
            self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
            self.combination_features[self.saved_csc_nb, area_idx] = surf
            self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
            self.combination_features[self.saved_csc_nb, in_arena_idx] = inside_pixels
            self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
            if biomask is not None:
                self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                    self.binary_image[biomask[0], biomask[1]])
            if backmask is not None:
                self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                    (1 - self.binary_image)[backmask[0], backmask[1]])
            self.saved_csc_nb += 1

            potentials = TDict()
            # One channel processing
            for csc_dict in color_space_dictionaries:
                self.image = combine_color_spaces(csc_dict, self.all_c_spaces, subtract_background)
                if kmeans_clust_nb is not None and (biomask is not None or backmask is not None):
                    self.binary_image, self.binary_image2, self.bio_label, self.bio_label2 = kmeans(self.image, self.image2, kmeans_clust_nb, biomask, backmask)
                else:
                    self.binary_image = otsu_thresholding(self.image)
                surf = np.sum(self.binary_image)
                if surf < total_surfarea:
                    nb, shapes = cv2.connectedComponents(self.binary_image)
                    outside_pixels = np.sum(self.binary_image * out_of_arenas)
                    inside_pixels = np.sum(self.binary_image * arenas_mask)
                    if outside_pixels < inside_pixels:
                        if (nb > concomp_nb[0] - 1) and (nb < concomp_nb[1]):
                            in_common = np.sum(ref_image * self.binary_image)
                            if in_common > 0:
                                nb, shapes, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image)
                                nb -= 1
                                if np.all(np.sort(stats[:, 4])[:-1] < max_shape_size):
                                    c_space = list(csc_dict.keys())[0]
                                    self.converted_images_list.append(self.image)
                                    self.saved_images_list.append(self.binary_image)
                                    self.saved_color_space_list.append(csc_dict)
                                    self.combination_features[self.saved_csc_nb, :3] = csc_dict[c_space]
                                    self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
                                    self.combination_features[self.saved_csc_nb, area_idx] = surf
                                    self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
                                    self.combination_features[self.saved_csc_nb, in_arena_idx] = inside_pixels
                                    self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
                                    if biomask is not None:
                                        self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                                            self.binary_image[biomask[0], biomask[1]])
                                    if backmask is not None:
                                        self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                                            (1 - self.binary_image)[backmask[0], backmask[1]])
                                    if np.isin(c_space, list(potentials.keys())):
                                        potentials[c_space] += csc_dict[c_space]
                                    else:
                                        potentials[c_space] = csc_dict[c_space]
                                    self.saved_csc_nb += 1
            if len(potentials) > 0:
                # All combination processing

                # Add a combination of all selected channels :
                self.saved_color_space_list.append(potentials)
                self.image = combine_color_spaces(potentials, self.all_c_spaces, subtract_background)
                if kmeans_clust_nb is not None and (biomask is not None or backmask is not None):
                    self.binary_image, self.binary_image2, self.bio_label, self.bio_label2 = kmeans(self.image, kmeans_clust_nb=kmeans_clust_nb, biomask=biomask, backmask=backmask)
                else:
                    self.binary_image = otsu_thresholding(self.image)
                surf = self.binary_image.sum()
                nb, shapes = cv2.connectedComponents(self.binary_image)
                nb -= 1
                outside_pixels = np.sum(self.binary_image * out_of_arenas)
                inside_pixels = np.sum(self.binary_image * arenas_mask)
                in_common = np.sum(ref_image * self.binary_image)
                self.converted_images_list.append(self.image)
                self.saved_images_list.append(self.binary_image)
                self.saved_color_space_list.append(potentials)
                self.combination_features[self.saved_csc_nb, :3] = list(potentials.values())[0]
                self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
                self.combination_features[self.saved_csc_nb, area_idx] = surf
                self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
                self.combination_features[self.saved_csc_nb, in_arena_idx] = inside_pixels
                self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
                if biomask is not None:
                    self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                        self.binary_image[biomask[0], biomask[1]])
                if backmask is not None:
                    self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                        (1 - self.binary_image)[backmask[0], backmask[1]])
                self.saved_csc_nb += 1
            # All combination processing
            # Try to remove color space one by one
            i = 0
            original_length = len(potentials)
            while np.logical_and(len(potentials) > 1, i < original_length // 2):
                color_space_to_remove = TList()
                # The while loop until one col space remains or the removal of one implies a strong enough area change
                previous_c_space = list(potentials.keys())[-1]
                for c_space in potentials.keys():
                    try_potentials = potentials.copy()
                    try_potentials.pop(c_space)
                    if i > 0:
                        try_potentials.pop(previous_c_space)
                    self.image = combine_color_spaces(try_potentials, self.all_c_spaces, subtract_background)
                    if kmeans_clust_nb is not None and (biomask is not None or backmask is not None):
                        self.binary_image, self.binary_image2, self.bio_label, self.bio_label2  = kmeans(self.image, kmeans_clust_nb=kmeans_clust_nb, biomask=biomask, backmask=backmask)
                    else:
                        self.binary_image = otsu_thresholding(self.image)
                    surf = np.sum(self.binary_image)
                    if surf < total_surfarea:
                        nb, shapes = cv2.connectedComponents(self.binary_image)
                        outside_pixels = np.sum(self.binary_image * out_of_arenas)
                        inside_pixels = np.sum(self.binary_image * arenas_mask)
                        if outside_pixels < inside_pixels:
                            if (nb > concomp_nb[0] - 1) and (nb < concomp_nb[1]):
                                in_common = np.sum(ref_image * self.binary_image)
                                if in_common > 0:
                                    nb, shapes, stats, centroids = cv2.connectedComponentsWithStats(self.binary_image)
                                    nb -= 1
                                    if np.all(np.sort(stats[:, 4])[:-1] < max_shape_size):
                                        # If a color space remove fits in the requirements, we store its values
                                        self.converted_images_list.append(self.image)
                                        self.saved_images_list.append(self.binary_image)
                                        self.saved_color_space_list.append(try_potentials)
                                        self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
                                        self.combination_features[self.saved_csc_nb, area_idx] = surf
                                        self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
                                        self.combination_features[self.saved_csc_nb, in_arena_idx] = inside_pixels
                                        self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
                                        if biomask is not None:
                                            self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                                                self.binary_image[biomask[0], biomask[1]])
                                        if backmask is not None:
                                            self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                                                (1 - self.binary_image)[backmask[0], backmask[1]])
                                        self.saved_csc_nb += 1
                                        color_space_to_remove.append(c_space)
                                        if i > 0:
                                            color_space_to_remove.append(previous_c_space)
                    # If it does not (if it did not pass every "if" layers), we definitely remove that color space
                    previous_c_space = c_space
                color_space_to_remove = np.unique(color_space_to_remove)
                for remove_col_space in color_space_to_remove:
                    potentials.pop(remove_col_space)
                i += 1
            if np.logical_and(len(potentials) > 0, i > 1):
                self.converted_images_list.append(self.image)
                self.saved_images_list.append(self.binary_image)
                self.saved_color_space_list.append(potentials)
                self.combination_features[self.saved_csc_nb, :3] = list(potentials.values())[0]
                self.combination_features[self.saved_csc_nb, cc_nb_idx] = nb
                self.combination_features[self.saved_csc_nb, area_idx] = surf
                self.combination_features[self.saved_csc_nb, out_of_arenas_idx] = outside_pixels
                self.combination_features[self.saved_csc_nb, in_arena_idx] = inside_pixels
                self.combination_features[self.saved_csc_nb, surf_in_common_idx] = in_common
                if biomask is not None:
                    self.combination_features[self.saved_csc_nb, biosum_idx] = np.sum(
                        self.binary_image[biomask[0], biomask[1]])
                if backmask is not None:
                    self.combination_features[self.saved_csc_nb, backsum_idx] = np.sum(
                        (1 - self.binary_image)[backmask[0], backmask[1]])
                self.saved_csc_nb += 1

            self.combination_features = self.combination_features[:self.saved_csc_nb, :]
            # Among all potentials, select the best one, according to criterion decreasing in importance
            cc_efficiency_order = np.argsort(self.combination_features[:, surf_in_common_idx] + self.combination_features[:, in_arena_idx] - self.combination_features[:, out_of_arenas_idx])

            # Save and return a dictionnary containing the selected color space combinations
            # and their corresponding binary images
            self.im_combinations = []
            for saved_csc in cc_efficiency_order:
                if len(self.saved_color_space_list[saved_csc]) > 0:
                    self.im_combinations.append({})
                    self.im_combinations[len(self.im_combinations) - 1]["csc"] = {}
                    self.im_combinations[len(self.im_combinations) - 1]["csc"]['logical'] = 'None'
                    for k, v in self.saved_color_space_list[saved_csc].items():
                        self.im_combinations[len(self.im_combinations) - 1]["csc"][k] = v
                    self.im_combinations[len(self.im_combinations) - 1]["binary_image"] = self.saved_images_list[saved_csc]
                    self.im_combinations[len(self.im_combinations) - 1]["converted_image"] = np.round(self.converted_images_list[
                        saved_csc]).astype(np.uint8)
            self.saved_color_space_list = []
            self.saved_images_list = None
            self.converted_images_list = None
            self.combination_features = None

    def network_detection(self, arenas_mask: NDArray=None, pseudopod_min_size: int=50, csc_dict: dict=None, biomask=None, backmask=None):
        """
        Network Detection Function

        Perform network detection and pseudopod analysis on an image.

        Parameters
        ----------
        arenas_mask : NDArray, optional
            The mask indicating the arena regions in the image.
        pseudopod_min_size : int, optional
            The minimum size for pseudopods to be detected.
        csc_dict : dict, optional
            A dictionary containing color space conversion parameters. If None,
            defaults to {'bgr': np.array((1, 1, 1), np.int8), 'logical': 'None'}
        biomask : NDArray, optional
            The mask for biological objects in the image.
        backmask : NDArray, optional
            The background mask.

        Notes
        -----
        This function modifies the object's state by setting `self.im_combinations`
        with the results of network detection and pseudopod analysis.
        """
        logging.info(f"Start automatic detection of network(s) in the last image")
        if len(self.bgr.shape) == 3:
            if csc_dict is None:
                csc_dict = {'bgr': np.array((1, 1, 1), np.int8), 'logical': 'None'}
            self._get_all_color_spaces()
            # csc_dict = translate_dict(csc_dict)
            # self.image = combine_color_spaces(csc_dict, self.all_c_spaces)
            first_dict, second_dict, c_spaces = split_dict(csc_dict)
            self.image, _, _, first_pc_vector = generate_color_space_combination(self.bgr, c_spaces, first_dict, second_dict, all_c_spaces=self.all_c_spaces)
            # if first_pc_vector is not None:
            #     csc_dict = {"bgr": first_pc_vector, "logical": 'None'}
        greyscale = self.image
        NetDet = NetworkDetection(greyscale, possibly_filled_pixels=arenas_mask)
        NetDet.get_best_network_detection_method()
        lighter_background = NetDet.greyscale_image[arenas_mask > 0].mean() < NetDet.greyscale_image[arenas_mask== 0].mean()
        NetDet.detect_pseudopods(lighter_background, pseudopod_min_size=pseudopod_min_size, only_one_connected_component=False)
        NetDet.merge_network_with_pseudopods()
        cc_efficiency_order = np.argsort(NetDet.quality_metrics)
        self.im_combinations = []
        for _i in cc_efficiency_order:
            res_i = NetDet.all_results[_i]
            self.im_combinations.append({})
            self.im_combinations[len(self.im_combinations) - 1]["csc"] = csc_dict
            self.im_combinations[len(self.im_combinations) - 1]["converted_image"] = bracket_to_uint8_image_contrast(res_i['filtered'])
            self.im_combinations[len(self.im_combinations) - 1]["binary_image"] = res_i['binary']
            self.im_combinations[len(self.im_combinations) - 1]['filter_spec']= {'filter1_type': res_i['filter'], 'filter1_param': [np.min(res_i['sigmas']), np.max(res_i['sigmas'])], 'filter2_type': "", 'filter2_param': [1., 1.]}
            self.im_combinations[len(self.im_combinations) - 1]['rolling_window']= res_i['rolling_window']

    def get_crop_coordinates(self):
        """
        Get the crop coordinates for image processing.

        This function projects the image on both x and y axes to detect rows
        and columns of arenas, calculates the boundaries for cropping,
        and determines if the arenas are zigzagged.-

        """
        logging.info("Project the image on the y axis to detect rows of arenas")
        self.y_boundaries, y_max_sum = self.projection_to_get_peaks_boundaries(axis=1)
        logging.info("Project the image on the x axis to detect columns of arenas")
        self.x_boundaries, x_max_sum = self.projection_to_get_peaks_boundaries(axis=0)
        logging.info("Get crop coordinates using the get_crop_coordinates method of OneImageAnalysis class")
        row_number = len(np.nonzero(self.y_boundaries)[0]) // 2
        col_number = len(np.nonzero(self.x_boundaries)[0]) // 2
        are_zigzag = None
        if col_number > 0 and row_number > 0:
            if (x_max_sum / col_number) * 2 < (y_max_sum / row_number):
                are_zigzag = "columns"
            elif (x_max_sum / col_number) > (y_max_sum / row_number) * 2:
                are_zigzag = "rows"
        # here automatically determine if are zigzag
        x_boundary_number = (self.x_boundaries == 1).sum()
        if x_boundary_number > 1:
            if x_boundary_number < 4:
                x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0]))) // 2
            else:
                if are_zigzag == "columns":
                    x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0][::2]))) // 2
                else:
                    x_interval = np.absolute(np.max(np.diff(np.where(self.x_boundaries == 1)[0]))) // 2
            cx_min = np.where(self.x_boundaries == - 1)[0][0] - x_interval.astype(int)
            cx_max = np.where(self.x_boundaries == 1)[0][col_number - 1] + x_interval.astype(int)
            if cx_min < 0: cx_min = 0
            if cx_max > len(self.x_boundaries): cx_max = len(self.x_boundaries) - 1
        else:
            cx_min = 0
            cx_max = len(self.x_boundaries)# - 1

        y_boundary_number = (self.y_boundaries == 1).sum()
        if y_boundary_number > 1:
            if y_boundary_number < 4:
                y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0]))) // 2
            else:
                if are_zigzag == "rows":
                    y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0][::2]))) // 2
                else:
                    y_interval = np.absolute(np.max(np.diff(np.where(self.y_boundaries == 1)[0]))) // 2
            cy_min = np.where(self.y_boundaries == - 1)[0][0] - y_interval.astype(int)
            cy_max = np.where(self.y_boundaries == 1)[0][row_number - 1] + y_interval.astype(int)
            if cy_min < 0: cy_min = 0
            if cy_max > len(self.y_boundaries): cy_max = len(self.y_boundaries) - 1
        else:
            cy_min = 0
            cy_max = len(self.y_boundaries)# - 1

        self.crop_coord = [cy_min, cy_max, cx_min, cx_max]

    def projection_to_get_peaks_boundaries(self, axis: int) -> Tuple[NDArray, int]:
        """

        Projection to get peaks' boundaries.

        Calculate the projection of an array along a specified axis and
        identify the boundaries of non-zero peaks.

        Args:
            axis: int,
                The axis along which to calculate the projection and identify
                peaks' boundaries.

        Returns:
            Tuple[NDArray, int]:
                A tuple containing two elements: an array representing the slopes
                of peaks' boundaries and an integer representing the maximum sum
                along the specified axis.

        """
        sums = np.sum(self.validated_shapes, axis)
        slopes = np.greater(sums, 0)
        slopes = np.append(0, np.diff(slopes))
        coord = np.nonzero(slopes)[0]
        for ci in np.arange(len(coord)):
            if ci % 2 == 0:
                slopes[coord[ci]] = - 1
        return slopes, sums.max()

    def automatically_crop(self, crop_coord):
        """
        Automatically crops the image using the given crop coordinates.

        This method crops various attributes of the image such as the main image,
        binary image, and color spaces. It also updates internal states related to
        cropping.

        Args:
            crop_coord (tuple): The coordinates for cropping in the format
                (start_y, end_y, start_x, end_x), representing the bounding box region
                to crop from the image.

        """
        if not self.cropped and crop_coord is not None:
            logging.info("Crop using the automatically_crop method of OneImageAnalysis class")
            self.cropped = True
            self.image = self.image[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            self.bgr = deepcopy(self.bgr[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...])
            self._get_all_color_spaces()
            if self.im_combinations is not None:
                for i in np.arange(len(self.im_combinations)):
                    self.im_combinations[i]["binary_image"] = self.im_combinations[i]["binary_image"][crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
                    self.im_combinations[i]["converted_image"] = self.im_combinations[i]["converted_image"][crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
            self.binary_image = self.binary_image[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]
            if self.greyscale is not None:
                self.greyscale = self.greyscale[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.greyscale2 is not None:
                self.greyscale2 = self.greyscale2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.image2 is not None:
                self.image2 = self.image2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.binary_image2 is not None:
                self.binary_image2 = self.binary_image2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.subtract_background is not None:
                self.subtract_background = self.subtract_background[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            if self.subtract_background2 is not None:
                self.subtract_background2 = self.subtract_background2[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            self.validated_shapes = self.validated_shapes[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3]]

            self.y_boundaries, y_max_sum = self.projection_to_get_peaks_boundaries(axis=1)
            self.x_boundaries, x_max_sum = self.projection_to_get_peaks_boundaries(axis=0)


