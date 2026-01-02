#!/usr/bin/env python3
"""Module implementing an image processing pipeline for segmentation and analysis.

This module defines functionality to process images using various color space
combinations, filters, and segmentation techniques like K-means clustering or
Otsu thresholding. It supports logical operations between images and iterative refinement of
color space combinations by adding/removing channels based on validation criteria.
The pipeline evaluates resulting binary masks for blob statistics and spatial relationships.

Classes
ProcessImage : Processes an image according to parameterized instructions, performing color space combination,
filtering, segmentation, and shape validation.

Notes
Relies on external functions from cellects.image_analysis module for core operations like
color space combination, filtering, PCA extraction, and segmentation.
"""
import threading
import logging
import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Tuple
from numba.typed import Dict, List
import pandas as pd
from cellects.image_analysis.image_segmentation import combine_color_spaces, apply_filter, kmeans, extract_first_pc, otsu_thresholding
from cellects.image_analysis.morphological_operations import shape_selection


class ProcessImage:
    """
    A class for processing an image according to a list of parameters.
    """
    def __init__(self, l):
        """
        Arguments:
            list : list
        """
        self.stats = None
        self.greyscale = None
        self.start_processing(l)

    def start_processing(self, l: list):
        """
        Begin processing tasks based on the given process type.

        Parameters
        ----------
        l : list
            A list containing parameters and instructions for processing.
            Expected structure:
                - First element: self.parent (object)
                - Second element: params (dict)
                - Third element: process type (str) ('one', 'PCA', 'add', 'subtract', or 'logical')
                - Fourth element (if applicable): additional parameters based on process type.

        Notes
        -----
        This function modifies instance attributes directly and performs various operations based on the `process` type.
        """
        self.parent = l[0]
        self.fact = pd.DataFrame(np.zeros((1, len(self.parent.factors)), dtype=np.uint32), columns=self.parent.factors)
        self.params = l[1]
        process = l[2]
        self.all_c_spaces = self.parent.all_c_spaces
        if process == 'one':
            self.csc_dict = l[3]
            self.combine_and_segment()
            self.evaluate_segmentation()
        elif process == 'PCA':
            self.pca_and_segment()
            self.evaluate_segmentation()
        elif process == 'add':
            self.try_adding_channels(l[3])
        elif process == 'subtract':
            self.try_subtracting_channels()
        elif process == 'logical':
            self.operation = l[3]
            self.apply_logical_operation()

    def combine_and_segment(self, csc_dict: Dict=None):
        """
        Combines color spaces and segments images into binary masks using either K-means clustering or Otsu's thresholding based on specified parameters.

        Parameters
        ----------
        csc_dict : dict or None
            Optional dictionary mapping color space abbreviations (e.g., 'bgr', 'hsv') to their relative contribution coefficients for combination. If not provided, uses the instance's `self.csc_dict` attribute instead.

        """
        if csc_dict is None:
            csc_dict = self.csc_dict
        self.image = combine_color_spaces(csc_dict, self.all_c_spaces)
        self.apply_filter_and_segment()

    def pca_and_segment(self):
        """
        Extract the first principal component and perform segmentation.

        This method extracts the first principal component from the 'bgr' color space
        and performs k-means clustering or Otsu thresholding to segment the image based on the
        parameters provided.
        """
        self.image, _, first_pc_vector = extract_first_pc(self.all_c_spaces['bgr'])
        self.csc_dict = Dict()
        self.csc_dict['bgr'] = first_pc_vector
        self.apply_filter_and_segment()

    def apply_filter_and_segment(self):
        self.greyscale = self.image
        if self.params['filter_spec'] is not None and self.params['filter_spec']["filter1_type"] != "":
            self.greyscale = apply_filter(self.greyscale, self.params['filter_spec']["filter1_type"], self.params['filter_spec']["filter1_param"])


        if self.params['kmeans_clust_nb'] is not None and (self.params['bio_mask'] is not None or self.params['back_mask'] is not None):
            self.binary_image, _, self.bio_label, _ = kmeans(self.greyscale, None, self.params['kmeans_clust_nb'],
                                                             self.params['bio_mask'], self.params['back_mask'])
        else:
            self.binary_image = otsu_thresholding(self.greyscale)
        self.validated_shapes = self.binary_image

    def evaluate_segmentation(self):
        """
        Use the filtering algorithm based on the kind of image to analyse
        """
        if self.params['is_first_image']:
            self.eval_first_image()
        else:
            self.eval_any_image()

    def eval_first_image(self):
        """
        First image filtering process for binary images.

        This method processes the first binary image by identifying connected components, computing
        the total area of the image and its potential size limits. If the number of blobs and their
        total area fall within acceptable thresholds, it proceeds with additional processing and saving.
        """
        self.fact['unaltered_blob_nb'], shapes = cv2.connectedComponents(self.binary_image)
        self.fact['unaltered_blob_nb'] -= 1
        if 1 <= self.fact['unaltered_blob_nb'].values[0] < 10000:
            self.fact['total_area'] = np.sum(self.binary_image)
            inf_lim = np.min((20, np.ceil(self.binary_image.size / 1000)))
            if inf_lim < self.fact['total_area'].values[0] < self.binary_image.size * 0.9:
                self.process_first_binary_image()
                self.save_combination()

    def eval_any_image(self):
        """

        Summarizes the binary image analysis and determines blob characteristics.

        Evaluates a binary image to determine various attributes like surface area,
        blob number, and their relative positions within specified masks. It also
        checks for common areas with a reference image and saves the combination if
        certain conditions are met.
        """
        surf = self.binary_image.sum()
        if surf < self.params['total_surface_area']:
            self.fact['unaltered_blob_nb'], shapes = cv2.connectedComponents(self.binary_image)
            self.fact['unaltered_blob_nb'] -= 1
            test: bool = True
            if self.params['arenas_mask'] is not None:
                self.fact['out_of_arenas'] = np.sum(self.binary_image * self.params['out_of_arenas_mask'])
                self.fact['in_arenas'] = np.sum(self.binary_image * self.params['arenas_mask'])
                test = self.fact['out_of_arenas'].values[0] < self.fact['in_arenas'].values[0]
            if test:
                if self.params['con_comp_extent'][0] <= self.fact['unaltered_blob_nb'].values[0] <= self.params['con_comp_extent'][1]:
                    if self.params['ref_image'] is not None:
                        self.fact['common_with_ref'] = np.sum(self.params['ref_image'] * self.binary_image)
                        test = self.fact['common_with_ref'].values[0] > 0
                    if test:
                        self.fact['blob_nb'], shapes, self.stats, centroids = cv2.connectedComponentsWithStats(self.binary_image)
                        self.fact['blob_nb'] -= 1
                        if np.all(np.sort(self.stats[:, 4])[:-1] < self.params['max_blob_size']):
                            self.save_combination()

    def apply_logical_operation(self):
        """
        Apply a logical operation between two saved images.

        This method applies a specified logical operation ('And' or 'Or')
        between two images stored in the parent's saved_images_list. The result
        is stored as a binary image and validated, with color space information
        updated accordingly.

        Notes
        -----
        This method modifies the following instance attributes:
        - `binary_image`
        - `validated_shapes`
        - `image`
        - `csc_dict`
        """
        im1 = self.parent.saved_images_list[self.operation[0]]
        im2 = self.parent.saved_images_list[self.operation[1]]
        if self.operation['logical'] == 'And':
            self.binary_image = np.logical_and(im1, im2).astype(np.uint8)
        elif self.operation['logical'] == 'Or':
            self.binary_image = np.logical_or(im1, im2).astype(np.uint8)
        self.validated_shapes = self.binary_image
        self.greyscale = self.parent.converted_images_list[self.operation[0]]
        csc1 = self.parent.saved_color_space_list[self.operation[0]]
        csc2 = self.parent.saved_color_space_list[self.operation[1]]
        self.csc_dict = {}
        for k, v in csc1.items():
            self.csc_dict[k] = v
        for k, v in csc2.items():
            self.csc_dict[k] = v
        self.csc_dict['logical'] = self.operation['logical']
        self.evaluate_segmentation()

    def try_adding_channels(self, i: int):
        """
        Try adding channels to the current color space combination.

        Extend the functionality of a selected color space combination by attempting
        to add channels from other combinations, evaluating the results based on
        the number of shapes and total area.

        Parameters
        ----------
        i : int
            The index of the saved color space and combination features to start with.
        """
        saved_color_space_list = self.parent.saved_color_space_list
        combination_features = self.parent.combination_features
        self.csc_dict = saved_color_space_list[i]
        previous_shape_number = combination_features.loc[i, 'blob_nb']
        previous_sum = combination_features.loc[i, 'total_area']
        for j in self.params['possibilities'][::-1]:
            csc_dict2 = saved_color_space_list[j]
            csc_dict = self.csc_dict.copy()
            keys = list(csc_dict.keys())

            k2 = list(csc_dict2.keys())[0]
            v2 = csc_dict2[k2]
            if np.isin(k2, keys) and np.sum(v2 * csc_dict[k2]) != 0:
                break
            for factor in [2, 1]:
                if np.isin(k2, keys):
                    csc_dict[k2] += v2 * factor
                else:
                    csc_dict[k2] = v2 * factor
                self.combine_and_segment(csc_dict)
                if self.params['is_first_image']:
                    self.process_first_binary_image()
                else:
                    self.fact['blob_nb'], shapes, self.stats, centroids = cv2.connectedComponentsWithStats(self.validated_shapes)
                    self.fact['blob_nb'] -= 1
                self.fact['total_area'] = self.validated_shapes.sum()
                if self.fact['blob_nb'].values[0] < previous_shape_number  and self.fact['total_area'].values[0] > previous_sum * 0.9:
                    previous_shape_number = self.fact['blob_nb'].values[0]
                    previous_sum = self.fact['total_area'].values[0]
                    self.csc_dict = csc_dict.copy()
                    self.fact['unaltered_blob_nb'] = combination_features.loc[i, 'unaltered_blob_nb']
                    self.save_combination()

    def try_subtracting_channels(self):
        """
        Tries to subtract channels to find the optimal color space combination.

        This method attempts to remove color spaces one by one from the image
        to find a combination that maintains the majority of areas while reducing
        the number of color spaces. This process is repeated until no further
        improvements are possible.
        """
        potentials = self.parent.all_combined
        # Try to remove color space one by one
        i = 0
        original_length = len(potentials)
        # The while loop until one col space remains or the removal of one implies a strong enough area change
        while np.logical_and(len(potentials) > 1, i < original_length - 1):
            self.combine_and_segment(potentials)
            if self.params['is_first_image']:
                self.process_first_binary_image()
                previous_blob_nb = self.fact['blob_nb'].values[0]
            else:
                previous_blob_nb, shapes, self.stats, centroids = cv2.connectedComponentsWithStats(
                    self.validated_shapes)
                previous_blob_nb -= 1
            previous_sum = self.validated_shapes.sum()
            color_space_to_remove = List()
            previous_c_space = list(potentials.keys())[-1]
            for c_space in potentials.keys():
                try_potentials = potentials.copy()
                try_potentials.pop(c_space)
                if i > 0:
                    try_potentials.pop(previous_c_space)
                self.combine_and_segment(try_potentials)
                if self.params['is_first_image']:
                    self.process_first_binary_image()
                else:
                    self.fact['blob_nb'], shapes, self.stats, centroids = cv2.connectedComponentsWithStats(
                        self.validated_shapes)
                    self.fact['blob_nb'] -= 1
                self.fact['total_area'] = self.validated_shapes.sum()
                if self.fact['blob_nb'].values[0] < previous_blob_nb and self.fact['total_area'].values[0] > previous_sum * 0.9:
                    previous_blob_nb = self.fact['blob_nb'].values[0]
                    previous_sum = self.fact['total_area'].values[0]
                    self.csc_dict = try_potentials.copy()
                    self.fact['unaltered_blob_nb'] = previous_blob_nb
                    self.save_combination()
                    # If removing that color space helps, we remove it from potentials
                    color_space_to_remove.append(c_space)
                    if i > 0:
                        color_space_to_remove.append(previous_c_space)
                previous_c_space = c_space
            if len(color_space_to_remove) == 0:
                break
            color_space_to_remove = np.unique(color_space_to_remove)
            for remove_col_space in color_space_to_remove:
                potentials.pop(remove_col_space)
            i += 1

    def process_first_binary_image(self):
        """
        Process the binary image to identify and validate shapes.

        This method processes a binary image to detect connected components,
        validate their sizes, and handle bio and back masks if specified.
        It ensures that the number of validated shapes matches the expected
        sample number or applies additional filtering if necessary.

        """
        shapes_features = shape_selection(self.binary_image, true_shape_number=self.params['blob_nb'],
                        horizontal_size=self.params['blob_size'], spot_shape=self.params['blob_shape'],
                        several_blob_per_arena=self.params['several_blob_per_arena'],
                        bio_mask=self.params['bio_mask'], back_mask=self.params['back_mask'])
        self.validated_shapes, self.fact['blob_nb'], self.stats, self.centroids = shapes_features

    def save_combination(self):
        """
        Saves the calculated features and masks for a combination of shapes.

        This method calculates various statistical properties (std) and sums for the
        validated shapes, and optionally computes sums for bio and back masks if they are
        specified in the parameters.

        Parameters
        ----------
        self : object
            The instance of the class containing this method. Must have attributes:
            - ``validated_shapes``: (ndarray) Array of validated shapes.
            - ``stats``: (ndarray) Statistics array containing width, height, and area.
            - ``params``: dictionary with parameters including 'bio_mask' and 'back_mask'.
            - ``fact``: dictionary to store the calculated features.
            - ``parent``: The parent object containing the method `save_combination_features`.
        """
        self.fact['total_area'] = self.validated_shapes.sum()
        self.fact['width_std'] = np.std(self.stats[1:, 2])
        self.fact['height_std'] = np.std(self.stats[1:, 3])
        self.fact['area_std'] = np.std(self.stats[1:, 4])
        if self.params['bio_mask'] is not None:
            self.fact['bio_sum'] = self.validated_shapes[self.params['bio_mask'][0], self.params['bio_mask'][1]].sum()
        if self.params['back_mask'] is not None:
            self.fact['back_sum'] = (1 - self.validated_shapes)[self.params['back_mask'][0], self.params['back_mask'][1]].sum()
        self.parent.save_combination_features(self)
        
        