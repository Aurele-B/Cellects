#!/usr/bin/env python3
"""Module containing classes for image processing and saving selected color space combinations.

This module provides two thread-based components for analyzing images and storing results:

ProcessFirstImage handles initial segmentation, thresholding, clustering, and shape validation
SaveCombinationThread stores processed features in parent objects asynchronously
The processing pipeline includes Otsu thresholding, k-means clustering, connected component analysis,
and geometric filtering based on size/shape constraints.

Classes
ProcessFirstImage : Processes image data with segmentation techniques and validates shapes.
SaveCombinationThread : Thread to save combination results while maintaining UI responsiveness.

Functions (in ProcessFirstImage)
shape_selection : Filters shapes by size thresholds and geometric criteria.
kmeans : Performs clustering-based image segmentation into specified number of clusters.
process_binary_image : Validates detected shapes against area constraints and spot count targets.

Notes
Uses threading.Thread for background operations to maintain application responsiveness during processing.
"""
import threading
import logging
import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Tuple
from cellects.image_analysis.image_segmentation import otsu_thresholding, combine_color_spaces


class ProcessFirstImage:
    """
    A class for processing lists.
    """
    def __init__(self, l):
        """
        Arguments:
            list : list

        """
        self.start_processing(l)

    def start_processing(self, l: list):
        """

        Start the processing based on given list input.

        The method processes the provided list to perform various operations
        on the image data. It sets up several attributes and performs different
        image processing tasks like Otsu thresholding or k-means clustering.

        The method does not return any value.
        """
        self.parent = l[0]
        get_one_channel_result = l[1]
        combine_channels = l[2]
        self.all_c_spaces = self.parent.all_c_spaces
        self.several_blob_per_arena = l[4]
        self.sample_number = l[5]
        self.spot_size = l[6]
        kmeans_clust_nb = l[7]
        self.biomask = l[8]
        self.backmask = l[9]
        if get_one_channel_result:
            self.csc_dict = l[3]
            self.image = combine_color_spaces(self.csc_dict, self.all_c_spaces)
            if kmeans_clust_nb is None:
                self.binary_image = otsu_thresholding(self.image)
            else:
                self.kmeans(kmeans_clust_nb, self.biomask, self.backmask)
                # self.parent.image = self.image
                # self.parent.kmeans(kmeans_clust_nb, self.biomask, self.backmask)
                # self.binary_image = self.parent.binary_image
            self.unaltered_concomp_nb, shapes = cv2.connectedComponents(self.binary_image)
            if 1 < self.unaltered_concomp_nb < 10000:
                self.total_area = np.sum(self.binary_image)
                inf_lim = np.min((100, np.ceil(self.binary_image.size / 1000)))
                if inf_lim < self.total_area < self.binary_image.size * 0.9:
                    self.process_binary_image()
                    self.parent.save_combination_features(self)
                    # except RuntimeWarning:
                    #     logging.info("Make sure that scaling and spot size are correct")
        if combine_channels:
            i = l[3]
            possibilities = l[10]
            saved_color_space_list = self.parent.saved_color_space_list
            combination_features = self.parent.combination_features
            self.csc_dict = saved_color_space_list[i]
            previous_shape_number = combination_features[i, 4]
            previous_sum = combination_features[i, 5]
            for j in possibilities[::-1]:
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
                    self.image = combine_color_spaces(csc_dict, self.all_c_spaces)
                    if kmeans_clust_nb is None:
                        self.binary_image = otsu_thresholding(self.image)
                    else:
                        self.kmeans(kmeans_clust_nb, self.biomask, self.backmask)
                    self.process_binary_image()
                    self.total_area = self.validated_shapes.sum()
                    if previous_shape_number >= self.shape_number and self.total_area > previous_sum * 0.9:
                        previous_shape_number = self.shape_number
                        previous_sum = self.total_area
                        self.csc_dict = csc_dict.copy()
                        self.unaltered_concomp_nb = combination_features[i, 3]
                        self.parent.save_combination_features(self)
                        logging.info(str(saved_color_space_list[i]) + "-->" + str(self.csc_dict ))

    def shape_selection(self, horizontal_size, shape: str, confint, do_not_delete: NDArray=None):
        """

        Selects and validates the shapes of stains based on their size and shape.

        This method performs two main tasks:
        1. Removes stains whose horizontal size varies too much from a reference value.
        2. Determines the shape of each remaining stain and only keeps those that correspond to a reference shape.

        The method first removes stains whose horizontal size is outside the specified confidence interval. Then, it identifies shapes that do not correspond to a predefined reference shape and removes them as well.

        Args:
            horizontal_size (int): The expected horizontal size of the stains to use as a reference.
            shape (str): The shape type ('circle' or 'rectangle')
                that the stains should match. Other shapes are not currently supported.
            confint (float): The confidence interval as a decimal
                representing the percentage within which the size of the stains should fall.
            do_not_delete (NDArray, optional): An array of stain indices that should not be deleted.
                Default is None.

        """
        # counter+=1;horizontal_size = self.spot_size; shape = self.parent.spot_shapes[counter];confint = self.parent.spot_size_confints[::-1][counter]
        # stats columns contain in that order:
        # - x leftmost coordinate of boundingbox
        # - y topmost coordinate of boundingbox
        # - The horizontal size of the bounding box.
        # - The vertical size of the bounding box.
        # - The total area (in pixels) of the connected component.

        # First, remove each stain which horizontal size varies too much from reference
        size_interval = [horizontal_size * (1 - confint), horizontal_size * (1 + confint)]
        cc_to_remove = np.argwhere(np.logical_or(self.stats[:, 2] < size_interval[0], self.stats[:, 2] > size_interval[1]))

        if do_not_delete is None:
            self.shapes2[np.isin(self.shapes2, cc_to_remove)] = 0
        else:
            self.shapes2[np.logical_and(np.isin(self.shapes2, cc_to_remove), np.logical_not(np.isin(self.shapes2, do_not_delete)))] = 0

        # Second, determine the shape of each stain to only keep the ones corresponding to the reference shape
        shapes = np.zeros(self.binary_image.shape, dtype=np.uint8)
        shapes[self.shapes2 > 0] = 1
        nb_components, self.shapes2, self.stats, self.centroids = cv2.connectedComponentsWithStats(shapes,
                                                                                   connectivity=8)
        if nb_components > 1:
            if shape == 'circle':
                surf_interval = [np.pi * np.square(horizontal_size // 2) * (1 - confint), np.pi * np.square(horizontal_size // 2) * (1 + confint)]
                cc_to_remove = np.argwhere(np.logical_or(self.stats[:, 4] < surf_interval[0], self.stats[:, 4] > surf_interval[1]))
            elif shape == 'rectangle':
                # If the smaller side is the horizontal one, use the user provided horizontal side
                if np.argmin((np.mean(self.stats[1:, 2]), np.mean(self.stats[1:, 3]))) == 0:
                    surf_interval = [np.square(horizontal_size) * (1 - confint), np.square(horizontal_size) * (1 + confint)]
                    cc_to_remove = np.argwhere(np.logical_or(self.stats[:, 4] < surf_interval[0], self.stats[:, 4] > surf_interval[1]))
                # If the smaller side is the vertical one, use the median vertical length shape
                else:
                    surf_interval = [np.square(np.median(self.stats[1:, 3])) * (1 - confint), np.square(np.median(self.stats[1:, 3])) * (1 + confint)]
                    cc_to_remove = np.argwhere(np.logical_or(self.stats[:, 4] < surf_interval[0], self.stats[:, 4] > surf_interval[1]))
            else:
                logging.info("Original blob shape not well written")

            if do_not_delete is None:
                self.shapes2[np.isin(self.shapes2, cc_to_remove)] = 0
            else:
                self.shapes2[np.logical_and(np.isin(self.shapes2, cc_to_remove),
                                            np.logical_not(np.isin(self.shapes2, do_not_delete)))] = 0
            # There was only that before:
            shapes = np.zeros(self.binary_image.shape, dtype=np.uint8)
            shapes[np.nonzero(self.shapes2)] = 1

            nb_components, self.shapes2, self.stats, self.centroids = cv2.connectedComponentsWithStats(shapes, connectivity=8)
        self.validated_shapes = shapes
        self.shape_number = nb_components - 1

    def kmeans(self, cluster_number: int, biomask: NDArray[np.uint8]=None, backmask: NDArray[np.uint8]=None, bio_label=None):
        """

        Perform k-means clustering on the image to segment it into a specified number of clusters.

        Args:
            cluster_number (int): The desired number of clusters.
            biomask (NDArray[np.uint8]): Optional mask for biological regions. Default is None.
            backmask (NDArray[np.uint8]): Optional mask for background regions. Default is None.
            bio_label (int): The label assigned to the biological region. Default is None.

        Returns:
            None

        Note:
            This method modifies the `binary_image` and `bio_label` attributes of the instance.

        """
        image = self.image.reshape((-1, 1))
        image = np.float32(image)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, label, center = cv2.kmeans(image, cluster_number, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        kmeans_image = np.uint8(label.flatten().reshape(self.image.shape[:2]))
        sum_per_label = np.zeros(cluster_number)
        self.binary_image = np.zeros(self.image.shape[:2], np.uint8)
        if bio_label is not None:
            self.binary_image[np.nonzero(kmeans_image == bio_label)] = 1
            self.bio_label = bio_label
        else:
            if biomask is not None:
                all_labels = kmeans_image[biomask[0], biomask[1]]
                for i in range(cluster_number):
                    sum_per_label[i] = (all_labels == i).sum()
                self.bio_label = np.nonzero(sum_per_label == np.max(sum_per_label))
            elif backmask is not None:
                all_labels = kmeans_image[backmask[0], backmask[1]]
                for i in range(cluster_number):
                    sum_per_label[i] = (all_labels == i).sum()
                self.bio_label = np.nonzero(sum_per_label == np.min(sum_per_label))
            else:
                for i in range(cluster_number):
                    sum_per_label[i] = (kmeans_image == i).sum()
                self.bio_label = np.nonzero(sum_per_label == np.min(sum_per_label))
            self.binary_image[np.nonzero(kmeans_image == self.bio_label)] = 1

    def process_binary_image(self, use_bio_and_back_masks: bool=False):
        """
        Process the binary image to identify and validate shapes.

        This method processes a binary image to detect connected components,
        validate their sizes, and handle bio and back masks if specified.
        It ensures that the number of validated shapes matches the expected
        sample number or applies additional filtering if necessary.

        Args:
            use_bio_and_back_masks (bool): Whether to use bio and back masks
                during the processing. Default is False.

        """
        self.shape_number, self.shapes, self.stats, self.centroids = cv2.connectedComponentsWithStats(
            self.binary_image, connectivity=8)
        do_not_delete = None
        if use_bio_and_back_masks:
            if self.backmask is not None:
                if np.any(self.shapes[self.backmask]):
                    self.shapes[np.isin(self.shapes, np.unique(self.shapes[self.backmask]))] = 0
                    self.shape_number, self.shapes, self.stats, self.centroids = cv2.connectedComponentsWithStats(
                        (self.shapes > 0).astype(np.uint8), connectivity=8)
            self.shape_number -= 1
            if self.biomask is not None:
                if np.any(self.shapes[self.biomask]):
                    do_not_delete = np.unique(self.shapes[self.biomask])
                    do_not_delete = do_not_delete[do_not_delete != 0]

        if not self.several_blob_per_arena and self.spot_size is not None:
            counter = 0
            self.shapes2 = self.shapes.copy()
            while self.shape_number != self.sample_number and counter < len(self.parent.spot_size_confints):
                self.shape_selection(horizontal_size=self.spot_size, shape=self.parent.spot_shapes[counter],
                                     confint=self.parent.spot_size_confints[counter], do_not_delete=do_not_delete)
                logging.info(f"Shape selection algorithm found {self.shape_number} disconnected shapes")
                counter += 1
            if self.shape_number == self.sample_number:
                self.shapes = self.shapes2
        if self.sample_number is None or (self.shape_number - 1) == self.sample_number:
            self.validated_shapes = np.zeros(self.shapes.shape, dtype=np.uint8)
            self.validated_shapes[self.shapes > 0] = 1
        else:
            max_size = self.binary_image.size * 0.75
            min_size = 10
            cc_to_remove = np.argwhere(np.logical_or(self.stats[1:, 4] < min_size, self.stats[1:, 4] > max_size)) + 1
            self.shapes[np.isin(self.shapes, cc_to_remove)] = 0
            self.validated_shapes = np.zeros(self.shapes.shape, dtype=np.uint8)
            self.validated_shapes[self.shapes > 0] = 1
            self.shape_number, self.shapes, self.stats, self.centroids = cv2.connectedComponentsWithStats(
                self.validated_shapes,
                connectivity=8)
            if not self.several_blob_per_arena and self.sample_number is not None and self.shape_number > self.sample_number:
                # Sort shapes by size and compare the largest with the second largest
                # If the difference is too large, remove that largest shape.
                cc_to_remove = np.array([], dtype=np.uint8)
                to_remove = np.array([], dtype=np.uint8)
                self.stats = self.stats[1:, :]
                while self.stats.shape[0] > self.sample_number and to_remove is not None:
                    # 1) rank by height
                    sorted_height = np.argsort(self.stats[:, 2])
                    # and only consider the number of shapes we want to detect
                    standard_error = np.std(self.stats[sorted_height, 2][-self.sample_number:])
                    differences = np.diff(self.stats[sorted_height, 2])
                    # Look for very big changes from one height to the next
                    if differences.any() and np.max(differences) > 2 * standard_error:
                        # Within these, remove shapes that are too large
                        to_remove = sorted_height[np.argmax(differences)]
                        cc_to_remove = np.append(cc_to_remove, to_remove + 1)
                        self.stats = np.delete(self.stats, to_remove, 0)

                    else:
                        to_remove = None
                self.shapes[np.isin(self.shapes, cc_to_remove)] = 0
                self.validated_shapes = np.zeros(self.shapes.shape, dtype=np.uint8)
                self.validated_shapes[self.shapes > 0] = 1
                self.shape_number, self.shapes, self.stats, self.centroids = cv2.connectedComponentsWithStats(
                    self.validated_shapes,
                    connectivity=8)

            self.shape_number -= 1


class SaveCombinationThread(threading.Thread):
    """
    SaveCombinationThread

    This class represents a thread for saving combinations.

    """
    def __init__(self, parent=None):
        """
        **Args:**

            - `parent`: The parent object that initiated the thread. This is an optional argument and defaults to 'None'.

        """
        # super(SaveCombinationThread, self).__init__()
        threading.Thread.__init__(self)
        self.parent = parent

    def run(self):
        """
        Runs the color space combination process and saves the results.

        This method performs several tasks to save intermediate and final
        results of the color space combination process. It logs messages,
        updates lists with valid shapes, converts images to a specific format,
        and updates combination features with various statistics. The method
        also handles biomask and backmask calculations if they are not None.
        Finally, it increments the saved color space number counter.
        """
        logging.info(f"Saving results from the color space combination: {self.process_i.csc_dict}. {self.process_i.shape_number} distinct spots detected.")
        self.parent.saved_images_list.append(self.process_i.validated_shapes)
        self.parent.converted_images_list.append(np.round(self.process_i.image).astype(np.uint8))
        self.parent.saved_color_space_list.append(self.process_i.csc_dict)
        self.parent.combination_features[self.parent.saved_csc_nb, :3] = list(self.process_i.csc_dict.values())[0]
        self.parent.combination_features[self.parent.saved_csc_nb, 3] = self.process_i.unaltered_concomp_nb - 1  # unaltered_cc_nb
        self.parent.combination_features[self.parent.saved_csc_nb, 4] = self.process_i.shape_number  # cc_nb
        self.parent.combination_features[self.parent.saved_csc_nb, 5] = self.process_i.total_area  # area
        self.parent.combination_features[self.parent.saved_csc_nb, 6] = np.std(self.process_i.stats[1:, 2])  # width_std
        self.parent.combination_features[self.parent.saved_csc_nb, 7] = np.std(self.process_i.stats[1:, 3])  # height_std
        self.parent.combination_features[self.parent.saved_csc_nb, 8] = np.std(self.process_i.stats[1:, 4])  # area_std
        if self.process_i.biomask is not None:
            self.parent.combination_features[self.parent.saved_csc_nb, 9] = np.sum(
                self.process_i.validated_shapes[self.process_i.biomask[0], self.process_i.biomask[1]])
        if self.process_i.backmask is not None:
            self.parent.combination_features[self.parent.saved_csc_nb, 10] = np.sum(
                (1 - self.process_i.validated_shapes)[self.process_i.backmask[0], self.process_i.backmask[1]])
        self.parent.saved_csc_nb += 1
        logging.info("end")
