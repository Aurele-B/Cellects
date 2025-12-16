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
from cellects.image_analysis.morphological_operations import shape_selection


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
        self.horizontal_size = l[6]
        self.spot_shape = l[7]
        kmeans_clust_nb = l[8]
        self.biomask = l[9]
        self.backmask = l[10]
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
                    #     Make sure that scaling and spot size are correct
        if combine_channels:
            i = l[3]
            possibilities = l[11]
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

    def process_binary_image(self):
        """
        Process the binary image to identify and validate shapes.

        This method processes a binary image to detect connected components,
        validate their sizes, and handle bio and back masks if specified.
        It ensures that the number of validated shapes matches the expected
        sample number or applies additional filtering if necessary.

        """
        shapes_features = shape_selection(self.binary_image, true_shape_number=self.sample_number, horizontal_size=self.horizontal_size,
                        spot_shape=self.spot_shape, several_blob_per_arena=self.several_blob_per_arena,
                        bio_mask=self.biomask, back_mask=self.backmask)
        self.validated_shapes, self.shape_number, self.stats, self.centroids = shapes_features


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
        logging.info(f"Saving results from the color space combination: {self.process_i.csc_dict}. {self.process_i.shape_number} distinct specimen(s) detected.")
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
