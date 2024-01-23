#!/usr/bin/env python3
"""
This script contains the class for detecting networks out of a grayscale image of Physarum polycephalum
"""

import cv2
import numpy as np
from numpy import nonzero, zeros_like, uint8, min, max, ptp
from cv2 import connectedComponents, CV_16U
from cellects.image_analysis.image_segmentation import get_otsu_threshold
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.image_analysis.shape_descriptors import ShapeDescriptors
from cellects.image_analysis.morphological_operations import cc


class NetworkDetection:
    def __init__(self, grayscale_image, binary_image, lighter_background, cross_33):
        """
        :param grayscale_image: current grayscale image to analyze
        :type grayscale_image: uint8
        :param binary_image: current binary image to analyze
        :type binary_image: uint8
        :param lighter_background: True if the background of the image is lighter than the shape to detect
        :type lighter_background: bool
        :param cross_33: A 3*3 matrix containing a binary cross
        :type cross_33: uint8
        """
        self.grayscale_image = grayscale_image
        self.binary_image = binary_image
        self.lighter_background = lighter_background
        self.cross_33 = cross_33

    def segment_locally(self, side_length=8, step=2, int_variation_thresh=20):
        """
        This method segment the network locally using otsu thresholding on a rolling window
        :param side_length: The size of the window to detect the tubes of the network
        :type side_length: uint8
        :param step:
        :type step: uint8
        :return:
        """
        self.side_length = side_length
        self.step = step
        y, x = nonzero(self.binary_image)
        self.min_y = min(y)
        if (self.min_y - 20) >= 0:
            self.min_y -= 20
        else:
            self.min_y = 0
        self.max_y = max(y)
        if (self.max_y + 20) < self.binary_image.shape[0]:
            self.max_y += 20
        else:
            self.max_y = self.binary_image.shape[0] - 1
        self.min_x = min(x)
        if (self.min_x - 20) >= 0:
            self.min_x -= 20
        else:
            self.min_x = 0
        self.max_x = max(x)
        if (self.max_x + 20) < self.binary_image.shape[1]:
            self.max_x += 20
        else:
            self.max_x = self.binary_image.shape[1] - 1

        # y_windows = np.arange(0, self.max_y - self.min_y + 1, side_length)
        # x_windows = np.arange(0, self.max_x - self.min_x + 1, side_length)
        y_size = self.max_y - self.min_y + 1
        x_size = self.max_x - self.min_x + 1
        network = np.zeros((self.max_y - self.min_y, self.max_x - self.min_x), np.uint64)
        self.homogeneities = np.zeros((self.max_y - self.min_y, self.max_x - self.min_x), np.uint64)
        cropped_binary_image = self.binary_image[self.min_y:self.max_y, self.min_x:self.max_x]
        cropped_grayscale_image = self.grayscale_image[self.min_y:self.max_y, self.min_x:self.max_x]
        # will be more efficient if it only loops over a zoom on self.binary_image == 1
        for to_add in np.arange(0, side_length, step):
            y_windows = np.arange(0, y_size, side_length)
            x_windows = np.arange(0, x_size, side_length)
            y_windows += to_add
            x_windows += to_add
            for y_start in y_windows:
                # y_start = 4
                if y_start < network.shape[0]:
                    y_end = y_start + side_length
                    if y_end < network.shape[0]:
                        for x_start in x_windows:
                            if x_start < network.shape[1]:
                                x_end = x_start + side_length
                                if x_end < network.shape[1]:
                                    if np.any(cropped_binary_image[y_start:y_end, x_start:x_end]):
                                        potential_network = cropped_grayscale_image[y_start:y_end, x_start:x_end]
                                        if ptp(potential_network[nonzero(potential_network)]) < int_variation_thresh:
                                            self.homogeneities[y_start:y_end, x_start:x_end] += 1
                                        threshold = get_otsu_threshold(potential_network)
                                        if self.lighter_background:
                                            net_coord = np.nonzero(potential_network < threshold)
                                        else:
                                            net_coord = np.nonzero(potential_network > threshold)
                                        network[y_start + net_coord[0], x_start + net_coord[1]] += 1

        self.network = np.zeros(self.binary_image.shape, np.uint8)
        self.network[self.min_y:self.max_y, self.min_x:self.max_x] = (network >= (side_length // step)).astype(np.uint8)
        self.network[self.min_y:self.max_y, self.min_x:self.max_x][self.homogeneities >= (((self.side_length // self.step) // 2) + 1)] = 0

    def segment_globally(self):
        """
        This method remove pixels whose intensity is too close from the average
        :return:
        """
        grayscale_network = self.grayscale_image * self.network
        average = grayscale_network[grayscale_network != 0].mean()

        nb, components = cv2.connectedComponents(self.network)
        network = np.zeros_like(self.network)
        for compo_i in np.arange(1, nb):
            coord = np.nonzero(components == compo_i)
            if self.lighter_background:
                if np.mean(self.grayscale_image[coord]) < 0.8 * average:
                    network[coord] = 1
            else:
                if np.mean(self.grayscale_image[coord]) > 1.25 * average:
                    network[coord] = 1
        self.network = network

    def selects_elongated_or_holed_shapes(self, hole_surface_ratio=0.1, eccentricity=0.65):
        """
        This method only keep the elongates or holed shapes.
        An elongated shape as a strong eccentricity
        A holed shape contain a holes of a decent size
        :param hole_surface_ratio: ratio threshold for hole selection
        :type hole_surface_ratio: float64
        :param eccentricity: eccentricity threshold for elongation selection
        :type eccentricity: float64
        :return:
        """
        potential_network = cv2.morphologyEx(self.network, cv2.MORPH_OPEN, self.cross_33)
        potential_network = cv2.morphologyEx(potential_network, cv2.MORPH_CLOSE, self.cross_33)

        # Filter out component that does not have the shape of the part of a network
        nb, components = cv2.connectedComponents(potential_network)
        self.network = np.zeros_like(self.network)
        for compo_i in np.arange(1, nb):
            #compo_i = 1
            #compo_i += 1
            matrix_i = components == compo_i
            # Accept as part of the network, the shapes that contains large holes and, if they don't,
            # that are eccentric enough
            SD = ShapeDescriptors(matrix_i, ['area', 'total_hole_area', 'eccentricity'])
            add_to_the_network: bool = False
            if SD.descriptors['total_hole_area'] > hole_surface_ratio * SD.descriptors['area']:
                add_to_the_network = True
            else:
                if SD.descriptors['eccentricity'] > eccentricity:
                    add_to_the_network = True
            if add_to_the_network:
                matrix_i = np.nonzero(matrix_i)
                self.network[matrix_i[0], matrix_i[1]] = 1

    def add_pixels_to_the_network(self, pixels_to_add):
        """
        When needed, consider all area occupied by the origin as part of the network
        :return:
        """
        self.network[pixels_to_add] = 1

    def remove_pixels_from_the_network(self, pixels_to_remove, remove_homogeneities=True):
        """
        When needed, consider all area occupied by the origin as part of the network
        :return:
        """
        self.network[pixels_to_remove] = 0
        if remove_homogeneities:
            self.network[self.min_y:self.max_y, self.min_x:self.max_x][self.homogeneities >= (((self.side_length // self.step) // 2) + 1)] = 0

    def connect_network(self, maximal_distance_connection=50):
        """
        Force all component of the network to be connected
        :param maximal_distance_connection:
        :type maximal_distance_connection: uint64
        :return:
        """
        # Force connection of the disconnected parts of the network
        if np.any(self.network):
            ordered_image, stats, centers = cc(self.network)
            # if self.lighter_background:
            #     intensity_valley = 255 - self.grayscale_image.copy()
            # else:
            #     intensity_valley = self.grayscale_image.copy()
            # new
            pads = ProgressivelyAddDistantShapes(self.network, ordered_image==1, maximal_distance_connection, self.cross_33)
            # If max_distance is non nul look for distant shapes
            # pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False, intensity_valley=intensity_valley)
            pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False)
            potential_network = np.nonzero(pads.expanded_shape)
            self.network[potential_network[0], potential_network[1]] = 1

            # Remove all disconnected components remaining
            ordered_image, stats, centers = cc(self.network)
            self.network[ordered_image != 1] = 0

            # prev:
            # nb, components = cv2.connectedComponents(self.network)
            # for compo_i in np.arange(1, nb):
            #     matrix_i = components == compo_i
            #
            #     # network_i = np.nonzero(change_thresh_until_one(self.grayscale_image, matrix_i, self.lighter_background))
            #     # self.network[network_i] = 1
            #     pads = ProgressivelyAddDistantShapes(matrix_i, self.network, maximal_distance_connection, self.cross_33)
            #     # If max_distance is non nul look for distant shapes
            #     pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False, intensity_valley=intensity_valley)
            #     potential_network = np.nonzero(pads.expanded_shape)
            #     self.network[potential_network[0], potential_network[1]] = 1
        # Visualize the network
        # img = self.grayscale_image.copy()
        # net_cc = np.nonzero(self.network)
        # img[net_cc[0], net_cc[1]] = 255
        # See(img)
        # Remains to really skeletonize that network !

    def skeletonize(self):
        """
        Skeletonize the network
        :return:
        """
        self.skeleton = skeletonize(self.network)
        # https://github.com/LingDong-/skeleton-tracing
        # https://runebook.dev/fr/docs/scikit_learn/modules/generated/sklearn.feature_extraction.image.img_to_graph

def skeletonize(binary_image):
    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        # Open the image
        open = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, element)
        # Substract open from the original image
        temp = cv2.subtract(binary_image, open)
        # Erode the original image and refine the skeleton
        eroded = cv2.erode(binary_image, element)
        skel = cv2.bitwise_or(binary_image, temp)
        binary_image = eroded.copy()
        # If there are no white pixels left, ie. the image has been completely eroded, quit the loop
        if cv2.countNonZero(binary_image) == 0:
            break
    return skel


def change_thresh_until_one(grayscale_image, binary_image, lighter_background):
    coord = nonzero(binary_image)
    min_cx = min(coord[0])
    max_cx = max(coord[0])
    min_cy = min(coord[1])
    max_cy = max(coord[1])
    gray_img = grayscale_image[min_cy:max_cy, min_cx:max_cx]
    threshold = get_otsu_threshold(gray_img)
    bin_img = (gray_img < threshold).astype(uint8)
    detected_shape_number, bin_img = connectedComponents(bin_img, ltype=CV_16U)
    while (detected_shape_number > 2) and (0 < threshold < 255):
        if lighter_background:
            threshold += 1
            bin_img = (gray_img < threshold).astype(uint8)
        else:
            threshold -= 1
            bin_img = (gray_img < threshold).astype(uint8)
        detected_shape_number, bin_img = connectedComponents(bin_img, ltype=CV_16U)
    binary_image = zeros_like(binary_image, uint8)
    binary_image[min_cy:max_cy, min_cx:max_cx] = bin_img
    return binary_image
