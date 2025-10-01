#!/usr/bin/env python3
"""
This script contains the class for analyzing fractals out of a binary image
"""


import os
from pickletools import uint8
from copy import deepcopy
import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.stats import linregress
from cellects.utils.formulas import linear_model
from cellects.core.cellects_paths import DATA_DIR
from cellects.image_analysis.morphological_operations import cross_33


def prepare_box_counting(binary_image, side_threshold=128, zoom_step=0, contours=True):
    side_lengths = None
    zoomed_binary = binary_image
    binary_idx = np.nonzero(binary_image)
    if binary_idx[0].size:
        min_y = np.min(binary_idx[0])
        min_y = np.max((min_y - 1, 0))

        min_x = np.min(binary_idx[1])
        min_x = np.max((min_x - 1, 0))

        max_y = np.max(binary_idx[0])
        max_y = np.min((max_y + 1, binary_image.shape[0] - 1))

        max_x = np.max(binary_idx[1])
        max_x = np.min((max_x + 1, binary_image.shape[1] - 1))

        zoomed_binary = deepcopy(binary_image[min_y:(max_y + 1), min_x: (max_x + 1)])
        min_side = np.min(zoomed_binary.shape)
        if min_side >= side_threshold:
            if contours:
                eroded_zoomed_binary = cv2.erode(zoomed_binary, cross_33)
                zoomed_binary = zoomed_binary - eroded_zoomed_binary
            if zoom_step == 0:
                max_power = int(np.floor(np.log2(min_side)))  # Largest integer power of 2
                side_lengths = 2 ** np.arange(max_power, -1, -1)
            else:
                side_lengths = np.arange(1, min_side, zoom_step)
    return zoomed_binary, side_lengths


def box_counting(zoomed_binary, side_lengths):
    """
    Let us take:
    - s: the side lengths of many boxes
    - N(s): the number of pixels belonging to the shape contained in a box of side length s.
    - c: a constant
    - D: the fractal dimension

    N(s) = C(1/s)^D
    log(N(s)) = D*log(1/s) + log(C)

    box_counting_dimension = log(N)/log(1/s)
    The line of y=log(N(r)) vs x=log(1/r) has a slope equal to the box_counting_dimension
    :param zoomed_binary:
    :return:
    """
    box_counting_dimension:float = 0.
    r_value:float = 0.
    box_nb:float = 0.
    if side_lengths is not None:
        box_counts = np.zeros(len(side_lengths), dtype=np.uint64)
        # Loop through side_lengths and compute block counts
        for idx, side_length in enumerate(side_lengths):
            S = np.add.reduceat(
                np.add.reduceat(zoomed_binary, np.arange(0, zoomed_binary.shape[0], side_length), axis=0),
                np.arange(0, zoomed_binary.shape[1], side_length),
                axis=1
            )
            box_counts[idx] = len(np.where(S > 0)[0])

        valid_indices = box_counts > 0
        if valid_indices.sum() >= 2:
            log_box_counts = np.log(box_counts)
            log_reciprocal_lengths = np.log(1 / side_lengths)
            slope, intercept, r_value, p_value, stderr = linregress(log_reciprocal_lengths, log_box_counts)
            # coefficients = np.polyfit(log_reciprocal_lengths, log_box_counts, 1)
            box_counting_dimension = slope
            box_nb = len(side_lengths)

    return box_counting_dimension, r_value, box_nb


class FractalAnalysis:
    def __init__(self, binary_image):
        """
        Initialize FractalAnalysis class
        :param binary_image: A 2D binary image. The two first dims are coordinates.
        :type binary_image: uint8
        """
        self.binary_image = binary_image
        self.fractal_contours = []
        self.fractal_box_lengths = []
        self.fractal_box_widths = []
        self.minkowski_dimension = None

    def detect_fractal(self, threshold=100):
        """
        Find the contours of the fractal
        :param threshold:
        :type threshold: int
        :return:
        """
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > threshold:
                self.fractal_contours.append(contour)

    def extract_fractal(self, image):
        """
        Draw the fractal mesh on a colored image
        :param image: 3D matrix of a bgr image. The two first dims are coordinates, le last is color.
        :type image: uint8
        :return:
        """
        # Créer un masque pour extraire la fractale
        self.fractal_mesh = np.zeros(self.binary_image.shape, dtype=np.uint8)

        for contour in self.fractal_contours:
            # i=0
            # i+=1
            # contour = self.fractal_contours[i]
            # cv2.drawContours(self.fractal_mesh, [contour], 0, 1, thickness=cv2.FILLED)
            cont = np.reshape(contour, (len(contour), 2))
            cont = np.transpose(cont)
            self.fractal_mesh[cont[1], cont[0]] = 1

            x, y, w, h = cv2.boundingRect(contour)
            self.fractal_box_lengths.append(np.max((w, h)))
            self.fractal_box_widths.append(np.min((w, h)))

    def get_dimension(self):
        """
        Compute the minkowski dimension of the binary image
        :return: Minkowski dimension
        :rtype: =np.float64
        """
        if np.any(self.fractal_mesh):
            fractal_mask = deepcopy(self.fractal_mesh[:, :])
            # fractal_mask = (self.fractal_mesh[:, :] > 0).astype(np.uint8)
            size = np.min(fractal_mask.shape)
            scales = np.arange(1, int(np.log2(size)) + 1)

            Ns = []
            for scale in scales:
                # i=0
                # i+=1
                # scale = scales[i]

                box_size = 2 ** scale
                boxes = np.add.reduceat(
                    np.add.reduceat(fractal_mask, np.arange(0, fractal_mask.shape[0], box_size), axis=0),
                    np.arange(0, fractal_mask.shape[1], box_size),
                    axis=1,
                )

                Ns.append(np.sum(boxes > 0))

            scales_inv = 1 / np.array(scales, dtype=np.float32)
            coeffs, _ = curve_fit(linear_model, scales_inv, np.log(np.array(Ns, dtype=np.float32)))
            self.minkowski_dimension = coeffs[1]
        else:
            self.minkowski_dimension = 0

    def save_fractal_mesh(self, image_save_path):
        """
        Save an image representing the fractal mesh
        :param image_save_path: path where to save the image
        :type image_save_path: str
        :return:
        """
        cv2.imwrite(image_save_path, self.fractal_mesh)



