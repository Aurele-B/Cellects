
import os
from pickletools import uint8
from copy import deepcopy

# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.stats import linregress
from cellects.utils.formulas import (linear_model)
from cellects.image_analysis.morphological_operations import cross_33


def display_boxes(binary_image, box_diameter):
    plt.imshow(binary_image, cmap='gray')
    height, width = binary_image.shape
    for x in range(0, width + 1, box_diameter):
        plt.axvline(x=x, color='white', linewidth=1)
    for y in range(0, height + 1, box_diameter):
        plt.axhline(y=y, color='white', linewidth=1)
    plt.show()


def prepare_box_counting(binary_image, min_im_side=128, min_mesh_side=8, zoom_step=0, contours=True):
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
        if min_side >= min_im_side:
            if contours:
                eroded_zoomed_binary = cv2.erode(zoomed_binary, cross_33)
                zoomed_binary = zoomed_binary - eroded_zoomed_binary
            if zoom_step == 0:
                max_power = int(np.floor(np.log2(min_side)))  # Largest integer power of 2
                side_lengths = 2 ** np.arange(max_power, int(np.log2(min_mesh_side // 2)), -1)
            else:
                side_lengths = np.arange(min_mesh_side, min_side, zoom_step)
    return zoomed_binary, side_lengths


def box_counting(zoomed_binary, side_lengths, display=False):
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
            if display:
                plt.scatter(log_reciprocal_lengths, log_box_counts, label="Box counting")
                plt.plot([0, log_reciprocal_lengths.min()], [intercept, intercept + slope * log_reciprocal_lengths.min()], label="Linear regression")
                plt.plot([], [], ' ', label=f"D = {slope:.2f}")
                plt.plot([], [], ' ', label=f"R2 = {r_value:.6f}")
                plt.plot([], [], ' ', label=f"p-value = {p_value:.2e}")
                plt.legend(loc='best')
                plt.xlabel(f"log(1/Diameter) | Diameter ⊆ [{side_lengths[0]}:{side_lengths[-1]}] (n={box_nb})")
                plt.ylabel(f"log(Box number) | Box number ⊆ [{box_counts[0]}:{box_counts[-1]}]")
                plt.show()
                # plt.close()

    return box_counting_dimension, r_value, box_nb