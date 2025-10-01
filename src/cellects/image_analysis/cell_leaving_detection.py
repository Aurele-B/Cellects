#!/usr/bin/env python3
"""Contains the function: cell_leaving_detection
This function considers the pixel intensity curve of each covered pixel and assesesed whether a covered pixel retrieved
-partially at least- its initial intensity.
"""
import cv2
import numpy as np
from cellects.image_analysis.morphological_operations import cross_33


def cell_leaving_detection(new_shape, covering_intensity, previous_binary, greyscale_image, fading_coefficient, lighter_background, several_blob_per_arena, erodila_disk, protect_from_fading=None, add_to_fading=None):
    # To look for fading pixels, only consider the internal contour of the shape at t-1
    fading = cv2.erode(previous_binary, erodila_disk)
    fading = previous_binary - fading
    # If the origin state is considered constant: origin pixels will never be fading
    if protect_from_fading is not None:
        fading *= (1 - protect_from_fading)
    if add_to_fading is not None:
        if protect_from_fading is not None:
            add_to_fading[np.nonzero(protect_from_fading)] = 0
        add_to_fading_coord = np.nonzero(add_to_fading)
        fading[add_to_fading_coord] = 1
        if lighter_background:
            covering_intensity[add_to_fading_coord] = 1 / (1 - fading_coefficient)  # 0.9 * covering_intensity[add_to_fading_coord]  #
        else:
            covering_intensity[add_to_fading_coord] = 255  # 1.1 * covering_intensity[add_to_fading_coord]
    # With a lighter background, fading them if their intensity gets higher than the covering intensity
    if lighter_background:
        fading = fading * np.greater((greyscale_image), (1 - fading_coefficient) * covering_intensity).astype(np.uint8)
    else:
        fading = fading * np.less((greyscale_image), (1 + fading_coefficient) * covering_intensity).astype(np.uint8)

    if np.any(fading):
        fading = cv2.morphologyEx(fading, cv2.MORPH_CLOSE, cross_33, iterations=1)
        if not several_blob_per_arena:
            # Check if uncov_potentials does not break the shape into several components
            uncov_nb, uncov_shapes = cv2.connectedComponents(fading, ltype=cv2.CV_16U)
            nb, garbage_img = cv2.connectedComponents(new_shape, ltype=cv2.CV_16U)
            i = 0
            while i <= uncov_nb:
                i += 1
                prev_nb = nb
                new_shape[np.nonzero(uncov_shapes == i)] = 0
                nb, garbage_img = cv2.connectedComponents(new_shape, ltype=cv2.CV_16U)
                if nb > prev_nb:
                    new_shape[np.nonzero(uncov_shapes == i)] = 1
                    nb, garbage_img = cv2.connectedComponents(new_shape, ltype=cv2.CV_16U)
            uncov_shapes = None
        else:
            new_shape[np.nonzero(fading)] = 0
        new_shape = cv2.morphologyEx(new_shape, cv2.MORPH_OPEN, cross_33, iterations=0)
        new_shape = cv2.morphologyEx(new_shape, cv2.MORPH_CLOSE, cross_33)

    return new_shape, covering_intensity