#!/usr/bin/env python3
"""Contains the function: cell_leaving_detection
This function considers the pixel intensity curve of each covered pixel and assesesed whether a covered pixel retrieved
-partially at least- its initial intensity.
"""
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from cellects.image_analysis.morphological_operations import cross_33


def cell_leaving_detection(new_shape: NDArray[np.uint8], covering_intensity:NDArray, previous_binary: NDArray[np.uint8], greyscale_image: NDArray, fading_coefficient: float, lighter_background: bool, several_blob_per_arena: bool, erodila_disk: NDArray[np.uint8], protect_from_fading: NDArray=None, add_to_fading: NDArray=None) -> Tuple[NDArray[np.uint8], NDArray]:
    """
    Perform cell leaving detection based on shape changes and intensity variations.

    Checks for fading pixels by considering the internal contour of a previous binary
    image, applies erosion and subtraction operations, and updates the shape based on
    fading detection. It handles cases where the background is lighter or darker and
    ensures that detected fading regions do not fragment the shape into multiple components,
    unless specified otherwise.

    Parameters
    ----------
    new_shape : NDArray[np.uint8]
        The current shape to be updated based on fading detection.

    covering_intensity : NDArray
        Intensity values used to determine if pixels are fading.
        Should have the same dimensions as new_shape.

    previous_binary : NDArray[np.uint8]
        Binary representation of the shape at the previous time step.
        Should have the same dimensions as new_shape.

    greyscale_image : NDArray
        Greyscale image used for intensity comparison.
        Should have the same dimensions as new_shape.

    fading_coefficient : float
        A coefficient to determine fading thresholds based on covering intensity.
        Should be between 0 and 1.

    lighter_background : bool
        Flag indicating if the background is lighter.
        True if background is lighter, False otherwise.

    several_blob_per_arena : bool
        Flag indicating if multiple blobs per arena are allowed.
        True to allow fragmentation, False otherwise.

    erodila_disk : NDArray[np.uint8]
        Disk used for erosion operations.
        Should be a valid structuring element.

    protect_from_fading : NDArray, optional
        An optional array to prevent certain pixels from being marked as fading.
        Should have the same dimensions as new_shape.

    add_to_fading : NDArray, optional
        An optional array to mark additional pixels as fading.
        Should have the same dimensions as new_shape.

    Returns
    -------
    new_shape : NDArray[np.uint8]
        Updated shape after applying fading detection and morphological operations.

    covering_intensity : NDArray
        Updated intensity values.
    """
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
            covering_intensity[add_to_fading_coord] = 1  # 0.9 * covering_intensity[add_to_fading_coord]  #
        else:
            covering_intensity[add_to_fading_coord] = 255  # 1.1 * covering_intensity[add_to_fading_coord]
    # With a lighter background, fading them if their intensity gets higher than the covering intensity
    if lighter_background:
        fading = fading * np.greater(greyscale_image, (1 - fading_coefficient) * covering_intensity).astype(np.uint8)
    else:
        fading = fading * np.less(greyscale_image, (1 + fading_coefficient) * covering_intensity).astype(np.uint8)

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