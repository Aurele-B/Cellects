#!/usr/bin/env python3
"""Contains the function: cell_leaving_detection"""
from cv2 import (
    connectedComponents, connectedComponentsWithStats, MORPH_CROSS,
    getStructuringElement, CV_16U, erode, dilate, morphologyEx, MORPH_OPEN,
    MORPH_CLOSE, MORPH_GRADIENT, BORDER_CONSTANT, resize, imshow, waitKey, destroyAllWindows,
    FONT_HERSHEY_SIMPLEX, putText)
from numpy import (
    c_, char, floor, pad, append, round, ceil, uint64, float32, absolute, sum,
    mean, median, quantile, ptp, diff, square, sqrt, convolve, gradient, zeros,
    ones, empty, array, arange, nonzero, min, max, argmin, argmax, unique,
    isin, repeat, tile, stack, concatenate, logical_and, logical_or,
    logical_xor, logical_not, less, greater, any, sign, uint8, int8, int16,
    uint32, float64, expand_dims, min, max, all, any)

def cell_leaving_detection(new_shape, covering_intensity, previous_binary, greyscale_image, fading_coefficient, lighter_background, several_blob_per_arena, erodila_disk, cross_33, protect_from_fading=None, add_to_fading=None):
    # To look for fading pixels, only consider the internal contour of the shape at t-1
    fading = erode(previous_binary, erodila_disk)
    # fading = logical_xor(self.binary[self.t - 1, :, :], fading)
    fading = previous_binary - fading
    # If the origin state is considered constant: origin pixels will never be fading
    if protect_from_fading is not None:
        fading *= (1 - protect_from_fading)
    if add_to_fading is not None:
        if protect_from_fading is not None:
            add_to_fading[nonzero(protect_from_fading)] = 0
        add_to_fading_coord = nonzero(add_to_fading)
        fading[add_to_fading_coord] = 1
        if lighter_background:
            covering_intensity[add_to_fading_coord] = 1 / (1 - fading_coefficient)  # 0.9 * covering_intensity[add_to_fading_coord]  #
        else:
            covering_intensity[add_to_fading_coord] = 255  # 1.1 * covering_intensity[add_to_fading_coord]
    # With a lighter background, fading them if their intensity gets higher than the covering intensity
    if lighter_background:
        # self.covering_intensity[new_idx[0], new_idx[1]] = ref[-round(self.vars['fading'] * 100), :]
        fading = fading * greater((greyscale_image), (1 - fading_coefficient) * covering_intensity).astype(uint8)
    else:
        # self.covering_intensity[new_idx[0], new_idx[1]] = ref[round(self.vars['fading'] * 100), :]
        fading = fading * less((greyscale_image), (1 + fading_coefficient) * covering_intensity).astype(uint8)

    if any(fading):
        fading = morphologyEx(fading, MORPH_CLOSE, cross_33, iterations=1)
        if not several_blob_per_arena:
            # Check if uncov_potentials does not break the shape into several components
            uncov_nb, uncov_shapes = connectedComponents(fading, ltype=CV_16U)
            nb, garbage_img = connectedComponents(new_shape, ltype=CV_16U)
            i = 0
            while i <= uncov_nb:
                i += 1
                prev_nb = nb
                new_shape[nonzero(uncov_shapes == i)] = 0
                nb, garbage_img = connectedComponents(new_shape, ltype=CV_16U)
                if nb > prev_nb:
                    new_shape[nonzero(uncov_shapes == i)] = 1
                    nb, garbage_img = connectedComponents(new_shape, ltype=CV_16U)
            uncov_shapes = None
        else:
            new_shape[nonzero(fading)] = 0
        # covering_intensity *= 1 - new_shape  # NEW
        new_shape = morphologyEx(new_shape, MORPH_OPEN, cross_33, iterations=0)
        new_shape = morphologyEx(new_shape, MORPH_CLOSE, cross_33)

    return new_shape, covering_intensity