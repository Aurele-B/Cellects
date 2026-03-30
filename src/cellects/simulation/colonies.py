#!/usr/bin/env python3
"""Generate synthetic colony growth videos.

This module provides a helper to synthesize a short video that mimics the
expansion of microbial colonies.  The implementation creates an initial
binary frame with a few random seed points, repeatedly dilates the mask
with a circular structuring element to simulate growth, and finally maps
the binary masks to RGB colours using :func:`colorize_mask`.

Functions
---------
growing_colonies : Generate a colony‑like video by applying dilation
    operations and random colour filling.
"""

import numpy as np
import cv2
from cellects.image.morphological_operations import create_ellipse
from cellects.simulation.coloring import colorize_mask

def growing_colonies():
    """
    Generate a colony-like video by applying dilation operations and random color filling.
    This function creates a binary video with randomized initial frames, dilates the
    frames using a circular kernel to simulate colony growth over time, and then converts
    the binary video into a colored RGB video.

    returns
    -------
    rgb_video : numpy.ndarray
        A video with shape `(20, 1000, 1000, 3)` where each frame is represented in RGB format.
        The video shows the growth and coloration of the colony over time.

    Examples
    --------
    >>> colony_rgb_video = growing_colonies()
    >>> print(colony_rgb_video.shape)
    (20, 1000, 1000, 3)
    """
    np.random.seed(42)
    ellipse = create_ellipse(7, 7).astype(np.uint8)
    binary_video = np.zeros((20, 1000, 1000), dtype=np.uint8)
    binary_video[0, np.random.randint(100, 900, 20), np.random.randint(100, 900, 20)] = 1
    binary_video[0, ...] = cv2.dilate(binary_video[0, ...], ellipse)
    for t in range(1, binary_video.shape[0]):
        binary_video[t, ...] = cv2.dilate(binary_video[t - 1, ...], ellipse, iterations=1)
    rgb_video = colorize_mask(binary_video)
    return rgb_video