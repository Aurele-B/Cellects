#!/usr/bin/env python3
"""
Generate synthetic binary test images.

Provides utilities to generate binary images or coordinates to generate binary images.
It can be either random either deterministic.

Functions
---------
sim_noisy_circle : Create a noisy circular binary image.
random_blob_coord : Sample top‑left coordinates for square blobs.

"""
from __future__ import annotations
import cv2
import numpy as np
from numpy.typing import NDArray
from cellects.image.morphological_operations import create_ellipse, get_contours
import itertools


def sim_noisy_circle(size: int) -> NDArray[np.uint8]:
    """
    Generate a binary image containing a noisy circular ellipse.

    The function creates a solid ellipse, extracts its contour, randomly
    removes a subset of contour pixels, and then applies a series of
    erosions whose kernel size grows with ``size``.  This yields a
    disc‑like shape with irregular edges, useful for testing image
    processing pipelines.

    Parameters
    ----------
    size : int
        The height and width of the output square image.  Must be a
        positive integer; larger values produce a smoother overall shape
        but increase processing time.

    Returns
    -------
    noisy_circle : NDArray[np.uint8]
        A ``size`` × ``size`` array of type ``uint8`` where pixel values
        are ``0`` (background) or ``255`` (foreground).  The foreground
        approximates a circle with random gaps and optional erosion.

    Notes
    -----
    * The erosion kernels are square arrays of ones whose side length is
      derived from ``log10(size)``; for ``size`` > 30 two additional
      directional erosions are applied.
    * The function relies on OpenCV (``cv2``) for the erosion operation.

    Examples
    --------
    >>> noisy = sim_noisy_circle(20)
    >>> noisy.shape
    (20, 20)
    """
    noisy_circle = create_ellipse(size, size, min_size=3).astype(np.uint8)
    contour = get_contours(noisy_circle)
    contour_coord = np.nonzero(contour)
    sample = np.random.choice(np.arange(len(contour_coord[0])), len(contour_coord[0]) // 3)
    noisy_circle[contour_coord[0][sample], contour_coord[1][sample]] = 0
    if size > 15:
        coef = int(round(np.log10(size))) * 2 // 2 + 3
        noisy_circle = cv2.erode(noisy_circle, np.ones((coef , coef), dtype=np.uint8), iterations=1)
        if size > 30:
            noisy_circle = cv2.erode(noisy_circle, np.ones((coef + 1, 1), dtype=np.uint8), iterations=1)
            noisy_circle = cv2.erode(noisy_circle, np.ones((1, coef + 1), dtype=np.uint8), iterations=1)
    return noisy_circle


def random_blob_coord(im_size: int, blob_size: int, blob_nb: int) -> np.ndarray:
    """
    Generate random top‑left coordinates for a set of square blobs within an
    image.

    Parameters
    ----------
    im_size : int
        Size of the (square) image in pixels (height = width = ``im_size``).
    blob_size : int
        Edge length of each square blob in pixels.
    blob_nb : int
        Number of blob positions to sample.

    Returns
    -------
    sample_coord : np.ndarray
        ``numpy.ndarray`` of shape ``(blob_nb, 2)`` containing the ``(row,
        column)`` coordinates of the selected blob corners.  The dtype is
        ``uint32``.

    Notes
    -----
    The function first builds all possible top‑left positions that keep a full
    blob inside the image, mirrors them to include both ``(y, x)`` and
    ``(x, y)`` orientations, and then draws ``blob_nb`` unique positions
    without replacement.

    Examples
    --------
    >>> coords = random_blob_coord(128, 16, 3)
    >>> print(coords)
    [[ 32, 64]
     [ 80, 16]
     [ 48, 96]]
    """
    possible_coord = np.arange(0, im_size, blob_size)
    possible_coord = possible_coord[possible_coord < im_size - blob_size]
    possible_coord = np.array(list(itertools.combinations(possible_coord, 2)))
    all_coord = np.zeros((possible_coord.shape[0] * 2, possible_coord.shape[1]), np.uint32)
    all_coord[:possible_coord.shape[0], :] = possible_coord
    all_coord[possible_coord.shape[0]:, :] = possible_coord[:, ::-1]
    sample_coord = np.random.choice(np.arange(len(all_coord)), blob_nb, replace=False)
    sample_coord = all_coord[sample_coord, :]
    return sample_coord
