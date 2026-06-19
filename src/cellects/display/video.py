#!/usr/bin/env python3
"""
This script contains functions display videos
"""
import numpy as np
import cv2
from cellects.utils.formulas import bracket_to_uint8_image_contrast


def movie(video, increase_contrast: bool=True):
    """
    Summary
    -------
    Processes a video to display each frame with optional contrast increase and resizing.

    Parameters
    ----------
    video : numpy.ndarray
        The input video represented as a 3D NumPy array.
    increase_contrast : bool, optional
        Flag to increase the contrast of each frame (default is True).

    Other Parameters
    ----------------
    keyboard : int, optional
        Key to wait for during the display of each frame.
    increase_contrast : bool, optional
        Whether to increase contrast for the displayed frames.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `video` is not a 3D NumPy array.

    Notes
    -----
    This function uses OpenCV's `imshow` to display each frame. Ensure that the required
    OpenCV dependencies are met.

    Examples
    --------
    >>> movie(video)
    Processes and displays a video with default settings.
    >>> movie(video, keyboard=0)
    Processes and displays a video waiting for the SPACE key between frames.
    >>> movie(video, increase_contrast=False)
    Processes and displays a video without increasing contrast.

    """
    for i in np.arange(video.shape[0]):
        image = video[i, :, :]
        if np.any(image):
            if increase_contrast:
                image = bracket_to_uint8_image_contrast(image)
            final_img = cv2.resize(image, (500, 500))
            cv2.imshow('Motion analysis', final_img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
