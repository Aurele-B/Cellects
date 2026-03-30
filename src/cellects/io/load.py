#!/usr/bin/env python3
"""
This script contains functions to load various files.

For example,
    - Images (e.g., tif, jpg)
    - Videos (e.g., mp4)
    - Arrays (h5)
"""

import logging
import os
import h5py
import json
import numpy as np
from numpy.typing import NDArray
from natsort import natsorted
import cv2
from pathlib import Path
import exifread
from exif import Image
import tifffile
from cellects.image.image_segmentation import generate_color_space_combination
from cellects.utils.formulas import sum_of_abs_differences
from cellects.utils.utilitarian import split_dict, insensitive_glob


def video2numpy(vid_name: str, conversion_dict=None, background: NDArray=None, background2: NDArray=None,
                true_frame_width: int=None):
    """
    Convert a video file to a NumPy array.

    Parameters
    ----------
    vid_name : str
        The path to the video file. Can be a `.mp4` or `.h5`.
    conversion_dict : dict, optional
        Dictionary containing color space conversion parameters.
    background : NDArray, optional
        Background image for processing.
    background2 : NDArray, optional
        Second background image for processing.
    true_frame_width : int, optional
        True width of the frame. If specified and the current width is double this value,
        adjusts to true_frame_width.

    Returns
    -------
    NDArray or tuple of NDArrays
        If conversion_dict is None, returns the video as a NumPy array.
        Otherwise, returns a tuple containing the original video and converted video.

    Notes
    -----
    This function uses OpenCV to read the contents of a `.mp4` video file.
    """
    h5_loading = vid_name[-3:] == ".h5"
    tif_loading = vid_name[-3:] == ".tif" or vid_name[-3:] == ".tiff"
    if h5_loading:
        video = read_h5(vid_name, 'video')
        dims = list(video.shape)
    elif tif_loading:
        video = read_tif_stack(vid_name)
        dims = video.shape
    else:
        cap = cv2.VideoCapture(vid_name)
        dims = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]

    if true_frame_width is not None:
        if dims[2] == 2 * true_frame_width:
            dims[2] = true_frame_width

    if conversion_dict is not None:
        first_dict, second_dict, c_spaces = split_dict(conversion_dict)
        converted_video = np.empty(dims[:3], dtype=np.uint8)
        if conversion_dict['logical'] == 'None':
            converted_video2 = np.empty(dims[:3], dtype=np.uint8)
        if h5_loading:
            for counter in np.arange(video.shape[0]):
                img = video[counter, :, :dims[2], :]
                greyscale_image, greyscale_image2, all_c_spaces, first_pc_vector = generate_color_space_combination(img, c_spaces,
                                                                                     first_dict, second_dict, background=background,background2=background2,
                                                                                     convert_to_uint8=True)
                converted_video[counter, ...] = greyscale_image
                if conversion_dict['logical'] == 'None':
                    converted_video2[counter, ...] = greyscale_image2
            video = video[:, :, :dims[2], ...]

    if not h5_loading:
        # 2) Create empty arrays to store video analysis data
        video = np.empty((dims[0], dims[1], dims[2], 3), dtype=np.uint8)
        # 3) Read and convert the video frame by frame
        counter = 0
        while cap.isOpened() and counter < dims[0]:
            ret, frame = cap.read()
            frame = frame[:, :dims[2], ...]
            video[counter, ...] = frame
            if conversion_dict is not None:
                greyscale_image, greyscale_image2, all_c_spaces, first_pc_vector = generate_color_space_combination(frame, c_spaces,
                                                                                     first_dict, second_dict, background=background,background2=background2,
                                                                                     convert_to_uint8=True)
                converted_video[counter, ...] = greyscale_image
                if conversion_dict['logical'] == 'None':
                    converted_video2[counter, ...] = greyscale_image2
            counter += 1
        cap.release()

    if conversion_dict is None:
        return video
    else:
        return video, converted_video


opencv_accepted_formats = [
    'bmp', 'BMP', 'dib', 'DIB', 'exr', 'EXR', 'hdr', 'HDR', 'jp2', 'JP2',
    'jpe', 'JPE', 'jpeg', 'JPEG', 'jpg', 'JPG', 'pbm', 'PBM', 'pfm', 'PFM',
    'pgm', 'PGM', 'pic', 'PIC', 'png', 'PNG', 'pnm', 'PNM', 'ppm', 'PPM',
    'ras', 'RAS', 'sr', 'SR', 'tif', 'TIF', 'tiff', 'TIFF', 'webp', 'WEBP'
    ]


def is_raw_image(image_path) -> bool:
    """
    Determine if the image path corresponds to a raw image.

    Parameters
    ----------
    image_path : str
        The file path of the image.

    Returns
    -------
    bool
        True if the image is considered raw, False otherwise.

    Examples
    --------
    >>> result = is_raw_image("image.jpg")
    >>> print(result)
    False
    """
    ext = image_path.split(".")[-1]
    if np.isin(ext, opencv_accepted_formats):
        raw_image = False
    else:
        raw_image = True
    return raw_image

def readim(image_path, raw_image: bool=False):
    """
    Read an image from a file and optionally process it.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    raw_image : bool, optional
        If True, logs an error message indicating that the raw image format cannot be processed. Default is False.

    Returns
    -------
    ndarray
        The decoded image represented as a NumPy array of shape (height, width, channels).

    Raises
    ------
    RuntimeError
        If `raw_image` is set to True, logs an error indicating that the raw image format cannot be processed.

    Notes
    -----
    Although `raw_image` is set to False by default, currently it does not perform any raw image processing.

    Examples
    --------
    >>> cv2.imread("example.jpg")
    array([[[255, 0, 0],
            [255, 0, 0]],

           [[  0, 255, 0],
            [  0, 255, 0]],

           [[  0,   0, 255],
            [  0,   0, 255]]], dtype=np.uint8)
    """
    if raw_image:
        logging.error("Cannot read this image format. If the rawpy package can, ask for a version of Cellects using it.")
        # import rawpy
        # raw = rawpy.imread(image_path)
        # raw = raw.postprocess()
        # return cv2.cvtColor(raw, COLOR_RGB2BGR)
        return cv2.imread(image_path)
    else:
        return cv2.imread(image_path)


def read_and_rotate(image_name, prev_img: NDArray=None, raw_images: bool=False, is_landscape: bool=True, crop_coord: NDArray=None) -> NDArray:
    """
    Read and rotate an image based on specified parameters.

    This function reads an image from the given file name, optionally rotates
    it by 90 degrees clockwise or counterclockwise based on its dimensions and
    the `is_landscape` flag, and applies cropping if specified. It also compares
    rotated images against a previous image to choose the best rotation.

    Parameters
    ----------
    image_name : str
        Name of the image file to read.
    prev_img : ndarray, optional
        Previous image for comparison. Default is `None`.
    raw_images : bool, optional
        Flag to read raw images. Default is `False`.
    is_landscape : bool, optional
        Flag to determine if the image should be considered in landscape mode.
        Default is `True`.
    crop_coord : ndarray, optional
        Coordinates for cropping the image. Default is `None`.

    Returns
    -------
    ndarray
        Rotated and optionally cropped image.

    Raises
    ------
    FileNotFoundError
        If the specified image file does not exist.

    Examples
    ------
    >>> pathway = Path(__name__).resolve().parents[0] / "data" / "single_experiment"
    >>> image_name = 'image1.tif'
    >>> image = read_and_rotate(pathway /image_name)
    >>> print(image.shape)
    (245, 300, 3)
    """
    if not os.path.exists(image_name):
        raise FileNotFoundError(image_name)
    img = readim(image_name, raw_images)
    if (img.shape[0] > img.shape[1] and is_landscape) or (img.shape[0] < img.shape[1] and not is_landscape):
        clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if crop_coord is not None:
            clockwise = clockwise[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
        if prev_img is not None:
            prev_img = np.int16(prev_img)
            clock_diff = sum_of_abs_differences(prev_img, np.int16(clockwise))
            counter_clockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if crop_coord is not None:
                counter_clockwise = counter_clockwise[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
            counter_clock_diff = sum_of_abs_differences(prev_img, np.int16(counter_clockwise))
            if clock_diff > counter_clock_diff:
                img = counter_clockwise
            else:
                img = clockwise
        else:
            img = clockwise
    else:
        if crop_coord is not None:
            img = img[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ...]
    return img

def read_rotate_crop_and_reduce_image(image_name: str, prev_img: NDArray=None, crop_coord: list=None, cr: list=None,
                                      raw_images: bool=False, is_landscape: bool=True, reduce_image_dim: bool=False):
    """
    Reads, rotates, crops (if specified), and reduces image dimensionality if required.

    Parameters
    ----------
    image_name : str
        Name of the image file to read.
    prev_img : NDArray
        Previous image array used for rotation reference or state tracking.
    crop_coord : list
        List of four integers [x_start, x_end, y_start, y_end] specifying cropping region. If None, no initial crop is applied.
    cr : list
        List of four integers [x_start, x_end, y_start, y_end] for final cropping after rotation.
    raw_images : bool
        Flag indicating whether to process raw image data (True) or processed image (False).
    is_landscape : bool
        Boolean determining if the image is landscape-oriented and requires specific rotation handling.
    reduce_image_dim : bool
        Whether to reduce the cropped image to a single channel (e.g., grayscale from RGB).

    Returns
    -------
    img : NDArray
        Processed image after rotation, cropping, and optional dimensionality reduction.
    prev_img : NDArray
        Copy of the image immediately after rotation but before any cropping operations.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(200, 300, 3)
    >>> new_img, prev = read_rotate_crop_and_reduce_image("example.jpg", img, [50, 150, 75, 225], [20, 180, 40, 250], False, True, True)
    >>> new_img.shape == (160, 210)
    True
    >>> prev.shape == (200, 300, 3)
    True
    """
    img = read_and_rotate(image_name, prev_img, raw_images, is_landscape)
    prev_img = img.copy()
    if crop_coord is not None:
        img = img[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], :]
    if cr is not None:
        img = img[cr[0]:cr[1], cr[2]:cr[3], :]
    if reduce_image_dim:
        img = img[:, :, 0]
    return img, prev_img


def read_json(file_name: str):
    if os.path.isfile(file_name):
        try:
            with open(file_name) as f:
                data = json.load(f)
            return data
        except:
            return None
    else:
        return None


def read_h5(file_name, key: str="data"):
    """
    Read data array from an HDF5 file.

    This function reads a specific dataset from an HDF5 file using the provided key.

    Parameters
    ----------
    file_name : str
        The path to the HDF5 file.
    key : str, optional, default: 'data'
        The dataset name within the HDF5 file.

    Returns
    -------
    ndarray
        The data array from the specified dataset in the HDF5 file.
    """
    if os.path.isfile(file_name):
        with h5py.File(file_name, 'r') as h5f:
            if key in h5f:
                data = h5f[key][:]
                return data
            else:
                return None
    else:
        return None


def get_h5_keys(file_name):
    """
    Retrieve all keys from a given HDF5 file.

    Parameters
    ----------
    file_name : str
        The path to the HDF5 file from which keys are to be retrieved.

    Returns
    -------
    list of str
        A list containing all the keys present in the specified HDF5 file.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    """
    try:
        with h5py.File(file_name, 'r') as h5f:
            all_keys = list(h5f.keys())
            return all_keys
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")

def read_tif_stack(vid_name: str, expected_channels: int=1):
    """
        Read video array from a tif file.

        This function reads a specific dataset from a tif file.

        Parameters
        ----------
        vid_name : str
            The path to the tif stack file.
        expected_channels : int
            The number of channel.

        Returns
        -------
        ndarray
            The data array from the tif file.
    """
    all_frames = None
    if os.path.isfile(vid_name):
        with tifffile.TiffFile(vid_name) as tif:
            # Count the number of pages (frames and channels)
            num_pages = len(tif.pages)

            # Determine the shape of a single frame
            example_page = tif.pages[0]
            height, width = example_page.asarray().shape

            # Calculate the number of frames per channel based on expected_channels parameter
            frames_per_channel = num_pages // expected_channels

            # Initialize an array to hold all frames for each channel
            all_frames = np.zeros((frames_per_channel, height, width, expected_channels),
                                  dtype=example_page.asarray().dtype)

            # Read and store each frame
            for i in range(frames_per_channel):
                for channel in range(expected_channels):
                    page_index = i * expected_channels + channel
                    frame = tif.pages[page_index].asarray()
                    all_frames[i, :, :, channel] = frame
    return all_frames

def list_image_dir(path_to_images='', img_extension: str='', img_radical: str='') -> list:
    """
    List files in an image directory based on optional naming patterns (extension and/or radical).

    Parameters
    ----------
    path_to_images : optional
        The path to the directory containing images. Default is an empty string.
    img_extension : str, optional
        The file extension of the images to be listed. Default is an empty string.
        When let empty, use the extension corresponding to the most numerous image file in the folder.
    img_radical : str, optional
        The radical part of the filenames to be listed. Default is an empty string.

    Returns
    -------
    list
        A list of image filenames that match the specified criteria,
        sorted in a natural order.

    Notes
    -----
    This function uses the `natsorted` and `insensitive_glob` utilities to ensure
    that filenames are sorted in a human-readable order.

    Examples
    --------
    >>> pathway = Path(__name__).resolve().parents[0] / "data" / "single_experiment"
    >>> image_list = list_image_dir(pathway)
    >>> print(image_list)
    """
    if isinstance(path_to_images, str):
        path_to_images = Path(path_to_images)
    os.chdir(path_to_images)
    if len(img_extension) == 0:
        imgs = insensitive_glob(f'{img_radical}*')
        matches = np.zeros(len(opencv_accepted_formats))
        for e_i, ext in enumerate(opencv_accepted_formats):
            matches[e_i] = np.char.endswith(imgs, ext).sum()
        img_extension = opencv_accepted_formats[np.argmax(matches)]
    imgs = insensitive_glob(f'{img_radical}*{img_extension}')
    imgs = natsorted(imgs)
    return imgs


def extract_time(pathway="", image_list: list=None, raw_images:bool=False):
    """
    Extract timestamps from a list of images.

    This function extracts the DateTimeOriginal or datetime values from
    the EXIF data of a list of image files, and computes the total time in seconds.

    Parameters
    ----------
    pathway : str, optional
        Path to the directory containing the images. Default is an empty string.
    image_list : list of str
        List of image file names.
    raw_images : bool, optional
        If True, use the exifread library. Otherwise, use the exif library.
        Default is False.

    Returns
    -------
    time : ndarray of int64
        Array containing the total time in seconds for each image.

    Examples
    --------
    >>> pathway = Path(__name__).resolve().parents[0] / "data" / "single_experiment"
    >>> image_list = ['image1.tif', 'image2.tif']
    >>> time = extract_time(pathway, image_list)
    >>> print(time)
    array([0, 0])

    """
    if isinstance(pathway, str):
        pathway = Path(pathway)
    os.chdir(pathway)
    if image_list is None:
        image_list = list_image_dir(pathway)
    nb = len(image_list)
    timings = np.zeros((nb, 6), dtype=np.int64)
    if raw_images:
        for i in np.arange(nb):
            with open(image_list, 'rb') as image_file:
                my_image = exifread.process_file(image_file, details=False, stop_tag='DateTimeOriginal')
                datetime = my_image["EXIF DateTimeOriginal"]
            datetime = datetime.values[:10] + ':' + datetime.values[11:]
            timings[i, :] = datetime.split(':')
    else:
        for i in np.arange(nb):
            with open(image_list[i], 'rb') as image_file:
                my_image = Image(image_file)
                if my_image.has_exif:
                    datetime = my_image.datetime
                    datetime = datetime[:10] + ':' + datetime[11:]
                    timings[i, :] = datetime.split(':')

    if np.all(timings[:, 0] == timings[0, 0]):
        if np.all(timings[:, 1] == timings[0, 1]):
            if np.all(timings[:, 2] == timings[0, 2]):
                time = timings[:, 3] * 3600 + timings[:, 4] * 60 + timings[:, 5]
            else:
                time = timings[:, 2] * 86400 + timings[:, 3] * 3600 + timings[:, 4] * 60 + timings[:, 5]
        else:
            days_per_month = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
            for j in np.arange(nb):
                month_number = timings[j, 1]#int(timings[j, 1])
                timings[j, 1] = days_per_month[month_number] * month_number
            time = (timings[:, 1] + timings[:, 2]) * 86400 + timings[:, 3] * 3600 + timings[:, 4] * 60 + timings[:, 5]
        #time = int(time)
    else:
        time = np.repeat(0, nb)#arange(1, nb * 60, 60)#"Do not experiment the 31th of december!!!"
    if time.sum() == 0:
        time = np.repeat(0, nb)#arange(1, nb * 60, 60)
    return time


def read_one_arena(arena_label, already_greyscale:bool, csc_dict: dict, videos_already_in_ram=None,
                   true_frame_width=None, vid_name: str=None, background: NDArray=None, background2: NDArray=None):
    """
    Read a single arena's video data, potentially converting it from color to greyscale.

    Parameters
    ----------
    arena_label : int
        The label of the arena.
    already_greyscale : bool
        Whether the video is already in greyscale format.
    csc_dict : dict
        Dictionary containing color space conversion settings.
    videos_already_in_ram : np.ndarray, optional
        Pre-loaded video frames in memory. Default is None.
    true_frame_width : int, optional
        The true width of the video frames. Default is None.
    vid_name : str, optional
        Name of the video file. Default is None.
    background : np.ndarray, optional
        Background image for subtractions. Default is None.
    background2 : np.ndarray, optional
        Second background image for subtractions. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
            - visu: np.ndarray or None, the visual frame.
            - converted_video: np.ndarray or None, the video data converted as needed.
            - converted_video2: np.ndarray or None, additional video data if necessary.

    Raises
    ------
    FileNotFoundError
        If the specified video file does not exist.
    ValueError
        If the video data shape is invalid.

    Notes
    -----
    This function assumes that `video2numpy` is a helper function available in the scope.
    For optimal performance, ensure all video data fits in RAM.
    """
    visu, converted_video, converted_video2 = None, None, None
    logging.info(f"Arena n°{arena_label}. Load images and videos")
    if videos_already_in_ram is not None:
        if already_greyscale:
            converted_video = videos_already_in_ram
        else:
            if csc_dict['logical'] == 'None':
                visu, converted_video = videos_already_in_ram
            else:
                visu, converted_video, converted_video2 = videos_already_in_ram
    else:
        if vid_name is not None:
            if already_greyscale:
                converted_video = video2numpy(vid_name, None, background, background2, true_frame_width)
                if len(converted_video.shape) == 4:
                    converted_video = converted_video[:, :, :, 0]
            else:
                visu = video2numpy(vid_name, None, background, background2, true_frame_width)
        else:
            vid_name = f"ind_{arena_label}.h5"
            if os.path.isfile(vid_name):
                h5_keys = get_h5_keys(vid_name)
                if os.path.isfile(vid_name) and 'video' in h5_keys:
                    if already_greyscale:
                        converted_video = video2numpy(vid_name, None, background, background2, true_frame_width)
                        if len(converted_video.shape) == 4:
                            converted_video = converted_video[:, :, :, 0]
                    else:
                        visu = video2numpy(vid_name, None, background, background2, true_frame_width)
    return visu, converted_video, converted_video2


def create_empty_videos(image_list: list, cr: list, lose_accuracy_to_save_memory: bool,
                        already_greyscale: bool, csc_dict: dict):
    """

    Create empty video arrays based on input parameters.

    Parameters
    ----------
    image_list : list
        List of images.
    cr : list
        Crop region defined by [x_start, y_start, x_end, y_end].
    lose_accuracy_to_save_memory : bool
        Boolean flag to determine if memory should be saved by using uint8 data type.
    already_greyscale : bool
        Boolean flag indicating if the images are already in greyscale format.
    csc_dict : dict
        Dictionary containing color space conversion settings, including 'logical' key.

    Returns
    -------
    tuple
        A tuple containing three elements:
            - `visu`: NumPy array with shape (len(image_list), cr[1] - cr[0] + 1, cr[3] - cr[2] + 1, 3) and dtype uint8 for RGB images.
            - `converted_video`: NumPy array with shape (len(image_list), cr[1] - cr[0] + 1, cr[3] - cr[2] + 1) and dtype uint8 or float according to `lose_accuracy_to_save_memory`.
            - `converted_video2`: NumPy array with shape same as `converted_video` and dtype uint8 or float according to `lose_accuracy_to_save_memory`.

    Notes
    -----
    Performance considerations:
        - If `lose_accuracy_to_save_memory` is True, the function uses np.uint8 for memory efficiency.
        - If `already_greyscale` is False, additional arrays are created to store RGB data.
    """
    visu, converted_video, converted_video2 = None, None, None
    dims = len(image_list), cr[1] - cr[0], cr[3] - cr[2]
    if lose_accuracy_to_save_memory:
        converted_video = np.zeros(dims, dtype=np.uint8)
    else:
        converted_video = np.zeros(dims, dtype=float)
    if not already_greyscale:
        visu = np.zeros((dims[0], dims[1], dims[2], 3), dtype=np.uint8)
        if csc_dict['logical'] != 'None':
            if lose_accuracy_to_save_memory:
                converted_video2 = np.zeros(dims, dtype=np.uint8)
            else:
                converted_video2 = np.zeros(dims, dtype=float)
    return visu, converted_video, converted_video2
