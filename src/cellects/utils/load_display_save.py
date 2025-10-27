#!/usr/bin/env python3
"""
This script contains functions and classes to load, display and save various files
For example:
    - PickleRick: to write and read files without conflicts
    - See: Display an image using opencv
    - write_video: Write a video on hard drive
"""
import logging
import os
import pickle
import time
import h5py
from timeit import default_timer
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import cv2
from numpy import any, unique, load, zeros, arange, empty, save, int16, isin, vstack, nonzero, concatenate, linspace
from cv2 import VideoWriter, imshow, waitKey, destroyAllWindows, resize, VideoCapture, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH
from matplotlib import pyplot as plt
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.image_segmentation import combine_color_spaces, get_color_spaces, generate_color_space_combination
from cellects.utils.formulas import bracket_to_uint8_image_contrast, sum_of_abs_differences
from cellects.utils.utilitarian import translate_dict


class PickleRick:
    """
    A class to handle safe file reading and writing operations using pickle.

    This class ensures that files are not being accessed concurrently by
    creating a lock file (PickleRickX.pkl) to signal that the file is open.
    It includes methods to check for the lock file, write data safely,
    and read data safely.
    """
    def __init__(self, pickle_rick_number=""):
        """
        Initialize a new instance of the class.

        This constructor sets up initial attributes for tracking Rick's state, including
        a boolean flag for waiting for Pickle Rick, a counter, the provided pickle Rick number,
        and the time when the first check was performed.

        Parameters
        ----------
        pickle_rick_number : str, optional
            The number associated with Pickle Rick. Defaults to an empty string.
        """
        self.wait_for_pickle_rick: bool = False
        self.counter = 0
        self.pickle_rick_number = pickle_rick_number
        self.first_check_time = default_timer()

    def _check_that_file_is_not_open(self):
        """
        Check if a specific pickle file exists and handle it accordingly.

        This function checks whether a file named `PickleRick{self.pickle_rick_number}.pkl`
        exists. If the file has not been modified for more than 2 seconds, it is removed.
        The function then updates an attribute to indicate whether the file exists.

        Parameters
        ----------
        self : PickleRickObject
            The instance of the class containing this method.

        Returns
        -------
        None
            This function does not return any value.
            It updates the `self.wait_for_pickle_rick` attribute.

        Notes
        -----
        This function removes the pickle file if it has not been modified for more than 2 seconds.
        The `self.wait_for_pickle_rick` attribute is updated based on the existence of the file.
        """
        if os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl"):
            if default_timer() - self.first_check_time > 2:
                os.remove(f"PickleRick{self.pickle_rick_number}.pkl")
            # logging.error((f"Cannot read/write, Trying again... tip: unlock by deleting the file named PickleRick{self.pickle_rick_number}.pkl"))
        self.wait_for_pickle_rick = os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl")

    def _write_pickle_rick(self):
        """
        Write pickle data to a file for Pickle Rick.

        Parameters
        ----------
        self : object
            The instance of the class that this method belongs to.
            This typically contains attributes and methods relevant to managing
            pickle operations for Pickle Rick.

        Raises
        ------
        Exception
            General exception raised if there is any issue with writing the file.
            The error details are logged.

        Notes
        -----
        This function creates a file named `PickleRick{self.pickle_rick_number}.pkl`
        with a dictionary indicating readiness for Pickle Rick.

        Examples
        --------
        >>> obj = PickleRick()  # Assuming `YourClassInstance` is the class containing this method
        >>> obj.pickle_rick_number = 1  # Set an example value for the attribute
        >>> obj._write_pickle_rick()     # Call the method to create and write to file
        """
        try:
            with open(f"PickleRick{self.pickle_rick_number}.pkl", 'wb') as file_to_write:
                pickle.dump({'wait_for_pickle_rick': True}, file_to_write)
        except Exception as exc:
            logging.error(f"Don't know how but Pickle Rick failed... Error is: {exc}")

    def _delete_pickle_rick(self):
        """

        Delete a specific Pickle Rick file.

        Deletes the pickle file associated with the current instance's
        `pickle_rick_number`.

        Raises
        ------
        FileNotFoundError
            If the file with name `PickleRick{self.pickle_rick_number}.pkl` does not exist.
        """
        if os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl"):
            os.remove(f"PickleRick{self.pickle_rick_number}.pkl")

    def write_file(self, file_content, file_name):
        """
        Write content to a file with error handling and retry logic.

        This function attempts to write the provided content into a file.
        If it fails, it retries up to 100 times with some additional checks
        and delays. Note that the content is serialized using pickle.

        Parameters
        ----------
        file_content : Any
            The data to be written into the file. This will be pickled.
        file_name : str
            The name of the file where data should be written.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the file cannot be written after 100 attempts, an error is logged.

        Notes
        -----
        This function uses pickle to serialize the data, which can introduce security risks
        if untrusted content is being written. It performs some internal state checks,
        such as verifying that the target file isn't open and whether it should delete
        some internal state, represented by `_delete_pickle_rick`.

        The function implements a retry mechanism with a backoff strategy that can include
        random delays, though the example code does not specify these details explicitly.

        Examples
        --------
        >>> result = PickleRick().write_file({'key': 'value'}, 'test.pkl')
        Success to write file
        """
        self.counter += 1
        if self.counter < 100:
            if self.counter > 95:
                self._delete_pickle_rick()
            # time.sleep(np.random.choice(np.arange(1, os.cpu_count(), 0.5)))
            self._check_that_file_is_not_open()
            if self.wait_for_pickle_rick:
                time.sleep(2)
                self.write_file(file_content, file_name)
            else:
                self._write_pickle_rick()
                try:
                    with open(file_name, 'wb') as file_to_write:
                        pickle.dump(file_content, file_to_write, protocol=0)
                    self._delete_pickle_rick()
                    logging.info(f"Success to write file")
                except Exception as exc:
                    logging.error(f"The Pickle error on the file {file_name} is: {exc}")
                    self._delete_pickle_rick()
                    self.write_file(file_content, file_name)
        else:
            logging.error(f"Failed to write {file_name}")

    def read_file(self, file_name):
        """
        Reads the contents of a file using pickle and returns it.

        Parameters
        ----------
        file_name : str
            The name of the file to be read.

        Returns
        -------
        Union[Any, None]
            The content of the file if successfully read; otherwise, `None`.

        Raises
        ------
        Exception
            If there is an error reading the file.

        Notes
        -----
        This function attempts to read a file multiple times if it fails.
        If the number of attempts exceeds 1000, it logs an error and returns `None`.

        Examples
        --------
        >>> PickleRick().read_file("example.pkl")
        Some content

        >>> read_file("non_existent_file.pkl")
        None
        """
        self.counter += 1
        if self.counter < 1000:
            if self.counter > 950:
                self._delete_pickle_rick()
            self._check_that_file_is_not_open()
            if self.wait_for_pickle_rick:
                time.sleep(2)
                self.read_file(file_name)
            else:
                self._write_pickle_rick()
                try:
                    with open(file_name, 'rb') as fileopen:
                        file_content = pickle.load(fileopen)
                except Exception as exc:
                    logging.error(f"The Pickle error on the file {file_name} is: {exc}")
                    file_content = None
                self._delete_pickle_rick()
                if file_content is None:
                    self.read_file(file_name)
                else:
                    logging.info(f"Success to read file")
                return file_content
        else:
            logging.error(f"Failed to read {file_name}")


def write_video(np_array: NDArray[np.uint8], vid_name: str, is_color: bool=True, fps: int=40):
    """
    Write video from numpy array.

    Save a numpy array as a video file. Supports .npy format for saving raw
    numpy arrays and various video formats (mp4, avi, mkv) using OpenCV.
    For video formats, automatically selects a suitable codec and handles
    file extensions.

    Parameters
    ----------
    np_array : ndarray of uint8
        Input array containing video frames.
    vid_name : str
        Filename for the output video. Can include extension or not (defaults to .mp4).
    is_color : bool, optional
        Whether the video should be written in color. Defaults to True.
    fps : int, optional
        Frame rate for the video in frames per second. Defaults to 40.

    Examples
    --------
    >>> video_array = np.random.randint(0, 255, size=(10, 100, 100, 3), dtype=np.uint8)
    >>> write_video(video_array, 'output.mp4', True, 30)
    Saves `video_array` as a color video 'output.mp4' with FPS 30.
    >>> video_array = np.random.randint(0, 255, size=(10, 100, 100), dtype=np.uint8)
    >>> write_video(video_array, 'raw_data.npy')
    Saves `video_array` as a raw numpy array file without frame rate.
    """
    #h265 ou h265 (mp4)
    # linux: fourcc = 0x00000021 -> don't forget to change it bellow as well
    if vid_name[-4:] == '.npy':
        with open(vid_name, 'wb') as file:
             np.save(file, np_array)
    else:
        valid_extensions = ['.mp4', '.avi', '.mkv']
        vid_ext = vid_name[-4:]
        if vid_ext not in valid_extensions:
            vid_name = vid_name[:-4]
            vid_name += '.mp4'
            vid_ext = '.mp4'
        if vid_ext =='.mp4':
            fourcc = 0x7634706d# VideoWriter_fourcc(*'FMP4') #(*'MP4V') (*'h265') (*'x264') (*'DIVX')
        else:
            fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')  # lossless
        size = np_array.shape[2], np_array.shape[1]
        vid = cv2.VideoWriter(vid_name, fourcc, float(fps), tuple(size), is_color)
        for image_i in np.arange(np_array.shape[0]):
            image = np_array[image_i, ...]
            vid.write(image)
        vid.release()


def video2numpy(vid_name: str, conversion_dict=None, background=None, true_frame_width=None):
    """
    Convert a video file to a NumPy array.

    This function reads a video file and converts it into a NumPy array.
    If a conversion dictionary is provided, the function also generates
    a converted version of the video using the specified color space conversions.
    If true_frame_width is provided, and it matches half of the actual frame width,
    the function adjusts the frame width accordingly.

    Parameters
    ----------
    vid_name : str
        Path to the video file or .npy file containing the video data.
    conversion_dict : dict, optional
        Dictionary specifying color space conversions. Default is None.
    background : bool, optional
        Whether to subtract the background from the video frames. Default is None.
    true_frame_width : int, optional
        The true width of the video frames. Default is None.

    Other Parameters
    ----------------
    background : bool, optional
        Whether to subtract the background from the video frames. Default is None.
    true_frame_width : int, optional
        The true width of the video frames. Default is None.

    Returns
    -------
    video : numpy.ndarray or tuple(numpy.ndarray, numpy.ndarray)
        If conversion_dict is None, returns the video as a NumPy array.
        Otherwise, returns a tuple containing the original and converted videos.

    Raises
    ------
    ValueError
        If the video file cannot be opened or if there is an error in processing.

    Notes
    -----
    - This function uses OpenCV to read video files.
    - If true_frame_width is provided and it matches half of the actual frame width,
      the function adjusts the frame width accordingly.
    - The conversion dictionary should contain color space mappings for transformation.

    Examples
    --------
    >>> vid_array = video2numpy('example_video.mp4')
    >>> print(vid_array.shape)
    (100, 720, 1280, 3)

    >>> vid_array, converted_vid = video2numpy('example_video.mp4', {'rgb': 'gray'}, True)
    >>> print(vid_array.shape, converted_vid.shape)
    (100, 720, 1280, 3) (100, 720, 640)

    >>> vid_array = video2numpy('example_video.npy')
    >>> print(vid_array.shape)
    (100, 720, 1920, 3)

    >>> vid_array = video2numpy('example_video.npy', true_frame_width=1920)
    >>> print(vid_array.shape)
    (100, 720, 960, 3)

    >>> vid_array = video2numpy('example_video.npy', {'rgb': 'gray'}, True, 960)
    >>> print(vid_array.shape)
    (100, 720, 960)"""
    if vid_name[-4:] == ".npy":
        video = np.load(vid_name) # , allow_pickle='TRUE'
        frame_width = video.shape[2]
        if true_frame_width is not None:
            if frame_width == 2 * true_frame_width:
                frame_width = true_frame_width
        if conversion_dict is not None:
            converted_video = np.zeros((video.shape[0], video.shape[1], frame_width), dtype=np.uint8)
            for counter in np.arange(video.shape[0]):
                img = video[counter, :, :frame_width, :]
                greyscale_image, greyscale_image2 = generate_color_space_combination(img, list(conversion_dict.keys()),
                                                                                     conversion_dict, background=background,
                                                                                     convert_to_uint8=True)
                converted_video[counter, ...] = greyscale_image
        video = video[:, :, :frame_width, ...]
    else:

        cap = cv2.VideoCapture(vid_name)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if true_frame_width is not None:
            if frame_width == 2 * true_frame_width:
                frame_width = true_frame_width

        # 2) Create empty arrays to store video analysis data

        video = np.empty((frame_number, frame_height, frame_width, 3), dtype=np.uint8)
        if conversion_dict is not None:
            converted_video = np.empty((frame_number, frame_height, frame_width), dtype=np.uint8)
        # 3) Read and convert the video frame by frame
        counter = 0
        while cap.isOpened() and counter < frame_number:
            ret, frame = cap.read()
            frame = frame[:, :frame_width, ...]
            video[counter, ...] = frame
            if conversion_dict is not None:
                conversion_dict = translate_dict(conversion_dict)
                c_spaces = get_color_spaces(frame, list(conversion_dict.keys()))
                csc = combine_color_spaces(conversion_dict, c_spaces, subtract_background=background)
                converted_video[counter, ...] = csc
            counter += 1
        cap.release()

    if conversion_dict is None:
        return video
    else:
        return video, converted_video
    

def movie(video, keyboard=1, increase_contrast: bool=True):
    """
    Summary
    -------
    Processes a video to display each frame with optional contrast increase and resizing.

    Parameters
    ----------
    video : numpy.ndarray
        The input video represented as a 3D NumPy array.
    keyboard : int, optional
        Key for waiting during display (default is 1).
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
            final_img = resize(image, (500, 500))
            cv2.imshow('Motion analysis', final_img)
            cv2.waitKey(keyboard)
    cv2.destroyAllWindows()


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


def read_and_rotate(image_name, prev_img: NDArray, raw_images: bool, is_landscape: bool):
    """
    Read and rotate an image based on landscape orientation.

    This function reads an image, checks its dimensions relative to the
    landscape orientation flag and rotates it if necessary. If a previous
    image is provided, it determines the optimal rotation to minimize
    differences between consecutive images.

    Parameters
    ----------
    image_name : str
        The name of the image file to read.
    prev_img : ndarray
        The previous image in int16 format for comparison. Defaults to None.
    raw_images : bool
        Flag indicating if the image should be read in raw format.
    is_landscape : bool
        Boolean flag indicating if the image should be considered as landscape.

    Returns
    -------
    img : ndarray
        The rotated image if necessary; otherwise, the original image.

    Examples
    --------
    >>> img = read_and_rotate("image1.tif", None, False, True)
    """
    img = readim(image_name, raw_images)
    if (img.shape[0] > img.shape[1] and is_landscape) or (img.shape[0] < img.shape[1] and not is_landscape):
        clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if prev_img is not None:
            prev_img = np.int16(prev_img)
            clock_diff = sum_of_abs_differences(prev_img, np.int16(clockwise))
            counter_clockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            counter_clock_diff = sum_of_abs_differences(prev_img, np.int16(counter_clockwise))
            if clock_diff > counter_clock_diff:
                img = counter_clockwise
            else:
                img = clockwise
        else:
            img = clockwise
    return img


def vstack_h5_array(file_name, table: NDArray, key: str="data"):
    """
    Stack tables vertically in an HDF5 file.

    This function either appends the input table to an existing dataset
    in the specified HDF5 file or creates a new dataset if the key doesn't exist.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file.
    table : NDArray[np.uint8]
        The table to be stacked vertically with the existing data.
    key : str, optional
        Key under which the dataset will be stored. Defaults to 'data'.

    Examples
    --------
    >>> table = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    >>> vstack_h5_array('example.h5', table)
    """
    if os.path.exists(file_name):
        # Open the file in append mode
        with h5py.File(file_name, 'a') as h5f:
            if key in h5f:
                # Append to the existing dataset
                existing_data = h5f[key][:]
                new_data = np.vstack((existing_data, table))
                del h5f[key]
                h5f.create_dataset(key, data=new_data)
            else:
                # Create a new dataset if the key doesn't exist
                h5f.create_dataset(key, data=table)
    else:
        with h5py.File(file_name, 'w') as h5f:
            h5f.create_dataset(key, data=table)


def read_h5_array(file_name, key: str="data"):
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
    try:
        with h5py.File(file_name, 'r') as h5f:
            if key in h5f:
                data = h5f[key][:]
                return data
            else:
                raise KeyError(f"Dataset '{key}' not found in file '{file_name}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")


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


def remove_h5_key(file_name, key: str="data"):
    """
    Remove a specified key from an HDF5 file.

    This function opens an HDF5 file in append mode and deletes the specified
    key if it exists. It handles exceptions related to file not found
    and other runtime errors.

    Parameters
    ----------
    file_name : str
        The path to the HDF5 file from which the key should be removed.
    key : str, optional
        The name of the dataset or group to delete from the HDF5 file.
        Default is "data".

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    RuntimeError
        If any other error occurs during file operations.

    Notes
    -----
    This function modifies the HDF5 file in place. Ensure you have a backup if necessary.
    """
    try:
        with h5py.File(file_name, 'a') as h5f:  # Open in append mode to modify the file
            if key in h5f:
                del h5f[key]
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")


def get_mpl_colormap(cmap_name: str):
    """
    Returns a linear color range array for the given matplotlib colormap.

    Parameters
    ----------
    cmap_name : str
        The name of the colormap to get.

    Returns
    -------
    numpy.ndarray
        A 256x1x3 array of bytes representing the linear color range.

    Examples
    --------
    >>> result = get_mpl_colormap('viridis')
    >>> print(result.shape)
    (256, 1, 3)

    """
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 1, 3)



def show(img, interactive: bool=True, cmap=None):
    """
    Display an image using Matplotlib with optional interactivity and colormap.

    Parameters
    ----------
    img : ndarray
        The image data to be displayed.
    interactive : bool, optional
        If ``True``, turn on interactive mode. Default is ``True``.
    cmap : str or Colormap, optional
        The colormap to be used. If ``None``, the default colormap will
        be used.

    Other Parameters
    ----------------
    interactive : bool, optional
        If ``True``, turn on interactive mode. Default is ``True``.
    cmap : str or Colormap, optional
        The colormap to be used. If ``None``, the default colormap will
        be used.

    Returns
    -------
    fig : Figure
        The Matplotlib figure object containing the displayed image.
    ax : AxesSubplot
        The axes on which the image is plotted.

    Raises
    ------
    ValueError
        If `cmap` is not a recognized colormap name or object.

    Notes
    -----
    If interactive mode is enabled, the user can manipulate the figure
    window interactively.

    Examples
    --------
    >>> img = np.random.rand(100, 50)
    >>> fig, ax = show(img)
    >>> print(fig) # doctest: +SKIP
    <Figure size ... with ... Axes>

    >>> fig, ax = show(img, interactive=False)
    >>> print(fig) # doctest: +SKIP
    <Figure size ... with ... Axes>

    >>> fig, ax = show(img, cmap='gray')
    >>> print(fig) # doctest: +SKIP
    <Figure size ... with .... Axes>
    """
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    sizes = img.shape[0] / 100,  img.shape[1] / 100
    fig = plt.figure(figsize=(sizes[1], sizes[0]))
    ax = fig.gca()
    if cmap is None:
        ax.imshow(img, interpolation="none", extent=(0, sizes[1], 0, sizes[0]))
    else:
        ax.imshow(img, cmap=cmap, interpolation="none", extent=(0, sizes[1], 0, sizes[0]))
    fig.tight_layout()
    fig.show()
    return fig, ax


def save_fig(img: NDArray, full_path, cmap=None):
    """
    Save an image figure to a file with specified options.

    This function creates a matplotlib figure from the given image,
    optionally applies a colormap, displays it briefly, saves the
    figure to disk at high resolution, and closes the figure.

    Parameters
    ----------
    img : array_like (M, N, 3)
        Input image to be saved as a figure. Expected to be in RGB format.
    full_path : str
        The complete file path where the figure will be saved. Must include
        extension (e.g., '.png', '.jpg').
    cmap : str or None, optional
        Colormap to be applied if the image should be displayed with a specific
        color map. If `None`, no colormap is applied.

    Returns
    -------
    None

        This function does not return any value. It saves the figure to disk
        at the specified location.

    Raises
    ------
    FileNotFoundError
        If the directory in `full_path` does not exist.

    Examples
    --------
    >>> img = np.random.rand(100, 100, 3) * 255
    >>> save_fig(img, 'test.png')
    Creates and saves a figure from the random image to 'test.png'.

    >>> save_fig(img, 'colored_test.png', cmap='viridis')
    Creates and saves a figure from the random image with 'viridis' colormap
    to 'colored_test.png'.
    """
    sizes = img.shape[0] / 100,  img.shape[1] / 100
    fig = plt.figure(figsize=(sizes[0], sizes[1]))
    ax = fig.gca()
    if cmap is None:
        ax.imshow(img, interpolation="none")
    else:
        ax.imshow(img, cmap=cmap, interpolation="none")
    plt.axis('off')
    fig.tight_layout()
    fig.show()
    fig.savefig(full_path, bbox_inches='tight', pad_inches=0., transparent=True, dpi=500)
    plt.close()


def display_boxes(binary_image: NDArray, box_diameter: int):
    """
    Display grid lines on a binary image at specified box diameter intervals.

    This function displays the given binary image with vertical and horizontal
    grid lines drawn at regular intervals defined by `box_diameter`. The function
    returns the total number of grid lines drawn.

    Parameters
    ----------
    binary_image : ndarray
        Binary image on which to draw the grid lines.
    box_diameter : int
        Diameter of each box in pixels.

    Returns
    -------
    line_nb : int
        Number of grid lines drawn, both vertical and horizontal.

    Examples
    --------
    >>> import numpy as np
    >>> binary_image = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    >>> display_boxes(binary_image, box_diameter=25)
    """
    plt.imshow(binary_image, cmap='gray', extent=(0, binary_image.shape[1], 0, binary_image.shape[0]))
    height, width = binary_image.shape
    line_nb = 0
    for x in range(0, width + 1, box_diameter):
        line_nb += 1
        plt.axvline(x=x, color='white', linewidth=1)
    for y in range(0, height + 1, box_diameter):
        line_nb += 1
        plt.axhline(y=y, color='white', linewidth=1)
    plt.show()
    return line_nb