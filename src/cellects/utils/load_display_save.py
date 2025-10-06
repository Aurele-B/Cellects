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

    Attributes
    ----------
    wait_for_pickle_rick : bool
        Flag indicating if the lock file is present.
    counter : int
        Counter to track the number of operations performed.
    pickle_rick_number : str
        Unique identifier for the lock file.
    first_check_time : float
        Timestamp of the first check for the lock file.

    """
    def __init__(self, pickle_rick_number=""):
        self.wait_for_pickle_rick: bool = False
        self.counter = 0
        self.pickle_rick_number = pickle_rick_number
        self.first_check_time = default_timer()

    def check_that_file_is_not_open(self):
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

        Examples
        --------
        >>> pickle_rick_instance.check_that_file_is_not_open()
        """
        if os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl"):
            if default_timer() - self.first_check_time > 2:
                os.remove(f"PickleRick{self.pickle_rick_number}.pkl")
            # logging.error((f"Cannot read/write, Trying again... tip: unlock by deleting the file named PickleRick{self.pickle_rick_number}.pkl"))
        self.wait_for_pickle_rick = os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl")

    def write_pickle_rick(self):
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
        >>> obj.write_pickle_rick()     # Call the method to create and write to file
        """
        try:
            with open(f"PickleRick{self.pickle_rick_number}.pkl", 'wb') as file_to_write:
                pickle.dump({'wait_for_pickle_rick': True}, file_to_write)
        except Exception as exc:
            logging.error(f"Don't know how but Pickle Rick failed... Error is: {exc}")

    def delete_pickle_rick(self):
        """

        Delete a specific Pickle Rick file.

        Deletes the pickle file associated with the current instance's
        `pickle_rick_number`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the file with name `PickleRick{self.pickle_rick_number}.pkl` does not exist.

        Notes
        -----
        This function attempts to delete the specified pickle file.
        If the file does not exist, a `FileNotFoundError` will be raised.

        Examples
        --------
        >>> obj = PickleRick()
        >>> obj.pickle_rick_number = 1  # Set an example value for the attribute
        >>> obj.write_pickle_rick()
        >>> delete_pickle_rick()
        >>> os.path.isfile("PickleRick1.pkl")
        False
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

        Other Parameters
        ----------------
        None

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
        some internal state, represented by `delete_pickle_rick`.

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
                self.delete_pickle_rick()
            # time.sleep(np.random.choice(np.arange(1, os.cpu_count(), 0.5)))
            self.check_that_file_is_not_open()
            if self.wait_for_pickle_rick:
                time.sleep(2)
                self.write_file(file_content, file_name)
            else:
                self.write_pickle_rick()
                try:
                    with open(file_name, 'wb') as file_to_write:
                        pickle.dump(file_content, file_to_write, protocol=0)
                    self.delete_pickle_rick()
                    logging.info(f"Success to write file")
                except Exception as exc:
                    logging.error(f"The Pickle error on the file {file_name} is: {exc}")
                    self.delete_pickle_rick()
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
                self.delete_pickle_rick()
            self.check_that_file_is_not_open()
            if self.wait_for_pickle_rick:
                time.sleep(2)
                self.read_file(file_name)
            else:
                self.write_pickle_rick()
                try:
                    with open(file_name, 'rb') as fileopen:
                        file_content = pickle.load(fileopen)
                except Exception as exc:
                    logging.error(f"The Pickle error on the file {file_name} is: {exc}")
                    file_content = None
                self.delete_pickle_rick()
                if file_content is None:
                    self.read_file(file_name)
                else:
                    logging.info(f"Success to read file")
                return file_content
        else:
            logging.error(f"Failed to read {file_name}")


def show(img, interactive=True, cmap=None):
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
    >>> img = np.random.rand(100, 100)
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
    fig = plt.figure(figsize=(sizes[0], sizes[1]))
    ax = fig.gca()
    if cmap is None:
        ax.imshow(img, interpolation="none")
    else:
        ax.imshow(img, cmap=cmap, interpolation="none")
    fig.tight_layout()
    fig.show()
    return fig, ax

def save_fig(img, full_path, cmap=None):
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


def write_video(np_array, vid_name, is_color=True, fps=40):
    """
    Write a video file from an array of images.

    This function saves the provided NumPy array as either a `.npy` file
    or encodes it into a video format such as .mp4, .avi, or .mkv based on
    the provided file name and other parameters.

    Parameters
    ----------
    np_array : numpy.ndarray
        A 4-d array representing a sequence of images. The shape should be
        (num_frames, height, width, channels) where `channels` is 3 for color
        images and 1 for grayscale.

    vid_name : str
        The name of the output file. If the extension is `.npy`, the array will be
        saved in NumPy's binary format. Otherwise, a video file with the specified
        extension will be created.

    is_color : bool, optional
        Whether the images are in color. Default is ``True``.

    fps : int, optional
        Frames per second for the video. Default is ``40``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the provided extension is not supported (.mp4, .avi, or .mkv).

    Notes
    -----
    - The function supports `.mp4`, `.avi`, and `.mkv` extensions for video files.
    - When specifying an extension, make sure it matches the intended codec.

    Examples
    --------
    >>> import numpy as np

    Create a dummy array of shape (10, 480, 640, 3):
    >>> dummy_array = np.random.rand(10, 480, 640, 3)

    Save it as a video file:
    >>> write_video(dummy_array, 'output.mp4')

    Save it as a .npy file:
    >>> write_video(dummy_array, 'data.npy')

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


def video2numpy(vid_name, conversion_dict=None, background=None, true_frame_width=None):
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
    

def movie(video, keyboard=1, increase_contrast=True):
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


def is_raw_image(image_path):
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


def readim(image_path, raw_image=False):
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
            [  0,   0, 255]]], dtype=uint8)
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


def read_and_rotate(image_name, prev_img, raw_images, is_landscape):
    """
    Reads an image from the given source and rotates it 90 degrees clockwise or counterclockwise if necessary.

    Parameters
    ----------
    image_name : str
        The name or path of the image to be read.
    prev_img : np.ndarray or None
        The previous image in int16 format to compare differences, if applicable.
    raw_images : dict
        A dictionary containing raw images for the given `image_name`.
    is_landscape : bool
        If True, assumes the image should be in landscape orientation.

    Returns
    -------
    np.ndarray
        The processed and potentially rotated image.

    Raises
    ------
    FileNotFoundError
        If the specified `image_name` does not exist in `raw_images`.
    ValueError
        If the image dimensions are inconsistent during rotation operations.

    Notes
    -----
    - This function assumes that raw images are stored in the `raw_images` dictionary with keys as image names.
    - Rotation decisions are based on whether the image is required to be in landscape orientation.

    Examples
    --------
    >>> img = read_and_rotate("sample.jpg", prev_img=None, raw_images=False, is_landscape=True)
    Rotated image of sample.jpg
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


def vstack_h5_array(file_name, table, key="data"):
    """
    Append a new table to an existing HDF5 dataset or create a new one if it doesn't exist.

    Given a file name, table data and an optional key, this function will
    check for existence of the HDF5 file. If it exists, append to the dataset
    identified by `key` in the file. Otherwise create a new HDF5 file and dataset.

    Parameters
    ----------
    file_name : str
        The name of the HDF5 file.
    table : np.ndarray
        New data to be added or stored in the HDF5 file.
    key : str, optional
        The dataset name within the HDF5 file. Default is "data".

    Returns
    -------
    None

    Raises
    ------
    OSError
        If there is an issue accessing the file system, e.g., due to permission errors.
    IOError
        If there is an issue writing the HDF5 file.

    Notes
    -----
    The dataset will be appended to if it already exists. If the file does not exist,
    a new HDF5 file will be created and the dataset will be initialized with `table`.

    Examples
    --------
    >>> import numpy as np
    >>> table1 = np.array([[1, 2], [3, 4]])
    >>> vstack_h5_array('example.h5', table1) # create file and dataset
    >>> table2 = np.array([[5, 6], [7, 8]])
    >>> vstack_h5_array('example.h5', table2) # append to dataset
    >>> with h5py.File('example.h5', 'r') as f:
    ...     print(f['data'][:])
    [[1 2]
     [3 4]
     [5 6]
     [7 8]]
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


def read_h5_array(file_name, key="data"):
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

    Raises
    ------
    KeyError
        If the specified dataset key does not exist in the HDF5 file.
    FileNotFoundError
        If the specified HDF5 file does not exist.

    Examples
    --------
    >>> data = read_h5_array('example.h5', 'data')
    >>> print(data)
    [[1 2 3]
     [4 5 6]]

    >>> data = read_h5_array('example.h5')
    >>> print(data)
    [[7 8 9]
     [10 11 12]]
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

    Examples
    --------
    >>> result = get_h5_keys("example.hdf5")  # Ensure "example.hdf5" exists
    >>> print(result)
    ['data', 'metadata']
    """
    try:
        with h5py.File(file_name, 'r') as h5f:
            all_keys = list(h5f.keys())
            return all_keys
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")


def remove_h5_key(file_name, key="data"):
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

    Examples
    --------
    >>> remove_h5_key("example.h5", "data")
    """
    try:
        with h5py.File(file_name, 'a') as h5f:  # Open in append mode to modify the file
            if key in h5f:
                del h5f[key]
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")


def get_mpl_colormap(cmap_name):
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