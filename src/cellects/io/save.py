#!/usr/bin/env python3
"""
This script contains functions save various files.

For example:
    - videos (e.g. mp4)
    - arrays (h5)
    - images (jpg)
"""
import logging
import os
import h5py
import json
import numpy as np
from numpy.typing import NDArray
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from cellects.utils.utilitarian import insensitive_glob
from cellects.io.load import get_h5_keys, list_image_dir, is_raw_image, read_and_rotate, read_rotate_crop_and_reduce_image



def write_video(np_array: NDArray[np.uint8], vid_name: str, is_color: bool=True, fps: int=40):
    """
    Write video from numpy array.

    Save a numpy array as a video file. Supports .h5 format for saving raw
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
    >>> write_video(video_array, 'raw_data.h5')
    Saves `video_array` as a raw numpy array file without frame rate.
    """
    #h265 ou h265 (mp4)
    # linux: fourcc = 0x00000021 -> don't forget to change it bellow as well
    if vid_name[-4:] == '.h5':
        write_h5(vid_name, np_array, 'video')
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


def write_video_sets(img_list: list, sizes: NDArray, vid_names: list, crop_coord, bounding_boxes,
                      bunch_nb: int, video_nb_per_bunch: int, remaining: int,
                      raw_images: bool, is_landscape: bool, use_list_of_vid: bool,
                      in_colors: bool=False, reduce_image_dim: bool=False, pathway: str=""):
    """
    Write video sets from a list of images, applying cropping and optional rotation.

    Parameters
    ----------
    img_list : list
        List of image file names.
    sizes : NDArray
        Array containing the dimensions of each video frame.
    vid_names : list
        List of video file names to be saved.
    crop_coord : dict or tuple
        Coordinates for cropping regions of interest in images/videos.
    bounding_boxes : tuple
        Bounding box coordinates to extract sub-images from the original images.
    bunch_nb : int
        Number of bunches to divide the videos into.
    video_nb_per_bunch : int
        Number of videos per bunch.
    remaining : int
        Number of videos remaining after the last full bunch.
    raw_images : bool
        Whether the images are in raw format.
    is_landscape : bool
        If true, rotate the images to landscape orientation before processing.
    use_list_of_vid : bool
        Flag indicating if the output should be a list of videos.
    in_colors : bool, optional
        If true, process images with color information. Default is False.
    reduce_image_dim : bool, optional
        If true, reduce image dimensions. Default is False.
    pathway : str, optional
        Path where the videos should be saved. Default is an empty string.
    """
    top, bot, left, right = bounding_boxes
    for bunch in np.arange(bunch_nb):
        print(f'\nSaving the video bunch n°{bunch + 1} (tot={bunch_nb})...', end=' ')
        if bunch == (bunch_nb - 1) and remaining > 0:
            arenas = np.arange(bunch * video_nb_per_bunch, bunch * video_nb_per_bunch + remaining, dtype=np.uint32)
        else:
            arenas = np.arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch, dtype=np.uint32)
        if use_list_of_vid:
            video_bunch = [np.zeros(sizes[i, :], dtype=np.uint8) for i in arenas]
        else:
            video_bunch = np.zeros(np.append(sizes[0, :], len(arenas)), dtype=np.uint8)
        prev_img = None
        images_done = bunch * len(img_list)
        for image_i, image_name in enumerate(img_list):
            img = read_and_rotate(image_name, prev_img, raw_images, is_landscape, crop_coord)
            prev_img = img.copy()
            if not in_colors and reduce_image_dim:
                img = img[:, :, 0]

            for arena_i, arena_name in enumerate(arenas):
                # arena_i = 0; arena_name = arena[arena_i]
                sub_img = img[top[arena_name]:bot[arena_name], left[arena_name]:right[arena_name], ...]
                if use_list_of_vid:
                    video_bunch[arena_i][image_i, ...] = sub_img
                else:
                    if len(video_bunch.shape) == 5:
                        video_bunch[image_i, :, :, :, arena_i] = sub_img
                    else:
                        video_bunch[image_i, :, :, arena_i] = sub_img
        for arena_i, arena_name in enumerate(arenas):
            if use_list_of_vid:
                 write_h5(pathway + vid_names[arena_name], video_bunch[arena_i], 'video')
            else:
                if len(video_bunch.shape) == 5:
                     write_h5(pathway + vid_names[arena_name], video_bunch[:, :, :, :, arena_i], 'video')
                else:
                     write_h5(pathway + vid_names[arena_name], video_bunch[:, :, :, arena_i], 'video')

def write_json(file_name: str, data: dict):
    with open(file_name, 'w') as f:
        json.dump(data, f)

def write_h5(file_name: str, table: NDArray, key: str="data"):
    """
    Write a file using the h5 format.

    Parameters
    ----------
    file_name : str
        Name of the file to write.
    table : NDArray[]
        An array.
    key: str
        The identifier of the data in this h5 file.
    """
    with h5py.File(file_name, 'a') as h5f:
        if key in h5f:
            del h5f[key]
        h5f.create_dataset(key, data=table)

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
    if os.path.isfile(file_name):
        with h5py.File(file_name, 'a') as h5f:  # Open in append mode to modify the file
            if key in h5f:
                del h5f[key]

def video_writing_decision(arena_nb: int, im_or_vid: int, overwrite_unaltered_videos: bool) -> bool:
    """
    Determine whether to write videos based on existing files and user preferences.

    Parameters
    ----------
    arena_nb : int
        Number of arenas to analyze.
    im_or_vid : int
        Indicates whether the analysis should be performed on images or videos.
    overwrite_unaltered_videos : bool
        Flag indicating whether existing unaltered videos should be overwritten.

    Returns
    -------
    bool
        True if videos should be written, False otherwise.
    """
    look_for_existing_videos = insensitive_glob('ind_' + '*' + '.h5')
    there_already_are_videos = len(look_for_existing_videos) > 0
    if there_already_are_videos:
        all_files_contain_video = np.all(['video' in get_h5_keys(vid_name) for vid_name in look_for_existing_videos])
        there_already_are_videos = all_files_contain_video and there_already_are_videos
        if not there_already_are_videos:
            look_for_existing_videos = []
        there_already_are_videos = len(look_for_existing_videos) == arena_nb and there_already_are_videos
    logging.info(f"Video files (h5) found: {len(look_for_existing_videos)} for {arena_nb} arenas to analyze")
    do_write_videos = not im_or_vid and (not there_already_are_videos or (there_already_are_videos and overwrite_unaltered_videos))
    return do_write_videos


def write_video_from_images(path_to_images='', vid_name: str='timelapse.mp4', fps: int=20, img_extension: str='',
                            img_radical: str='', crop_coord: list=None):
    """
    Write a video file from a sequence of images.

     Extended Description
     --------------------
     This function creates a video from a list of image files in the specified directory.
     To prevent the most comon issues:
     - The image list is sorted
     - mp4 files are removed
     - If they do not have the same orientation, rotate the images accordingly
     - Images are cropped
     - Color vs greyscale is automatically determined

     After processing, images are compiled into a video file.

     Parameters
     ----------
     path_to_images : str
         The directory where the images are located.
     vid_name : str, optional
         The name of the output video file. Default is 'video.mp4'.
     fps : int, optional
         The frames per second for the video. Default is 20.
     img_extension : str, optional
         The file extension of the images. Default is an empty string.
     img_radical : str, optional
         The common prefix of the image filenames. Default is an empty string.
     crop_coord : list, optional
         list containing four crop coordinates: [top, bot, left, right]. Default is None and takes the whole image.

     Examples
     --------
     >>> write_video_from_images('path/to/images', vid_name='timelapse.mp4')
     This will create a video file named 'timelapse.mp4' from the images in the specified directory.
    """
    if isinstance(path_to_images, str):
        path_to_images = Path(path_to_images)
    os.chdir(path_to_images)
    imgs = list_image_dir(path_to_images, img_extension, img_radical)
    is_raw = is_raw_image(imgs[0])
    image, prev_img = read_rotate_crop_and_reduce_image(imgs[0], crop_coord=crop_coord, raw_images=is_raw)
    is_landscape = image.shape[0] < image.shape[1]
    is_color: bool = True
    if len(image.shape) == 2:
        is_color = False
    video = np.zeros((len(imgs), image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i_, img in enumerate(imgs):
        video[i_], prev_img = read_rotate_crop_and_reduce_image(img, prev_img=prev_img, crop_coord=crop_coord,
                                                                raw_images=is_raw, is_landscape=is_landscape)
    write_video(video, vid_name=vid_name, is_color=is_color, fps=fps)

def save_im(img: NDArray, full_path: str=None, cmap=None):
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
    >>> save_im(img, 'test.png')
    Creates and saves a figure from the random image to 'test.png'.

    >>> save_im(img, 'colored_test.png', cmap='viridis')
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
    if np.min(img.shape) > 50:
        fig.tight_layout()
    if full_path is None:
        plt.show()
    else:
        fig.savefig(full_path, bbox_inches='tight', pad_inches=0., transparent=True, dpi=500)
        plt.close(fig)
        