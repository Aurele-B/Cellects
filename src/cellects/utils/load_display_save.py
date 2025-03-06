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
from numpy import any, uint8, unique, load, zeros, arange, empty, save, int16, isin, vstack, nonzero, concatenate
from cv2 import getStructuringElement, MORPH_CROSS, morphologyEx, MORPH_GRADIENT, rotate, ROTATE_90_COUNTERCLOCKWISE, ROTATE_90_CLOCKWISE, cvtColor, COLOR_RGB2BGR, imread, VideoWriter_fourcc, VideoWriter, imshow, waitKey, destroyAllWindows, resize, VideoCapture, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.image_segmentation import combine_color_spaces, get_color_spaces, generate_color_space_combination
from cellects.utils.formulas import bracket_to_uint8_image_contrast, sum_of_abs_differences
from cellects.utils.utilitarian import translate_dict


class PickleRick:
    def __init__(self, pickle_rick_number=""):
        self.wait_for_pickle_rick: bool = False
        self.counter = 0
        self.pickle_rick_number = pickle_rick_number
        self.first_check_time = default_timer()

    def check_that_file_is_not_open(self):
        """
        This method checks whether the file is open in another thread,
        if so, there is a PickleRickX.pkl
        :return: None
        """
        if os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl"):
            if default_timer() - self.first_check_time > 2:
                os.remove(f"PickleRick{self.pickle_rick_number}.pkl")
            # logging.error((f"Cannot read/write, Trying again... tip: unlock by deleting the file named PickleRick{self.pickle_rick_number}.pkl"))
        self.wait_for_pickle_rick = os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl")

    def write_pickle_rick(self):
        """
        This method checks a PickleRickX.pkl to flag that the file is open
        :return: None
        """
        try:
            with open(f"PickleRick{self.pickle_rick_number}.pkl", 'wb') as file_to_write:
                pickle.dump({'wait_for_pickle_rick': True}, file_to_write)
        except Exception as exc:
            logging.error(f"Don't know how but Pickle Rick failed... Error is: {exc}")

    def delete_pickle_rick(self):
        """
        This method deletes the PickleRickX.pkl once the file is closed
        :return: None
        """
        if os.path.isfile(f"PickleRick{self.pickle_rick_number}.pkl"):
            os.remove(f"PickleRick{self.pickle_rick_number}.pkl")

    def write_file(self, file_content, file_name):
        """
        This method write a file safely
        :param file_content: Any data to write
        :param file_name: A string containing the file name
        :type file_name: str
        :return: None
        """
        self.counter += 1
        if self.counter < 100:
            if self.counter > 95:
                self.delete_pickle_rick()
            # time.sleep(random.choice(arange(1, os.cpu_count(), 0.5)))
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
        This method read a file safely
        :param file_name: A string containing the file name
        :type file_name: str
        :return: whatever file that has been pickled
        """
        self.counter += 1
        if self.counter < 1000:
            if self.counter > 950:
                self.delete_pickle_rick()
            # time.sleep(random.choice(arange(1, os.cpu_count(), 0.5)))
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


def See(image, img_name="", size=None, keep_display=0):
    """
    Display an image using opencv
    :param image: the image to display, if not uint8, will be converted
    :type image: uint8
    :param img_name:
    :type img_name: str
    :param size: two element list saying the size of the image during display
    :type size: int
    :return:
    """
    # image = resize(image, (960, 540))
    if image.dtype != 'uint8':
        image = image.astype(uint8)
    if size is None:
        size = (1000, 1000)
    image = resize(image, size)
    if not isinstance(image, uint8):
        image = image.astype(uint8)
    img_content_diversity = len(unique(image))
    if img_content_diversity < 10:
        image *= 255 // img_content_diversity
    imshow(img_name, image)
    waitKey(keep_display)
    if not keep_display:
        destroyAllWindows()


def write_video(np_array, vid_name, is_color=True, fps=40):
    """
    Write a video on hard drive
    :param np_array: the video to write
    :type np_array: uint8
    :param vid_name: path and file name of the video to write
    :type vid_name: str
    :param is_color: if True, the fourth dimension of the array is the color
    :type is_color: bool
    :param fps: frame per second
    :type fps: uint64
    :return: 
    """
    #h265 ou h265 (mp4)
    # linux: fourcc = 0x00000021 -> don't forget to change it bellow as well
    if vid_name[-4:] == '.npy':
        with open(vid_name, 'wb') as file:
            save(file, np_array)
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
            fourcc = VideoWriter_fourcc('F', 'F', 'V', '1')  # lossless
        size = np_array.shape[2], np_array.shape[1]
        vid = VideoWriter(vid_name, fourcc, float(fps), tuple(size), is_color)
        for image_i in arange(np_array.shape[0]):
            image = np_array[image_i, ...]
            vid.write(image)
        vid.release()


def video2numpy(vid_name, conversion_dict=None, background=None, true_frame_width=None):
    """
    Read a video from hard drive
    :param vid_name: path and file name of the video to read
    :type vid_name: str
    :param conversion_dict: dictionary containing the color space combination to modify the bgr image before writing
    :type conversion_dict: TDict[str: float64]
    :param background: grayscale image
    :type background: uint8
    :param true_frame_width: widht of one frame, if the video is twice that width returns the left side of it
    :type true_frame_width: int
    :return: a numpy array of the video and its converted version if required
    """
    if vid_name[-4:] == ".npy":
        video = load(vid_name) # , allow_pickle='TRUE'
        frame_width = video.shape[2]
        if true_frame_width is not None:
            if frame_width == 2 * true_frame_width:
                frame_width = true_frame_width
        if conversion_dict is not None:
            converted_video = zeros((video.shape[0], video.shape[1], frame_width), dtype=uint8)
            for counter in arange(video.shape[0]):
                img = video[counter, :, :frame_width, :]
                greyscale_image, greyscale_image2 = generate_color_space_combination(img, list(conversion_dict.keys()),
                                                                                     conversion_dict, background=background,
                                                                                     convert_to_uint8=True)
                converted_video[counter, ...] = greyscale_image
                # csc = OneImageAnalysis(csc)
                # csc.generate_color_space_combination(conversion_dict, background)
                # converted_video[counter, ...] = csc.image.astype(uint8)
        video = video[:, :, :frame_width, ...]
    else:

        cap = VideoCapture(vid_name)
        frame_number = int(cap.get(CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(CAP_PROP_FRAME_WIDTH))
        if true_frame_width is not None:
            if frame_width == 2 * true_frame_width:
                frame_width = true_frame_width

        # 2) Create empty arrays to store video analysis data

        video = empty((frame_number, frame_height, frame_width, 3), dtype=uint8)
        if conversion_dict is not None:
            converted_video = empty((frame_number, frame_height, frame_width), dtype=uint8)
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
    Display a 3D or 4D numpy array as a video in a new window
    :param video: a numpy array with time, cy, cx
    :param keyboard: If 1, the video will automatically switch from one frame to the next,
    if 0, the user will have to press a key (e.g. Enter) to switch from one frame to the next
    :type keyboard: int
    :param increase_contrast: If True, an algorithm that increases image contrast will be applied
    :type increase_contrast: bool
    :return: 
    """
    for i in arange(video.shape[0]):
        image = video[i, :, :]
        if any(image):
            if increase_contrast:
                image = bracket_to_uint8_image_contrast(image)
            final_img = resize(image, (500, 500))
            imshow('Motion analysis', final_img)
            waitKey(keyboard)
    destroyAllWindows()


opencv_accepted_formats = [
    'bmp', 'BMP', 'dib', 'DIB', 'exr', 'EXR', 'hdr', 'HDR', 'jp2', 'JP2',
    'jpe', 'JPE', 'jpeg', 'JPEG', 'jpg', 'JPG', 'pbm', 'PBM', 'pfm', 'PFM',
    'pgm', 'PGM', 'pic', 'PIC', 'png', 'PNG', 'pnm', 'PNM', 'ppm', 'PPM',
    'ras', 'RAS', 'sr', 'SR', 'tif', 'TIF', 'tiff', 'TIFF', 'webp', 'WEBP'
    ]


def is_raw_image(image_path):
    """
    This function checks whether an image is in raw format
    :param image_path: path (and image name) toward the image to read
    :type image_path: str
    :return: True if it is a raw image
    :rtype: bool
    """
    ext = image_path.split(".")[-1]
    if isin(ext, opencv_accepted_formats):
        raw_image = False
    else:
        raw_image = True
    return raw_image


def readim(image_path, raw_image=False):
    """
    Read an image
    Uses opencv for usual images and rawpy for raw images
    :param image_path: path (and image name) toward the image to read
    :type image_path: str
    :param raw_image: True if the image is in raw format
    :type raw_image: bool
    :return:
    """
    if raw_image:
        logging.error("Cannot read this image format. If the rawpy package can, ask for a version of Cellects using it.")
        # import rawpy
        # raw = rawpy.imread(image_path)
        # raw = raw.postprocess()
        # return cvtColor(raw, COLOR_RGB2BGR)
        return imread(image_path)
    else:
        return imread(image_path)


def read_and_rotate(image_name, prev_img, raw_images, is_landscape):
    """
    Read an image and correct its orientation if necessary
    :param image_name: path (and image name) toward the image to read
    :param prev_img: reference image to use to orentate the current image correctly
    :param raw_images: True if the image is in raw format
    :type raw_image: bool
    :param is_landscape: True if the image is landscape
    :type raw_image: bool
    :return:
    """
    img = readim(image_name, raw_images)
    if (img.shape[0] > img.shape[1] and is_landscape) or (img.shape[0] < img.shape[1] and not is_landscape):
        clockwise = rotate(img, ROTATE_90_CLOCKWISE)
        if prev_img is not None:
            prev_img = int16(prev_img)
            clock_diff = sum_of_abs_differences(prev_img, int16(clockwise))
            # clock_diff = sum(absolute(int16(prev_img) - int16(clockwise)))
            counter_clockwise = rotate(img, ROTATE_90_COUNTERCLOCKWISE)
            counter_clock_diff = sum_of_abs_differences(prev_img, int16(counter_clockwise))
            # counter_clock_diff = sum(absolute(int16(prev_img) - int16(counter_clockwise)))
            if clock_diff > counter_clock_diff:
                img = counter_clockwise
            else:
                img = clockwise
        else:
            img = clockwise
    return img


def vstack_h5_array(file_name, table, key="data"):
    """
        Append a DataFrame to an HDF5 file. If the file or dataset doesn't exist, it creates them.
        :param file_name: str, path to the HDF5 file
        :param table: pd.DataFrame, the DataFrame to append
        :param key: str, the dataset key in the HDF5 file
    """
    if os.path.exists(file_name):
        # Open the file in append mode
        with h5py.File(file_name, 'a') as h5f:
            if key in h5f:
                # Append to the existing dataset
                existing_data = h5f[key][:]
                new_data = vstack((existing_data, table))
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
    Reads a NumPy array from a specified key in an HDF5 file.

    :param file_name: str, path to the HDF5 file
    :param key: str, name of the array to read
    :return: np.ndarray, the data from the specified key
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
    try:
        with h5py.File(file_name, 'r') as h5f:
            all_keys = list(h5f.keys())
            return all_keys
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")


def remove_h5_key(file_name, key="data"):
    """
    Removes a specific key from an HDF5 file.

    :param file_name: str, path to the HDF5 file
    :param key: str, name of the dataset to remove
    """
    try:
        with h5py.File(file_name, 'a') as h5f:  # Open in append mode to modify the file
            if key in h5f:
                del h5f[key]
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")