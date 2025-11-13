"""Process image data to detect experiment arenas and generate cropped video arrays.

This module implements arena detection algorithms for experimental tracking, using initial images
to define region-of-interest boundaries. It provides fast/slow algorithms for arena localization
and handles memory constraints when writing output as .npy files.

Classes
OneVideoPerBlob : Finds bounding boxes containing blob motion over time and generates videos

Functions
Notes
Includes RAM usage monitoring during video processing
Provides both static shape-based and motion-tracking arena detection methods
Generates numpy arrays of cropped video regions to reduce file size
"""
import os
import logging
from copy import deepcopy
import numpy as np
import cv2
import psutil
from numpy.typing import NDArray
from cellects.image_analysis.morphological_operations import cross_33, Ellipse, get_minimal_distance_between_2_shapes, \
    rank_from_top_to_bottom_from_left_to_right, \
    expand_until_neighbor_center_gets_nearer_than_own, get_line_points
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.utils.load_display_save import read_and_rotate


class OneVideoPerBlob:
    """
        This class finds the bounding box containing all pixels covered by one blob over time
        and create a video from it.
        It does that, for each blob, considering a few information.
    """

    def __init__(self, first_image: object, starting_blob_hsize_in_pixels: int, raw_images: bool):
        """
        Initialize an instance of a class that processes video frames.

        Parameters
        ----------
        first_image : object
            The first frame in the video sequence.
        starting_blob_hsize_in_pixels : int
            The initial size of a blob in pixels.
        raw_images : bool
            Whether the images are raw.

        Attributes
        ----------
        ordered_first_image : None or object
            Placeholder for the ordered first image in the sequence.
        motion_list : list
            List to store motion events or states.
        shapes_to_remove : None or object
            Placeholder for shapes that need to be removed from analysis.
        not_analyzed_individuals : None or object
            Placeholder for individuals not yet analyzed.

        Notes
        -----
        This class initializes various variables and sets up arrays necessary for the
        analysis of video frames.

        """
        # Initialize all variables used in the following methods
        self.first_image = first_image
        if self.first_image.crop_coord is None:
            self.first_image.get_crop_coordinates()
        self.original_shape_hsize = starting_blob_hsize_in_pixels
        self.raw_images = raw_images
        if self.original_shape_hsize is not None:
            self.k_size = int((np.ceil(self.original_shape_hsize / 5) * 2) + 1)

        # 7) Create required empty arrays: especially the bounding box coordinates of each video
        self.ordered_first_image = None
        self.motion_list = list()
        self.shapes_to_remove = None
        self.not_analyzed_individuals = None

    def get_bounding_boxes(self, are_gravity_centers_moving: bool, img_list: list, color_space_combination: dict,
                           color_number: int=2, sample_size: int=5, all_specimens_have_same_direction: bool=True,
                           display: bool=False, filter_spec: dict=None):
        """
        Get the coordinates of all arenas using one or several image(s).

        Parameters
        ----------
        are_gravity_centers_moving : bool
            Flag indicating if the gravity centers are moving.
        img_list : list
            List of images to process.
        color_space_combination : dict
            Dictionary containing color space combinations.
        color_number : int, optional
            Number of colors to consider, by default 2.
        sample_size : int, optional
            Size of the sample, by default 5.
        all_specimens_have_same_direction : bool, optional
            Flag indicating if all specimens have the same direction, by default True.
        display : bool, optional
            Flag to display results, by default False.
        filter_spec : dict, optional
            Filter specifications, by default None.

        Notes
        -----
        - This function uses Numba's @njit decorator for performance.
        """
        logging.info("Get the coordinates of all arenas using the get_bounding_boxes method of the VideoMaker class")
        if self.first_image.validated_shapes.any():
            self.big_kernel = Ellipse((self.k_size, self.k_size)).create()
            self.big_kernel = self.big_kernel.astype(np.uint8)
            self.small_kernel = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)
            self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                self.first_image.validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
            self.unchanged_ordered_fimg = deepcopy(self.ordered_first_image)
            self.modif_validated_shapes = deepcopy(self.first_image.validated_shapes)
            self.standard = - 1
            counter = 0
            while np.any(np.less(self.standard, 0)) and counter < 20:
                counter += 1
                self.left = np.zeros(self.first_image.shape_number, dtype=np.int64)
                self.right = np.repeat(self.modif_validated_shapes.shape[1], self.first_image.shape_number)
                self.top = np.zeros(self.first_image.shape_number, dtype=np.int64)
                self.bot = np.repeat(self.modif_validated_shapes.shape[0], self.first_image.shape_number)
                if are_gravity_centers_moving:
                    self._get_bb_with_moving_centers(img_list, color_space_combination, color_number, sample_size,
                                                     all_specimens_have_same_direction, display,
                                                     filter_spec=filter_spec)
                    new_ordered_first_image = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)

                    for i in np.arange(1, self.first_image.shape_number + 1):
                        previous_shape = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
                        previous_shape[np.nonzero(self.unchanged_ordered_fimg == i)] = 1
                        new_potentials = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
                        new_potentials[np.nonzero(self.ordered_first_image == i)] = 1
                        new_potentials[np.nonzero(self.unchanged_ordered_fimg == i)] = 0

                        pads = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance=2)
                        pads.consider_shapes_sizes(min_shape_size=10)
                        pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False)
                        new_ordered_first_image[np.nonzero(pads.expanded_shape)] = i
                    self.ordered_first_image = new_ordered_first_image
                    self.modif_validated_shapes = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
                    self.modif_validated_shapes[np.nonzero(self.ordered_first_image)] = 1
                    self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                        self.modif_validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
                    self._get_quick_bb()
                else:
                    self._get_quick_bb()
                self.standardize_video_sizes()
            if counter == 20:
                self.top[self.top < 0] = 1
                self.bot[self.bot >= self.ordered_first_image.shape[0] - 1] = self.ordered_first_image.shape[0] - 2
                self.left[self.left < 0] = 1
                self.right[self.right >= self.ordered_first_image.shape[1] - 1] = self.ordered_first_image.shape[1] - 2
        else:
            self.top = np.array(0, dtype=np.int64)
            self.bot = np.array(self.first_image.image.shape[0], dtype=np.int64)
            self.left = np.array(0, dtype=np.int64)
            self.right = np.array(self.first_image.image.shape[1], dtype=np.int64)

    def _get_quick_bb(self):
        """
        Get quick bounding boxes for the validated shapes.

        This method calculates the minimal distance between pairs of components
        in the validated shapes and uses this information to determine bounding boxes.

        Notes
        -----
        This method modifies the following attributes:
        - `left`, `right`, `top`, and `bot` to store the coordinates of the bounding boxes.
        """
        shapes = deepcopy(self.modif_validated_shapes)
        eroded_shapes = cv2.erode(self.modif_validated_shapes, cross_33)
        shapes = shapes - eroded_shapes
        x_min = self.ordered_stats[:, 0]
        y_min = self.ordered_stats[:, 1]
        x_max = self.ordered_stats[:, 0] + self.ordered_stats[:, 2]
        y_max = self.ordered_stats[:, 1] + self.ordered_stats[:, 3]
        x_min_dist = shapes.shape[1]
        y_min_dist = shapes.shape[0]

        shapes *= self.ordered_first_image
        shape_nb = (len(np.unique(shapes)) - 1)
        i = 0
        a_indices, b_indices = np.triu_indices(shape_nb, 1)
        a_indices, b_indices = a_indices + 1, b_indices + 1
        all_distances = np.zeros((len(a_indices), 3), dtype=float)
        # For every pair of components, find the minimal distance
        for (a, b) in zip(a_indices, b_indices):
            x_dist = np.absolute(x_max[a - 1] - x_min[b - 1])
            y_dist = np.absolute(y_max[a - 1] - y_min[b - 1])
            if x_dist < 2 * x_min_dist and y_dist < 2 * y_min_dist:
                sub_shapes = np.logical_or(shapes == a, shapes == b) * shapes
                sub_shapes = sub_shapes[np.min((y_min[a - 1], y_min[b - 1])):np.max((y_max[a - 1], y_max[b - 1])),
                             np.min((x_min[a - 1], x_min[b - 1])):np.max((x_max[a - 1], x_max[b - 1]))]
                sub_shapes[sub_shapes == a] = 1
                sub_shapes[sub_shapes == b] = 2
                if np.any(sub_shapes == 1) and np.any(sub_shapes == 2):
                    all_distances[i, :] = a, b, get_minimal_distance_between_2_shapes(sub_shapes, False)

                    if x_dist > y_dist:
                        x_min_dist = np.min((x_min_dist, x_dist))
                    else:
                        y_min_dist = np.min((y_min_dist, y_dist))
                    i += 1
        for shape_i in np.arange(1, shape_nb + 1):
            # Get where the shape i appear in pairwise comparisons
            idx = np.nonzero(np.logical_or(all_distances[:, 0] == shape_i, all_distances[:, 1] == shape_i))
            # Compute the minimal distance related to shape i and divide by 2
            if len(all_distances[idx, 2]) > 0:
                dist = all_distances[idx, 2].min() // 2
            else:
                dist = 1
                # Save the coordinates of the arena around shape i
            self.left[shape_i - 1] = x_min[shape_i - 1] - dist.astype(np.int64)
            self.right[shape_i - 1] = x_max[shape_i - 1] + dist.astype(np.int64)
            self.top[shape_i - 1] = y_min[shape_i - 1] - dist.astype(np.int64)
            self.bot[shape_i - 1] = y_max[shape_i - 1] + dist.astype(np.int64)

    def standardize_video_sizes(self):
        """
        Standardize video sizes by adjusting bounding boxes.

        Extended Description
        --------------------
        This function adjusts the bounding boxes of detected shapes in a video frame.
        It ensures that all bounding boxes are within the frame's boundaries and
        standardizes their sizes to avoid issues with odd dimensions during video writing.

        Parameters
        ----------
        distance_threshold_to_consider_an_arena_out_of_the_picture : int, optional
            Threshold in pixels to consider a bounding box out of the picture.
            If `None`, defaults to the minimum value in `out_of_pic`.

        Returns
        -------
        None
            The function modifies the following attributes of the class instance:

        Attributes Modified
        ------------------
        standard : numpy.ndarray
            Standardized bounding boxes.
        shapes_to_remove : numpy.ndarray
            Indices of shapes to be removed from the image.
        modif_validated_shapes : numpy.ndarray
            Modified validated shapes after removing out-of-picture areas.
        ordered_stats : list of float
            Updated order statistics for the shapes.
        ordered_centroids : numpy.ndarray
            Centroids of the ordered shapes.
        ordered_first_image : numpy.ndarray
            First image with updated order statistics and centroids.
        first_image.shape_number : int
            Updated number of shapes in the first image.
        not_analyzed_individuals : numpy.ndarray
            Indices of individuals not analyzed after modifications.

        """
        distance_threshold_to_consider_an_arena_out_of_the_picture = None# in pixels, worked nicely with - 50

        # The modifications allowing to not make videos of setups out of view, do not work for moving centers
        y_diffs = self.bot - self.top
        x_diffs = self.right - self.left
        add_to_y = ((np.max(y_diffs) - y_diffs) / 2)
        add_to_x = ((np.max(x_diffs) - x_diffs) / 2)
        self.standard = np.zeros((len(self.top), 4), dtype=np.int64)
        self.standard[:, 0] = self.top - np.uint8(np.floor(add_to_y))
        self.standard[:, 1] = self.bot + np.uint8(np.ceil(add_to_y))
        self.standard[:, 2] = self.left - np.uint8(np.floor(add_to_x))
        self.standard[:, 3] = self.right + np.uint8(np.ceil(add_to_x))

        # Monitor if one bounding box gets out of picture shape
        out_of_pic = deepcopy(self.standard)
        out_of_pic[:, 1] = self.ordered_first_image.shape[0] - out_of_pic[:, 1] - 1
        out_of_pic[:, 3] = self.ordered_first_image.shape[1] - out_of_pic[:, 3] - 1

        if distance_threshold_to_consider_an_arena_out_of_the_picture is None:
            distance_threshold_to_consider_an_arena_out_of_the_picture = np.min(out_of_pic) - 1

        # If it occurs at least one time, apply a correction, otherwise, continue and write videos
        # If the overflow is strong, remove the corresponding individuals and remake bounding_box finding
        if np.any(np.less(out_of_pic, distance_threshold_to_consider_an_arena_out_of_the_picture)):
            # Remove shapes
            self.standard = - 1
            self.shapes_to_remove = np.nonzero(np.less(out_of_pic, - 20))[0]
            for shape_i in self.shapes_to_remove:
                self.ordered_first_image[self.ordered_first_image == (shape_i + 1)] = 0
            self.modif_validated_shapes = np.zeros(self.ordered_first_image.shape, dtype=np.uint8)
            self.modif_validated_shapes[np.nonzero(self.ordered_first_image)] = 1
            self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                self.modif_validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)

            self.first_image.shape_number = self.first_image.shape_number - len(self.shapes_to_remove)
            self.not_analyzed_individuals = np.unique(self.unchanged_ordered_fimg -
                                                      (self.unchanged_ordered_fimg * self.modif_validated_shapes))[1:]

        else:
            # Reduce all box sizes if necessary and proceed
            if np.any(np.less(out_of_pic, 0)):
                # When the overflow is weak, remake standardization with lower "add_to_y" and "add_to_x"
                overflow = np.nonzero(np.logical_and(np.less(out_of_pic, 0), np.greater_equal(out_of_pic, distance_threshold_to_consider_an_arena_out_of_the_picture)))[0]
                # Look if overflow occurs on the y axis
                if np.any(np.less(out_of_pic[overflow, :2], 0)):
                    add_to_top_and_bot = np.min(out_of_pic[overflow, :2])
                    self.standard[:, 0] = self.standard[:, 0] - add_to_top_and_bot
                    self.standard[:, 1] = self.standard[:, 1] + add_to_top_and_bot
                # Look if overflow occurs on the x axis
                if np.any(np.less(out_of_pic[overflow, 2:], 0)):
                    add_to_left_and_right = np.min(out_of_pic[overflow, 2:])
                    self.standard[:, 2] = self.standard[:, 2] - add_to_left_and_right
                    self.standard[:, 3] = self.standard[:, 3] + add_to_left_and_right
            # If x or y sizes are odd, make them even :
            # Don't know why, but opencv remove 1 to odd shapes when writing videos
            if (self.standard[0, 1] - self.standard[0, 0]) % 2 != 0:
                self.standard[:, 1] -= 1
            if (self.standard[0, 3] - self.standard[0, 2]) % 2 != 0:
                self.standard[:, 3] -= 1
            self.top = self.standard[:, 0]
            self.bot = self.standard[:, 1]
            self.left = self.standard[:, 2]
            self.right = self.standard[:, 3]


    def _get_bb_with_moving_centers(self, img_list: list, color_space_combination: dict, color_number: int,
                                    sample_size: int=2, all_specimens_have_same_direction: bool=True,
                                    display: bool=False, filter_spec: dict=None):
        """
            _get_bb_with_moving_centers(self, img_list: list, color_space_combination: dict,
                                        color_number: int, sample_size: int=2,
                                        all_specimens_have_same_direction: bool=True,
                                        display: bool=False, filter_spec: dict=None)

            Summary
            -------
            Read, segment, and rank shapes across multiple images to detect moving centers,
            adjusting centroids based on movement.

            Extended Description
            --------------------
            Processes a list of images to identify and rank shapes, segmenting motion-blurred
            shapes using kernels and updating centroids. Adjustments are made if all specimens
            move in the same direction.

            Parameters
            ----------
            img_list : list
                List of images or file paths to be processed.
            color_space_combination : dict
                Dictionary detailing the color space combination used for segmentation.
            color_number : int
                Number of colors to be considered in the segmentation process.
            sample_size : int, optional
                Number of frames to sample from `img_list`. Default is 2.
            all_specimens_have_same_direction : bool, optional
                Flag indicating if all specimens move in the same direction. Default is True.
            display : bool, optional
                Flag to enable visualization of intermediate results. Default is False.
            filter_spec : dict, optional
                Additional filtering specifications for shape segmentation. Default is None.

            Raises
            ------
            FileNotFoundError
                If any image in `img_list` is a path and does not exist.
            ValueError
                If `color_space_combination` or other parameters are invalid.

            Notes
            -----
            This function processes image sequences to identify shapes, track their movement,
            and adjust centroids accordingly. Intermediate results can be displayed for debugging.

        """
        print("Read and segment each sample image and rank shapes from top to bot and from left to right")

        self.motion_list = list()
        if isinstance(img_list, list):
            frame_number = len(img_list)
        else:
            frame_number = img_list.shape[0]
        sample_numbers = np.floor(np.linspace(0, frame_number, sample_size)).astype(int)
        for frame_idx in np.arange(sample_size):
            if frame_idx == 0:
                self.motion_list.insert(frame_idx, self.first_image.validated_shapes)
            else:
                if isinstance(img_list[0], str):
                    image = img_list[sample_numbers[frame_idx] - 1]
                else:
                    image = img_list[sample_numbers[frame_idx] - 1]
                self.motion_list.insert(frame_idx, self._segment_blob_motion(image, color_space_combination,
                                                                             color_number, filter_spec=filter_spec))

        self.big_kernels = Ellipse((self.k_size, self.k_size)).create().astype(np.uint8)
        self.small_kernels = cross_33
        self.small_kernels = self.small_kernels.astype(np.uint8)

        ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
            self.first_image.validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
        ordered_image_i = deepcopy(self.ordered_first_image)
        logging.info("For each frame, expand each previously confirmed shape to add area to its maximal bounding box")
        for step_i in np.arange(1, sample_size):
            previously_ordered_centroids = deepcopy(ordered_centroids)
            new_image_i = deepcopy(self.motion_list[step_i])
            detected_shape_number = self.first_image.shape_number + 1
            c = 0
            while c < 5 and detected_shape_number == self.first_image.shape_number + 1:
                c += 1
                image_i = new_image_i
                new_image_i = cv2.dilate(image_i, self.small_kernels, iterations=1)
                detected_shape_number, _ = cv2.connectedComponents(new_image_i, connectivity=8)
            if c == 0:
                break
            else:
                for shape_i in range(self.first_image.shape_number):
                    shape_to_expand = np.zeros(image_i.shape, dtype=np.uint8)
                    shape_to_expand[ordered_image_i == (shape_i + 1)] = 1
                    without_shape_i = ordered_image_i.copy()
                    without_shape_i[ordered_image_i == (shape_i + 1)] = 0
                    if self.k_size != 3:
                        test_shape = expand_until_neighbor_center_gets_nearer_than_own(shape_to_expand, without_shape_i,
                                                                                       ordered_centroids[shape_i, :],
                                                                                       np.delete(ordered_centroids, shape_i,
                                                                                                 axis=0), self.big_kernels)
                    else:
                        test_shape = shape_to_expand
                    test_shape = expand_until_neighbor_center_gets_nearer_than_own(test_shape, without_shape_i,
                                                                                   ordered_centroids[shape_i, :],
                                                                                   np.delete(ordered_centroids, shape_i,
                                                                                             axis=0), self.small_kernels)
                    confirmed_shape = test_shape * image_i
                    ordered_image_i[confirmed_shape > 0] = shape_i + 1


                mask_to_display = np.zeros(image_i.shape, dtype=np.uint8)
                mask_to_display[ordered_image_i > 0] = 1

                # If the blob moves enough to drastically change its gravity center,
                # update the ordered centroids at each frame.
                detected_shape_number, mask_to_display = cv2.connectedComponents(mask_to_display,
                                                                                 connectivity=8)

                mask_to_display = mask_to_display.astype(np.uint8)
                while np.logical_and(detected_shape_number - 1 != self.first_image.shape_number,
                                     np.sum(mask_to_display > 0) < mask_to_display.size):
                    mask_to_display = cv2.dilate(mask_to_display, self.small_kernels, iterations=1)
                    detected_shape_number, mask_to_display = cv2.connectedComponents(mask_to_display,
                                                                                     connectivity=8)
                    mask_to_display[np.nonzero(mask_to_display)] = 1
                    mask_to_display = mask_to_display.astype(np.uint8)
                ordered_stats, ordered_centroids = rank_from_top_to_bottom_from_left_to_right(mask_to_display,
                                                                                              self.first_image.y_boundaries)

                if all_specimens_have_same_direction:
                    # Adjust each centroid position according to the maximal centroid displacement.
                    x_diffs = ordered_centroids[:, 0] - previously_ordered_centroids[:, 0]
                    if np.mean(x_diffs) > 0: # They moved left, we add to x
                        add_to_x = np.max(x_diffs) - x_diffs
                    else: #They moved right, we remove from x
                        add_to_x = np.min(x_diffs) - x_diffs
                    ordered_centroids[:, 0] = ordered_centroids[:, 0] + add_to_x

                    y_diffs = ordered_centroids[:, 1] - previously_ordered_centroids[:, 1]
                    if np.mean(y_diffs) > 0:  # They moved down, we add to y
                        add_to_y = np.max(y_diffs) - y_diffs
                    else:  # They moved up, we remove from y
                        add_to_y = np.min(y_diffs) - y_diffs
                    ordered_centroids[:, 1] = ordered_centroids[:, 1] + add_to_y

                ordered_image_i = mask_to_display

        # Normalize each bounding box

        for shape_i in range(self.first_image.shape_number):
            shape_i_indices = np.where(ordered_image_i == shape_i + 1)
            self.left[shape_i] = np.min(shape_i_indices[1])
            self.right[shape_i] = np.max(shape_i_indices[1])
            self.top[shape_i] = np.min(shape_i_indices[0])
            self.bot[shape_i] = np.max(shape_i_indices[0])
        self.ordered_first_image = ordered_image_i

    def _segment_blob_motion(self, image, color_space_combination: dict, color_number: int, filter_spec: dict) -> NDArray[np.uint8]:
        """

        Segment one image using the OneImageAnalysis class

        Parameters
        ----------
        image : str or np.ndarray
            The input image, either as a file path (str) or an already loaded image array.
        color_space_combination : dict
            A dictionary specifying the color space combination to be used for segmentation.
        color_number : int
            The number representing the specific hue or color to segment in the image.
        filter_spec : dict
            A dictionary containing filter specifications for post-processing of the segmented image.

        Returns
        -------
        ndarray[np.uint8]
            A binary image array representing the segmented blob motion.

        Notes
        -----
        This function processes an input image to segment blobs based on color information.
        The method utilizes various color space conversions and filtering techniques
        to isolate the desired object from the background.

        """
        if isinstance(image, str):
            is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
            image = read_and_rotate(image, self.first_image.bgr, self.raw_images,
                                             is_landscape, self.first_image.crop_coord)
            # image = readim(image)
        In = OneImageAnalysis(image)#, self.raw_images
        In.convert_and_segment(color_space_combination, color_number, None, None, self.first_image.subtract_background,
                               self.first_image.subtract_background2, filter_spec=filter_spec)
        return In.binary_image


    def prepare_video_writing(self, img_list: list, min_ram_free: float, in_colors: bool=False, pathway: str=""):
        """

        Prepare the raw video (.npy) writing process for Cellects.

        Parameters
        ----------
        img_list : list
            List of images to be processed.
        min_ram_free : float
            Minimum amount of RAM in GB that should remain free.
        in_colors : bool, optional
            Whether the images are in color. Default is False.
        pathway : str, optional
            Path to save the video files. Default is an empty string.

        Returns
        -------
        tuple
            A tuple containing:
            - bunch_nb: int, number of bunches needed for video writing.
            - video_nb_per_bunch: int, number of videos per bunch.
            - sizes: ndarray, dimensions of each video.
            - video_bunch: list or ndarray, initialized video arrays.
            - vid_names: list, names of the video files.
            - rom_memory_required: None or float, required ROM memory.
            - analysis_status: dict, status and message of the analysis process.
            - remaining: int, remainder videos that do not fit in a complete bunch.

        Notes
        -----
        - The function calculates necessary memory and ensures 10% extra to avoid issues.
        - It checks for available RAM and adjusts the number of bunches accordingly.
        - If using color images, memory requirements are tripled.

        expected output depends on the provided images and RAM availability
        """
        # 1) Create a list of video names
        if self.not_analyzed_individuals is not None:
            number_to_add = len(self.not_analyzed_individuals)
        else:
            number_to_add = 0
        vid_names = list()
        ind_i = 0
        counter = 0
        while ind_i < (self.first_image.shape_number + number_to_add):
            ind_i += 1
            while np.any(np.isin(self.not_analyzed_individuals, ind_i)):
                ind_i += 1
            vid_names.append(pathway + "ind_" + str(ind_i) + ".npy")
            counter += 1
        img_nb = len(img_list)

        # 2) Create a table of the dimensions of each video
        # Add 10% to the necessary memory to avoid problems
        necessary_memory = img_nb * np.multiply((self.bot - self.top + 1).astype(np.uint64), (self.right - self.left + 1).astype(np.uint64)).sum() * 8 * 1.16415e-10
        if in_colors:
            sizes = np.column_stack(
                (np.repeat(img_nb, self.first_image.shape_number), self.bot - self.top + 1, self.right - self.left + 1,
                 np.repeat(3, self.first_image.shape_number)))
            necessary_memory *= 3
        else:
            sizes = np.column_stack(
                (np.repeat(img_nb, self.first_image.shape_number), self.bot - self.top + 1, self.right - self.left + 1))
        self.use_list_of_vid = True
        if np.all(sizes[0, :] == sizes):
            self.use_list_of_vid = False
        available_memory = (psutil.virtual_memory().available >> 30) - min_ram_free
        bunch_nb = int(np.ceil(necessary_memory / available_memory))
        if bunch_nb > 1:
            # The program will need twice the memory to create the second bunch.
            bunch_nb = int(np.ceil(2 * necessary_memory / available_memory))

        video_nb_per_bunch = np.floor(self.first_image.shape_number / bunch_nb).astype(np.uint8)
        analysis_status = {"continue": True, "message": ""}
        try:
            if self.use_list_of_vid:
                video_bunch = [np.zeros(sizes[i, :], dtype=np.uint8) for i in range(video_nb_per_bunch)]
            else:
                video_bunch = np.zeros(np.append(sizes[0, :], video_nb_per_bunch), dtype=np.uint8)
        except ValueError as v_err:
            analysis_status = {"continue": False, "message": "Probably failed to detect the right cell(s) number, do the first image analysis manually."}
            logging.error(f"{analysis_status['message']} error is: {v_err}")
        # Check for available ROM memory
        if (psutil.disk_usage('/')[2] >> 30) < (necessary_memory + 2):
            rom_memory_required = necessary_memory + 2
        else:
            rom_memory_required = None
        remaining = self.first_image.shape_number % bunch_nb
        if remaining > 0:
            bunch_nb += 1
        logging.info(f"Cellects will start writing {self.first_image.shape_number} videos. Given available memory, it will do it in {bunch_nb} time(s)")
        return bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining

    def write_videos_as_np_arrays(self, img_list: list, min_ram_free: float, in_colors: bool=False,
                                  reduce_image_dim: bool=False, pathway: str=""):
        """
        Save video frames as NumPy arrays.

        Write the given list of images into NumPy array format for video storage,
        considering memory constraints and optional color reduction.

        Parameters
        ----------
        img_list : list
            List of image file names to process.
        min_ram_free : float
            Minimum amount of RAM that should be free in MB to start processing.
        in_colors : bool, optional
            Whether to keep images in color. Default is False.
        reduce_image_dim : bool, optional
            Whether to reduce image dimensions by keeping only one color channel.
            Default is False. Only applicable if `in_colors` is False.
        pathway : str, optional
            Path to save the video arrays. Default is an empty string.

        Returns
        -------
        None

        Notes
        -----
        This method manages memory usage by processing the videos in batches. It checks available RAM,
        reads and rotates images, crops them as specified, and saves the processed portions as NumPy arrays.

        Examples
        --------
        >>> write_videos_as_np_arrays(img_list=['image1.png', 'image2.png'], min_ram_free=500)
        """
        is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
        bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining = self.prepare_video_writing(img_list, min_ram_free, in_colors, pathway=pathway)
        for bunch in np.arange(bunch_nb):
            print(f'\nSaving the bunch n: {bunch + 1} / {bunch_nb} of videos:', end=' ')
            if bunch == (bunch_nb - 1) and remaining > 0:
                arena = np.arange(bunch * video_nb_per_bunch, bunch * video_nb_per_bunch + remaining)
            else:
                arena = np.arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch)
            if self.use_list_of_vid:
                video_bunch = [np.zeros(sizes[i, :], dtype=np.uint8) for i in arena]
            else:
                video_bunch = np.zeros(np.append(sizes[0, :], len(arena)), dtype=np.uint8)
            prev_img = None
            images_done = bunch * len(img_list)
            for image_i, image_name in enumerate(img_list):
                img = read_and_rotate(image_name, prev_img, self.raw_images, is_landscape, self.first_image.crop_coord)
                prev_img = deepcopy(img)
                if not in_colors and reduce_image_dim:
                    img = img[:, :, 0]

                for arena_i, arena_name in enumerate(arena):
                    # arena_i = 0; arena_name = arena[arena_i]
                    sub_img = img[self.top[arena_name]: (self.bot[arena_name] + 1),
                                                             self.left[arena_name]: (self.right[arena_name] + 1), ...]
                    if self.use_list_of_vid:
                        video_bunch[arena_i][image_i, ...] = sub_img
                    else:
                        if len(video_bunch.shape) == 5:
                            video_bunch[image_i, :, :, :, arena_i] = sub_img
                        else:
                            video_bunch[image_i, :, :, arena_i] = sub_img
            for arena_i, arena_name in enumerate(arena):
                if self.use_list_of_vid:
                     np.save(vid_names[arena_name], video_bunch[arena_i])
                else:
                    if len(video_bunch.shape) == 5:
                         np.save(vid_names[arena_name], video_bunch[:, :, :, :, arena_i])
                    else:
                         np.save(vid_names[arena_name], video_bunch[:, :, :, arena_i])
