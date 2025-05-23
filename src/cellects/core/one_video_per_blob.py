"""
    This class uses the first (and if is required more accuracy, other) image(s)
    in order to make one video per detected arena from a bunch of chronologically ordered images.
"""

import os
import logging
from copy import deepcopy

from numpy import arange, any, round, uint8, int8, int16, int64, uint64, save, array, zeros, zeros_like, \
    nonzero, less, greater_equal, logical_and, logical_or, repeat, floor, ceil, unique, \
    delete, linspace, all, mean, sum, absolute, where, append, in1d, \
    multiply, column_stack, triu_indices, str_, min, max
import cv2
import psutil
from cellects.image_analysis.morphological_operations import cross_33, Ellipse, get_minimal_distance_between_2_shapes, get_every_coord_between_2_points, rank_from_top_to_bottom_from_left_to_right, expand_until_neighbor_center_gets_nearer_than_own
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.utils.load_display_save import readim
from cellects.utils.formulas import sum_of_abs_differences


class OneVideoPerBlob:
    """
        This class finds the bounding box containing all pixels covered by one blob over time
        and create a video from it.
        It does that, for each blob, considering a few information.
    """

    def __init__(self, first_image, starting_blob_hsize_in_pixels, raw_images):
        """

        """
        # Initialize all variables used in the following methods
        self.first_image = first_image
        self.original_shape_hsize = starting_blob_hsize_in_pixels
        self.raw_images = raw_images
        if self.original_shape_hsize is not None:
            self.k_size = int(((self.original_shape_hsize // 5) * 2) + 1)

        # 7) Create required empty arrays: especially the bounding box coordinates of each video
        self.ordered_first_image = None
        self.motion_list = list()
        self.shapes_to_remove = None
        self.not_analyzed_individuals = None

    def get_bounding_boxes(self, are_gravity_centers_moving, img_list, color_space_combination, color_number=2, sample_size=5, all_same_direction=True, display=False):
        logging.info("Get the coordinates of all arenas using the get_bounding_boxes method of the VideoMaker class")
        # are_gravity_centers_moving=self.all['are_gravity_centers_moving'] == 1; img_list=self.data_list; color_space_combination=self.vars['convert_for_origin']; color_number=self.vars["color_number"]; sample_size=5

        self.big_kernel = Ellipse((self.k_size, self.k_size)).create()  # fromfunction(self.circle_fun, (self.k_size, self.k_size))
        self.big_kernel = self.big_kernel.astype(uint8)
        self.small_kernel = array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=uint8)
        self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
            self.first_image.validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
        self.unchanged_ordered_fimg = deepcopy(self.ordered_first_image)
        self.modif_validated_shapes = deepcopy(self.first_image.validated_shapes)
        self.standard = - 1
        counter = 0
        while any(less(self.standard, 0)) and counter < 20:
            counter += 1
            self.left = zeros(self.first_image.shape_number, dtype=int64)
            self.right = repeat(self.modif_validated_shapes.shape[1], self.first_image.shape_number)
            self.top = zeros(self.first_image.shape_number, dtype=int64)
            self.bot = repeat(self.modif_validated_shapes.shape[0], self.first_image.shape_number)
            if are_gravity_centers_moving:
                self.get_bb_with_moving_centers(img_list, color_space_combination, color_number, sample_size, all_same_direction, display)
                # new:
                new_ordered_first_image = zeros(self.ordered_first_image.shape, dtype=uint8)
                #
                for i in arange(1, self.first_image.shape_number + 1):
                    previous_shape = zeros(self.ordered_first_image.shape, dtype=uint8)
                    previous_shape[nonzero(self.unchanged_ordered_fimg == i)] = 1
                    new_potentials = zeros(self.ordered_first_image.shape, dtype=uint8)
                    new_potentials[nonzero(self.ordered_first_image == i)] = 1
                    new_potentials[nonzero(self.unchanged_ordered_fimg == i)] = 0

                    pads = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance=2)
                    pads.consider_shapes_sizes(min_shape_size=10)
                    pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False)
                    new_ordered_first_image[nonzero(pads.expanded_shape)] = i
                self.ordered_first_image = new_ordered_first_image
                self.modif_validated_shapes = zeros(self.ordered_first_image.shape, dtype=uint8)
                self.modif_validated_shapes[nonzero(self.ordered_first_image)] = 1
                self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                    self.modif_validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
                self.get_quick_bb()
                # self.print_bounding_boxes()
            else:
                self.get_quick_bb()
            self.standardize_video_sizes()
        if counter == 20:
            self.top[self.top < 0] = 1
            self.bot[self.bot >= self.ordered_first_image.shape[0] - 1] = self.ordered_first_image.shape[0] - 2
            self.left[self.left < 0] = 1
            self.right[self.right >= self.ordered_first_image.shape[1] - 1] = self.ordered_first_image.shape[1] - 2


    def get_quick_bb(self):
        """
        Compute euclidean distance between cell(s) to get each arena bounding box
            To earn computation time:
            1) We use triu_indices to consider one time each pairwise distance
            2) We only compute distances when x and y distances are small enough
            (i.e. 3 * the minimal distance already calculated)

        :return:
        """
        from timeit import default_timer
        tic = default_timer()
        shapes = deepcopy(self.modif_validated_shapes)
        eroded_shapes = cv2.erode(self.modif_validated_shapes, cross_33)
        shapes = shapes - eroded_shapes
        # shapes = cv2.morphologyEx(self.modif_validated_shapes, cv2.MORPH_GRADIENT, cross_33)
        x_min = self.ordered_stats[:, 0]
        y_min = self.ordered_stats[:, 1]
        x_max = self.ordered_stats[:, 0] + self.ordered_stats[:, 2]
        y_max = self.ordered_stats[:, 1] + self.ordered_stats[:, 3]
        x_min_dist = shapes.shape[1]
        y_min_dist = shapes.shape[0]

        shapes *= self.ordered_first_image
        shape_nb = (len(unique(shapes)) - 1)
        i = 0
        a_indices, b_indices = triu_indices(shape_nb, 1)
        a_indices, b_indices = a_indices + 1, b_indices + 1
        all_distances = zeros((len(a_indices), 3), dtype=float)
        # For every pair of components, find the minimal distance
        for (a, b) in zip(a_indices, b_indices):
            x_dist = absolute(x_max[a - 1] - x_min[b - 1])
            y_dist = absolute(y_max[a - 1] - y_min[b - 1])
            if x_dist < 2 * x_min_dist and y_dist < 2 * y_min_dist:
                sub_shapes = logical_or(shapes == a, shapes == b) * shapes
                sub_shapes = sub_shapes[min((y_min[a - 1], y_min[b - 1])):max((y_max[a - 1], y_max[b - 1])),
                             min((x_min[a - 1], x_min[b - 1])):max((x_max[a - 1], x_max[b - 1]))]
                sub_shapes[sub_shapes == a] = 1
                sub_shapes[sub_shapes == b] = 2
                if any(sub_shapes == 1) and any(sub_shapes == 2):
                    all_distances[i, :] = a, b, get_minimal_distance_between_2_shapes(sub_shapes, False)

                    if x_dist > y_dist:
                        x_min_dist = min((x_min_dist, x_dist))
                    else:
                        y_min_dist = min((y_min_dist, y_dist))
                    i += 1
        for shape_i in arange(1, shape_nb + 1):
            # Get where the shape i appear in pairwise comparisons
            idx = nonzero(logical_or(all_distances[:, 0] == shape_i, all_distances[:, 1] == shape_i))
            # print(all_distances[idx, 2])
            # Compute the minimal distance related to shape i and divide by 2
            if len(all_distances[idx, 2]) > 0:
                dist = all_distances[idx, 2].min() // 2
            else:
                dist = 1
                # Save the coordinates of the arena around shape i
            self.left[shape_i - 1] = x_min[shape_i - 1] - dist.astype(int64)
            self.right[shape_i - 1] = x_max[shape_i - 1] + dist.astype(int64)
            self.top[shape_i - 1] = y_min[shape_i - 1] - dist.astype(int64)
            self.bot[shape_i - 1] = y_max[shape_i - 1] + dist.astype(int64)
        print((default_timer() - tic))

    def standardize_video_sizes(self):
        distance_threshold_to_consider_an_arena_out_of_the_picture = None# in pixels, worked nicely with - 50

        # The modifications allowing to not make videos of setups out of view, do not work for moving centers
        y_diffs = self.bot - self.top
        x_diffs = self.right - self.left
        add_to_y = ((max(y_diffs) - y_diffs) / 2)
        add_to_x = ((max(x_diffs) - x_diffs) / 2)
        self.standard = zeros((len(self.top), 4), dtype=int64)
        self.standard[:, 0] = self.top - int8(floor(add_to_y))
        self.standard[:, 1] = self.bot + int8(ceil(add_to_y))
        self.standard[:, 2] = self.left - int8(floor(add_to_x))
        self.standard[:, 3] = self.right + int8(ceil(add_to_x))

        # Monitor if one bounding box gets out of picture shape
        out_of_pic = deepcopy(self.standard)
        out_of_pic[:, 1] = self.ordered_first_image.shape[0] - out_of_pic[:, 1] - 1
        out_of_pic[:, 3] = self.ordered_first_image.shape[1] - out_of_pic[:, 3] - 1

        if distance_threshold_to_consider_an_arena_out_of_the_picture is None:
            distance_threshold_to_consider_an_arena_out_of_the_picture = min(out_of_pic) - 1

        # If it occurs at least one time, apply a correction, otherwise, continue and write videos
        # If the overflow is strong, remove the corresponding individuals and remake bounding_box finding

        if any(less(out_of_pic, distance_threshold_to_consider_an_arena_out_of_the_picture)):
            # Remove shapes
            self.standard = - 1
            self.shapes_to_remove = nonzero(less(out_of_pic, - 20))[0]
            for shape_i in self.shapes_to_remove:
                self.ordered_first_image[self.ordered_first_image == (shape_i + 1)] = 0
            self.modif_validated_shapes = zeros(self.ordered_first_image.shape, dtype=uint8)
            self.modif_validated_shapes[nonzero(self.ordered_first_image)] = 1
            self.ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
                self.modif_validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)

            self.first_image.shape_number = self.first_image.shape_number - len(self.shapes_to_remove)
            self.not_analyzed_individuals = unique(self.unchanged_ordered_fimg -
                                                      (self.unchanged_ordered_fimg * self.modif_validated_shapes))[1:]

        else:
            # Reduce all box sizes if necessary and proceed
            if any(less(out_of_pic, 0)):
                # When the overflow is weak, remake standardization with lower "add_to_y" and "add_to_x"
                overflow = nonzero(logical_and(less(out_of_pic, 0), greater_equal(out_of_pic, distance_threshold_to_consider_an_arena_out_of_the_picture)))[0]
                # Look if overflow occurs on the y axis
                if any(less(out_of_pic[overflow, :2], 0)):
                    add_to_top_and_bot = min(out_of_pic[overflow, :2])
                    self.standard[:, 0] = self.standard[:, 0] - add_to_top_and_bot
                    self.standard[:, 1] = self.standard[:, 1] + add_to_top_and_bot
                # Look if overflow occurs on the x axis
                if any(less(out_of_pic[overflow, 2:], 0)):
                    add_to_left_and_right = min(out_of_pic[overflow, 2:])
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


    def get_bb_with_moving_centers(self, img_list, color_space_combination, color_number, sample_size=2, all_same_direction=True, display=False):
        """
        Starting with the first image, this function try to make each shape grow to see if it covers segmented pixels
        on following images. i.e. it segment evenly spaced images (See self.segment_blob_motion and OneImageAnalysis)
        to make a rough tracking of blob motion allowing to be sure that the video will only contain the shapes that
        have a chronological link with the shape as it was on the first image.

        :param img_list: The whole list of image names
        :type img_list: list
        :param sample_size: The picture number to analyse. The higher it is, the higher bath accuracy and computation
        time are
        :type sample_size: int
        :param all_same_direction: Whether all specimens move roughly in the same direction or not
        :type all_same_direction: bool
        :return: For each shapes, the coordinate of a bounding box including all shape movements
        """
        print("Read and segment each sample image and rank shapes from top to bot and from left to right")

        self.motion_list = list()
        if img_list.dtype.type is str_:
            frame_number = len(img_list)
        sample_numbers = floor(linspace(0, frame_number, sample_size)).astype(int)
        for frame_idx in arange(sample_size):
            if frame_idx == 0:
                self.motion_list.insert(frame_idx, self.first_image.validated_shapes)
            else:
                # image_obj = OneImageAnalysis(cv2.imread(img_list[sample_numbers[image] - 1]))  # image_name=img_list[10]
                # image_obj.conversion(rgb_hsv_lab=[[1, 0, 0], [0, 0, 1], [0, 0, 0]])
                # image_obj.crop_images(self.first_image.crop_coord)
                # image_obj.thresholding()
                # self.motion_list.insert(image, image_obj.binary_image)
                if img_list.dtype.type is str_:
                    image = img_list[sample_numbers[frame_idx] - 1]
                else:
                    image = img_list[sample_numbers[frame_idx] - 1, ...]
                self.motion_list.insert(frame_idx, self.segment_blob_motion(image, color_space_combination, color_number))


        self.big_kernels = Ellipse((self.k_size, self.k_size)).create().astype(uint8)
        # big_kernels = fromfunction(self.circle_fun, (self.k_size, self.k_size))
        # big_kernels = big_kernels.astype(uint8)
        self.small_kernels = array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=uint8)
        self.small_kernels = self.small_kernels.astype(uint8)

        ordered_stats, ordered_centroids, self.ordered_first_image = rank_from_top_to_bottom_from_left_to_right(
            self.first_image.validated_shapes, self.first_image.y_boundaries, get_ordered_image=True)
        #self.ordered_first_image = self.ordered_first_image.astype(uint8)
        previous_ordered_image_i = deepcopy(self.ordered_first_image)
        if img_list.dtype.type is str_:
            img_to_display = self.read_and_rotate(img_list[sample_numbers[1] - 1], self.first_image.bgr)
            # img_to_display = readim(img_list[sample_numbers[1] - 1], self.raw_images)
        else:
            img_to_display = img_list[sample_numbers[1] - 1, ...]
            if self.first_image.cropped:
                img_to_display = img_to_display[self.first_image.crop_coord[0]:self.first_image.crop_coord[1],
                                                self.first_image.crop_coord[2]:self.first_image.crop_coord[3], :]
        print("For each frame, expand each previously confirmed shape to add area to its maximal bounding box")
        for step_i in arange(1, sample_size):
            print(step_i)

            previously_ordered_centroids = deepcopy(ordered_centroids)
            image_i = deepcopy(self.motion_list[step_i])
            image_i = cv2.dilate(image_i, self.small_kernels, iterations=5)

            # Display the segmentation result for all shapes at this frame
            if img_list.dtype.type is str_:
                img_to_display = self.read_and_rotate(img_list[sample_numbers[step_i] - 1], self.first_image.bgr)
                # img_to_display = readim(img_list[sample_numbers[step_i] - 1], self.raw_images)
            else:
                img_to_display = img_list[sample_numbers[step_i] - 1, ...]
                if self.first_image.cropped:
                    img_to_display = img_to_display[self.first_image.crop_coord[0]: self.first_image.crop_coord[1],
                                     self.first_image.crop_coord[2]: self.first_image.crop_coord[3], :]

            for shape_i in range(self.first_image.shape_number):
                shape_to_expand = zeros(image_i.shape, dtype=uint8)
                shape_to_expand[previous_ordered_image_i == (shape_i + 1)] = 1
                without_shape_i = deepcopy(previous_ordered_image_i)
                without_shape_i[previous_ordered_image_i == (shape_i + 1)] = 0
                test_shape = expand_until_neighbor_center_gets_nearer_than_own(shape_to_expand, without_shape_i,
                                                                               ordered_centroids[shape_i, :],
                                                                               delete(ordered_centroids, shape_i,
                                                                                         axis=0), self.big_kernels)
                test_shape = expand_until_neighbor_center_gets_nearer_than_own(test_shape, without_shape_i,
                                                                               ordered_centroids[shape_i, :],
                                                                               delete(ordered_centroids, shape_i,
                                                                                         axis=0), self.small_kernels)
                confirmed_shape = test_shape * image_i
                previous_ordered_image_i[nonzero(confirmed_shape)] = shape_i + 1
                # update the image by putting a purple mask around the current shape
                contours, useless = cv2.findContours(confirmed_shape, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img_to_display, contours, -1, (255, 0, 180), 3)
                if display:
                    imtoshow = cv2.resize(img_to_display.astype(uint8), (960, 540))
                    cv2.imshow('Rough detection', imtoshow)
                    cv2.waitKey(1)
            if display:
                cv2.destroyAllWindows()


            mask_to_display = zeros(image_i.shape, dtype=uint8)
            mask_to_display[nonzero(previous_ordered_image_i)] = 1
            contours_to_display, useless = cv2.findContours(mask_to_display,
                                                          cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img_to_display, contours_to_display, -1, (255, 0, 0), 3)
            if display:
                imtoshow = cv2.resize(img_to_display.astype(uint8), (960, 540))
                cv2.imshow('Rough detection', imtoshow)
                cv2.waitKey(1)

            # If the blob moves enough to drastically change its gravity center,
            # update the ordered centroids at each frame.
            detected_shape_number, mask_to_display = cv2.connectedComponents(mask_to_display,
                                                                             connectivity=8)
            mask_to_display = mask_to_display.astype(uint8)
            while logical_and(detected_shape_number - 1 != self.first_image.shape_number,
                                 sum(mask_to_display > 0) < mask_to_display.size):
                mask_to_display = cv2.dilate(mask_to_display, self.small_kernels, iterations=1)
                detected_shape_number, mask_to_display = cv2.connectedComponents(mask_to_display,
                                                                                 connectivity=8)
                mask_to_display[nonzero(mask_to_display)] = 1
                mask_to_display = mask_to_display.astype(uint8)
                if display:
                    imtoshow = cv2.resize(mask_to_display * 255, (960, 540))
                    cv2.imshow('expansion', imtoshow)
                    cv2.waitKey(1)
            if display:
                cv2.destroyAllWindows()
            ordered_stats, ordered_centroids = rank_from_top_to_bottom_from_left_to_right(mask_to_display,
                                                                                          self.first_image.y_boundaries)

            # # Only keep one centroid per shape
            # previous_binary_image = self.motion_list[step_i - 1]
            # prev_bin_idx = nonzero(previous_binary_image)
            # centroimage = zeros(image_i.shape, dtype=uint8)
            # oc = round(ordered_centroids).astype(int64)
            # centroimage[oc[:, 1], oc[:, 0]] = 1
            # # Remove those that are not among the previous binary image
            # centroimage = centroimage * previous_binary_image
            # oc = nonzero(centroimage)
            # nb_of_potential_centroids = len(oc[0])

            # # Only keep centroids that moved the less from their previous position
            # new_ordered_centroids = zeros_like(previously_ordered_centroids)
            # for shape_i in range(self.first_image.shape_number):
            #     euclidean_distances = zeros(nb_of_potential_centroids, dtype=float)
            #     for i in range(nb_of_potential_centroids):
            #         potential_centroid = [oc[1][i], oc[0][i]]
            #         euclidean_distances[i] = eudist(previously_ordered_centroids[shape_i, :], potential_centroid)
            #     idx = argmin(euclidean_distances)
            #     new_ordered_centroids[shape_i, :] = oc[1][idx], oc[0][idx]

            new_ordered_centroids = ordered_centroids
            if all_same_direction:
                # Adjust each centroid position according to the maximal centroid displacement.
                x_diffs = new_ordered_centroids[:, 0] - previously_ordered_centroids[:, 0]
                if mean(x_diffs) > 0: # They moved left, we add to x
                    add_to_x = max(x_diffs) - x_diffs
                else: #They moved right, we remove from x
                    add_to_x = min(x_diffs) - x_diffs
                new_ordered_centroids[:, 0] = new_ordered_centroids[:, 0] + add_to_x

                y_diffs = new_ordered_centroids[:, 1] - previously_ordered_centroids[:, 1]
                if mean(y_diffs) > 0:  # They moved down, we add to y
                    add_to_y = max(y_diffs) - y_diffs
                else:  # They moved up, we remove from y
                    add_to_y = min(y_diffs) - y_diffs
                new_ordered_centroids[:, 1] = new_ordered_centroids[:, 1] + add_to_y

            ordered_centroids = new_ordered_centroids

            # Normalize each bounding box

        for shape_i in range(self.first_image.shape_number):
            shape_i_indices = where(previous_ordered_image_i == shape_i + 1)
            self.left[shape_i] = min(shape_i_indices[1])
            self.right[shape_i] = max(shape_i_indices[1])
            self.top[shape_i] = min(shape_i_indices[0])
            self.bot[shape_i] = max(shape_i_indices[0])
        #new See(previous_ordered_image_i)
        self.ordered_first_image = previous_ordered_image_i

    def segment_blob_motion(self, image, color_space_combination, color_number):
        if isinstance(image, str):
            image = self.read_and_rotate(image, self.first_image.bgr)
            # image = readim(image)
        In = OneImageAnalysis(image)#, self.raw_images
        In.convert_and_segment(color_space_combination, color_number, None, None, self.first_image.subtract_background,
                               self.first_image.subtract_background2)
        # In.generate_color_space_combination(color_space_combination)
        # In.thresholding()
        return In.binary_image


    def print_bounding_boxes(self, display_or_return=0):
        imtoshow = deepcopy(self.first_image.bgr)
        segments = zeros((2, 1), dtype=uint8)
        for i in arange(self.first_image.shape_number):
            j = i * 4
            segments = append(segments, get_every_coord_between_2_points(array((self.top[i], self.left[i])),
                                                                            array((self.bot[i], self.left[i]))),
                                 axis=1)
            j = j + 1
            segments = append(segments, get_every_coord_between_2_points(array((self.top[i], self.right[i])),
                                                                            array((self.bot[i], self.right[i]))),
                                 axis=1)
            j = j + 1
            segments = append(segments, get_every_coord_between_2_points(array((self.top[i], self.left[i])),
                                                                            array((self.top[i], self.right[i]))),
                                 axis=1)
            j = j + 1
            segments = append(segments,  get_every_coord_between_2_points(array((self.bot[i], self.left[i])),
                                                                             array((self.bot[i], self.right[i]))),
                                 axis=1)

            text = f"{i + 1}"
            position = (self.left[i] + 25, self.top[i] + (self.bot[i] - self.top[i]) // 2)
            imtoshow = cv2.putText(imtoshow,  # numpy array on which text is written
                            text,  # text
                            position,  # position at which writing has to start
                            cv2.FONT_HERSHEY_SIMPLEX,  # font family
                            1,  # font size
                            (0, 0, 0, 255),  # (209, 80, 0, 255),  # repeat(self.vars["contour_color"], 3),# font color
                            2)  # font stroke

        mask = zeros(self.first_image.validated_shapes.shape, dtype=uint8)
        mask[segments[0], segments[1]] = 1
        mask = cv2.dilate(mask, array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=uint8), iterations=3)
        if display_or_return == 0:
            imtoshow[mask == 1, :] = 0
            imtoshow = cv2.resize(imtoshow, (2000, 1000))
            cv2.imshow('Video contour', imtoshow)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return mask


    def prepare_video_writing(self, img_list, min_ram_free, in_colors=False):
        #https://stackoverflow.com/questions/48672130/saving-to-hdf5-is-very-slow-python-freezing
        #https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
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
            while any(in1d(self.not_analyzed_individuals, ind_i)):
                ind_i += 1
            vid_names.append("ind_" + str(ind_i) + ".npy")
            counter += 1
        img_nb = len(img_list)

        # 2) Create a table of the dimensions of each video
        # Add 10% to the necessary memory to avoid problems
        necessary_memory = img_nb * multiply((self.bot - self.top + 1).astype(uint64), (self.right - self.left + 1).astype(uint64)).sum() * 8 * 1.16415e-10
        if in_colors:
            sizes = column_stack(
                (repeat(img_nb, self.first_image.shape_number), self.bot - self.top + 1, self.right - self.left + 1,
                 repeat(3, self.first_image.shape_number)))
            necessary_memory *= 3
        else:
            sizes = column_stack(
                (repeat(img_nb, self.first_image.shape_number), self.bot - self.top + 1, self.right - self.left + 1))
        self.use_list_of_vid = True
        if all(sizes[0, :] == sizes):
            self.use_list_of_vid = False
        available_memory = (psutil.virtual_memory().available >> 30) - min_ram_free
        bunch_nb = int(ceil(necessary_memory / available_memory))
        if bunch_nb > 1:
            # The program will need twice the memory to create the second bunch.
            bunch_nb = int(ceil(2 * necessary_memory / available_memory))

        video_nb_per_bunch = floor(self.first_image.shape_number / bunch_nb).astype(uint8)
        analysis_status = {"continue": True, "message": ""}
        try:
            if self.use_list_of_vid:
                video_bunch = [zeros(sizes[i, :], dtype=uint8) for i in range(video_nb_per_bunch)]
            else:
                video_bunch = zeros(append(sizes[0, :], video_nb_per_bunch), dtype=uint8)
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

    def write_videos_as_np_arrays(self, img_list, min_ram_free, in_colors=False, reduce_image_dim=False):
        #self=self.videos
        #img_list = self.data_list
        #min_ram_free = self.vars['min_ram_free']
        #in_colors = not self.vars['already_greyscale']

        bunch_nb, video_nb_per_bunch, sizes, video_bunch, vid_names, rom_memory_required, analysis_status, remaining = self.prepare_video_writing(img_list, min_ram_free, in_colors)
        for bunch in arange(bunch_nb):
            print(f'\nSaving the bunch n: {bunch + 1} / {bunch_nb} of videos:', end=' ')
            if bunch == (bunch_nb - 1) and remaining > 0:
                arena = arange(bunch * video_nb_per_bunch, bunch * video_nb_per_bunch + remaining)
            else:
                arena = arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch)
            # print(arena)
            if self.use_list_of_vid:
                video_bunch = [zeros(sizes[i, :], dtype=uint8) for i in arena]
            else:
                video_bunch = zeros(append(sizes[0, :], len(arena)), dtype=uint8)

            # # Add the remaining videos to the last bunch if necessary
            # if bunch == (bunch_nb - 1):
            #     remaining = self.first_image.shape_number % bunch_nb
            #     arena = arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch + remaining)
            #     if bunch > 0:
            #         # The last bunch is larger if there is a remaining to division
            #         if self.use_list_of_vid:
            #             video_bunch = [zeros(sizes[i, :], dtype=uint8) for i in arena]
            #         else:
            #             video_bunch = zeros(append(sizes[0, :], len(arena)), dtype=uint8)
            #         # Add the remaining videos to the last bunch if necessary
            #     if self.use_list_of_vid:
            #         for i in arange(self.first_image.shape_number - remaining,
            #                         self.first_image.shape_number):
            #             video_bunch.append(zeros(sizes[i, :], dtype=uint8))
            #     else:
            #         video_bunch = zeros(append(sizes[0, :], len(arena) + remaining), dtype=uint8)
            # # Otherwise, use the same video_bunch and loop over the right individuals
            # else:
            #     arena = arange(bunch * video_nb_per_bunch, (bunch + 1) * video_nb_per_bunch)
            #     if bunch > 0:
            #         if self.use_list_of_vid:
            #             video_bunch = [zeros(sizes[i, :], dtype=uint8) for i in arena]
            #         else:
            #             video_bunch = zeros(append(sizes[0, :], len(arena)), dtype=uint8)
            prev_img = None
            images_done = bunch * len(img_list)
            for image_i, image_name in enumerate(img_list):
                # image_i = 0; image_name = img_list[image_i]
                # print(str(image_i), end=' ')
                img = self.read_and_rotate(image_name, prev_img)
                prev_img = deepcopy(img)
                if not in_colors and reduce_image_dim:
                    img = img[:, :, 0]
                # if not in_colors:
                #     csc = OneImageAnalysis(img)
                #     csc.generate_color_space_combination(convert_for_motion)
                #     img = csc.image
                # if self.first_image.crop_coord is not None:
                #     img = img[self.first_image.crop_coord[0]:self.first_image.crop_coord[1],
                #               self.first_image.crop_coord[2]:self.first_image.crop_coord[3], ...]

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
                    save(vid_names[arena_name], video_bunch[arena_i])
                else:
                    if len(video_bunch.shape) == 5:
                        save(vid_names[arena_name], video_bunch[:, :, :, :, arena_i])
                    else:
                        save(vid_names[arena_name], video_bunch[:, :, :, arena_i])

    def read_and_rotate(self, image_name, prev_img):
        """ This method read an image from its name and:
        - Make sure to properly crop it
        - Rotate the image if ir is not in the same orientation as the reference
        - Make sure that the rotation is on the good direction (clockwise or counterclockwise)"""

        # Read the image
        if not os.path.exists(image_name):
            raise FileNotFoundError(image_name)
        img = readim(image_name, self.raw_images)

        # Use a reference image to make sure that the read image is landscape or not
        is_landscape = self.first_image.image.shape[0] < self.first_image.image.shape[1]
        if (img.shape[0] > img.shape[1] and is_landscape) or (img.shape[0] < img.shape[1] and not is_landscape):
            # Try to turn it clockwise and (if necessary crop it)
            clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if self.first_image.cropped:
                clockwise = clockwise[self.first_image.crop_coord[0]:self.first_image.crop_coord[1],
                      self.first_image.crop_coord[2]:self.first_image.crop_coord[3], ...]
            if prev_img is not None:
                # Quantify the similarity between the clockwised turned image and the reference
                prev_img = int16(prev_img)
                clock_diff = sum_of_abs_differences(prev_img, int16(clockwise))
                # clock_diff = sum(absolute(int16(prev_img) - int16(clockwise)))
                # Try to turn it counterclockwise and (if necessary crop it)
                counter_clockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if self.first_image.cropped:
                    counter_clockwise = counter_clockwise[self.first_image.crop_coord[0]:self.first_image.crop_coord[1],
                                self.first_image.crop_coord[2]:self.first_image.crop_coord[3], ...]
                # Quantify the similarity between the counterclockwised turned image and the reference
                counter_clock_diff = sum_of_abs_differences(prev_img, int16(counter_clockwise))
                # counter_clock_diff = sum(absolute(int16(prev_img) - int16(counter_clockwise)))
                # The image that has the lower difference is kept.
                if clock_diff > counter_clock_diff:
                    img = counter_clockwise
                else:
                    img = clockwise
            else:
                img = clockwise
        else:
            if self.first_image.cropped:
                img = img[self.first_image.crop_coord[0]:self.first_image.crop_coord[1],
                                    self.first_image.crop_coord[2]:self.first_image.crop_coord[3], ...]
        return img

    def make_videos(self, img_list, extension, fps=40):
        is_color = True
        sizes = column_stack((self.right - self.left + 1, self.bot - self.top + 1))
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        if extension == '.mp4':
            fourcc = 0x7634706d#fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # (*'MP4V') (*'h265')
        else:
            fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')  # lossless
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '5')
        # fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
        # fourcc = cv2.VideoWriter_fourcc(*'FFV1')# lossless

        # 1) Create a list of video names
        if self.not_analyzed_individuals is not None:
            number_to_add = len(self.not_analyzed_individuals)
        else:
            number_to_add = 0
        vid_list = list()
        ind_i = 0
        counter = 0
        while ind_i < (self.first_image.shape_number + number_to_add):
            ind_i += 1
            while any(in1d(self.not_analyzed_individuals, ind_i)):
                ind_i += 1
            vid_name = f"ind_{ind_i}{extension}"
            vid_list.insert(counter, cv2.VideoWriter(vid_name, fourcc, float(fps), tuple(sizes[counter, :]), is_color))
            counter += 1

        # 2) loop over images and save videos frame by frame
        print("Image number: ")
        prev_img = None
        for image_i in arange(len(img_list)):
            print(str(image_i), end=' ')
            image_name = img_list[image_i]
            if not os.path.exists(image_name):
                raise FileNotFoundError(image_name)
            img = self.read_and_rotate(image_name, prev_img)
            prev_img = deepcopy(img)
            for blob_i in arange(self.first_image.shape_number):
                blob_img = deepcopy(img)
                if self.first_image.crop_coord is not None:
                    blob_img = blob_img[self.first_image.crop_coord[0]:self.first_image.crop_coord[1],
                               self.first_image.crop_coord[2]:self.first_image.crop_coord[3], :]
                blob_img = blob_img[self.top[blob_i]: (self.bot[blob_i] + 1),
                                    self.left[blob_i]: (self.right[blob_i] + 1), :]
                vid = vid_list[blob_i]
                vid.write(blob_img)

        for blob_i in arange(self.first_image.shape_number):
            vid_list[blob_i].release()


if __name__ == "__main__":
    from glob import glob
    from pathlib import Path
    from cellects.core.cellects_paths import TEST_DIR
    from cellects.utils.load_display_save import *
    from cellects.utils.utilitarian import insensitive_glob
    from cellects.image_analysis.one_image_analysis_threads import ProcessFirstImage
    from numpy import sort, array
    # os.chdir(TEST_DIR / "experiment")
    # image = readim("IMG_7653.jpg")
    os.chdir(Path("D:/Directory/Data/100/101-104/"))
    img_list = sort(insensitive_glob("IMG_" + '*' + ".jpg"))
    image = readim(img_list[0])
    first_image = OneImageAnalysis(image)
    first_im_color_space_combination = {"lab": array((1, 0, 0), uint8)}
    last_im_color_space_combination = {"lab": array((0, 0, 1), uint8)}
    first_image.convert_and_segment(first_im_color_space_combination)
    first_image.set_spot_shapes_and_size_confint('circle')
    process_i = ProcessFirstImage(
        [first_image, False, False, None, False, 8, None, 2, None, None, None])
    process_i.binary_image = first_image.binary_image
    process_i.process_binary_image()
    first_image.validated_shapes = process_i.validated_shapes
    first_image.shape_number = 8
    first_image.get_crop_coordinates()
    # See(first_image.binary_image)
    self = OneVideoPerBlob(first_image, 100, False)
    are_gravity_centers_moving=1; color_space_combination=last_im_color_space_combination; color_number=2; sample_size=5; all_same_direction=True
    self.get_bounding_boxes(are_gravity_centers_moving=1, img_list=img_list, color_space_combination=last_im_color_space_combination, color_number=2, sample_size=5, all_same_direction=False, display=True)
    self.print_bounding_boxes()

