#!/usr/bin/env python3

"""
Progressively Add Distant Shapes Module

This module contains the `ProgressivelyAddDistantShapes` class which is designed to analyze
and connect shapes in binary images based on their size and distance from a main shape. It can
progressively grow bridges between shapes in binary video sequences, with growth speeds that depend on neighboring growth speed.

The module provides functionality to:
- Check and adjust main shape labels
- Consider shapes based on size criteria
- Connect shapes that meet distance and size requirements
- Expand small shapes toward the main shape
- Modify past analysis by progressively filling pixels based on shape growth patterns

Classes:
    ProgressivelyAddDistantShapes: Main class for analyzing and connecting shapes in binary images.

Functions:
    make_gravity_field: Creates a gravity field around the main shape.
    CompareNeighborsWithValue: Compares neighbor values in an array.
    get_radius_distance_against_time: Calculates the relationship between distance and time for shape expansion.

This module is particularly useful in image analysis tasks where shapes need to be tracked and connected over time based on spatial relationships.
"""


from copy import deepcopy
import numpy as np
import cv2
from cellects.image_analysis.morphological_operations import cross_33, make_gravity_field, CompareNeighborsWithValue, get_radius_distance_against_time, cc, Ellipse, keep_shape_connected_with_ref



class ProgressivelyAddDistantShapes:
    """
    This class checks new potential shapes sizes and distance to a main shape.

    If these sizes and distance match requirements, create a bridge between
    these and the main shape. Then, the `modify_past_analysis` method progressively grows that bridge
    in a binary video. Bridge growth speed depends on neighboring growth speed.

    Attributes
    ----------
    new_order : numpy.ndarray
        A binary image of all shapes detected at t.
    main_shape : numpy.ndarray
        A binary image of the main shape (1) at t - 1.
    stats : numpy.ndarray
        Statistics about the connected components found in `new_order`.
    max_distance : int
        The maximal distance for a shape from new_potentials to get bridged.
    gravity_field : numpy.ndarray
        The gravity field used for connecting shapes.

    Parameters
    ----------
    new_potentials : numpy.ndarray
        A binary image of all shapes detected at t.
    previous_shape : numpy.ndarray
        A binary image of the main shape (1) at t - 1.
    max_distance : int
        The maximal distance for a shape from new_potentials to get bridged.

    Methods
    -------
    check_main_shape_label(previous_shape)
        Check if the main shape label is correctly set.
    consider_shapes_sizes(min_shape_size=None, max_shape_size=None)
        Consider shapes sizes and eliminate too small or large ones.
    connect_shapes(only_keep_connected_shapes, rank_connecting_pixels, intensity_valley=None)
        Connect shapes that are within the maximal distance and of appropriate size.
    expand_smalls_toward_main()
        Expand small shapes toward the main shape.

    Example
    -------
    >>> new_potentials = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> previous_shape = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    >>> max_distance = 2
    >>> bridge_shapes = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance)
    >>> bridge_shapes.consider_shapes_sizes(min_shape_size=2, max_shape_size=10)
    >>> bridge_shapes.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False)
    >>> print(bridge_shapes.expanded_shape)
    [[0 1 0]
     [1 1 1]
     [0 1 0]]
    """
    def __init__(self, new_potentials, previous_shape, max_distance):
        """
        This class check new potential shapes sizes and distance to a main
        (first called previous) shape.
        If these sizes and distance match requirements,
        create a bridge between these and the main shape
        Then, the modify_past_analysis method progressively grows that bridge
        in a binary video. Bridge growth speed depends on neighboring growth speed

        :param new_potentials: A binary image of all shapes detected at t
        :param previous_shape: A binary image of the main shape (1) at t - 1
        :param min_shape_size: The minimal size for a shape from new_potentials to get bridged
        :param max_shape_size: The maximal size for a shape from new_potentials to get bridged
        :param max_distance: The maximal distance for a shape from new_potentials to get bridged
        :param cross_33: A binary crux
        """
        self.new_order = np.logical_or(new_potentials, previous_shape).astype(np.uint8)
        self.new_order, self.stats, centers = cc(self.new_order)
        self.main_shape = np.zeros(self.new_order.shape, np.uint8)
        self.max_distance = max_distance
        self.check_main_shape_label(previous_shape)

    def check_main_shape_label(self, previous_shape):
        """
        Check and update main shape label based on previous shape data when multiple shapes exist in new_order.

        This method ensures consistent labeling of the primary shape (labeled 1) in `new_order` by analyzing overlaps
        with labels from a prior segmentation step. If multiple candidate labels exist for the main shape, it selects
        the one with the highest pixel count and swaps its label with '1' in both `new_order` and associated statistics.

        Parameters
        ----------
        previous_shape
            Input array representing previous segmentation labels used to identify the primary shape when
            `new_order` contains multiple potential candidates (labels > 1).

        Examples
        --------
        >>> new_potentials = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> previous_shape = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        >>> max_distance = 2
        >>> pads = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance)
        >>> pads.main_shape
        array([[0, 0, 0],
               [0, 0, 0],
               [0, 1, 1]], dtype=uint8)
        """
        if np.any(self.new_order > 1):
            # If there is at least one pixel of the previous shape that is not among pixels labelled 1,
            # clarify who's main shape
            main_shape_label = np.unique(previous_shape * self.new_order)
            main_shape_label = main_shape_label[main_shape_label != 0]

            # If the main shape is not labelled 1 in main_shape:
            if not np.isin(1, main_shape_label):
                # If it is not 1, find which label correspond to the previous shape
                if len(main_shape_label) > 1:
                    pixel_sum_per_label = np.zeros(len(main_shape_label), dtype =np.uint64)
                    # Find out the label corresponding to the largest shape
                    for li, label in enumerate(main_shape_label):
                        pixel_sum_per_label[li] = self.new_order[self.new_order == label].sum()
                    main_shape_label = main_shape_label[np.argmax(pixel_sum_per_label)]
                # Attribute the correct main shape
                self.main_shape[self.new_order == main_shape_label] = 1
                # Exchange the 1 and the main shape label in new_order image
                not_one_idx = np.nonzero(self.new_order == main_shape_label)
                one_idx = np.nonzero(self.new_order == 1)
                self.new_order[not_one_idx[0], not_one_idx[1]] = 1
                self.new_order[one_idx[0], one_idx[1]] = main_shape_label
                # Do the same for stats
                not_one_stats = deepcopy(self.stats[main_shape_label, :])
                self.stats[main_shape_label, :] = self.stats[1, :]
                self.stats[1, :] = not_one_stats
            else:
            #if np.any(previous_shape * (self.new_order == 1)):
                # Create an image of the principal shape
                self.main_shape[self.new_order == 1] = 1
        else:
            self.main_shape[np.nonzero(self.new_order)] = 1

    def consider_shapes_sizes(self, min_shape_size=None, max_shape_size=None):
        """Filter shapes based on minimum and maximum size thresholds.

        This method adjusts `new_order` by excluding indices of shapes that are either
        smaller than `min_shape_size` or larger than `max_shape_size`. The main shape index
        (1) is preserved even if it meets the filtering criteria. When no constraints apply,
        the expanded shape defaults to the main shape.

        Parameters
        ----------
        min_shape_size : int, optional
            Minimum allowed size for shapes (compared against 4th column of `self.stats`).
        max_shape_size : int, optional
            Maximum allowed size for shapes (compared against 4th column of `self.stats`).

        Examples
        --------
        >>> new_potentials = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> previous_shape = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        >>> max_distance = 2
        >>> pads = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance)
        >>> pads.consider_shapes_sizes(min_shape_size=2, max_shape_size=10)
        >>> pads.new_order
        array([[2, 2, 0],
       [0, 0, 0],
       [0, 1, 1]], dtype=uint8)
        """
        if self.max_distance != 0:
            # Eliminate too small and too large shapes
            if min_shape_size is not None or max_shape_size is not None:
                if min_shape_size is not None:
                    small_shapes = self.stats[:, 4] < min_shape_size
                    extreme_shapes = deepcopy(small_shapes)
                if max_shape_size is not None:
                    large_shapes = self.stats[:, 4] > max_shape_size
                    extreme_shapes = deepcopy(large_shapes)
                if min_shape_size is not None and max_shape_size is not None:
                    extreme_shapes = np.nonzero(np.logical_or(small_shapes, large_shapes))[0]
                is_main_in_it = np.isin(extreme_shapes, 1)
                if np.any(is_main_in_it):
                    extreme_shapes = np.delete(extreme_shapes, is_main_in_it)
                for extreme_shape in extreme_shapes:
                    self.new_order[self.new_order == extreme_shape] = 0
        else:
            self.expanded_shape = self.main_shape

    def expand_smalls_toward_main(self):
        """Expands small shapes toward a main shape using morphological operations and gravity field analysis.

        The method dilates the main shape to determine an order of expansion for connected regions.
        Each identified region is iteratively expanded until overlapping with the main shape, guided by a gravity field gradient.
        Results include both the final expanded binary mask and peak values from the gravity field during expansion phases.

        Returns
        -------
        numpy.ndarray[numpy.uint8]
            Binary array where small shapes are fully expanded to connect with the main shape.
        numpy.ndarray[numpy.uint32]
            Array containing maximum detected field strengths for each expanded region, in order of connection.

        Examples
        --------
        >>> new_potentials = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> previous_shape = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        >>> max_distance = 3
        >>> pads = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance)
        >>> pads.consider_shapes_sizes(min_shape_size=2, max_shape_size=10)
        >>> pads.gravity_field = make_gravity_field(pads.main_shape, max_distance=pads.max_distance, with_erosion=0)
        >>> expanded_main, max_field_feelings = pads.expand_smalls_toward_main()
        >>> print(expanded_main)
        [[1 1 0]
         [0 1 1]
         [0 1 1]]
        """
        other_shapes = np.zeros(self.main_shape.shape, np.uint8)
        other_shapes[self.new_order > 1] = 1
        simple_disk = cross_33
        kernel = Ellipse((5, 5)).create().astype(np.uint8)
        # Dilate the main shape, progressively to infer in what order other shapes should be expanded toward it
        main_shape = deepcopy(self.main_shape)
        new_order = deepcopy(self.new_order)
        order_of_shapes_to_expand = np.empty(0, dtype=np.uint32)
        nb = 3
        while nb > 2:
            main_shape = cv2.dilate(main_shape, kernel)
            connections = deepcopy(main_shape)
            connections *= new_order
            new_connections = np.unique(connections)[2:]
            new_order[np.isin(new_order, new_connections)] = 1
            order_of_shapes_to_expand = np.append(order_of_shapes_to_expand, new_connections)
            connections[main_shape > 0] = 1
            connections[other_shapes > 0] = 1
            nb, connections = cv2.connectedComponents(connections)

        expanded_main = deepcopy(self.main_shape)
        max_field_feelings = np.empty(0, dtype=np.uint32)
        max_field_feeling = 0
        # Loop over each shape to connect, from the nearest to the furthest to the main shape
        for shape_i in order_of_shapes_to_expand:#  np.unique(self.new_order)[2:]:
            current_shape = np.zeros(self.main_shape.shape, np.uint8)
            current_shape[self.new_order == shape_i] = 1
            dil = 0
            # Dilate that shape until it overlaps the main shape
            while np.logical_and(dil <= self.max_distance, not np.any(current_shape * expanded_main)):
                dil += 1
                rings = cv2.dilate(current_shape, simple_disk, iterations=1, borderType=cv2.BORDER_CONSTANT,
                               borderValue=0)

                rings = self.gravity_field * (rings - current_shape)
                max_field_feeling = np.max(rings) # np.min(rings[rings>0])
                max_field_feelings = np.append(max_field_feeling, max_field_feelings)
                if max_field_feeling > 0:  # If there is no shape within max_distance range, quit the loop

                    if dil == 1:
                        initial_pixel_number = np.sum(rings == max_field_feeling)
                    while np.sum(rings == max_field_feeling) > initial_pixel_number:
                        shrinking_stick = CompareNeighborsWithValue(rings, 8, np.uint32)
                        shrinking_stick.is_equal(max_field_feeling, True)
                        rings[shrinking_stick.equal_neighbor_nb < 2] = 0
                    current_shape[rings == max_field_feeling] = 1
                else:
                    break

            expanded_main[current_shape != 0] = 1
        return expanded_main, max_field_feelings


    def connect_shapes(self, only_keep_connected_shapes, rank_connecting_pixels, intensity_valley=None):
        """Connects small shapes to a main shape using gravity field expansion and filtering based on distance and intensity conditions.

        Extended Description
        --------------------
        When distant shapes of sufficient size are present, this method generates a gravity field around the main shape. It then expands smaller shapes toward the main one according to gradient values. If shapes fall within the gravity field range:
        - Shapes not connected to the main one (via `only_keep_connected_shapes`) are filtered out.
        - Connecting pixels between small and main shapes (via `rank_connecting_pixels`) receive distance-based ranking.

        Parameters
        ----------
        only_keep_connected_shapes : bool
            If True, filters expanded shapes to retain only those connected directly to the main shape.
        rank_connecting_pixels : bool
            If True, ranks connecting pixel extensions based on distance between small/main shapes.
        intensity_valley : array-like, optional
            Optional intensity values defining a valley region for gravity field calculation. Default is None.

        Attributes
        ----------
        gravity_field : ndarray or array-like
            Stores the computed gravity field used to guide shape expansion.
        expanded_shape : ndarray of dtype uint8
            Final combined shape after processing; contains main and connected small shapes.
        Examples
        --------
        >>> new_potentials = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> previous_shape = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        >>> max_distance = 3
        >>> pads = ProgressivelyAddDistantShapes(new_potentials, previous_shape, max_distance)
        >>> pads.consider_shapes_sizes(min_shape_size=2, max_shape_size=10)
        >>> pads.gravity_field = make_gravity_field(pads.main_shape, max_distance=pads.max_distance, with_erosion=0)
        >>> pads.connect_shapes(only_keep_connected_shapes=False, rank_connecting_pixels=True)
        >>> expanded_main, max_field_feelings = pads.expand_smalls_toward_main()
        >>> print(expanded_main)
        [[1 1 0]
         [0 1 1]
         [0 1 1]]
        """
        # If there are distant shapes of the good size, run the following:
        if self.max_distance != 0 and np.any(self.new_order > 1):
            # The intensity valley method does not work yet, don't use it
            if intensity_valley is not None:
                self.gravity_field = intensity_valley # make sure that the values correspond to the coord
            else:
                # 1) faire un champ gravitationnel autour de la forme principale
                self.gravity_field = make_gravity_field(self.main_shape, max_distance=self.max_distance, with_erosion=1)

                # If there are near enough shapes, run the following
                # 2) Dilate other shapes toward the main according to the gradient
            other_shapes, max_field_feelings = self.expand_smalls_toward_main()


            # plt.imshow(other_shapes)
            # If there are shapes within gravity field range
            if np.any(max_field_feelings > 0):
                self.expanded_shape = np.zeros(self.main_shape.shape, np.uint8)
                self.expanded_shape[np.nonzero(self.main_shape + other_shapes)] = 1
                if only_keep_connected_shapes:
                    # Make sure that only shapes connected with the main one remain on the final image
                    expanded_shape = keep_shape_connected_with_ref(self.expanded_shape, self.main_shape)
                    if expanded_shape is not None:
                        self.expanded_shape = expanded_shape
                    self.keep_connected_shapes()
                    if rank_connecting_pixels:
                        # Rate the extension of small shapes according to the distance between the small and the main shapes
                        self.distance_ranking_of_connecting_pixels()
                #self.expanded_shape
                # plt.imshow(self.expanded_shape)
            else:
                self.expanded_shape = self.main_shape
        # Otherwise, end by putting the main shape as output
        else:
            self.expanded_shape = self.main_shape

        # else:
        #     self.expanded_shape = other_shapes + self.main_shape
        # self.expanded_shape[self.expanded_shape > 1] = 1

    def distance_ranking_of_connecting_pixels(self):
        rated_extension = np.zeros(self.main_shape.shape, np.uint8)
        rated_extension[(self.main_shape - self.expanded_shape) == 255] = 1
        rated_extension = rated_extension * self.gravity_field
        if np.any(rated_extension):
            rated_extension[np.nonzero(rated_extension)] -= np.min(
                rated_extension[np.nonzero(rated_extension)]) - 1
        self.expanded_shape += rated_extension

    #binary_video = self.binary[(self.step // 2):(self.t + 1), :, :]
    #draft_seg = self.segmentation[(self.step // 2):(self.t + 1), :, :]
    def modify_past_analysis(self, binary_video, draft_seg):
        self.binary_video = binary_video
        self.draft_seg = draft_seg
        self.expanded_shape[self.expanded_shape == 1] = 0
        # Find the time at which the shape became connected to the expanded shape
        # (i.e. the time to start looking for a growth)
        distance_against_time, time_start, time_end = self.find_expansion_timings()

        # Use that vector to progressively fill pixels at the same speed as shape grows
        for t in np.arange(len(distance_against_time)):
            image_garbage = (self.expanded_shape >= distance_against_time[t]).astype(np.uint8)
            new_order, stats, centers = cc(image_garbage)
            for comp_i in np.arange(1, stats.shape[0]):
                past_image = deepcopy(self.binary_video[time_start + t, :, :])
                with_new_comp = new_order == comp_i
                past_image[with_new_comp] = 1
                nb_comp, image_garbage = cv2.connectedComponents(past_image)
                if nb_comp == 2:
                    self.binary_video[time_start + t, :, :][with_new_comp] = 1
        #self.expanded_shape[self.expanded_shape > 0] = 1
        #self.binary_video[time_end:, :, :] += self.expanded_shape
        for t in np.arange(time_end, self.binary_video.shape[0]):
            self.binary_video[t, :, :][np.nonzero(self.expanded_shape)] = 1
        last_image = self.binary_video[t, :, :] + self.binary_video[t - 1, :, :]
        last_image[last_image > 0] = 1
        self.binary_video[-1, :, :] = last_image
        return self.binary_video

    def find_expansion_timings(self):
        max_t = self.binary_video.shape[0] - 1
        dilated_one = cv2.dilate(self.expanded_shape, cross_33)
        # Find the time at which the nearest pixel of the expanded_shape si reached by the main shape
        closest_pixels = np.zeros(self.main_shape.shape, dtype=np.uint8)
        closest_pixels[self.expanded_shape == np.max(dilated_one)] = 1
        expand_start = max_t
        # Loop until there is no overlap between the dilated added shape and the original shape
        # Stop one frame before in order to obtain the exact reaching moment.
        while np.any(self.binary_video[expand_start - 1, :, :] * closest_pixels):
            expand_start -= 1

        # Find the relationship between distance and time
        distance_against_time, time_start, time_end = get_radius_distance_against_time(
            self.draft_seg[expand_start:(max_t + 1), :, :], dilated_one)
        time_start += expand_start
        time_end += expand_start
        return distance_against_time, time_start, time_end
