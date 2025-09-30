#!/usr/bin/env python3
"""
This script contains the class for detecting networks out of a grayscale image of Physarum polycephalum
"""

# A completely different strategy could be to segment the network by layers of luminosity.
# The first layer captures the brightest veins and replace their pixels by background pixels.
# The second layer captures other veins, (make sure that they are connected to the first?) and replace their pixels too.
# During one layer segmentation, the algorithm make sure that all detected veins are as long as possible
# but less long than and connected to the previous.

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from poetry.console.commands import self

from cellects.image_analysis.morphological_operations import square_33, cross_33, cc, Ellipse, CompareNeighborsWithValue, get_contours, get_all_line_coordinates, get_line_points
from cellects.utils.utilitarian import remove_coordinates
from cellects.utils.formulas import *
from cellects.utils.load_display_save import *
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.image_segmentation import generate_color_space_combination, rolling_window_segmentation, binary_quality_index
from numba.typed import Dict as TDict
from skimage import morphology
from skimage.segmentation import flood_fill
from skimage.filters import frangi, sato, threshold_otsu
from skimage.measure import perimeter
from collections import deque
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
import networkx as nx

# 8-connectivity neighbors
neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]
neighbors_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]



class NetworkDetection:
    def __init__(self, greyscale_image, possibly_filled_pixels, lighter_background, add_rolling_window=False, origin_to_add=None, best_result=None):
        self.greyscale_image = greyscale_image
        self.lighter_background = lighter_background
        self.possibly_filled_pixels = possibly_filled_pixels
        self.best_result = best_result
        self.add_rolling_window = add_rolling_window
        self.origin_to_add = origin_to_add
        self.frangi_beta = 1.
        self.frangi_gamma = 1.
        self.black_ridges = True

    def apply_frangi_variations(self):
        """Apply 12 variations of Frangi filter"""
        results = []

        # Parameter variations for Frangi filter
        frangi_sigmas = {
            's_fine_vessels': [0.75],
            'fine_vessels': [0.5, 1.0],  # Very fine capillaries, thin fibers
            'small_vessels': [1.0, 2.0],  # Small vessels, fine structures
            'multi_scale_medium': [1.0, 2.0, 3.0],  # Standard multi-scale
            'ultra_fine': [0.3, 0.5, 0.8],  # Ultra-fine structures
            'comprehensive': [0.5, 1.0, 2.0, 4.0],  # Multi-scale
            'retinal_vessels': [1.0, 2.0, 4.0, 8.0],  # Optimized for retinal imaging
            'microscopy': [0.5, 1.0, 1.5, 2.5],  # Microscopy applications
            'broad_spectrum': [0.5, 1.5, 3.0, 6.0, 10.0]
        }

        for i, (key, sigmas) in enumerate(frangi_sigmas.items()):
            # Apply Frangi filter
            frangi_result = frangi(self.greyscale_image, sigmas=sigmas, beta=self.frangi_beta, gamma=self.frangi_gamma, black_ridges=self.black_ridges)

            # Apply both thresholding methods
            # Method 1: Otsu thresholding
            thresh_otsu = threshold_otsu(frangi_result)
            binary_otsu = frangi_result > thresh_otsu
            # binary_otsu = binary_otsu * self.possibly_filled_pixels
            # if self.origin_to_add is not None:
            #     binary_otsu = np.logical_or(binary_otsu, self.origin_to_add)
            # binary_otsu = cv2.morphologyEx(binary_otsu.astype(np.uint8), cv2.MORPH_CLOSE, cross_33) # circle_55
            quality_otsu = binary_quality_index(self.possibly_filled_pixels * binary_otsu)

            # Method 2: Rolling window thresholding

            # Store results
            results.append({
                'method': f'f_{sigmas}_thresh',
                'binary': binary_otsu,
                'quality': quality_otsu,
                'filtered': frangi_result,
                'filter': f'frangi',
                'rolling_window': False,
                'sigmas': sigmas
            })
            # Method 2: Rolling window thresholding
            if self.add_rolling_window:
                binary_rolling = rolling_window_segmentation(frangi_result, self.possibly_filled_pixels, patch_size=(10, 10))
                quality_rolling = binary_quality_index(binary_rolling)
                results.append({
                    'method': f'f_{sigmas}_roll',
                    'binary': binary_rolling,
                    'quality': quality_rolling,
                    'filtered': frangi_result,
                    'filter': f'frangi',
                    'rolling_window': True,
                    'sigmas': sigmas
                })

        return results


    def apply_sato_variations(self):
        """Apply 12 variations of sato filter"""
        results = []

        # Parameter variations for Frangi filter
        sigmas_list = [
            [1], [2], [3], [1, 2], [2, 3], [1, 3],
            [1, 2, 3], [0.5, 1], [1, 4], [0.5, 2],
            [2, 4], [1, 2, 4]
        ]
        sato_sigmas = {
            'super_small_tubes': [0.01, 0.05, 0.1, 0.15],  #
            'small_tubes': [0.1, 0.2, 0.4, 0.8],  #
            's_thick_ridges': [0.25, 0.75],  # Thick ridges/tubes
            'small_multi_scale': [0.1, 0.2, 0.4, 0.8, 1.6],  #
            'fine_ridges': [0.8, 1.5],  # Fine ridge detection
            'medium_ridges': [1.5, 3.0],  # Medium ridge structures
            'multi_scale_fine': [0.8, 1.5, 2.5],  # Multi-scale fine detection
            'multi_scale_standard': [1.0, 2.5, 5.0],  # Standard multi-scale
            'edge_enhanced': [0.5, 1.0, 2.0],  # Edge-enhanced detection
            'noise_robust': [1.5, 2.5, 4.0],  # Robust to noise
            'fingerprint': [1.0, 1.5, 2.0, 3.0],  # Fingerprint ridge detection
            'geological': [2.0, 5.0, 10.0, 15.0]  # Geological structures
        }

        for i, (key, sigmas) in enumerate(sato_sigmas.items()):
            # Apply sato filter
            sato_result = sato(self.greyscale_image, sigmas=sigmas, black_ridges=self.black_ridges, mode='reflect')

            # Apply both thresholding methods
            # Method 1: Otsu thresholding
            thresh_otsu = threshold_otsu(sato_result)
            binary_otsu = sato_result > thresh_otsu
            # binary_otsu = binary_otsu * self.possibly_filled_pixels
            # if self.origin_to_add is not None:
            #     binary_otsu = np.logical_or(binary_otsu, self.origin_to_add)
            # binary_otsu = cv2.morphologyEx(binary_otsu.astype(np.uint8), cv2.MORPH_CLOSE, cross_33)
            quality_otsu = binary_quality_index(self.possibly_filled_pixels * binary_otsu)


            # Store results
            results.append({
                'method': f's_{sigmas}_thresh',
                'binary': binary_otsu,
                'quality': quality_otsu,
                'filtered': sato_result,
                'filter': f'sato',
                'rolling_window': False,
                'sigmas': sigmas
            })

            # Method 2: Rolling window thresholding
            if self.add_rolling_window:
                binary_rolling = rolling_window_segmentation(sato_result, self.possibly_filled_pixels, patch_size=(10, 10))
                quality_rolling = binary_quality_index(binary_rolling)

                results.append({
                    'method': f's_{sigmas}_roll',
                    'binary': binary_rolling,
                    'quality': quality_rolling,
                    'filtered': sato_result,
                    'filter': f'sato',
                    'rolling_window': True,
                    'sigmas': sigmas
                })

        return results


    def get_best_network_detection_method(self):
        frangi_res = self.apply_frangi_variations()
        sato_res = self.apply_sato_variations()
        self.all_results = frangi_res + sato_res
        self.quality_metrics = np.array([result['quality'] for result in self.all_results])
        self.best_idx = np.argmax(self.quality_metrics)
        self.best_result = self.all_results[self.best_idx]
        self.incomplete_network = self.best_result['binary'] * self.possibly_filled_pixels


    def detect_network(self):
        if self.best_result['filter'] == 'frangi':
            filtered_result = frangi(self.greyscale_image, sigmas=self.best_result['sigmas'], beta=self.frangi_beta, gamma=self.frangi_gamma, black_ridges=self.black_ridges)
        else:
            filtered_result = sato(self.greyscale_image, sigmas=self.best_result['sigmas'], black_ridges=self.black_ridges, mode='reflect')

        if self.best_result['rolling_window']:
            binary_image = rolling_window_segmentation(filtered_result, self.possibly_filled_pixels, patch_size=(10, 10))
        else:
            thresh_otsu = threshold_otsu(filtered_result)
            binary_image = filtered_result > thresh_otsu
        return binary_image

    def change_greyscale(self, img, c_space_dict):
        self.greyscale_image, g2 = generate_color_space_combination(img, list(c_space_dict.keys()), c_space_dict)

    def get_best_pseudopod_detection_method(self):
        # This adds a lot of noise in the network instead of only detecting pseudopods
        pad_skeleton, pad_distances = morphology.medial_axis(self.incomplete_network, return_distance=True, rng=0)
        pad_skeleton = pad_skeleton.astype(np.uint8)

        unique_distances = np.unique(pad_distances)
        counter = 1
        while pad_skeleton.sum() > 1000:
            counter += 1
            width_threshold = unique_distances[counter]
            pad_skeleton[pad_distances < width_threshold] = 0
        self.best_result['width_threshold'] = width_threshold
        potential_tips = np.nonzero(pad_skeleton)
        if self.lighter_background:
            max_tip_int = self.greyscale_image[potential_tips].max()
            low_pixels = self.greyscale_image <= max_tip_int  # mean_tip_int# max_tip_int
        else:
            min_tip_int = self.greyscale_image[potential_tips].min()
            high_pixels = self.greyscale_image >= min_tip_int  # mean_tip_int

        not_in_cell = 1 - self.possibly_filled_pixels
        error_threshold = not_in_cell.sum() * 0.01
        tolerances = np.arange(150, 0, - 1)
        for t_i, tolerance in enumerate(tolerances):
            potential_network = self.incomplete_network.copy()
            for y, x in zip(potential_tips[0],
                            potential_tips[1]):  # y, x =potential_tips[0][0], potential_tips[1][0]
                filled = flood_fill(image=self.greyscale_image, seed_point=(y, x), new_value=255, tolerance=tolerance)
                filled = filled == 255
                if (filled * not_in_cell).sum() > error_threshold:
                    break
                if self.lighter_background:
                    filled *= low_pixels
                else:
                    filled *= high_pixels
                potential_network[filled] = 1
            # show(potential_network)
            if not np.array_equal(potential_network, self.incomplete_network):
                break
        self.best_result['tolerance'] = tolerance

        complete_network = potential_network * self.possibly_filled_pixels
        complete_network = cv2.morphologyEx(complete_network, cv2.MORPH_CLOSE, cross_33)
        self.complete_network, stats, centers = cc(complete_network)
        self.complete_network[self.complete_network > 1] = 0


    def detect_pseudopods(self):
        pad_skeleton, pad_distances = morphology.medial_axis(self.incomplete_network, return_distance=True, rng=0)
        pad_skeleton = pad_skeleton.astype(np.uint8)
        pad_skeleton[pad_distances < self.best_result['width_threshold']] = 0
        potential_tips = np.nonzero(pad_skeleton)
        if self.lighter_background:
            max_tip_int = self.greyscale_image[potential_tips].max()
        else:
            min_tip_int = self.greyscale_image[potential_tips].min()

        complete_network = self.incomplete_network.copy()
        for y, x in zip(potential_tips[0],
                        potential_tips[1]):  # y, x =potential_tips[0][0], potential_tips[1][0]
            filled = flood_fill(image=self.greyscale_image, seed_point=(y, x), new_value=255, tolerance=self.best_result['tolerance'])
            filled = filled == 255
            if self.lighter_background:
                filled *= self.greyscale_image <= max_tip_int  # mean_tip_int# max_tip_int
            else:
                filled *= self.greyscale_image >= min_tip_int  # mean_tip_int
            complete_network[filled] = 1

        # Check that the current parameters do not produce images full of ones
        # If so, update the width_threshold and tolerance parameters
        not_in_cell = 1 - self.possibly_filled_pixels
        error_threshold = not_in_cell.sum() * 0.1
        if (complete_network * not_in_cell).sum() > error_threshold:
            self.get_best_network_detection_method()
        else:
            complete_network = complete_network * self.possibly_filled_pixels
            complete_network = cv2.morphologyEx(complete_network, cv2.MORPH_CLOSE, cross_33)
            self.complete_network, stats, centers = cc(complete_network)
            self.complete_network[self.complete_network > 1] = 0


def get_skeleton_and_widths(pad_network, pad_origin=None, pad_origin_centroid=None):
    pad_skeleton, pad_distances = morphology.medial_axis(pad_network, return_distance=True, rng=0)
    pad_skeleton = pad_skeleton.astype(np.uint8)
    if pad_origin is not None:
        pad_skeleton, pad_distances, pad_origin_contours = add_central_contour(pad_skeleton, pad_distances, pad_origin, pad_network, pad_origin_centroid)
    else:
        pad_origin_contours = None
        # a = pad_skeleton[821:828, 643:649]
        # new_skeleton2[821:828, 643:649]
    pad_skeleton, pad_distances = remove_small_loops(pad_skeleton, pad_distances)

    # width = 10
    # pad_skeleton[pad_distances > width] = 0
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_distances *= pad_skeleton
    # print(pad_skeleton.sum())
    return pad_skeleton, pad_distances, pad_origin_contours
    # width = 10
    # skel_size  = skeleton.sum()
    # while width > 0 and skel_size > skeleton.sum() * 0.75:
    #     width -= 1
    #     skeleton = skeleton.copy()
    #     skeleton[distances > width] = 0
    #     # Only keep the largest connected component
    #     skeleton, stats, _ = cc(skeleton)
    #     skeleton[skeleton > 1] = 0
    #     skel_size = skeleton.sum()
    # skeleton = pad_skeleton.copy()
    # Remove the origin

def remove_small_loops(pad_skeleton, pad_distances=None):
    """
    New version:
    New rule to add: when there is the pattern
    [[x, 1, x],
    [1, 0, 1],
    [x, 1, x]]
    Add 1 at the center when all other 1 are 3 connected
    Otherwise, just remove the 1 that are 2 connected

    Previous version:
    When zeros are surrounded by 4-connected ones and only contain 0 on their diagonal, replace 1 by 0
    and put 1 in the center



    :param pad_skeleton:
    :return:
    """

    # # New version:
    # cnv8 = CompareNeighborsWithValue(pad_skeleton, 8)
    # cnv8.is_equal(1, and_itself=False)
    # cnv4_false = CompareNeighborsWithValue(pad_skeleton, 4)
    # cnv4_false.is_equal(1, and_itself=False)
    # loop_y, loop_x = np.nonzero(cnv4_false.equal_neighbor_nb == 4)
    # for y, x in zip(loop_y, loop_x):
    #     top, bot, left, right = cnv8.equal_neighbor_nb[y - 1, x], cnv8.equal_neighbor_nb[y + 1, x], cnv8.equal_neighbor_nb[y, x - 1], cnv8.equal_neighbor_nb[y, x + 1]
    #     connected_with_2_pixels = np.array((top, bot, left, right)) == 2
    #
    #     if connected_with_2_pixels.any():
    #         if top == 2:
    #             pad_skeleton[y - 1, x] = 0
    #             if pad_distances is not None:
    #                 pad_distances[y - 1, x] = 0.
    #         if bot == 2:
    #             pad_skeleton[y + 1, x] = 0
    #             if pad_distances is not None:
    #                 pad_distances[y + 1, x] = 0.
    #         if left == 2:
    #             pad_skeleton[y, x - 1] = 0
    #             if pad_distances is not None:
    #                 pad_distances[y, x - 1] = 0.
    #         if right == 2:
    #             pad_skeleton[y, x + 1] = 0
    #             if pad_distances is not None:
    #                 pad_distances[y, x + 1] = 0.
    #     else:
    #         pad_skeleton[y, x] = 1
    #         pad_distances[y, x] = 2.
    # if pad_distances is None:
    #     return pad_skeleton
    # else:
    #     return pad_skeleton, pad_distances

    # Previous version:
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    # potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)

    cnv_diag_0 = CompareNeighborsWithValue(pad_skeleton, 0)
    cnv_diag_0.is_equal(0, and_itself=True)

    cnv4_false = CompareNeighborsWithValue(pad_skeleton, 4)
    cnv4_false.is_equal(1, and_itself=False)

    loop_centers = np.logical_and((cnv4_false.equal_neighbor_nb == 4), cnv_diag_0.equal_neighbor_nb > 2).astype(np.uint8)
    # loop_centers = np.logical_and((cnv4_false.equal_neighbor_nb == 4), cnv_diag_0.equal_neighbor_nb == 4).astype(np.uint8)

    surrounding = cv2.dilate(loop_centers, kernel=square_33)
    surrounding -= loop_centers
    surrounding = surrounding * cnv8.equal_neighbor_nb

    # Every 2 can be replaced by 0 if the loop center becomes 1
    filled_loops = pad_skeleton.copy()
    filled_loops[surrounding == 2] = 0
    filled_loops += loop_centers

    # Prev:
    # new_pad_skeleton = morphology.skeletonize(filled_loops, method='zhang')
    new_pad_skeleton = morphology.skeletonize(filled_loops, method='lee')
    # # Now
    # new_pad_skeleton = filled_loops
    # new_cnv4, new_cnv8 = get_neighbor_comparisons(filled_loops)
    # new_cnv8.equal_neighbor_nb

    # Put the new pixels in pad_distances
    new_pixels = new_pad_skeleton * (1 - pad_skeleton)
    pad_skeleton = new_pad_skeleton.astype(np.uint8)
    if pad_distances is None:
        return pad_skeleton
    else:
        pad_distances[np.nonzero(new_pixels)] = np.nan # 2. # Put nearest value instead?
        pad_distances *= pad_skeleton
        # for yi, xi in zip(npY, npX): # yi, xi = npY[0], npX[0]
        #     distances[yi, xi] = 2.
        return pad_skeleton, pad_distances


def get_neighbor_comparisons(pad_skeleton):
    cnv4 = CompareNeighborsWithValue(pad_skeleton, 4)
    cnv4.is_equal(1, and_itself=True)
    cnv8 = CompareNeighborsWithValue(pad_skeleton, 8)
    cnv8.is_equal(1, and_itself=True)
    return cnv4, cnv8


def keep_one_connected_component(pad_skeleton):
    """
    """
    nb_pad_skeleton, stats, _ = cc(pad_skeleton)
    pad_skeleton[nb_pad_skeleton > 1] = 0
    # nb, nb_pad_skeleton = cv2.connectedComponents(pad_skeleton)
    # pad_skeleton[nb_pad_skeleton > 1] = 0
    return pad_skeleton
# def get_vertices_and_edges_from_skeleton(pad_skeleton):


def keep_components_larger_than_one(pad_skeleton):
    """
    """
    nb_pad_skeleton, stats, _ = cc(pad_skeleton)
    for i in np.nonzero(stats[:, 4] == 1)[0]:
        pad_skeleton[nb_pad_skeleton == i] = 0
    # nb, nb_pad_skeleton = cv2.connectedComponents(pad_skeleton)
    # pad_skeleton[nb_pad_skeleton > 1] = 0
    return pad_skeleton


def get_vertices_and_tips_from_skeleton(pad_skeleton):
    """
    Find the vertices from a skeleton according to the following rules:
    - Network terminations at the border are nodes
    - The 4-connected nodes have priority over 8-connected nodes
    :return:
    """
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, pad_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    # pad_edges = (1 - pad_vertices) * pad_skeleton
    # pathway = Path(f"/Users/Directory/Data/dossier1/plots")
    # save_vertices_and_tips_image(pad_skeleton, pad_vertices, pad_potential_tips, pathway)
    return pad_vertices, pad_tips
    # return pad_skeleton, pad_vertices, pad_edges, pad_tips


def get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8):
    # All pixels having only one neighbor, and containing the value 1, are terminations for sure
    potential_tips = np.zeros(pad_skeleton.shape, dtype=np.uint8)
    potential_tips[cnv8.equal_neighbor_nb == 1] = 1
    # Add more terminations using 4-connectivity
    # If a pixel is 1 (in 4) and all its neighbors are neighbors (in 4), it is a termination

    coord1_4 = cnv4.equal_neighbor_nb == 1
    if np.any(coord1_4):
        coord1_4 = np.nonzero(coord1_4)
        for y1, x1 in zip(coord1_4[0], coord1_4[1]):
            # y1, x1 = 3,5
            # If, in the neighborhood of the 1 (in 4), all (in 8) its neighbors are 4-connected together, and none of them are terminations, the 1 is a termination
            is_4neigh = cnv4.equal_neighbor_nb[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] != 0
            all_4_connected = pad_skeleton[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] == is_4neigh
            is_not_term = 1 - potential_tips[y1, x1]
            # is_not_term = (1 - potential_tips[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)])
            if np.all(all_4_connected * is_not_term):
                is_4neigh[1, 1] = 0
                is_4neigh = np.pad(is_4neigh, [(1,), (1,)], mode='constant')
                cnv_4con = CompareNeighborsWithValue(is_4neigh, 4)
                cnv_4con.is_equal(1, and_itself=True)
                all_connected = (is_4neigh.sum() - (cnv_4con.equal_neighbor_nb > 0).sum()) == 0
                # If they are connected, it can be a termination
                if all_connected:
                    # print('h',y1, x1)
                    # If its closest neighbor is above 3 (in 8), this one is also a node
                    is_closest_above_3 = cnv8.equal_neighbor_nb[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] * cross_33 > 3
                    if np.any(is_closest_above_3):
                        # print('h',y1, x1)
                        Y, X = np.nonzero(is_closest_above_3)
                        Y += y1 - 1
                        X += x1 - 1
                        potential_tips[Y, X] = 1
                    potential_tips[y1, x1] = 1
    return potential_tips


def get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8): # potential_tips=pad_tips
    """
    1. Find connected vertices using the number of 8-connected neighbors
    2. The ones having 3 neighbors:
        - are connected when in the neighborhood of the 3, there is at least a 2 (in 8) that is 0 (in 4), and not a termination
        but this, only when it does not create an empty cross.... To do
    """

    # Initiate the vertices final matrix as a copy of the potential_tips
    pad_vertices = deepcopy(potential_tips)
    for neighbor_nb in [8, 7, 6, 5, 4]:
        # All pixels having neighbor_nb neighbor are potential vertices
        potential_vertices = np.zeros(potential_tips.shape, dtype=np.uint8)

        potential_vertices[cnv8.equal_neighbor_nb == neighbor_nb] = 1
        # remove the false intersections that are a neighbor of a previously detected intersection
        # Dilate vertices to make sure that no neighbors of the current potential vertices are already vertices.
        dilated_previous_intersections = cv2.dilate(pad_vertices, cross_33, iterations=1)
        # dilated_previous_intersections = cv2.dilate((cnv8.equal_neighbor_nb > neighbor_nb).astype(np.uint8), cross_33, iterations=1)
        potential_vertices *= (1 - dilated_previous_intersections)
        pad_vertices[np.nonzero(potential_vertices)] = 1

    # Having 3 neighbors is ambiguous
    with_3_neighbors = cnv8.equal_neighbor_nb == 3
    if np.any(with_3_neighbors):
        # We compare 8-connections with 4-connections
        # We loop over all 3 connected
        coord_3 = np.nonzero(with_3_neighbors)
        for y3, x3 in zip(coord_3[0], coord_3[1]):
            # y3, x3 = 3,7
            # If, in the neighborhood of the 3, there is at least a 2 (in 8) that is 0 (in 4), and not a termination: the 3 is a node
            has_2_8neigh = cnv8.equal_neighbor_nb[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)] > 0  # 1
            # is_term = potential_tips[y3, x3]
            # is_not_term = np.logical_not(potential_tips[(y3-1):(y3+2), (x3-1):(x3+2)])
            has_2_8neigh_without_focal = has_2_8neigh.copy()
            has_2_8neigh_without_focal[1, 1] = 0
            # all_are_nodes = np.array_equal(has_2_8neigh_without_focal, pad_vertices[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)])
            node_but_not_term = pad_vertices[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)] * (1 - potential_tips[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)])
            all_are_node_but_not_term = np.array_equal(has_2_8neigh_without_focal, node_but_not_term)
            # has_0_4neigh = cnv4.equal_neighbor_nb[(y3-1):(y3+2), (x3-1):(x3+2)] == 0
            if np.any(has_2_8neigh * (1 - all_are_node_but_not_term)):
                # At least 3 of the 8neigh are not connected:
                has_2_8neigh_without_focal = np.pad(has_2_8neigh_without_focal, [(1,), (1,)], mode='constant')
                cnv_8con = CompareNeighborsWithValue(has_2_8neigh_without_focal, 4)
                cnv_8con.is_equal(1, and_itself=True)
                disconnected_nb = has_2_8neigh_without_focal.sum() - (cnv_8con.equal_neighbor_nb > 0).sum()
                # disconnected_nb, shape = cv2.connectedComponents(has_2_8neigh_without_focal.astype(np.uint8), connectivity=4)
                # nb_not_connected = has_2_8neigh_without_focal.sum() - (disconnected_nb - 1)
                if disconnected_nb > 2:
                    # print(y3, x3)
                    pad_vertices[y3, x3] = 1
    # Now there may be too many vertices:
    # - Those that are 4-connected:
    nb, sh, st, ce = cv2.connectedComponentsWithStats(pad_vertices, connectivity=4)
    problematic_vertices = np.nonzero(st[:, 4] > 1)[0][1:]
    for prob_v in problematic_vertices: # prob_v = problematic_vertices[0]
        vertices_group = sh == prob_v
        # If there is a tip in the group, do
        if np.any(potential_tips[vertices_group]):
            # Change the most connected one from tip to vertex
            curr_neighbor_nb = cnv8.equal_neighbor_nb * vertices_group
            wrong_tip = np.nonzero(curr_neighbor_nb == curr_neighbor_nb.max())
            potential_tips[wrong_tip] = 0
        else:
            #  otherwise do:
            # Find the most 4-connected one, and check whether
            # its 4 connected neighbors have 1 or more other connexions
            # 1. # Find the most 4-connected one:
            vertices_group_4 = cnv4.equal_neighbor_nb * vertices_group
            max_con = vertices_group_4.max()
            most_con = np.nonzero(vertices_group_4 == max_con)
            # 2. Check its 4-connected neighbors and remove those having only 1 other 8-connexion
            # cnv8.equal_neighbor_nb
            skel_copy = pad_skeleton.copy()
            skel_copy[most_con] = 0
            skel_copy[most_con[0] - 1, most_con[1]] = 0
            skel_copy[most_con[0] + 1, most_con[1]] = 0
            skel_copy[most_con[0], most_con[1] - 1] = 0
            skel_copy[most_con[0], most_con[1] + 1] = 0
            sub_cnv8 = CompareNeighborsWithValue(skel_copy, 8)
            sub_cnv8.is_equal(1, and_itself=False)
            # Remove those having
            # v_to_remove = (vertices_group_4 * sub_cnv8.equal_neighbor_nb) == 1
            v_to_remove = ((vertices_group_4 > 0) * sub_cnv8.equal_neighbor_nb) == 1
            pad_vertices[v_to_remove] = 0
            # Remove the less (4-vertex) connected ones
            # Faire des dessins, trouver le truc...

            # if st[prob_v, 2] < 3 and st[prob_v, 3] < 3: # 3 in a row condition
                # If they are not longer than 2 pixels wide
                # And if they have at least one vertex connexion
            # print(vertices_group_4)
            # max_con = vertices_group_4.max()
            # if max_con > 2:
            #     less_connected_ones = np.logical_and(vertices_group_4 > 0, vertices_group_4 < max_con)
            #     pad_vertices[less_connected_ones] = 0
    # Other vertices to remove:
    # - Those that are forming a cross with 0 at the center while the skeleton contains 1
    cnv4_false = CompareNeighborsWithValue(pad_vertices, 4)
    cnv4_false.is_equal(1, and_itself=False)
    cross_vertices = cnv4_false.equal_neighbor_nb == 4
    wrong_cross_vertices = cross_vertices * pad_skeleton
    if wrong_cross_vertices.any():
        pad_vertices[np.nonzero(wrong_cross_vertices)] = 1
        cross_fix = cv2.dilate(wrong_cross_vertices, kernel=cross_33, iterations=1)
        # Remove the 4-connected vertices that have no more than 4 8-connected neighbors
        # i.e. the three on the side of the surrounded 0 and only one on edge on the other side
        cross_fix = ((cnv8.equal_neighbor_nb * cross_fix) == 4) * (1 - wrong_cross_vertices)
        pad_vertices *= (1 - cross_fix)
    return pad_vertices, potential_tips


def get_branches_and_tips_coord(pad_vertices, pad_tips):
    pad_branches = pad_vertices - pad_tips
    branch_v_coord = np.transpose(np.array(np.nonzero(pad_branches)))
    tips_coord = np.transpose(np.array(np.nonzero(pad_tips)))
    return branch_v_coord, tips_coord


class EdgeIdentification:
    def __init__(self, pad_skeleton):
        self.pad_skeleton = pad_skeleton
        self.remaining_vertices = None
        self.vertices = None
        self.growing_vertices = None
        self.im_shape = pad_skeleton.shape

    def get_vertices_and_tips_coord(self):
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(self.pad_skeleton)
        self.non_tip_vertices, self.tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)

    def get_tipped_edges(self):
        self.pad_skeleton = keep_one_connected_component(self.pad_skeleton)
        self.vertices_branching_tips, self.edge_lengths, self.edge_pix_coord = find_closest_vertices(self.pad_skeleton,
                                                                                        self.non_tip_vertices,
                                                                                        self.tips_coord[:, :2])
        # Remove the tips that did not manage to connect another
        # np.isnan(self.edge_lengths).sum()

    def remove_tipped_edge_smaller_than_branch_width(self, pad_distances):
        """
        Problem: when an edge is removed and its branching vertex is not anymore a vertex,
        if another tipped edge was connected to this vertex, its length, pixel coord, and branching v are wrong.
        Solution, re-run self.get_tipped_edges() after that

        a=pad_skeleton.copy()
        a[self.tips_coord[:, 0],self.tips_coord[:, 1]] = 2
        aa=a[632:645, 638:651]
        Yt,Xt=632+10,638+1
        np.nonzero(np.all(self.tips_coord[:, :2] == [t1Y, t1X], axis=1))
        np.nonzero(np.all(self.tips_coord[:, :2] == [t2Y, t2X], axis=1))
        i = 3153
        """
        self.pad_distances = pad_distances
        # Identify edges that are smaller than the width of the branch it is attached to
        tipped_edges_to_remove = np.zeros(self.edge_lengths.shape[0], dtype=bool)
        # connecting_vertices_to_remove = np.zeros(self.vertices_branching_tips.shape[0], dtype=bool)
        branches_to_remove = np.zeros(self.non_tip_vertices.shape[0], dtype=bool)
        new_edge_pix_coord = []
        remaining_tipped_edges_nb = 0
        for i in range(len(self.edge_lengths)): # i = 3142 #1096 # 974 # 222
            Y, X = self.vertices_branching_tips[i, 0], self.vertices_branching_tips[i, 1]
            edge_bool = self.edge_pix_coord[:, 2] == i + 1
            eY, eX = self.edge_pix_coord[edge_bool, 0], self.edge_pix_coord[edge_bool, 1]
            if np.nanmax(pad_distances[(Y - 1): (Y + 2), (X - 1): (X + 2)]) >= self.edge_lengths[i]:
                # print(i)
                tipped_edges_to_remove[i] = True
                # Remove the edge
                self.pad_skeleton[eY, eX] = 0
                # Remove the tip
                self.pad_skeleton[self.tips_coord[i, 0], self.tips_coord[i, 1]] = 0
                # # and its 4-connected neighborhood:
                # self.pad_skeleton[self.tips_coord[i, 0] - 1, self.tips_coord[i, 1]] = 0
                # self.pad_skeleton[self.tips_coord[i, 0] + 1, self.tips_coord[i, 1]] = 0
                # self.pad_skeleton[self.tips_coord[i, 0], self.tips_coord[i, 1] - 1] = 0
                # self.pad_skeleton[self.tips_coord[i, 0], self.tips_coord[i, 1] + 1] = 0
                #
                # nb, sh = cv2.connectedComponents(self.pad_skeleton)
                # if nb != 2:
                #     break

                # Remove the coordinates corresponding to that edge
                self.edge_pix_coord = np.delete(self.edge_pix_coord, edge_bool, 0)

                # check whether the connecting vertex remains a vertex of not
                pad_sub_skeleton = np.pad(self.pad_skeleton[(Y - 2): (Y + 3), (X - 2): (X + 3)], [(1,), (1,)],
                                          mode='constant')
                sub_vertices, sub_tips = get_vertices_and_tips_from_skeleton(pad_sub_skeleton)
                # If the vertex does not connect at least 3 edges anymore, remove its vertex label
                # if sub_tips[3, 3] + (1 - sub_vertices[3, 3]):
                if sub_vertices[3, 3] == 0:
                    # connecting_vertices_to_remove[i] = True

                    vertex_to_remove = np.nonzero(np.logical_and(self.non_tip_vertices[:, 0] == Y, self.non_tip_vertices[:, 1] == X))[0]
                    branches_to_remove[vertex_to_remove] = True
                # If that pixel became a tip connected to another vertex remove it from the skeleton
                if sub_tips[3, 3]:
                    if sub_vertices[2:5, 2:5]. sum() > 1:
                        self.pad_skeleton[Y, X] = 0
                        self.edge_pix_coord = np.delete(self.edge_pix_coord, np.all(self.edge_pix_coord[:, :2] == [Y, X], axis=1), 0)
                        vertex_to_remove = np.nonzero(np.logical_and(self.non_tip_vertices[:, 0] == Y, self.non_tip_vertices[:, 1] == X))[0]
                        branches_to_remove[vertex_to_remove] = True
            else:
                remaining_tipped_edges_nb += 1
                new_edge_pix_coord.append(np.stack((eY, eX, np.repeat(remaining_tipped_edges_nb, len(eY))), axis=1))

        # Check that excedent connected components are 1 pixel size, if so:
        # It means that they were neighbors to removed tips and not necessary for the skeleton
        nb, sh = cv2.connectedComponents(self.pad_skeleton)
        if nb > 2:
            for i in range(2, nb):
                excedent = sh == i
                if (excedent).sum() == 1:
                    self.pad_skeleton[excedent] = 0
                # else:
                #     print("More than one pixel area excedent components exists")

        # Remove in distances the pixels removed in skeleton:
        self.pad_distances *= self.pad_skeleton

        # update edge_pix_coord
        self.edge_pix_coord = np.vstack(new_edge_pix_coord)

        # Remove tips connected to very small edges
        self.tips_coord = np.delete(self.tips_coord, tipped_edges_to_remove, 0)
        # Add corresponding edge names
        self.tips_coord = np.hstack((self.tips_coord, np.arange(1, len(self.tips_coord) + 1)[:, None]))

        # Within all branching (non-tip) vertices, keep those that did not lose their vertex status because of the edge removal
        self.non_tip_vertices = np.delete(self.non_tip_vertices, branches_to_remove, 0)

        # Get the branching vertices who kept their typped edge
        self.vertices_branching_tips = np.delete(self.vertices_branching_tips, tipped_edges_to_remove, 0)

        # Within all branching (non-tip) vertices, keep those that do not connect a tipped edge.
        v_branching_tips_in_branching_v = find_common_coord(self.non_tip_vertices, self.vertices_branching_tips[:, :2])
        self.remaining_vertices = np.delete(self.non_tip_vertices, v_branching_tips_in_branching_v, 0)
        ordered_v_coord = np.vstack((self.tips_coord[:, :2], self.vertices_branching_tips[:, :2], self.remaining_vertices))

        # tips = self.tips_coord
        # branching_any_edge = self.non_tip_vertices
        # branching_typped_edges = self.vertices_branching_tips
        # branching_no_typped_edges = self.remaining_vertices

        self.get_vertices_and_tips_coord()
        self.get_tipped_edges()

    def label_tipped_edges_and_their_vertices(self):
        self.tip_number = self.tips_coord.shape[0]

        # Stack vertex coordinates in that order: 1. Tips, 2. Vertices branching tips, 3. All remaining vertices
        ordered_v_coord = np.vstack((self.tips_coord[:, :2], self.vertices_branching_tips[:, :2], self.non_tip_vertices))
        ordered_v_coord = np.unique(ordered_v_coord, axis=0)

        # Create arrays to store edges and vertices labels
        self.numbered_vertices = np.zeros(self.im_shape, dtype=np.uint32)
        self.numbered_vertices[ordered_v_coord[:, 0], ordered_v_coord[:, 1]] = np.arange(1, ordered_v_coord.shape[0] + 1)
        self.vertices = None

        # Name edges from 1 to the number of edges connecting tips and set the vertices labels from all tips to their connected vertices:
        self.edges_labels = np.zeros((self.tip_number, 3), dtype=np.uint32)
        # edge label:
        self.edges_labels[:, 0] = np.arange(self.tip_number) + 1
        # tip label:
        self.edges_labels[:, 1] = self.numbered_vertices[self.tips_coord[:, 0], self.tips_coord[:, 1]]
        # vertex branching tip label:
        self.edges_labels[:, 2] = self.numbered_vertices[self.vertices_branching_tips[:, 0], self.vertices_branching_tips[:, 1]]

        # Remove duplicates in vertices_branching_tips
        self.vertices_branching_tips = np.unique(self.vertices_branching_tips[:, :2], axis=0)

    def identify_all_other_edges(self):
        # I. Identify edges connected to connected vertices and their own connexions:
        # II. Identify all remaining edges
        self.obsn = np.zeros_like(self.numbered_vertices)  # DEBUG
        self.obsn[np.nonzero(self.pad_skeleton)] = 1  # DEBUG
        self.obsn[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 2  # DEBUG
        self.obi = 2  # DEBUG

        # I.1. Identify edges connected to touching vertices:
        # First, create another version of these arrays, where we remove every already detected edge and their tips
        cropped_skeleton = self.pad_skeleton.copy()
        cropped_skeleton[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 0
        cropped_skeleton[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 0

        # non_tip_vertices does not need to be updated yet, because it only contains verified branching vertices
        cropped_non_tip_vertices = self.non_tip_vertices.copy()

        self.new_level_vertices = None
        # Fix the vertex_to_vertex_connexion problem
        # The problem with vertex_to_vertex_connexion is that since they are not separated by zeros,
        # they always atract each other instead of exploring other paths.
        # To fix this, we loop over each vertex group to
        # 1. Add one edge per inter-vertex connexion inside the group
        # 2. Remove all except one, and loop as many time as necessary.
        # Inside that second loop, we explore and identify every edge nearby.
        # Find every vertex_to_vertex_connexion
        v_grp_nb, v_grp_lab, v_grp_stats, vgc = cv2.connectedComponentsWithStats(
            (self.numbered_vertices > 0).astype(np.uint8), connectivity=8)
        max_v_nb = np.max(v_grp_stats[1:, 4])
        cropped_skeleton_list = []
        starting_vertices_list = []
        for v_nb in range(2, max_v_nb + 1):
            labels = np.nonzero(v_grp_stats[:, 4] == v_nb)[0]
            # pos = v_grp_stats[v_grp_stats[:, 4] == v_nb, :4]
            # y_start, y_end, x_start, x_end = pos[:, 1], pos[:, 1] + pos[:, 3], pos[:, 0], pos[:, 0] + pos[:, 2]
            coord_list = []
            for lab in labels:  # lab=labels[0]
                coord_list.append(np.nonzero(v_grp_lab == lab))
            for iter in range(v_nb):
                for lab_ in range(labels.shape[0]): # lab=labels[0]
                    cs = cropped_skeleton.copy()
                    sv = [] # np.empty((0, 2), dtype=np.int64)
                    v_c = coord_list[lab_]
                    # Save the current coordinate in the starting vertices array of this iteration
                    sv.append([v_c[0][iter], v_c[1][iter]])# = np.vstack((sv, [v_c[0][iter], v_c[1][iter]]))
                    # Remove one vertex coordinate to keep it from cs
                    v_y, v_x = np.delete(v_c[0], iter), np.delete(v_c[1], iter)
                    cs[v_y, v_x] = 0
                    cropped_skeleton_list.append(cs)
                    starting_vertices_list.append(np.array(sv))

        for cropped_skeleton, starting_vertices in zip(cropped_skeleton_list, starting_vertices_list):
            # cropped_skeleton, starting_vertices_coord = cropped_skeleton_list[0], starting_vertices_list[0]
            _, _ = self.identify_edges_connecting_a_vertex_list(cropped_skeleton, cropped_non_tip_vertices, starting_vertices)

        # obsn = np.zeros_like(self.obsn)
        # obsn[np.nonzero(self.pad_skeleton)] = 1
        # sub_pix = self.edge_pix_coord[self.edge_pix_coord[:, 2] > self.tips_coord[:, 2].max(), :]
        # obsn[sub_pix[:, 0], sub_pix[:, 1]] = 5
        # for group_i in all_connected_vertices:
        #     obsn[np.nonzero(v_grp_lab == group_i)] = 7
        # show(obsn)

        # I.2. Identify the connexions between connected vertices:
        all_connected_vertices = np.nonzero(v_grp_stats[:, 4] > 1)[0][1:]
        all_con_v_im = np.zeros_like(cropped_skeleton)
        for v_group in all_connected_vertices:
            all_con_v_im[v_grp_lab == v_group] = 1
        cropped_skeleton = all_con_v_im
        vertex_groups_coord = np.transpose(np.array(np.nonzero(cropped_skeleton)))
        # cropped_non_tip_vertices, starting_vertices_coord = vertex_groups_coord, vertex_groups_coord
        _, _ = self.identify_edges_connecting_a_vertex_list(cropped_skeleton, vertex_groups_coord, vertex_groups_coord)
        # self.edges_labels

        # II/ Identify all remaining edges
        self.obsn = np.zeros_like(self.numbered_vertices)  # DEBUG
        self.obsn[np.nonzero(self.pad_skeleton)] = 1  # DEBUG
        self.obsn[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 2  # DEBUG
        self.obi = 2  # DEBUG
        # show(self.obsn > 0) # DEBUG
        if self.new_level_vertices is not None:
            starting_vertices_coord = np.vstack((self.new_level_vertices[:, :2], self.vertices_branching_tips))
            starting_vertices_coord = np.unique(starting_vertices_coord, axis=0)
            # I get better coverage by starting only with this:
            # starting_vertices_coord = np.unique(new_level_vertices, axis=0)
            # Or this
            # starting_vertices_coord = np.unique(vertices_branching_tips, axis=0)
        else:
            # We start from the vertices connecting tips
            starting_vertices_coord = self.vertices_branching_tips.copy()

        # Remove the detected edges from cropped_skeleton:
        cropped_skeleton = self.pad_skeleton.copy()
        cropped_skeleton[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 0
        cropped_skeleton[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 0
        cropped_skeleton[vertex_groups_coord[:, 0], vertex_groups_coord[:, 1]] = 0

        # Reinitialize cropped_non_tip_vertices to browse all vertices except tips and groups
        cropped_non_tip_vertices = self.non_tip_vertices.copy()
        cropped_non_tip_vertices = remove_coordinates(cropped_non_tip_vertices, vertex_groups_coord)
        # Y, X = 730 + 4, 451 + 6
        # Y1, X1 = 730 + 18, 451 + 5
        # np.all(self.tips_coord[:,:2] == [Y,X], axis=1).any()
        # np.all(self.non_tip_vertices == [Y,X], axis=1).any()
        # np.all(starting_vertices_coord == [Y,X], axis=1).any()
        # np.all(starting_vertices_coord == [Y1,X1], axis=1).any()
        # np.all(cropped_non_tip_vertices == [Y,X], axis=1).any()
        # np.all(cropped_non_tip_vertices == [Y1,X1], axis=1).any()
        # np.nonzero(np.all(cropped_non_tip_vertices == [Y, X], axis=1))
        # np.nonzero(np.all(cropped_non_tip_vertices == [Y1, X1], axis=1))
        remaining_v = cropped_non_tip_vertices.shape[0] + 1
        while remaining_v > cropped_non_tip_vertices.shape[0]:
            remaining_v = cropped_non_tip_vertices.shape[0]
            cropped_skeleton, cropped_non_tip_vertices = self.identify_edges_connecting_a_vertex_list(cropped_skeleton, cropped_non_tip_vertices, starting_vertices_coord)
            # starting_vertices_coord = np.unique(self.new_level_vertices[:, :2], axis=0)
            # print(cropped_non_tip_vertices.shape)
            if self.new_level_vertices is None:
                break
            else:
                starting_vertices_coord = np.unique(self.new_level_vertices[:, :2], axis=0)


        # self.edge_pix_coord[:, 2].max()
        # len(self.edge_lengths)
        # self.edges_labels.shape[0]

        identified_skeleton = np.zeros_like(self.numbered_vertices)
        identified_skeleton[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 1
        identified_skeleton[self.non_tip_vertices[:, 0], self.non_tip_vertices[:, 1]] = 1
        identified_skeleton[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 1
        # identified_skeleton[np.nonzero(self.numbered_vertices)] = self.edges_labels.shape[0] + 1
        # not_identified = (identified_skeleton > 0) != self.pad_skeleton
        not_identified = (1 - identified_skeleton) * self.pad_skeleton
        # self.obsn[np.nonzero(not_identified)] = self.obi + 10
        # show(self.obsn)
        # show(not_identified)
        #a=not_identified[1190:1201,  543:553]

        # Find out the remaining non-identified pixels
        nb, sh, st, ce = cv2.connectedComponentsWithStats(not_identified.astype(np.uint8))

        # Handle the cases were edges are loops over only one vertex
        looping_edges = np.nonzero(st[:, 4 ] > 2)[0][1:]
        for loop_i in looping_edges: # loop_i = looping_edges[0]
            edge_i = (sh == loop_i).astype(np.uint8)
            dil_edge_i = cv2.dilate(edge_i, square_33)
            unique_vertices_im = self.numbered_vertices.copy()
            unique_vertices_im[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 0
            unique_vertices_im = dil_edge_i * unique_vertices_im
            unique_vertices = np.unique(unique_vertices_im)
            unique_vertices = unique_vertices[unique_vertices > 0]
            if len(unique_vertices) == 1:
                start, end = unique_vertices[0], unique_vertices[0]
                new_edge_lengths = edge_i.sum()
                new_edge_pix_coord = np.transpose(np.vstack((np.nonzero(edge_i))))
                new_edge_pix_coord = np.hstack((new_edge_pix_coord, np.repeat(1, new_edge_pix_coord.shape[0])[:, None])) # np.arange(1, new_edge_pix_coord.shape[0] + 1)[:, None]))
                self.update_edge_data(start, end, new_edge_lengths, new_edge_pix_coord)
                self.obsn[new_edge_pix_coord[:, 0], new_edge_pix_coord[:, 1]] -= 10  # DEBUG
            else:
                print(f"Other long edges cannot be identified: i={loop_i} of len={edge_i.sum()}")
        identified_skeleton[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 1

        # Check whether the 1 or 2 pixel size non-identified areas can be removed without breaking the skel
        one_pix = np.nonzero(st[:, 4 ] <= 2)[0] # == 1)[0]
        cutting_removal = []
        for pix_i in one_pix: #pix_i=one_pix[0]
            skel_copy = self.pad_skeleton.copy()
            y1, y2, x1, x2 = st[pix_i, 1], st[pix_i, 1] + st[pix_i, 3], st[pix_i, 0], st[pix_i, 0] + st[pix_i, 2]
            skel_copy[y1:y2, x1:x2][sh[y1:y2, x1:x2] == pix_i] = 0
            # skel_copy[st[pix_i, 1], st[pix_i, 0]] = 0
            nb1, sh1 = cv2.connectedComponents(skel_copy.astype(np.uint8), connectivity=8)
            if nb1 > 2:
                cutting_removal.append(pix_i)
            else:
                self.pad_skeleton[y1:y2, x1:x2][sh[y1:y2, x1:x2] == pix_i] = 0
                # self.pad_skeleton[st[pix_i, 1], st[pix_i, 0]] = 0
        if len(cutting_removal) > 0:
            print(f"These pixels break the skeleton when removed: {cutting_removal}")
        print(100 * (identified_skeleton > 0).sum() / self.pad_skeleton.sum())
        # show(identified_skeleton)
        # not_identified = (1 - identified_skeleton) * self.pad_skeleton
        # show(not_identified)

        # Protocol: show this, find prob area, its vertices, fill X1, X2,
        # look for when they disappear from starting, go in find_closest_vertices, update params
        # find the i with np.all, break, find if found_vertex, if written, check everything

        # edge_visu = not_identified * 10
        # # sub_pix = self.edge_pix_coord[self.edge_pix_coord[:, 2] <= self.tips_coord[:, 2].max(), :]
        # # not_identified[sub_pix[:, 0], sub_pix[:, 1]] = 1
        # edge_visu[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 4
        # edge_visu[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = self.edge_pix_coord[:, 2]
        # edge_visu[starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]] = 3
        # edge_visu[self.non_tip_vertices[:, 0], self.non_tip_vertices[:, 1]] = 2
        # edge_visu[np.nonzero(self.numbered_vertices)] = self.numbered_vertices[np.nonzero(self.numbered_vertices)]
        # a=pad_skeleton[1190:1201,  543:553]# all
        # a=edge_visu[1190:1201,  543:553]# all
        # Y, X = 700+30, 1040+37
        # Y, X = 700+30, 1040+36
        # self.numbered_vertices[Y,X]
        # Y1, X1 = 700+31, 1040+37
        # self.numbered_vertices[Y1, X1]
        # aaa=pad_skeleton[(Y-3):(Y+4), (X-3):(X+4)]# branching
        # aa=self.pad_skeleton[972:978, 565:569]# branching
        # np.any(a==10)
        # aa=pad_skeleton[1157:1179, 709:727]
        # aaa=self.pad_skeleton[1157:1179, 709:727]
        # np.array_equal(aa, aaa)
        #
        #
        # # Find the non identified shapes
        # nb, sh, st, ce = cv2.connectedComponentsWithStats(not_identified.astype(np.uint8))
        # marg = 4
        # xmin, xmax, ymin, ymax = st[:, 0] - marg, st[:, 0] + st[:, 2] + marg, st[:, 1] - marg, st[:, 1] + st[:, 3] + marg
        # large_sizes = np.nonzero(st[:, 4 ] > 1)[0][1:]
        # for i in large_sizes:
        #     visu = self.pad_skeleton.copy()
        #     visu[np.nonzero(not_identified)] = 2
        #     visu[self.non_tip_vertices[:, 0], self.non_tip_vertices[:, 1]] = 10
        #     visu[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 11
        #     print(visu[ymin[i]:ymax[i], xmin[i]:xmax[i]])
        # visu[ymin[i]:ymax[i], xmin[i]:xmax[i]]
        # self.pad_skeleton[ymin[i]:ymax[i], xmin[i]:xmax[i]]
        # pad_skeleton[ymin[i]:ymax[i], xmin[i]+10:xmax[i]]
        # ymin[i] + 2, ymax[i] - 3, xmin[i] + 10, xmax[i]
        # ymin[i], xmin[i]
        # ymi, yma, xmi, xma = 614, 623, 1097, 1109
        # ymi, yma, xmi, xma = 687, 692, 1054, 1060
        # ymi, yma, xmi, xma = 677, 693, 1045, 1060
        # sav= self.pad_skeleton[ymi:yma, xmi:xma].copy()
        # self.pad_skeleton[ymin[i]:ymax[i], xmin[i]:xmax[i]]
        # pad_skeleton[ymin[i]:ymax[i], xmin[i]:xmax[i]]
        # Y, X = 614+5, 1097+3
        # Y1, X1 = 614+5, 1097+8
        # self.numbered_vertices[Y, X]
        # self.numbered_vertices[Y1, X1]
        self.pad_distances *= self.pad_skeleton

    def identify_edges_connecting_a_vertex_list(self, cropped_skeleton, cropped_non_tip_vertices, starting_vertices_coord):
        explored_connexions_per_vertex = 0  # the maximal edge number that can connect a vertex
        new_connexions = True
        while new_connexions and explored_connexions_per_vertex < 5 and np.any(cropped_non_tip_vertices) and np.any(starting_vertices_coord):
            # print(new_connexions)
            explored_connexions_per_vertex += 1
            # 1. Find the ith closest vertex to each focal vertex
            ending_vertices_coord, new_edge_lengths, new_edge_pix_coord = find_closest_vertices(
                cropped_skeleton, cropped_non_tip_vertices, starting_vertices_coord)
            # if np.isnan(new_edge_lengths).sum()  == new_edge_lengths.shape[0]:
            # print(f"{np.isnan(new_edge_lengths).sum()} nans, {(new_edge_lengths == 0).sum()} zeros, {(new_edge_lengths == 0).sum()} ones, / {new_edge_lengths.shape[0]}")
            if np.isnan(new_edge_lengths).sum() + (new_edge_lengths == 0).sum() == new_edge_lengths.shape[0]:
                new_connexions = False
            else:
                # In new_edge_lengths, zeros are duplicates and nan are lone vertices (from starting_vertices_coord)
                # Find out which starting_vertices_coord should be kept and which one should be used to save edges
                no_new_connexion = np.isnan(new_edge_lengths)
                no_found_connexion = np.logical_or(no_new_connexion, new_edge_lengths == 0)
                found_connexion = np.logical_not(no_found_connexion)

                # Any vertex_to_vertex_connexions must be analyzed only once. We remove them with the non-connectable vertices
                vertex_to_vertex_connexions = new_edge_lengths == 1

                # Save edge data
                start = self.numbered_vertices[
                    starting_vertices_coord[found_connexion, 0], starting_vertices_coord[found_connexion, 1]]
                end = self.numbered_vertices[
                    ending_vertices_coord[found_connexion, 0], ending_vertices_coord[found_connexion, 1]]
                new_edge_lengths = new_edge_lengths[found_connexion]
                self.update_edge_data(start, end, new_edge_lengths, new_edge_pix_coord)

                # new_edge_pix_coord[:, 2].max()
                # len(new_edge_lengths)
                # self.edges_labels.shape[0]

                no_new_connexion = np.logical_or(no_new_connexion, vertex_to_vertex_connexions)
                vertices_to_crop = starting_vertices_coord[no_new_connexion, :]

                # # Remove the unconnectable from: starting_vertices_coord and ending_vertices_coord
                # starting_vertices_coord = starting_vertices_coord[still_connected, :2]
                # ending_vertices_coord = ending_vertices_coord[still_connected, :]

                # Remove non-connectable and connected_vertices from:
                # cropped_skeleton, cropped_non_tip_vertices,  starting_vertices_coord and not ending_vertices_coord
                cropped_non_tip_vertices = remove_coordinates(cropped_non_tip_vertices, vertices_to_crop)
                starting_vertices_coord = remove_coordinates(starting_vertices_coord, vertices_to_crop)

                if new_edge_pix_coord.shape[0] > 0:
                    # Update cropped_skeleton to not identify each edge more than once
                    cropped_skeleton[new_edge_pix_coord[:, 0], new_edge_pix_coord[:, 1]] = 0
                    self.obi += 1  # DEBUG
                    self.obsn[new_edge_pix_coord[:, 0], new_edge_pix_coord[:, 1]] = self.obi  # DEBUG

                # And the starting vertices that cannot connect anymore
                # fvY, fvX = starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]
                cropped_skeleton[vertices_to_crop[:, 0], vertices_to_crop[:, 1]] = 0

                if self.new_level_vertices is None:
                    self.new_level_vertices = ending_vertices_coord[found_connexion, :].copy()
                else:
                    self.new_level_vertices = np.vstack((self.new_level_vertices, ending_vertices_coord[found_connexion, :]))
        return cropped_skeleton, cropped_non_tip_vertices

    def update_edge_data(self, start, end, new_edge_lengths, new_edge_pix_coord):
        if isinstance(start, np.ndarray):
            end_idx = len(start)
            self.edge_lengths = np.concatenate((self.edge_lengths, new_edge_lengths))
        else:
            end_idx = 1
            self.edge_lengths = np.append(self.edge_lengths, new_edge_lengths)
        start_idx = self.edges_labels.shape[0]
        new_edges = np.zeros((end_idx, 3), dtype=np.uint32)
        new_edges[:, 0] = np.arange(start_idx, start_idx + end_idx) + 1  # edge label
        new_edges[:, 1] = start  # starting vertex label
        new_edges[:, 2] = end  # ending vertex label
        self.edges_labels = np.vstack((self.edges_labels, new_edges))
        # Add the new edge coord
        if new_edge_pix_coord.shape[0] > 0:
            # Add the new edge pixel coord
            new_edge_pix_coord[:, 2] += start_idx
            self.edge_pix_coord = np.vstack((self.edge_pix_coord, new_edge_pix_coord))

    def remove_edge_duplicates(self):
        edges_to_remove = []
        duplicates = find_duplicates_coord(np.vstack((self.edges_labels[:, 1:], self.edges_labels[:, :0:-1])))
        duplicates = np.logical_or(duplicates[:len(duplicates)//2], duplicates[len(duplicates)//2:])
        for v in self.edges_labels[duplicates, 1:]: #v = self.edges_labels[duplicates, 1:][4]
            edge_lab1 = self.edges_labels[np.all(self.edges_labels[:, 1:] == v, axis=1), 0]
            edge_lab2 = self.edges_labels[np.all(self.edges_labels[:, 1:] == v[::-1], axis=1), 0]
            edge_labs = np.unique(np.concatenate((edge_lab1, edge_lab2)))
            for edge_i in range(0, len(edge_labs) - 1):  #  edge_i = 0
                edge_i_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_labs[edge_i], :2]
                for edge_j in range(edge_i + 1, len(edge_labs)):  #  edge_j = 1
                    edge_j_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_labs[edge_j], :2]
                    if np.array_equal(edge_i_coord, edge_j_coord):
                        edges_to_remove.append(edge_labs[edge_j])
            # edge_coord1 = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_labs[0], :2]
            # edge_coord2 = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_labs[1], :2]
            # if np.array_equal(edge_coord1, edge_coord2):
            #     edges_to_remove.append(edge_labs[1])

        for edge in edges_to_remove:
            edge_bool = self.edges_labels[:, 0] != edge
            self.edges_labels = self.edges_labels[edge_bool, :]
            self.edge_lengths = self.edge_lengths[edge_bool]
            self.edge_pix_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] != edge, :]


    def remove_vertices_connecting_2_edges(self):
        """
        Find all vertices connecting 2 edges
        For each:
            If it is a tip, NOW THIS CANNOT NOT BE A TIP
                remove one connexion
            else
                Get the 2 edges id and their 2nd vertex
                Make them have the same edge_id and update (with edge and vertex):
                self.edges_labels, self.edge_lengths, self.edge_pix_coord
                Remove the vertex in
                self.numbered_vertices, self.non_tip_vertices
        """
        v_labels, v_counts = np.unique(self.edges_labels[:, 1:], return_counts=True)
        vertices2 = v_labels[v_counts == 2]
        for vertex2 in vertices2:  # vertex2 = vertices2[0]
            edge_indices = np.nonzero(self.edges_labels[:, 1:] == vertex2)[0]
            edge_names = [self.edges_labels[edge_indices[0], 0], self.edges_labels[edge_indices[1], 0]]
            v_names = np.concatenate((self.edges_labels[edge_indices[0], 1:], self.edges_labels[edge_indices[1], 1:]))
            v_names = v_names[v_names != vertex2]
            kept_edge = int(self.edge_lengths[edge_indices[1]] >= self.edge_lengths[edge_indices[0]])

            # Rename the removed edge by the kept edge name in pix_coord:
            self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_names[1 - kept_edge], 2] = edge_names[kept_edge]
            # Add the removed edge length to the kept edge length
            self.edge_lengths[self.edges_labels[:, 0] == edge_names[kept_edge]] += self.edge_lengths[self.edges_labels[:, 0] == edge_names[1 - kept_edge]]
            # Remove the corresponding edge length from the list
            self.edge_lengths = self.edge_lengths[self.edges_labels[:, 0] != edge_names[1 - kept_edge]]
            # Rename the vertex of the kept edge in edges_labels
            self.edges_labels[self.edges_labels[:, 0] == edge_names[kept_edge], 1:] = v_names[1 - kept_edge], v_names[kept_edge]
            # Remove the removed edge from the edges_labels array
            self.edges_labels = self.edges_labels[self.edges_labels[:, 0] != edge_names[1 - kept_edge], :]
            # self.edges_labels = np.delete(self.edges_labels, edge_indices[1 - kept_edge], axis=0)
            # self.edge_lengths = np.delete(self.edge_lengths, edge_indices[1 - kept_edge], axis=0)

            vY, vX = np.nonzero(self.numbered_vertices == vertex2)
            v_idx = np.nonzero(np.all(self.non_tip_vertices == [vY[0], vX[0]], axis=1))
            self.non_tip_vertices = np.delete(self.non_tip_vertices, v_idx, axis=0)

        # #
        # v_labels, v_counts = np.unique(self.edges_labels[:, 1:], return_counts=True)
        # vertex_visu = np.zeros_like(self.pad_skeleton)
        # for vert in v_labels[v_counts == 2]:
        #     print(vert)
        #     vertex_visu[self.numbered_vertices == vert] = 1
        # # show(vertex_visu)
        # print(f"There is {vertex_visu.sum()} vertices connecting 2 edges")
        #
        # #
        # v_labels, v_counts = np.unique(self.edges_labels[:, 1:], return_counts=True)
        # vertices2 = v_labels[v_counts == 2]
        # # should_not_be_here = [2728, 5182, 5190, 5039, 3517, 3505, 773]
        # for vertex2 in vertices2: # vertex2 = vertices2[0]
        #     # Check all cases to make sure that the proposed algorithm works well in all !!!
        #     # vertex2 = 773
        #     vY, vX = np.nonzero(self.numbered_vertices == vertex2)
        #     id2 = self.numbered_vertices > 0
        #     id = np.nonzero(self.edges_labels[:, 1:] == vertex2)
        #     edge1 = self.edges_labels[id[0][0], 0]
        #     edge2 = self.edges_labels[id[0][1], 0]
        #     print(f"v: {vertex2}; 1:{edge1}, 2:{edge2}")
        #     # print(id2[(vY[0] - 2):(vY[0] + 3), (vX[0] - 2):(vX[0] + 3)])
        #     idskel = np.zeros_like(self.numbered_vertices)
        #     idskel[self.numbered_vertices > 0] = 10
        #     idskel[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = self.edge_pix_coord[:, 2]
        #     print(pad_skeleton0[(vY[0] - 3):(vY[0] + 4), (vX[0] - 3):(vX[0] + 4)])
        #     print(idskel[(vY[0] - 2):(vY[0] + 3), (vX[0] - 2):(vX[0] + 3)])
        # #
        # v1, v2 = self.edges_labels[self.edges_labels[:, 0] == 2125, 1:][0]
        # pc = self.edge_pix_coord[self.edge_pix_coord[:, 2] == 2125, :]
        # ymin, ymax, xmin, xmax = pc[:, 0].min() - 2, pc[:, 0].max() + 3, pc[:, 1].min()-2, pc[:, 1].max() + 3
        # a = self.pad_skeleton.copy()
        # a[self.numbered_vertices == v1] = 10
        # a[self.numbered_vertices == v2] = 20
        # aa = a[ymin:ymax,xmin:xmax]
        #
        # idskel = np.zeros_like(self.numbered_vertices)
        # idskel[self.numbered_vertices > 0] = self.numbered_vertices[self.numbered_vertices > 0]
        # idskel[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = self.edge_pix_coord[:, 2]
        # print(idskel[(vY[0] - 5):(vY[0] + 6), (vX[0] - 5):(vX[0] + 6)])
        # print(self.pad_skeleton[(vY[0] - 5):(vY[0] + 6), (vX[0] - 5):(vX[0] + 6)])
        # # print(pad_skeleton[(vY[0] - 5):(vY[0] + 6), (vX[0] - 5):(vX[0] + 6)])

    def remove_padding(self):
        self.edge_pix_coord[:, :2] -= 1
        self.tips_coord[:, :2] -= 1
        self.non_tip_vertices[:, :2] -= 1
        self.skeleton, self.distances, self.vertices = remove_padding(
            [self.pad_skeleton, self.pad_distances, self.numbered_vertices])

    def find_growing_vertices(self, origin_contours=None, origin_centeroid=None):
        if origin_contours is not None:
            if self.vertices is None:
                self.remove_padding()
            edge_widths_copy = self.distances.copy()
            edge_widths_copy[origin_contours > 0] = 0
            pot_growing_skel = edge_widths_copy > np.quantile(edge_widths_copy[edge_widths_copy > 0], .9)
            # show(pot_growing_skel)
            dist_from_center = np.ones(self.vertices.shape, dtype=np.float64)
            dist_from_center[origin_centeroid[0], origin_centeroid[1]] = 0
            dist_from_center = distance_transform_edt(dist_from_center)
            dist_from_center *= pot_growing_skel
            growing_skel = dist_from_center > np.quantile(dist_from_center[dist_from_center > 0], .7)
            # show(growing_skel)
            self.growing_vertices = np.unique(self.vertices * growing_skel)[1:]


    def make_vertex_table(self, origin_contours=None):
        """
        Gives coordinates, labels, and natures of each vertex
        The nature can be:
        - a tip or a branching vertex
        - if it is network/food/growing
        """
        if self.vertices is None:
            self.remove_padding()
        # y_coord, x_coord, vertex_label, is_tip, network/food/growing, vertex_connected
        self.vertex_table = np.zeros((self.tips_coord.shape[0] + self.non_tip_vertices.shape[0], 6), dtype=self.vertices.dtype)
        self.vertex_table[:self.tips_coord.shape[0], :2] = self.tips_coord
        self.vertex_table[self.tips_coord.shape[0]:, :2] = self.non_tip_vertices
        self.vertex_table[:self.tips_coord.shape[0], 2] = self.vertices[self.tips_coord[:, 0], self.tips_coord[:, 1]]
        self.vertex_table[self.tips_coord.shape[0]:, 2] = self.vertices[self.non_tip_vertices[:, 0], self.non_tip_vertices[:, 1]]
        self.vertex_table[:self.tips_coord.shape[0], 3] = 1
        if origin_contours is not None:
            food_vertices = self.vertices[origin_contours > 0]
            food_vertices = food_vertices[food_vertices > 0]
        self.vertex_table[np.isin(self.vertex_table[:, 2], food_vertices), 4] = 1

        if self.growing_vertices is not None:
            self.vertex_table[:, 4] = 0
            growing = np.all(self.vertex_table[:, 2] == self.growing_vertices, axis=1)
            self.vertex_table[growing, 4] = 2

        nb, sh, stats, cent = cv2.connectedComponentsWithStats((self.vertices > 0).astype(np.uint8))
        for i, v_i in enumerate(np.nonzero(stats[:, 4] > 1)[0][1:]):
            v_labs = self.vertices[sh == v_i]
            for v_lab in v_labs: # v_lab = v_labs[0]
                self.vertex_table[self.vertex_table[:, 2] == v_lab, 5] = 1


    def make_edge_table(self, greyscale):
        self.edge_table = np.zeros((self.edges_labels.shape[0], 7), float)
        # label, length, width, int, BC
        self.edge_table[:, :3] = self.edges_labels[:, :]
        self.edge_table[:, 3] = self.edge_lengths
        for idx, edge_lab in enumerate(self.edges_labels[:, 0]):
            edge_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_lab, :]
            v_id = self.edges_labels[self.edges_labels[:, 0] == edge_lab, 1:][0]
            v1_width, v2_width = self.distances[self.vertices == v_id[0]], self.distances[self.vertices == v_id[1]]
            pix_widths = np.concatenate((v1_width, v2_width))
            v1_int, v2_int = greyscale[self.vertices == v_id[0]], greyscale[self.vertices == v_id[1]]
            pix_ints = np.concatenate((v1_int, v2_int))
            if len(edge_coord) > 0:
                pix_widths = np.append(pix_widths, self.distances[edge_coord[:, 0], edge_coord[:, 1]])
                pix_ints = np.append(pix_widths, greyscale[edge_coord[:, 0], edge_coord[:, 1]])
            self.edge_table[idx, 4] = pix_widths.mean()
            self.edge_table[idx, 5] = pix_ints.mean()

        G = nx.from_edgelist(self.edges_labels[:, 1:])
        e_BC = nx.edge_betweenness_centrality(G, seed=0)
        self.BC_net = np.zeros_like(self.distances)
        for v, k in e_BC.items(): # v=(81, 80)
            v1_coord = self.vertex_table[self.vertex_table[:, 2] == v[0], :2]
            v2_coord = self.vertex_table[self.vertex_table[:, 2] == v[1], :2]
            full_coord = np.concatenate((v1_coord, v2_coord))
            edge_lab1 = self.edges_labels[np.all(self.edges_labels[:, 1:] == v[::-1], axis=1), 0]
            edge_lab2 = self.edges_labels[np.all(self.edges_labels[:, 1:] == v, axis=1), 0]
            edge_lab = np.unique(np.concatenate((edge_lab1, edge_lab2)))
            if len(edge_lab) == 1:
                edge_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_lab, :2]
                full_coord = np.concatenate((full_coord, edge_coord))
                self.BC_net[full_coord[:, 0], full_coord[:, 1]] = k
                self.edge_table[self.edge_table[:, 0] == edge_lab, 6] = k
            elif len(edge_lab) > 1:
                edge_coord0 = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_lab[0], :2]
                for edge_i in range(len(edge_lab)): #  edge_i=1
                    edge_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_lab[edge_i], :2]
                    self.edge_table[self.edge_table[:, 0] == edge_lab[edge_i], 6] = k
                    full_coord = np.concatenate((full_coord, edge_coord))
                    self.BC_net[full_coord[:, 0], full_coord[:, 1]] = k
                    if edge_i > 0 and np.array_equal(edge_coord0, edge_coord):
                        print(f"There still is two identical edges: {edge_lab} of len: {len(edge_coord)} linking v={v}")
                        break
            #
            #     edge_coord1 = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_lab[0], :2]
            #     edge_coord2 = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_lab[1], :2]
            #     self.edge_table[edge_lab[0] - 1, 3] = k
            #     self.edge_table[edge_lab[1] - 1, 3] = k
            #     full_coord1 = np.concatenate((full_coord, edge_coord1))
            #     full_coord2 = np.concatenate((full_coord, edge_coord2))
            #     self.BC_net[full_coord1[:, 0], full_coord1[:, 1]] = k
            #     self.BC_net[full_coord2[:, 0], full_coord2[:, 1]] = k
            # else:
            #     print(f"len(edge_lab)={len(edge_lab)}")
            #     break


    def make_graph(self, cell_img, computed_network, pathway, i):
        if self.vertices is None:
            self.remove_padding()
        self.graph = np.zeros_like(self.distances)
        self.graph[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = self.distances[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]]
        self.graph = bracket_to_uint8_image_contrast(self.graph)
        cell_contours = get_contours(cell_img)
        net_contours = get_contours(computed_network)

        self.graph[np.nonzero(cell_contours)] = 9
        self.graph[np.nonzero(net_contours)] = 255
        vertices = np.zeros_like(self.graph)
        vertices[self.non_tip_vertices[:, 0], self.non_tip_vertices[:, 1]] = 1
        vertices = cv2.dilate(vertices, cross_33)
        self.graph[vertices > 0] = 240
        self.graph[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 140
        food = self.vertex_table[self.vertex_table[:, 4] == 1]
        self.graph[food[:, 0], food[:, 1]] = 190
        self.graph[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 140
        sizes = self.graph.shape[0] / 50, self.graph.shape[1] / 50
        fig = plt.figure(figsize=(sizes[0], sizes[1]))
        plt.imshow(self.graph, cmap='nipy_spectral')
        fig.tight_layout()
        fig.show()
        fig.savefig(pathway / f"contour network with medial axis{i}.png", dpi=1500)
        plt.close()


def find_closest_vertices(skeleton, all_vertices_coord, starting_vertices_coord):
    """
    skeleton, all_vertices_coord, starting_vertices_coord = cropped_skeleton, cropped_non_tip_vertices, starting_vertices_coord
    skeleton, all_vertices_coord, starting_vertices_coord = self.pad_skeleton, self.non_tip_vertices, self.tips_coord
    skeleton, all_vertices_coord, starting_vertices_coord = pad_skeleton, non_tip_vertices, starting_vertices_coord
    For each vertex, find the nearest branching vertex along the skeleton.

    UPDATE TO MAKE:
    - vertex pixels should not be included in all_path_pixels, maybe added afterward?
    - When the edge only contains two vertices, it takes a length of 1 and is saved in starting_vertices_coord
    - update remove_tipped_edge_smaller_than_branch_width to cope with that change


    Parameters:
    - skeleton (2D np.ndarray): Binary skeleton image (0 and 1)
    - all_vertices_coord (tuple): (array_y, array_x) coordinates of the first vertices
    - starting_vertices_coord (tuple): (array_y, array_x) coordinates of the second vertices

    Returns:
    - dict: keys are tip coordinates, values are (branch_vertex_coords, geodesic_distance)
    """

    # Convert branching vertices to set for quick lookup
    branch_set = set(zip(all_vertices_coord[:, 0], all_vertices_coord[:, 1]))
    n = starting_vertices_coord.shape[0]

    ending_vertices_coord = np.zeros((n, 3), np.uint32)  # next_vertex_y, next_vertex_x, edge_id
    edge_lengths = np.zeros(n, np.float64)
    # edge_lengths[:] = np.nan
    all_path_pixels = []  # Will hold rows of (y, x, edge_id)
    i = 0
    edge_i = 0
    for tip_y, tip_x in zip(starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]):
        # tip_y, tip_x = starting_vertices_coord[i, 0], starting_vertices_coord[i, 1]
        # tip_y, tip_x =[Y, X]
        # ky = 20
        # ly = ky + 1
        # kx = 20
        # lx = kx + 1
        # d = skeleton.copy()
        # d[tip_y, tip_x] = 2
        # d[all_vertices_coord[:, 0], all_vertices_coord[:, 1]] = 3
        # aa = d[800:860, 469:530]
        # aa = d[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]
        # d[tip_y, tip_x] = 4
        # d[Y, X] = 4
        # d[Y1, X1] = 4
        # c = pad_skeleton[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]
        # c = numbered_vertices[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]
        # obsn[tip_y, tip_x] = 25
        # obsn[fy, fx] = 26
        # c = obsn[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]

        # v_set = set(zip([tip_y - 1, tip_y - 1, tip_y, tip_y, tip_y + 1, tip_y + 1, tip_y, tip_y - 1, tip_y + 1], [tip_x - 1, tip_x, tip_x - 1, tip_x, tip_x + 1, tip_x, tip_x + 1, tip_x + 1, tip_x - 1]))
        visited = np.zeros_like(skeleton, dtype=bool)
        parent = {}
        q = deque()

        q.append((tip_y, tip_x))
        visited[tip_y, tip_x] = True
        parent[(tip_y, tip_x)] = None
        found_vertex = None

        while q:
            r, c = q.popleft()

            # # Check for branching vertex (ignore the starting tip itself)
            if (r, c) in branch_set and (r, c) != (tip_y, tip_x):
                # if (r, c) in branch_set and (r, c) not in v_set:
                found_vertex = (r, c)
                break  # stop at first encountered (shortest due to BFS)

            for dr, dc in neighbors_8:
                nr, nc = r + dr, c + dc
                if (0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and
                    not visited[nr, nc] and skeleton[nr, nc] > 0): # This does not work:  and (nr, nc) not in v_set
                    visited[nr, nc] = True
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        #
        # np.nonzero(np.all(starting_vertices_coord == [Y, X], axis=1))
        # if i == 36:
        #     break
        if found_vertex:
            fy, fx = found_vertex
            # Do not add the connection if has already been detected from the other way:
            from_start = np.all(starting_vertices_coord[:i, :] == [fy, fx], axis=1).any()
            to_end = np.all(ending_vertices_coord[:i, :2] == [tip_y, tip_x], axis=1).any()
            if not from_start or not to_end:
                edge_i += 1
                ending_vertices_coord[i, :] = [fy, fx, i + 1]
                # Reconstruct path from found_vertex back to tip
                path = []
                current = (fy, fx)
                while current is not None:
                    path.append((i, *current))
                    current = parent[current]

                # path.reverse()  # So path goes from starting tip to found vertex

                for _, y, x in path[1:-1]: # Exclude no vertices from the edge pixels path
                    all_path_pixels.append((y, x, edge_i))

                edge_lengths[i] = len(path) - 1  # exclude one node for length computation
            # else:
            #     # Remove the ending vertex from skeleton and retry to be sure that there is no other way
            #     # before forgetting this starting vertex:
            #

        else:
            edge_lengths[i] = np.nan
        i += 1

    edges_coords = np.array(all_path_pixels, dtype=np.uint32)
    # nan_edges = np.nonzero(np.isnan(edge_lengths))[0]
    # y_c, x_c = starting_vertices_coord[nan_edges, 0], starting_vertices_coord[nan_edges, 1]
    return ending_vertices_coord, edge_lengths, edges_coords


def identify_other_edges(pad_skeleton, edges_labels, edge_pix_coord, edge_lengths, numbered_vertices, tips_coord, non_tip_vertices, vertices_branching_tips):

    # First, create another version of these arrays, where we remove every already detected edge and their tips
    cropped_skeleton = pad_skeleton.copy()
    cropped_skeleton[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = 0
    cropped_skeleton[tips_coord[:, 0], tips_coord[:, 1]] = 0

    # non_tip_vertices does not need to be updated yet, because it only contains verified branching vertices
    cropped_non_tip_vertices = non_tip_vertices.copy()

    new_level_vertices = None
    # Fix the vertex_to_vertex_connexion problem
    # The problem with vertex_to_vertex_connexion is that since they are not separated by zeros,
    # they always atract each other instead of exploring other paths.
    # To fix this, we loop over each vertex group to
        # 1. Add one edge per inter-vertex connexion inside the group
        # 2. Remove all except one, and loop as many time as necessary.
            # Inside that second loop, we explore and identify every edge nearby.
    # Find every vertex_to_vertex_connexion
    v_grp_nb, v_grp_lab, v_grp_stats, vgc = cv2.connectedComponentsWithStats((numbered_vertices > 0).astype(np.uint8), connectivity=8)
    v_groups = np.nonzero(v_grp_stats[:, 4] > 1)[0][1:]
    for group_i in v_groups: #group_i=v_groups[0]
        v_coord = np.nonzero(v_grp_lab == group_i)
        for v_j, (v_y, v_x) in enumerate(zip(v_coord[0], v_coord[1])): #  v_j=1; v_y, v_x = v_coord[0][v_j], v_coord[1][v_j]
            hiden_y, hiden_x = np.delete(v_coord[0], v_j), np.delete(v_coord[1], v_j)
            # a = pad_skeleton.copy()
            # a = c2_skeleton.copy()
            # a[np.nonzero(numbered_vertices)] = numbered_vertices[np.nonzero(numbered_vertices)]
            # aa = a[v_y - 20:v_y + 21, v_x - 20:v_x + 20]
            c2_skeleton = cropped_skeleton.copy()
            c2_skeleton[hiden_y, hiden_x] = 0
            c2_non_tip_vertices = cropped_non_tip_vertices.copy()
            c2_non_tip_vertices = remove_coordinates(c2_non_tip_vertices, np.vstack((hiden_y, hiden_x)))
            starting_vertices_coord = np.array((v_y, v_x))[None, :]

            explored_connexions_per_vertex = 0
            new_connexions = True
            while new_connexions and explored_connexions_per_vertex < 5 and np.any(c2_non_tip_vertices):
                explored_connexions_per_vertex += 1
                # 1. Find the ith closest vertex to each focal vertex
                ending_vertices_coord, new_edge_lengths, new_edge_pix_coord = find_closest_vertices(
                    c2_skeleton, c2_non_tip_vertices, starting_vertices_coord)
                if np.isnan(new_edge_lengths).sum() == new_edge_lengths.shape[0]:  # starting_vertices_coord.shape[0] == 0:#:
                    new_connexions = False
                else:
                    # Find out which starting_vertices_coord should be kept and which one should be used to save edges
                    no_more_connexion = np.isnan(new_edge_lengths)
                    no_new_connexion = starting_vertices_coord[no_more_connexion, :]
                    vertices_to_keep = np.logical_not(no_more_connexion)
                    edges_to_save = np.logical_and(vertices_to_keep, new_edge_lengths > 0)

                    # Save edge data
                    start = numbered_vertices[
                        starting_vertices_coord[edges_to_save, 0], starting_vertices_coord[edges_to_save, 1]]
                    end = numbered_vertices[
                        ending_vertices_coord[edges_to_save, 0], ending_vertices_coord[edges_to_save, 1]]
                    new_edge_lengths = new_edge_lengths[edges_to_save]
                    edges_labels, edge_lengths, edge_pix_coord = update_edge_data(edges_labels, start, end,
                                                                                  edge_lengths, new_edge_lengths,
                                                                                  edge_pix_coord, new_edge_pix_coord)

                    # Remove the unconnectable from: starting_vertices_coord, ending_vertices_coord, c2_non_tip_vertices
                    starting_vertices_coord = starting_vertices_coord[vertices_to_keep, :2]
                    ending_vertices_coord = ending_vertices_coord[vertices_to_keep, :2]

                    c2_non_tip_vertices = remove_coordinates(c2_non_tip_vertices, no_new_connexion)

                    if new_edge_pix_coord.shape[0] > 0:
                        # Update c2_skeleton to not identify each edge more than once
                        c2_skeleton[new_edge_pix_coord[:, 0], new_edge_pix_coord[:, 1]] = 0
                    # And the starting vertices that cannot connect anymore
                    fvY, fvX = starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]
                    c2_skeleton[fvY, fvX] = 0

                    if new_level_vertices is None:
                        new_level_vertices = ending_vertices_coord.copy()
                    else:
                        new_level_vertices = np.vstack((new_level_vertices, ending_vertices_coord))
    if new_level_vertices is not None:
        starting_vertices_coord = np.vstack((new_level_vertices, vertices_branching_tips))
        starting_vertices_coord = np.unique(starting_vertices_coord, axis=0)
        # I get better coverage by starting only with this:
        # starting_vertices_coord = np.unique(new_level_vertices, axis=0)
        # Or this
        # starting_vertices_coord = np.unique(vertices_branching_tips, axis=0)
    else:
        # We will start from the vertices connecting tips
        starting_vertices_coord = vertices_branching_tips.copy()

    # Clearly differentiate the loop that make sure that we explore all connexions of the current starting_vertices_coord
    # And the loop that update starting_vertices_coord until we explored all vertices.
    obsn = np.zeros_like(numbered_vertices) # DEBUG
    obsn[np.nonzero(pad_skeleton)] = 1 # DEBUG
    obsn[edge_pix_coord[:,0], edge_pix_coord[:,1]] = 2 # DEBUG
    obi=2 # DEBUG
    while np.any(cropped_non_tip_vertices):
        explored_connexions_per_vertex = 0 # the maximal edge number that can connect a vertex
        new_level_vertices = None
        new_connexions = True
        while new_connexions and explored_connexions_per_vertex < 5 and np.any(cropped_non_tip_vertices):
            explored_connexions_per_vertex += 1
            # 1. Find the ith closest vertex to each focal vertex
            ending_vertices_coord, new_edge_lengths, new_edge_pix_coord = find_closest_vertices(
                cropped_skeleton, cropped_non_tip_vertices, starting_vertices_coord[:, :2])
            if np.isnan(new_edge_lengths).sum() == new_edge_lengths.shape[0]: #starting_vertices_coord.shape[0] == 0:#:
                new_connexions = False
            else:
                # In new_edge_lengths, zeros are duplicates and nan are vertices (from starting_vertices_coord)
                # that cannot connect anymore. We want to remove those that cannot connect from:
                # starting_vertices_coord, ending_vertices_coord, cropped_non_tip_vertices, cropped_skeleton
                # And we don't want to add duplicates in edge data

                # 2. Find which starting vertex cannot connect with any ending vertex anymore
                # y, x = starting_vertices_coord[:, 0], ending_vertices_coord[:, 1]
                # cnv8 = CompareNeighborsWithValue(cropped_skeleton, 8)
                # cnv8.is_equal(1, and_itself=True)
                # cnv8.equal_neighbor_nb[starting_vertices_coord[:, 0], ending_vertices_coord[:, 1]]
                # cropped_skeleton[starting_vertices_coord[:, 0], ending_vertices_coord[:, 1]]

                # Find out which starting_vertices_coord should be kept and which one should be used to save edges
                no_more_connexion = np.isnan(new_edge_lengths)
                no_new_connexion = starting_vertices_coord[no_more_connexion, :]
                vertices_to_keep = np.logical_not(no_more_connexion)
                edges_to_save = np.logical_and(vertices_to_keep, new_edge_lengths > 0)

                # Save edge data
                start = numbered_vertices[starting_vertices_coord[edges_to_save, 0], starting_vertices_coord[edges_to_save, 1]]
                end = numbered_vertices[ending_vertices_coord[edges_to_save, 0], ending_vertices_coord[edges_to_save, 1]]
                new_edge_lengths = new_edge_lengths[edges_to_save]
                edges_labels, edge_lengths, edge_pix_coord = update_edge_data(edges_labels, start, end, edge_lengths, new_edge_lengths, edge_pix_coord, new_edge_pix_coord)

                # Remove the unconnectable from: starting_vertices_coord, ending_vertices_coord, cropped_non_tip_vertices
                starting_vertices_coord = starting_vertices_coord[vertices_to_keep, :2]
                ending_vertices_coord = ending_vertices_coord[vertices_to_keep, :]

                cropped_non_tip_vertices = remove_coordinates(cropped_non_tip_vertices, no_new_connexion)

                if new_edge_pix_coord.shape[0] > 0:
                    # Update cropped_skeleton to not identify each edge more than once
                    cropped_skeleton[new_edge_pix_coord[:, 0], new_edge_pix_coord[:, 1]] = 0
                    obi += 1  # DEBUG
                    obsn[new_edge_pix_coord[:, 0], new_edge_pix_coord[:, 1]] = obi  # DEBUG

                # And the starting vertices that cannot connect anymore
                fvY, fvX = starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]
                cropped_skeleton[fvY, fvX] = 0


                if new_level_vertices is None:
                    new_level_vertices = ending_vertices_coord.copy()
                else:
                    new_level_vertices = np.vstack((new_level_vertices, ending_vertices_coord))

        # Update the starting_vertices_coord array
        if new_level_vertices is None:
            break
        else:
            starting_vertices_coord = np.unique(new_level_vertices, axis=0) # new_level_vertices
        # a[edge_pix_coord[:, 1],edge_pix_coord[:, 2]]=1
        #show(a)
    # Do this, it is good, and go find why some nodes are not 12. Look for their name, study their evolution
    obsn[tips_coord[:,0],tips_coord[:,1]] = 21
    obsn[non_tip_vertices[:,0],non_tip_vertices[:,1]] = 20
    obsn[new_level_vertices[:,0],new_level_vertices[:,1]] = 15
    obsn[starting_vertices_coord[:,0],starting_vertices_coord[:,1]] = 12
    obsn[ending_vertices_coord[:,0],ending_vertices_coord[:,1]] = 10
    # obsn[1200,739] = 16
    # obsn[np.nonzero(cropped_numbered_vertices)] = 8
    a=obsn[1090:1112, 971:1010].copy()
    # a=obsn[1190:1210, 730:749].copy()

    cs2 = cropped_skeleton.copy()
    # cs2[cropped_non_tip_vertices[:,0],cropped_non_tip_vertices[:,1]] = 10
    aa = cs2[1090:1112, 971:1010]
    simp=numbered_vertices.copy()
    simp[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = 1
    # simp[ending_vertices_coord[:,0], ending_vertices_coord[:,1]] = 11
    aaa = simp[1090:1112, 971:1010]
    # Find the corresponding number and coordinates
    yv,xv = np.nonzero(numbered_vertices==5107)
    [yv[0], xv[0]]
    # Find where it is in the coordinates array
    idx_coord = np.nonzero(np.all(starting_vertices_coord == [yv[0], xv[0]], axis=1))[0]
    idx_coord = np.nonzero(np.all(no_new_connexion == [yv[0], xv[0]], axis=1))[0]
    # Check if it gets removed:
    no_more_connexion[idx_coord]
    idx_coord in no_new_connexion
    # If it remains:
    np.nonzero(np.all(starting_vertices_coord[:, :] == [yv[0], xv[0]], axis=1))[0]
    np.nonzero(np.all(starting_vertices_coord[vertices_to_keep, :] == [yv[0], xv[0]], axis=1))[0]
    np.nonzero(np.all(ending_vertices_coord[vertices_to_keep, :2] == [yv[0], xv[0]], axis=1))[0]
    np.nonzero(np.all(ending_vertices_coord[:, :2] == [yv[0], xv[0]], axis=1))[0]

    # Find duplicated edges:
    uni, cou = np.unique(edges_labels[:, 1:], axis=0, return_counts=True)
    duplicated_edges = np.nonzero(cou > 1)
    # Find their coordinates:
    edge_pix_coord[edge_pix_coord[:, 2] == edges_labels[duplicated_edges, 0][0][5], :2]

    # obsn[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = edge_pix_coord[:, 2]
    # obsn[no_new_connexion[:,0],no_new_connexion[:,1]] = 15
    # aa= a + cropped_skeleton[1090:1112, 971:1010] *20
    # aaa= cropped_skeleton[1090:1112, 971:1010] *20
    # # Check a to find a place where an edge should have been added
    # b=numbered_vertices[1090:1112, 971:1010]
    # Check at what step it can be find in cropped_non_tip_vertices and starting_vertices_coord
    # np.nonzero(np.logical_and(cropped_non_tip_vertices[:, 1] == xv, cropped_non_tip_vertices[:, 0] == yv))
    # np.nonzero(np.logical_and(starting_vertices_coord[:, 1] == xv, starting_vertices_coord[:, 0] == yv))
    # np.nonzero(np.logical_and(ending_vertices_coord[:, 1] == xv, ending_vertices_coord[:, 0] == yv))
    # np.nonzero(np.logical_and(no_new_connexion[:, 1] == xv, no_new_connexion[:, 0] == yv))
    # # At the second step,
    #
    # yv,xv = np.nonzero(numbered_vertices==5576)
    #
    #
    # numbered_edges = np.zeros_like(numbered_vertices)
    # numbered_edges[edge_pix_coord[edge_pix_coord[:,0] == 9,1], edge_pix_coord[edge_pix_coord[:,0] == 9,2]] =  edge_pix_coord[edge_pix_coord[:,0] == 9,0]
    # numbered_edges[edge_pix_coord[:,1], edge_pix_coord[:,2]] =  edge_pix_coord[:,0]
    # show(obsn)
    #
    # obsn[edge_pix_coord[:,1], edge_pix_coord[:,2]] =  edge_pix_coord[:,0]
    # obsn[new_edge_pix_coord[:,1], new_edge_pix_coord[:,2]] =  new_edge_pix_coord[:,0]

    identified_skeleton = np.zeros_like(numbered_vertices)
    identified_skeleton[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = edge_pix_coord[:, 2]
    identified_skeleton[np.nonzero(numbered_vertices)] = edges_labels.shape[0] + 1
    not_identified = (identified_skeleton > 0) != pad_skeleton
    obsn[np.nonzero(not_identified)] = 100
    print(100* (identified_skeleton > 0).sum() /pad_skeleton.sum())
    show(obsn)
    # Save the coordinates containing vertex to vertex edges, will be useful below
    # vertex_to_vertex_connexions = new_edge_lengths == 1
    # not_vertex_to_vertex_connexions = new_edge_lengths != 1
    # vertex_to_vertex_connexions = starting_vertices_coord[vertex_to_vertex_connexions, :]

    # We remove the vertices that did not manage to make a new connection
    # The vertex to vertex connexion cannot be removed here: if so, adjacent edges will never find this vertex
    # cropped_numbered_vertices[vertex_to_vertex_connexions[:, 0], vertex_to_vertex_connexions[:, 1]] = 0
    # However, they can be removed as a starting point:

    """
    # identified_skeleton[edge_pix_coord[:, 0], edge_pix_coord[:, 1]] = 50
    # identified_skeleton[not_identified] = 200
    # identified_skeleton[non_tip_vertices[:, 0], non_tip_vertices[:, 1]] = 250
    # show(identified_skeleton)
    # identified_skeleton[603:610,606:613]
    # numbered_vertices[603:610,606:613]
    # pad_skeleton[603:610,606:613]
    # (cropped_skeleton*not_identified).sum() == not_identified.sum()

    # THIS IS A SUPER SLOW SOLUTION, hence the break
    if not_identified.any():
        not_ident_Y, not_ident_X = np.nonzero(not_identified)
        # for not_id_y, not_id_x in zip(not_id_coord[0], not_id_coord[1]): # not_id_y, not_id_x = not_id_coord[0][0], not_id_coord[1][0]
        ci = 0
        while ci <= len(not_ident_X):
            not_id_y, not_id_x = not_ident_Y[ci], not_ident_X[ci]
            neighbors_label = identified_skeleton[(not_id_y - 1):(not_id_y + 2), (not_id_x - 1):(not_id_x + 2)]
            neighbors_label = neighbors_label[neighbors_label > 0]
            if neighbors_label.any():
                unique_nei, counts = np.unique(neighbors_label, return_counts=True)
                not_id_new_label = unique_nei[np.argmax(counts)]
                identified_skeleton[not_id_y, not_id_x] = not_id_new_label
                edge_pix_coord = np.vstack((edge_pix_coord, np.array((not_id_new_label, not_id_y, not_id_x))))
            else:
                print("here")
                break
                not_ident_X = np.append(not_ident_X, not_id_x)
                not_ident_Y = np.append(not_ident_Y, not_id_y)
            ci += 1

    if ((identified_skeleton > 0) != pad_skeleton).any():
        print("Skeleton parts are still not identified")
    """
    # 3. Find another closest vertex to each focal vertex
    # 4. Remove all edges connecting these two vertices
    # 5. Find -if existing- another closest vertex to each focal vertex
    # 6. Do 4. and 5. four times
    # 7. Remove all edges, use the new_connecting_vertices as the first vertices

    # show(a)
    return edges_labels, edge_pix_coord, edge_lengths


def update_edge_data(edges_labels, starting_vertices_labels, ending_vertices_labels, edge_lengths, new_edge_lengths, edge_pix_coord, new_edge_pix_coord):
    end_idx = len(starting_vertices_labels)
    start_idx = edges_labels.shape[0]
    new_edges = np.zeros((end_idx, 3), dtype=np.uint32)
    new_edges[:, 0] = np.arange(start_idx, start_idx + end_idx) + 1  # edge label
    new_edges[:, 1] = starting_vertices_labels  # starting vertex label
    new_edges[:, 2] = ending_vertices_labels  # ending vertex label
    edges_labels = np.vstack((edges_labels, new_edges))
    edge_lengths = np.concatenate((edge_lengths, new_edge_lengths))
    # Add the new edge coord
    if new_edge_pix_coord.shape[0] > 0:
        # Add the new edge pixel coord
        new_edge_pix_coord[:, 2] += start_idx
        edge_pix_coord = np.vstack((edge_pix_coord, new_edge_pix_coord))
    return edges_labels, edge_lengths, edge_pix_coord


def get_numbered_edges_and_vertices(pad_skeleton, pad_vertices, pad_tips):
    """ TO REMOVE THIS FUNCTION """
    pad_edges = (1 - pad_vertices) * pad_skeleton
    pad_branches = pad_vertices - pad_tips
    bY, bX = np.nonzero(pad_branches)
    numbered_branches = np.zeros(pad_branches.shape, np.int64)
    for bi, (bYi, bXi) in enumerate(zip(bY, bX)):
        numbered_branches[bYi, bXi] = bi + 1
    tY, tX = np.nonzero(pad_tips)
    numbered_tips = np.zeros(pad_branches.shape, np.int64)
    for ti, (tYi, tXi) in enumerate(zip(tY, tX)):
        numbered_tips[tYi, tXi] = ti + 1
    # nb_v, tempo_numbered_vertices = cv2.connectedComponents(vertices, connectivity=4)  # Connectivity is 4 to avoid having the same label for two nodes
    nb_e, tempo_numbered_edges = cv2.connectedComponents(pad_edges, connectivity=8)
    return nb_e, tempo_numbered_edges, numbered_branches, numbered_tips


def add_padding(array_list):
    new_array_list = []
    for arr in array_list:
        new_array_list.append(np.pad(arr, [(1, ), (1, )], mode='constant'))
    return new_array_list


def remove_padding(array_list):
    new_array_list = []
    for arr in array_list:
        new_array_list.append(arr[1:-1, 1:-1])
    return new_array_list


def edges_directly_connecting_two_vertices(nb_e, tempo_numbered_edges, tempo_numbered_vertices):
    tempo_edges_labels = []
    problematic_edges = {}
    for i in range(1, nb_e):  # nb_e
        edge_i = (tempo_numbered_edges == i).astype(np.uint8)
        dil_edge_i = cv2.dilate(edge_i, square_33)
        unique_vertices_im = dil_edge_i * tempo_numbered_vertices
        unique_vertices = np.unique(unique_vertices_im)
        unique_vertices = unique_vertices[unique_vertices > 0]
        # In most cases, the edge is connected to 2 vertices
        if len(unique_vertices) == 2:
            tempo_edges_labels.append((i, unique_vertices[0], unique_vertices[1]))
        # When the edge is connected to 1 vertex, it forms a loop
        elif len(unique_vertices) == 1:
            tempo_edges_labels.append((i, unique_vertices[0], unique_vertices[0]))
        else:
            problematic_edges[i] = unique_vertices

    return tempo_edges_labels, problematic_edges



def get_graph_from_vertices_and_edges(pad_vertices, pad_edges, pad_distances):
    """ TO REMOVE THIS FUNCTION
    Remaining problems:
    1. when 3 or more nodes contour one edge,
    find those that are close together and pick one (the closest to the edge)
    2. When nodes appear along the edge,
    split the edge into two parts

    :param vertices:
    :param edges:
    :return:
    """
    nb_e, tempo_numbered_edges, tempo_numbered_vertices = get_numbered_edges_and_vertices(pad_vertices, pad_edges, pad_tips)
    tempo_edges_labels, problematic_edges = edges_directly_connecting_two_vertices(nb_e, tempo_numbered_edges, tempo_numbered_vertices)

    tempo_edges_labels = []
    # All this does not work because I use morphological dilatation too much.
    # Start as it is to find the edges that are already connected to only two vertices
    # Then use distance transform on the remaining edges.
    # Start from each tip to get the non-tip vertex that is the nearest to that tip along the edge connected to the tip
    # X/ Start from each vertex to get the non-tip vertex that is the nearest to that tip along the edge connected to the first vertex
    # Do X/ as long as necessary
    # Difficulty is to adjust the distance transform so that it works along an edge
    for i in range(1, nb_e):  # nb_e   i=335 436 2563
        # i += 1
        edge_i = (tempo_numbered_edges == i).astype(np.uint8)
        dil_edge_i = cv2.dilate(edge_i, square_33)
        unique_vertices_im = dil_edge_i * tempo_numbered_vertices
        unique_vertices = np.unique(unique_vertices_im)
        unique_vertices = unique_vertices[unique_vertices > 0]
        # In most cases, the edge is connected to 2 vertices
        if len(unique_vertices) == 2:
            tempo_edges_labels.append((i, unique_vertices[0], unique_vertices[1]))
        # When the edge is connected to 1 vertex, it forms a loop
        elif len(unique_vertices) == 1:
            tempo_edges_labels.append((i, unique_vertices[0], unique_vertices[0]))
        # When the edge is connected to more than two vertices, we need to split it into more edges:
        else:
            eY, eX = np.nonzero(edge_i)
            vY, vX = np.nonzero(np.isin(tempo_numbered_vertices, unique_vertices))
            # Take a marge window around the edge to correctly compute the neighbor number
            eY_min, eY_max, eX_min, eX_max = np.min(np.concatenate([eY, vY])) - 1, np.max(
                np.concatenate([eY, vY])) + 2, np.min(np.concatenate([eX, vX])) - 1, np.max(
                np.concatenate([eX, vX])) + 2
            # eY_min, eY_max, eX_min, eX_max = np.min(eY) - 2, np.max(eY) + 3, np.min(eX) - 2, np.max(eX) + 3
            # print(f"{i}: {unique_vertices}")
            # First remove vertices that are too close to each other
            sub_edge_i = edge_i[eY_min:eY_max, eX_min:eX_max]
            sub_tempo_numbered_edges = tempo_numbered_edges[eY_min:eY_max, eX_min:eX_max]
            sub_tempo_numbered_vertices = tempo_numbered_vertices[eY_min:eY_max, eX_min:eX_max].copy()

            # a = sub_tempo_numbered_vertices + sub_edge_i
            # cc_vertex_nb, _ = cv2.connectedComponents(sub_tempo_numbered_vertices.astype(np.uint8), connectivity=8)
            # Make sure that we are not in a situation where there are 2 tips among 3 vertices
            # if len(unique_vertices) != 3 or cc_vertex_nb != 4: #  and sub_edge_i.sum() == 2
            # if not (len(unique_vertices) == 3 and cc_vertex_nb == 4): #  and sub_edge_i.sum() == 2
            unique_vertices_im = np.isin(sub_tempo_numbered_vertices, unique_vertices)
            for vertex in unique_vertices: # vertex=unique_vertices[1]
                vertex_i = sub_tempo_numbered_vertices == vertex
                dil_vertex = (sub_edge_i + vertex_i) * cv2.dilate((vertex_i).astype(np.uint8), square_33)
                dil_vertex = cv2.dilate(dil_vertex, cross_33)
                duplicate_vertices = dil_vertex * unique_vertices_im
                dup_vert_nb = duplicate_vertices.sum()
                if dup_vert_nb > 1:
                    # print("h")
                    vertices_coord = np.nonzero(sub_tempo_numbered_vertices * duplicate_vertices)
                    # Remove the vertex with the lower connection number with any edge
                    # connexions = sub_edge_i.astype(np.int32)
                    connexions = sub_tempo_numbered_edges.copy() # NEW
                    connexions[vertices_coord] = sub_tempo_numbered_vertices[vertices_coord]
                    c_cnv = CompareNeighborsWithValue(connexions > 0, 8)
                    c_cnv.is_equal(1, and_itself=False)
                    connexion_nb = c_cnv.equal_neighbor_nb[vertices_coord]
                    # if len(connexion_nb) > 1 + (connexion_nb == 1).sum():
                    vertices_to_remove = np.argsort(connexion_nb)[::-1][1:]
                    # vertices_to_remove = vertices_to_remove[vertices_to_remove != vertex]
                    remove_nb = 0
                    while len(unique_vertices) > 2 and remove_nb < len(vertices_to_remove):
                        vertex_to_remove = vertices_to_remove[remove_nb]
                        vertex_name = sub_tempo_numbered_vertices[vertices_coord[0][vertex_to_remove], vertices_coord[1][vertex_to_remove]]
                        removed_vertex = (sub_tempo_numbered_vertices == vertex_name).astype(np.uint8)
                        rv_Y, rv_X = np.nonzero(removed_vertex)
                        sub_tempo_numbered_vertices[rv_Y, rv_X] = 0
                        unique_vertices = unique_vertices[unique_vertices != vertex_name]
                        remove_nb += 1
                        # Replace that pixel by the value of the most connected edge nearby
                        # removed_vertex = cv2.dilate(removed_vertex, square_33)
                        # new_ids, counts = np.unique(removed_vertex * sub_tempo_numbered_edges, return_counts=True)
                        # new_id = new_ids[counts[1:].argmax() + 1]
                        # sub_tempo_numbered_edges[rv_Y, rv_X] = new_id
                        # sub_edge_i[rv_Y, rv_X] = 1
                        # numbered_edge_i[eY_min:eY_max, eX_min:eX_max] = new_id
                # for vertex_to_remove in vertices_to_remove: # vertex_to_remove=1

            if len(unique_vertices) == 2:
                tempo_edges_labels.append((i, unique_vertices[0], unique_vertices[1]))
            else:
                # Split the edge:
                # Second if removing vertices was not enough, cut the edges according to the remaining vertices
                vY, vX = np.nonzero(sub_tempo_numbered_vertices)
                terminations = np.logical_or(
                    np.logical_or(np.logical_or(vY == 0, vY == (sub_tempo_numbered_vertices.shape[0] - 1)), vX == 0),
                    vX == (sub_tempo_numbered_vertices.shape[1] - 1))

                not_terminations = np.logical_not(terminations)
                dil_vertices = np.zeros(sub_tempo_numbered_vertices.shape, dtype=np.uint8)
                dil_vertices[vY[not_terminations], vX[not_terminations]] = 1
                dil_vertices = cv2.dilate(dil_vertices, cross_33)
                cut_edge_i = sub_edge_i * (1 - dil_vertices)
                nb_ei, numbered_edge_i = cv2.connectedComponents(cut_edge_i, connectivity=8)
                if nb_ei <= 2:
                    nb_ei = 3
                    new_edge_names = []
                    new_edge_names.append(i)
                    if sub_edge_i.sum() == 2:
                        Y, X = np.nonzero(sub_edge_i)
                        sub_tempo_numbered_edges[Y[1], X[1]] = nb_e
                        new_edge_names.append(nb_e)
                        nb_e += 1
                    else:
                        # split_edge = cv2.dilate(numbered_edge_i.astype(np.uint8), square_33)
                        # split_edge = (1 - split_edge) * sub_edge_i
                        # nb_diff_edges, different_edges = cv2.connectedComponents(split_edge)
                        nb_diff_edges, different_edges = cv2.connectedComponents((sub_edge_i - numbered_edge_i).astype(np.uint8))
                        for diff_edge in range(1, nb_diff_edges):
                            # Add the pixels of sub_edge_i connected to it but not in other edges
                            Y, X = np.nonzero(different_edges == diff_edge)
                            sub_tempo_numbered_edges[Y, X] = nb_e
                            new_edge_names.append(nb_e)
                            nb_e += 1
                else:
                    # new_edge_number = (nb_ei - 1)
                    # not_terminations = np.nonzero(not_terminations)[0]
                    # edge_numbers = np.zeros(len(unique_vertices) - 1)
                    new_tempo_numbered_edges = (cv2.dilate(numbered_edge_i.astype(np.uint8), square_33) * sub_edge_i).astype(np.uint32)
                    # If any sub_edge_i pixel is missing from new_tempo_numbered_edges
                    sub_edge_bis = new_tempo_numbered_edges > 0
                    if sub_edge_bis.sum() != sub_edge_i.sum():
                    # Create another edge
                        hidden_segments = (sub_edge_i * (1 - sub_edge_bis)).astype(np.uint8)
                        dil_sub_edge_bis = cv2.dilate(hidden_segments, square_33)
                        nb_dseb, im_dseb = cv2.connectedComponents(dil_sub_edge_bis)
                        for dseb_i in range(1, nb_dseb): # dseb_i=1
                            im_dseb_i = im_dseb == dseb_i
                            # hidden_segment_name = np.unique(im_dseb_i * new_tempo_numbered_edges)[1]
                            hiddenY, hiddenX = np.nonzero(im_dseb_i * hidden_segments)
                            new_tempo_numbered_edges[hiddenY, hiddenX] = nb_ei
                            nb_ei += 1

                    new_edge_names = []
                    for sub_ei in range(1, nb_ei):
                        # sub_ei+=1
                        Y, X = np.nonzero(new_tempo_numbered_edges == sub_ei)
                        # Make sure that the first sub_edge to be treated receive the value of the initial edge
                        if len(new_edge_names) == 0:
                            new_edge_names.append(i)
                        else:
                            # Others will take another value, starting above the current total number of edges
                            new_edge_names.append(nb_e)
                            sub_tempo_numbered_edges[Y, X] = nb_e
                            nb_e += 1

                if not np.any(nb_ei > 2):
                    print('here', i)
                    cnv = CompareNeighborsWithValue(sub_edge_i, 4)
                    cnv.is_equal(1, and_itself=True)
                    one_edge = cnv.equal_neighbor_nb
                    numbered_edge_i = sub_edge_i.copy()
                    numbered_edge_i[np.nonzero(one_edge)] = 0
                    nb_ei, numbered_edge_i = cv2.connectedComponents(numbered_edge_i, connectivity=8)
                    nb_ei2, numbered_edge_i2 = cv2.connectedComponents(one_edge, connectivity=8)
                    numbered_edge_i2[numbered_edge_i2 > 0] += (nb_ei - 1)
                    numbered_edge_i[numbered_edge_i2 > 0] = numbered_edge_i2[numbered_edge_i2 > 0]
                    nb_ei = len(np.unique(numbered_edge_i))
                    for sub_ei in range(1, nb_ei):
                        # Make sure that the first sub_edge to be treated receive the value of the initial edge
                        if len(new_edge_names) == 0:
                            new_edge_names.append(i)
                        else:
                            # Others will take another value, starting above the current total number of edges
                            Y, X = np.nonzero(numbered_edge_i == sub_ei)
                            new_edge_names.append(nb_e)
                            sub_tempo_numbered_edges[Y, X] = nb_e
                            nb_e += 1

                # Loop over the new edges to add them to the graph:
                for j in new_edge_names:
                    edge_i = (sub_tempo_numbered_edges == j).astype(np.uint8)
                    dil_edge_i = cv2.dilate(edge_i, square_33)
                    unique_vertices_im = dil_edge_i * tempo_numbered_vertices[eY_min:eY_max, eX_min:eX_max]
                    # unique_vertices_im = dil_edge_i * sub_tempo_numbered_vertices
                    new_unique_vertices = np.unique(unique_vertices_im)
                    new_unique_vertices = new_unique_vertices[new_unique_vertices > 0]
                    new_unique_vertices = new_unique_vertices[np.isin(new_unique_vertices, unique_vertices)]
                    # In most cases, the edge is connected to 2 vertices
                    if len(new_unique_vertices) == 2:
                        tempo_edges_labels.append((j, new_unique_vertices[0], new_unique_vertices[1]))
                    # When the edge is connected to 1 vertex, it forms a loop
                    elif len(new_unique_vertices) == 1:
                        tempo_edges_labels.append((j, new_unique_vertices[0], new_unique_vertices[0]))
                    # When the edge is connected to more than two vertices, we need to split it into more edges:
                    else:
                        print(f"i={i}, j={j}: {new_unique_vertices}")
    # Create edges between each connected vertices
    # cc(tempo_numbered_vertices > 0)
    # The case with two tips branching from one vertex cannot be handled in the pipeline above.
    # Because there, we don't want to remove tips to correctly get edge id. We want to remove the branching
    # Or, if we remove the branching, we create an edge between two tips and we don't want that. Do it later on.

    tempo_edges_labels = np.array(tempo_edges_labels, dtype=np.uint64)
    # # Debugging stuff:
    # i = 151 # 12 17 239 462
    # edge_i = (tempo_numbered_edges == i).astype(np.uint8)
    # edge_i = (tempo_numbered_edges == edge_with_tip).astype(np.uint8)
    # Y, X = np.nonzero(edge_i)
    # a=skeleton[(np.min(Y) - 2):(np.max(Y) + 3), (np.min(X) - 2):(np.max(X) + 3)].astype(np.uint8)
    # a=edge_i[(np.min(Y) - 2):(np.max(Y) + 3), (np.min(X) - 2):(np.max(X) + 3)].astype(np.uint8)
    # b = tempo_numbered_vertices[(np.min(Y) - 2):(np.max(Y) + 3), (np.min(X) - 2):(np.max(X) + 3)]
    # a=tempo_numbered_edges[(np.min(Y) - 2):(1456 + 3), (np.min(X) - 2):(682 + 3)]
    # b = tempo_numbered_vertices[(np.min(Y) - 2):(1456 + 3), (np.min(X) - 2):(682 + 3)]
    # a[b > 0] = b[b > 0]
    # tempo_vertices[(np.min(Y) - 2):(np.max(Y) + 3), (np.min(X) - 2):(np.max(X) + 3)]
    # a = tempo_numbered_edges[(np.min(Y) - 2):(np.max(Y) + 3), (np.min(X) - 2):(np.max(X) + 3)]
    # a = edge_i[(np.min(Y) - 2):(np.max(Y) + 3), (np.min(X) - 2):(np.max(X) + 3)]

    # Remove small tips:
    # Get the vertices appearing only once
    tempo_unique_vertices, counts = np.unique(tempo_edges_labels[:, 1:], return_counts=True)
    # Tips are vertices appearing only once AND not connected to any vertices:
    potential_tips_labels = tempo_unique_vertices[counts == 1]
    tips = np.isin(tempo_numbered_vertices, potential_tips_labels)
    tips = tips.astype(np.uint8)
    cnv8 = CompareNeighborsWithValue((tempo_numbered_vertices + tempo_numbered_edges) > 0, 8)
    cnv8.is_equal(1, and_itself=True)
    tips[cnv8.equal_neighbor_nb != 1] = 0

    # _, connected_vertices, cv_stats,_ = cv2.connectedComponentsWithStats((tempo_numbered_vertices > 0).astype(np.uint8), connectivity=8)
    # false_tips = np.nonzero(cv_stats[:, 4] > 1)[0][1:]
    # for false_tip in false_tips:
    #     tips[connected_vertices == false_tip] = 0

    numbered_tips = tips.astype(np.int64)
    numbered_tips *= tempo_numbered_vertices
    dil_tips = cv2.dilate(tips, square_33)
    edges_with_tip = np.unique(tempo_numbered_edges * dil_tips)[1:]
    # Loop over every edge with tip, and remove those that are shorter than the width of the other edge pixels
    # that are connected to the vertex connecting the edge with tip.
    for edge_with_tip in edges_with_tip: # edge_with_tip = 5
        # Get the vertices connected to that edge:
        include_a_tip = tempo_edges_labels[:, 0] == edge_with_tip
        vertices_pair = tempo_edges_labels[include_a_tip, 1:][0]
        tip_pos = np.nonzero(np.isin(vertices_pair, numbered_tips))[0]
        if len(tip_pos) == 2:
            print(f"PROBLEM HERE: {edge_with_tip}")
            # The edge connects both two tips and one branching vertex
            # Find the branching vertex
            eY, eX = np.nonzero(tempo_numbered_edges == edge_with_tip)
            sub_edge = (tempo_numbered_edges == edge_with_tip)[(np.min(eY) - 2):(np.max(eY) + 3), (np.min(eX) - 2):(np.max(eX) + 3)]
            sub_tempo_numbered_vertices = tempo_numbered_vertices[(np.min(eY) - 2):(np.max(eY) + 3), (np.min(eX) - 2):(np.max(eX) + 3)]
            all_in = (sub_tempo_numbered_vertices + sub_edge) > 0
            _, sub_edges, stats, _ = cv2.connectedComponentsWithStats(all_in.astype(np.uint8))
            branching_vertex = sub_tempo_numbered_vertices[sub_edges == (np.argmax(stats[1:, 4]) + 1)]
            branching_vertex = branching_vertex[np.logical_and(branching_vertex > 0, np.logical_not(np.isin(branching_vertex, vertices_pair)))][0]
            # Split the edge
            cv2.dilate((sub_tempo_numbered_vertices == branching_vertex).astype(np.uint8), square_33) * sub_edge
            # Rename one part of the edge
            # add these to the edges_with_tip list to treat them later in the while loop (transform the loop)
        else:
            non_tip_vertex = vertices_pair[1 - tip_pos]
            non_tip_im = (tempo_numbered_vertices == non_tip_vertex).astype(np.uint8)
            dil_non_tip_im = cv2.dilate(non_tip_im, square_33, iterations=1)
            non_tip_skel_width = pad_distances * dil_non_tip_im
            if edge_i.sum() < non_tip_skel_width.max():
                edge_i_coord = np.nonzero(tempo_numbered_edges == edge_with_tip)
                tempo_numbered_edges[edge_i_coord] = 0
                tempo_edges_labels = tempo_edges_labels[np.logical_not(include_a_tip)]
                # If the remaining vertex only connects two edges, remove it and fuse the edges
                if counts[tempo_unique_vertices == non_tip_vertex] == 3:
                    rows, colomns = np.nonzero(tempo_edges_labels[:, 1:] == non_tip_vertex)
                    edge_names = tempo_edges_labels[rows, 0]
                    if len(edge_names) == 2:
                        # Use the first edge label to label both edges:
                        tempo_numbered_edges[tempo_numbered_edges == edge_names[1]] = edge_names[0]
                        # Remove the vertex and give it that same edge label
                        tempo_numbered_edges[tempo_numbered_vertices == non_tip_vertex] = edge_names[0]
                        tempo_numbered_vertices[tempo_numbered_vertices == non_tip_vertex] = 0


        # edge_i = (tempo_numbered_edges == edge_with_tip).astype(np.uint8)
        # dil_edge_i = cv2.dilate(edge_i, square_33, iterations=2)
        # closer_edge_distance = distances * dil_edge_i
        # closer_edge_distance[tips_coord[0], tips_coord[1]] = 0
        # closer_edge_distance[edge_i_coord] = 0
        # if edge_i.sum() < closer_edge_distance.max():
        #     tempo_numbered_edges[edge_i_coord] = 0
        #     tempo_edges_labels[tempo_edges_labels[:, 0] == edge_with_tip, 1:]

    # Remove the vertices that are not connecting any edge, and rename labels everywhere
    tempo_unique_vertices, counts = np.unique(tempo_edges_labels[:, 1:], return_counts=True)

    nb_v = len(tempo_unique_vertices)
    vertices_table = np.zeros((nb_v, 4), dtype=np.uint64)
    unique_vertices = np.zeros_like(tempo_unique_vertices)
    edges_labels = np.zeros_like(tempo_edges_labels)
    numbered_vertices = np.zeros_like(tempo_numbered_vertices)

    # CURRENT PROBLEM : WE have some tempo_numbered_vertices that have been removed while they still exist in tempo_edges_labels
    #IDEA: Do not remove vertices from tempo_numbered_vertices and do not replace vertices pixels by edges. Instead, create edges between each connected vertices
    for i, old_label in enumerate(tempo_unique_vertices):
        new_label = i + 1
        # Update old vertex labels with new ones on all tables
        edges_labels[:, 1:][tempo_edges_labels[:, 1:] == old_label] = new_label
        Vyi, Vxi = np.nonzero(tempo_numbered_vertices == old_label)
        numbered_vertices[Vyi, Vxi] = new_label
        vertices_table[i, 1] = Vyi[0]
        vertices_table[i, 2] = Vxi[0]
        vertex_bool = tempo_unique_vertices == old_label
        unique_vertices[vertex_bool] = new_label
        vertices_table[i, 0] = new_label
        if counts[vertex_bool] == 1:
            vertices_table[i, 3] = 1

    # Rename labels to not miss any number between 1 and edges_labels.shape[0]
    numbered_edges = np.zeros_like(tempo_numbered_edges)
    for i in range(1, edges_labels.shape[0] + 1):
        old_name = tempo_edges_labels[i - 1, 0]
        numbered_edges[tempo_numbered_edges == old_name] = i
        edges_labels[i - 1, 0] = i
    return numbered_vertices, numbered_edges, vertices_table, edges_labels


def add_central_contour(pad_skeleton, pad_distances, pad_origin, pad_network, pad_origin_centroid):
    """

    """
    pad_net_contour = get_contours(pad_network)

    # Make a hole at the skeleton center and find the vertices connecting it
    holed_skeleton = pad_skeleton * (1 - pad_origin)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    dil_origin = cv2.dilate(pad_origin, Ellipse((5, 5)).create().astype(np.uint8), iterations=20)
    pad_vertices *= dil_origin
    connecting_pixels = np.transpose(np.array(np.nonzero(pad_vertices)))

    skeleton_without_vertices = pad_skeleton.copy()
    skeleton_without_vertices[pad_vertices > 0] = 0

    # # Connect each vertex to the nearest on the contour of the origin
    import scipy
    # line_coordinates = []
    # for vertex in connecting_pixels: #vertex = connecting_pixels[0]
    #     dist_transform = np.zeros_like(pad_skeleton)
    #     dist_transform[vertex[0], vertex[1]] = 1
    #     dist_transform = scipy.ndimage.distance_transform_edt(1 - dist_transform)
    #     dist_transform *= pad_origin_contours
    #     nearest_pix_on_the_ring = dist_transform[dist_transform > 0].min()
    #     nearest_y, nearest_x = np.nonzero(dist_transform == nearest_pix_on_the_ring)
    #     line_coords = get_line_points(np.array([nearest_y[0], nearest_x[0]]), vertex)
    #     line_coordinates.append(line_coords)

    # Previously was connected to the center of the shape.
    line_coordinates = get_all_line_coordinates(pad_origin_centroid, connecting_pixels)
    with_central_contour = holed_skeleton.copy()
    for vertex, new_edge in zip(connecting_pixels, line_coordinates): # nei = 65; new_edge=line_coordinates[nei]
        new_edge_im = np.zeros_like(pad_origin)
        new_edge_im[new_edge[:, 0], new_edge[:, 1]] = 1
        if not np.any(new_edge_im * pad_net_contour) and not np.any(new_edge_im * skeleton_without_vertices):# and not np.any(new_edge_im * holed_skeleton):
            # if np.any(new_edge_im * holed_skeleton):
            #     # break
            #     # Find the nearest points in holed_skeleton to connect this vertex
            #     dist_transform = np.zeros_like(pad_skeleton)
            #     dist_transform[vertex[0], vertex[1]] = 1
            #     dist_transform = scipy.ndimage.distance_transform_edt(1 - dist_transform)
            #     dist_transform *= pad_vertices
            #     nearest_pix_on_mask = dist_transform[dist_transform > 0].min()
            #     nearest_y, nearest_x = np.nonzero(dist_transform == nearest_pix_on_mask)
            #     new_edge = get_line_points(np.array([nearest_y[0], nearest_x[0]]), np.array([vertex[0], vertex[1]]))
            # holed_skeleton = with_central_contour.copy()
            # holed_skeleton[new_edge[:, 0], new_edge[:, 1]] = 1
            # holed_skeleton = cv2.dilate(holed_skeleton, square_33)
            # holed_skeleton *= (1 - pad_origin)

            with_central_contour[new_edge[:, 0], new_edge[:, 1]] = 1
            # pad_distances[new_edge[:, 0], new_edge[:, 1]] = pad_distances[vertex[0], vertex[1]]

    # Add dilated contour
    pad_origin_contours = get_contours(pad_origin)
    with_central_contour *= (1 - pad_origin)
    with_central_contour += pad_origin_contours
    if np.any(with_central_contour == 2):
        with_central_contour[with_central_contour > 0] = 1

    # show(dil_origin * with_central_contour)
    # Capture only the new contour and its neighborhood, get its skeleton and update the final skeleton
    new_contour = cv2.morphologyEx(dil_origin * with_central_contour, cv2.MORPH_CLOSE, square_33)
    new_contour = morphology.medial_axis(new_contour, rng=0).astype(np.uint8)
    new_skeleton = with_central_contour * (1 - dil_origin)
    new_skeleton += new_contour
    # nb, sh = cv2.connectedComponents(new_skeleton)
    # new_pixels = cv2.dilate(with_central_contour * (1 - pad_skeleton), square_33)
    # new_skeleton = np.logical_or(with_central_contour, new_pixels).astype(np.uint8)

    # new_skeleton = with_central_contour * (1 - pad_skeleton)
    # new_skeleton = cv2.morphologyEx(new_skeleton, cv2.MORPH_CLOSE, square_33)
    # # new_skeleton = morphology.medial_axis(new_skeleton)
    # new_skeleton = morphology.medial_axis(new_skeleton, rng=0).astype(np.uint8)
    # new_skeleton1 = morphology.skeletonize(new_skeleton, method='lee').astype(np.uint8)
    #
    # new_skeleton2 = pad_skeleton + new_skeleton
    # new_skeleton2 *= (1 - pad_origin)
    # new_skeleton2 += pad_origin_contours
    # new_skeleton2[new_skeleton2 > 0] = 1
    # new_skeleton2 = morphology.skeletonize(new_skeleton2, method='lee').astype(np.uint8)
    # pad_origin_contours = new_skeleton*cv2.dilate(pad_origin_contours, cross_33)
    # nb, sh = cv2.connectedComponents((new_skeleton2*pad_origin_contours).astype(np.uint8))
    # new_pixels = (1 - pad_origin_contours) * np.logical_and(pad_distances == 0, new_skeleton2 == 1)
    new_pixels = np.logical_and(pad_distances == 0, new_skeleton == 1)
    new_pix_coord = np.transpose(np.array(np.nonzero(new_pixels)))
    dist_coord = np.transpose(np.array(np.nonzero(pad_distances)))

    dist_from_dist = cdist(new_pix_coord[:, :], dist_coord)
    for np_i, dist_i in enumerate(dist_from_dist): # dist_i=dist_from_dist[0]
        close_i = dist_i.argmin()
        pad_distances[new_pix_coord[np_i, 0], new_pix_coord[np_i, 1]] = pad_distances[dist_coord[close_i, 0], dist_coord[close_i, 1]]
    pad_distances *= new_skeleton
    # pad_distances[np.logical_and(pad_distances == 0, new_skeleton2 == 1)] = pad_distances.max() + 1
    # show(pad_distances)
    # show(pad_skeleton)
    # show(new_skeleton2)
    # Update distances
    dil_pad_origin_contours = cv2.dilate(pad_origin_contours, cross_33, iterations=1)
    new_pad_origin_contours = dil_pad_origin_contours * new_skeleton
    nb, sh = cv2.connectedComponents(new_pad_origin_contours)
    while nb > 2:
        dil_pad_origin_contours = cv2.dilate(dil_pad_origin_contours, cross_33, iterations=1)
        new_pad_origin_contours = dil_pad_origin_contours * new_skeleton
        nb, sh = cv2.connectedComponents(new_pad_origin_contours)
    pad_origin_contours = new_pad_origin_contours
    pad_distances[pad_origin_contours > 0] = np.nan # pad_distances.max() + 1 #
    # test1 = ((pad_distances > 0) * (1 - new_skeleton)).sum() == 0
    # test2 = ((1 - (pad_distances > 0)) *  new_skeleton).sum() == 0

    return new_skeleton, pad_distances, pad_origin_contours


def add_central_vertex(numbered_vertices, numbered_edges, vertices_table, edges_labels, origin, network_contour):
    """
    Make links between the center of the origin and the leaves touching it
    :return:
    """
    cnv4, cnv8 = get_neighbor_comparisons(holed_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(holed_skeleton, cnv4, cnv8)
    leaves_im = np.zeros_like(potential_tips)
    # leaf_id = np.nonzero(vertices_table[:, 3] == 1)[0] + 1
    # leaves_coord = np.nonzero(np.isin(numbered_vertices, leaf_id))
    # dil_origin = cv2.dilate(origin, square_33)
    dil_origin = cv2.dilate(pad_origin, Ellipse((5, 5)).create().astype(np.uint8), iterations=30)
    potential_tips *= dil_origin
    connecting_pixels = np.transpose(np.array(np.nonzero(potential_tips)))

    # Get center coord
    _, _, _, origin_centroid = cv2.connectedComponentsWithStats(pad_origin)
    origin_centroid = np.round(origin_centroid[1, :]).astype(np.uint64)
    # add this new vertex to vertices_table, and numbered_vertices
    # new_vertex_label = vertices_table.shape[0] + 1
    # numbered_vertices[origin_centroid[0], origin_centroid[1]] = new_vertex_label
    # new_vertex = np.array((new_vertex_label, origin_centroid[0], origin_centroid[1], 0), dtype=np.uint64)
    # vertices_table = np.vstack((vertices_table, new_vertex))

    # Get skeleton pixels connecting center:
    # connecting_pixels = vertices_table[np.unique(leaves_im)[1:] - 1, :3]
    # Same as: np.transpose(np.array(np.nonzero(leaves_im), dtype=np.uint64))
    # Draw lines between these and the center
    # line_coordinates = get_all_line_coordinates(origin_centroid, connecting_pixels[:, 1:])
    e_nb = edges_labels.shape[0]
    line_coordinates = get_all_line_coordinates(origin_centroid, connecting_pixels)
    e_nb = edges_labels.shape[0]
    nei = 0
    for new_edge in line_coordinates: # nei = 65; new_edge=line_coordinates[nei]
        # if np.any(np.logical_and(new_edge[:, 0] == 781, new_edge[:, 1] == 604)): #797,  563
        #     print(nei)
        new_edge_im = np.zeros_like(origin)
        new_edge_im[new_edge[:, 0], new_edge[:, 1]] = 1
        if not np.any(new_edge_im * network_contour):
            nei += 1
            numbered_edges[new_edge[:, 0], new_edge[:, 1]] = e_nb + nei + 1
            # new_edge_label = np.array((skel_coord[nei, 0], skel_coord[nei, 1]), dtype=np.uint64)
            new_edge_label = np.array((e_nb + nei + 1, new_vertex[0], connecting_pixels[nei, 0]), dtype=np.uint64)
            edges_labels = np.vstack((edges_labels, new_edge_label))
            # Specify that this vertex is not a leaf anymore
            vertices_table[vertices_table[:, 0] == connecting_pixels[nei, 0], 3] = 0

    return numbered_vertices, numbered_edges, vertices_table, edges_labels

def get_edges_table(numbered_edges, distances, greyscale_img):
    nb_e = len(np.unique(numbered_edges)) - 1
    edges_table = []
    for ei in range(1, nb_e + 1):
        Eyi, Exi = np.nonzero(numbered_edges == ei)
        edges_table.append(np.stack((np.repeat(ei, len(Eyi)), Eyi, Exi, distances[Eyi, Exi], greyscale_img[Eyi, Exi]), axis=1, dtype=np.float64))
    edges_table = np.vstack(edges_table)

    return edges_table

def save_network_as_csv(full_network, skeleton, vertices_table, edges_table, edges_labels, pathway):
    # node_labels = np.arange(1, nb_v + 1)
    # vertices_table = np.zeros((nb_v, 4), dtype=np.uint64)
    # terms = np.zeros_like(numbered_vertices)
    # terms[terminations > 0] = numbered_vertices[terminations > 0]
    # for i, node in enumerate(node_labels):
    #     vertices_table[i, 0] = node
    #     vertices_table[i, 1] = np.nonzero(numbered_vertices == node)[0][0]
    #     vertices_table[i, 2] = np.nonzero(numbered_vertices == node)[1][0]
    #     if np.any(terms == node):
    #         vertices_table[i, 3] = 1
    pd.DataFrame(np.transpose(np.array(np.nonzero(full_network))), columns=["y_coord", "x_coord"]).to_csv(
        pathway / f"full_net_coord_imshape={full_network.shape}.csv", index=False)
    pd.DataFrame(np.transpose(np.array(np.nonzero(skeleton))), columns=["y_coord", "x_coord"]).to_csv(
        pathway / f"skeleton_coord_imshape={full_network.shape}.csv", index=False)

    pd.DataFrame(edges_labels, columns=["edge_id", "vertex1", "vertex2"]).to_csv(pathway / f"edges_labels_imshape={full_network.shape}.csv", index=False)
    pd.DataFrame(vertices_table, columns=["vertex_id", "y_coord", "x_coord", "is_leaf"]).to_csv(pathway / f"vertices_coord_imshape={full_network.shape}.csv", index=False)
    pd.DataFrame(edges_table, columns=["edge_id", "y_coord", "x_coord", "width", "height"]).to_csv(pathway / f"skeleton_coord_imshape={full_network.shape}.csv", index=False)


def save_graph_image(binary_im, full_network, numbered_edges, distances, origin, vertices_table, pathway):
    valued_skeleton = np.zeros_like(distances)
    valued_skeleton[numbered_edges > 0] = 9
    valued_skeleton[np.nonzero(numbered_edges * (1 - origin))] = distances[np.nonzero(numbered_edges * (1 - origin))]
    valued_skeleton = bracket_to_uint8_image_contrast(valued_skeleton)
    cell_contours = get_contours(binary_im)
    net_contours = get_contours(full_network)
    valued_skeleton[np.nonzero(cell_contours)] = 9
    valued_skeleton[np.nonzero(net_contours)] = 255
    vertices_coord = vertices_table[:, 1:3]
    leaves_coord = vertices_table[vertices_table[:, 3] == 1, 1:3]
    vertices = np.zeros_like(binary_im)
    vertices[vertices_coord[:, 0], vertices_coord[:, 1]] = 1
    vertices = cv2.dilate(vertices, cross_33)
    valued_skeleton[np.nonzero(vertices)] = 240
    valued_skeleton[leaves_coord[:, 0], leaves_coord[:, 1]] = 140
    plt.imshow(valued_skeleton, cmap='nipy_spectral')
    plt.tight_layout()
    plt.show()
    plt.savefig(pathway / f"contour network with medial axis.png", dpi=1500)
    plt.close()


    # # Remove all edges of only one pixel
    # _, numbered_edges, stats, _ = cv2.connectedComponentsWithStats(edges)
    # too_small_edges = np.nonzero(stats[:, -1] == 1)[0]
    # c = 0
    # while len(too_small_edges) > 0 and c < 100:
    #     for too_small_edge in too_small_edges:
    #         skeleton[numbered_edges == too_small_edge] = 0
    #     vertices = get_vertices_from_skeleton(skeleton)
    #     edges = (1 - vertices) * skeleton
    #     _, numbered_edges, stats, _ = cv2.connectedComponentsWithStats(edges)
    #     too_small_edges = np.nonzero(stats[:, -1] == 1)[0]
    #     c += 1
    # print(c)

def save_vertices_and_tips_image(pad_skeleton, pad_vertices, pad_potential_tips, pathway):
    plt.figure(figsize=(15, 15))
    valued_skeleton = pad_skeleton.copy()
    valued_skeleton *= 20
    valued_skeleton[np.nonzero(pad_vertices)] = 200
    valued_skeleton[np.nonzero(pad_potential_tips)] = 40
    plt.imshow(valued_skeleton, cmap='nipy_spectral')
    plt.tight_layout()
    plt.show()
    plt.savefig(pathway / f"vertices and tips on skeleton.png", dpi=1500)
    plt.close()



def old_get_vertices_from_skeleton(skeleton):
    """
    Find the vertices from a skeleton according to the following rules:
    - Network terminations at the border are nodes
    - The 4-connected nodes have priority over 8-connected nodes
    :return:
    """
    # O-padding to allow boundary nodes
    pad_skeleton = np.pad(skeleton, [(1, ), (1, )], mode='constant')
    # pad_skeleton = skeleton
    cnv8 = CompareNeighborsWithValue(pad_skeleton, 8)
    cnv8.is_equal(1, and_itself=True)
    # All pixels having only one neighbor, and containing the value 1, are terminations for sure
    potential_tips = np.zeros(pad_skeleton.shape, dtype=np.uint8)
    potential_tips[cnv8.equal_neighbor_nb == 1] = 1

    # Add more terminations using 4-connectivity
    # If a pixel is 1 (in 4) and all its neighbors are neighbors (in 4), it is a termination
    cnv4 = CompareNeighborsWithValue(pad_skeleton, 4)
    cnv4.is_equal(1, and_itself=True)
    coord1_4 = cnv4.equal_neighbor_nb == 1
    if np.any(coord1_4):
        coord1_4 = np.nonzero(coord1_4)
        for y1, x1 in zip(coord1_4[0], coord1_4[1]):
            # If, in the neighborhood of the 1 (in 4), all (in 8) its neighbors are 4-connected, and none of them are terminations, the 1 is a termination
            is_4neigh = cnv4.equal_neighbor_nb[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] != 0
            all_4_connected = pad_skeleton[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] == is_4neigh
            is_not_term = 1 - potential_tips[y1, x1]
            # is_not_term = (1 - potential_tips[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)])
            if np.all(all_4_connected * is_not_term):
                is_4neigh[1, 1] = 0
                is_4neigh = np.pad(is_4neigh, [(1,), (1,)], mode='constant')
                cnv_4con = CompareNeighborsWithValue(is_4neigh, 4)
                cnv_4con.is_equal(1, and_itself=True)
                all_connected = cnv_4con.equal_neighbor_nb.sum()
                # If they are connected, it can be a termination
                if all_connected:
                    # print(y1,x1)
                    # If its closest neighbor is above 3 (in 8), this one is a node
                    is_closest_above_3 = cnv8.equal_neighbor_nb[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] * cross_33 > 3
                    if np.any(is_closest_above_3):
                        Y, X = np.nonzero(is_closest_above_3)
                        Y += y1 - 1
                        X += x1 - 1
                        potential_tips[Y, X] = 1
                    # Otherwise, it is a termination
                    else:
                        potential_tips[y1, x1] = 1
    # Initiate the vertices final matrix as a copy of the potential_tips
    pad_vertices = deepcopy(potential_tips)
    for i, neighbor_nb in enumerate([8, 7, 6, 5, 4]):
        # All pixels having neighbor_nb neighbor are potential vertices
        potential_vertices = np.zeros(pad_skeleton.shape, dtype=np.uint8)
        potential_vertices[cnv8.equal_neighbor_nb == neighbor_nb] = 1
        # remove the false intersections that are a neighbor of a previously detected intersection
        # Dilate vertices to make sure that no neighbors of the current potential vertices are already vertices.
        dilated_previous_intersections = cv2.dilate(pad_vertices, cross_33, iterations=1)
        potential_vertices *= (1 - dilated_previous_intersections)
        pad_vertices[np.nonzero(potential_vertices)] = 1

    # Having 3 neighbors is ambiguous
    with_3_neighbors = cnv8.equal_neighbor_nb == 3
    if np.any(with_3_neighbors):
        # We compare 8-connections with 4-connections
        # We loop over all 3 connected
        coord_3 = np.nonzero(with_3_neighbors)
        for y3, x3 in zip(coord_3[0], coord_3[1]):
            # If, in the neighborhood of the 3, there is at least a 2 (in 8) that is 0 (in 4), and not a termination: the 3 is a node
            has_2_8neigh = cnv8.equal_neighbor_nb[(y3-1):(y3+2), (x3-1):(x3+2)] > 0#1
            is_not_term = 1 - potential_tips[y3,x3]
            # is_not_term = np.logical_not(potential_tips[(y3-1):(y3+2), (x3-1):(x3+2)])

            # has_0_4neigh = cnv4.equal_neighbor_nb[(y3-1):(y3+2), (x3-1):(x3+2)] == 0
            if np.any(has_2_8neigh * is_not_term):
                # At least 3 of the 8neigh are not connected:
                has_2_8neigh[1, 1] = 0
                has_2_8neigh = np.pad(has_2_8neigh, [(1,), (1,)], mode='constant')
                cnv_8con = CompareNeighborsWithValue(has_2_8neigh, 4)
                cnv_8con.is_equal(1, and_itself=True)
                disconnected_nb = has_2_8neigh.sum() - cnv_8con.equal_neighbor_nb.sum()
                # disconnected_nb, shape = cv2.connectedComponents(has_2_8neigh.astype(np.uint8), connectivity=4)
                # nb_not_connected = has_2_8neigh.sum() - (disconnected_nb - 1)
                if disconnected_nb > 2:
                    # print(y3, x3)
                    pad_vertices[y3, x3] = 1
        # potential_vertices = np.zeros(pad_skeleton.shape, dtype=np.uint8)
        # pad_vertices[np.logical_and(cnv4.equal_neighbor_nb == 2, cnv8.equal_neighbor_nb == 3)] = 1

    cnvv = CompareNeighborsWithValue(pad_vertices, 4)
    cnvv.is_equal(1, and_itself=True)
    if np.any(cnvv.equal_neighbor_nb):
        nb, numbered_nodes = cv2.connectedComponents((cnvv.equal_neighbor_nb > 0).astype(np.uint8))
        for i in range(1, nb):
            node_i = (numbered_nodes == i).astype(np.uint8)
            node_i *= cnvv.equal_neighbor_nb
            if np.any(node_i):
                pad_vertices[np.logical_and(node_i > 0, potential_tips == 0)] = 0
                dil_node_i = cv2.dilate(node_i, square_33, iterations=1)
                dil_node_i *= pad_skeleton
                bary = np.round(np.mean(np.array(np.nonzero(dil_node_i)), 1)).astype(np.uint64)
                # pad_vertices[node_i == 1] = 0
                pad_vertices[bary[0], bary[1]] = 1
                # pad_vertices[np.logical_and(node_i == 1, potential_tips == 0)] = 0
                # if not np.any(node_i > 1):
                #     dil_node_i = cv2.dilate(node_i, square_33, iterations=1)
                #     dil_node_i *= pad_skeleton
                #     bary = np.round(np.mean(np.array(np.nonzero(dil_node_i)), 1)).astype(np.uint64)
                #     # pad_vertices[node_i == 1] = 0
                #     pad_vertices[bary[0], bary[1]] = 1

    # Remove 0-padding
    vertices = pad_vertices[1:-1, 1:-1]
    return vertices

    # We first detect the 4 connected vertices and add them
    # All pixels that have neighbor_nb neighbors, none of which is already detected as a vertex.
    # for neighbor_nb in [4, 3]:
    #     # All pixels having neighbor_nb neighbor are potential vertices
    #     potential_vertices = np.zeros(im_shape, dtype=np.uint8)
    #     potential_vertices[cnv4.equal_neighbor_nb == neighbor_nb] = 1
    #     pad_vertices[np.nonzero(potential_vertices)] = 1

    # # Then, add all 8 connected vertices that are not inside a dilatation of the 4-connected previously detected vertices
    # dilated_previous_intersections = cv2.dilate(pad_vertices, cross_33)# square_33 cross_33
    # for neighbor_nb in [8, 7, 6, 5, 4, 3]:
    #     # All pixels having neighbor_nb neighbor are potential vertices
    #     potential_vertices = np.zeros(im_shape, dtype=np.uint8)
    #     potential_vertices[cnv8.equal_neighbor_nb == neighbor_nb] = 1
    #     potential_vertices[np.nonzero(np.logical_and(cnv8.array == 1, cnv8.equal_neighbor_nb == 0))] = 1
    #     # remove the false intersections that are a neighbor of a previously detected intersection
    #     # Dilate vertices to make sure that no neighbors of the current potential vertices are already vertices.
    #     # # dilated_previous_intersections = cv2.dilate(vertices, Ellipse((5, 5)).create().astype(np.uint8))# square_33 cross_33
    #     potential_vertices *= (1 - dilated_previous_intersections)
    #     pad_vertices[np.nonzero(potential_vertices)] = 1

    # real_vertices = np.zeros_like(vertices)
    # nb, shapes = cv2.connectedComponents(vertices, connectivity=4)
    # for j in range(1, nb):
    #     shape = shapes == j
    #     if shape.sum() == 1:
    #         real_vertices[np.nonzero(shape)] = 1
    #     else:
    #         cnv_shape = CompareNeighborsWithValue(shape, 4)
    #         cnv_shape.is_equal(1, and_itself=True)
    #         real_vertices[np.nonzero(cnv_shape.equal_neighbor_nb == np.max(cnv_shape.equal_neighbor_nb))] = 1



def get_segments_from_vertices_skeleton(skeleton, vertices_coord):
    # skeleton = test_skel; vertices_coord = vertices_positions
    im_shape = skeleton.shape
    vertices = np.zeros(im_shape, dtype=np.uint8)
    vertices[vertices_coord[:, 0], vertices_coord[:, 1]] = 1
    # I dilate the vertices to avoid any connection between segments
    vertices = cv2.dilate(vertices, np.ones((3, 3), np.uint8))
    segments = (1 - vertices) * skeleton
    return segments

