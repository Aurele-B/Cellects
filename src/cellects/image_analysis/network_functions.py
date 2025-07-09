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
from cellects.image_analysis.morphological_operations import square_33, cross_33, cc, Ellipse, CompareNeighborsWithValue, get_contours, get_all_line_coordinates
from cellects.utils.formulas import *
from numba.typed import Dict as TDict
from skimage import morphology
from collections import deque

# 8-connectivity neighbors
neighbors = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]


def get_skeleton_and_widths(pad_network, origin=None):
    pad_skeleton, pad_distances = morphology.medial_axis(pad_network, return_distance=True)
    pad_skeleton = pad_skeleton.astype(np.uint8)
    pad_skeleton, pad_distances = remove_small_loops(pad_skeleton, pad_distances)

    width = 10
    pad_skeleton[pad_distances > width] = 0
    if origin is not None:
        pad_skeleton = pad_skeleton * (1 - origin)
    # Only keep the largest connected component
    pad_skeleton, stats, _ = cc(pad_skeleton)
    pad_skeleton[pad_skeleton > 1] = 0

    pad_distances *= pad_skeleton
    return pad_skeleton, pad_distances
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
    When zeros are surrounded by 4-connected ones and only contain 0 on their diagonal, replace 1 by 0
    and put 1 in the center
    Does not work because it cuts the skeleton!
    :param pad_skeleton:
    :return:
    """
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    # sure_terminations = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)

    cnv_diag_0 = CompareNeighborsWithValue(pad_skeleton, 0)
    cnv_diag_0.is_equal(0, and_itself=True)

    cnv4_false = CompareNeighborsWithValue(pad_skeleton, 4)
    cnv4_false.is_equal(1, and_itself=False)

    loop_centers = np.logical_and((cnv4_false.equal_neighbor_nb == 4), cnv_diag_0.equal_neighbor_nb == 4).astype(np.uint8)

    surrounding = cv2.dilate(loop_centers, kernel=square_33)
    surrounding -= loop_centers
    surrounding = surrounding * cnv8.equal_neighbor_nb

    # Every 2 can be replaced by 0 if the loop center becomes 1
    filled_loops = pad_skeleton.copy()
    filled_loops[surrounding == 2] = 0
    filled_loops += loop_centers

    new_pad_skeleton = morphology.medial_axis(filled_loops)
    # Put the new pixels in pad_distances
    new_pixels = new_pad_skeleton * (1 - pad_skeleton)
    if pad_distances is None:
        return pad_skeleton
    else:
        pad_distances[np.nonzero(new_pixels)] = 2.
        # for yi, xi in zip(npY, npX): # yi, xi = npY[0], npX[0]
        #     distances[yi, xi] = 2.
        return pad_skeleton, pad_distances


def get_neighbor_comparisons(pad_skeleton):
    cnv4 = CompareNeighborsWithValue(pad_skeleton, 4)
    cnv4.is_equal(1, and_itself=True)
    cnv8 = CompareNeighborsWithValue(pad_skeleton, 8)
    cnv8.is_equal(1, and_itself=True)
    return cnv4, cnv8


# def get_vertices_and_edges_from_skeleton(pad_skeleton):
def keep_one_connected_component(pad_skeleton):
    """
    """
    nb, nb_pad_skeleton = cv2.connectedComponents(pad_skeleton)
    pad_skeleton[nb_pad_skeleton > 1] = 0
    return pad_skeleton


def get_vertices_and_tips_from_skeleton(pad_skeleton):
    """
    Find the vertices from a skeleton according to the following rules:
    - Network terminations at the border are nodes
    - The 4-connected nodes have priority over 8-connected nodes
    :return:
    """
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    pad_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices = get_inner_vertices(pad_tips, cnv8)
    # pad_edges = (1 - pad_vertices) * pad_skeleton
    # pathway = Path(f"/Users/Directory/Data/dossier1/plots")
    # save_vertices_and_tips_image(pad_skeleton, pad_vertices, pad_sure_terminations, pathway)
    return pad_vertices, pad_tips
    # return pad_skeleton, pad_vertices, pad_edges, pad_tips


def get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8):
    # All pixels having only one neighbor, and containing the value 1, are terminations for sure
    sure_terminations = np.zeros(pad_skeleton.shape, dtype=np.uint8)
    sure_terminations[cnv8.equal_neighbor_nb == 1] = 1
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
            is_not_term = 1 - sure_terminations[y1, x1]
            # is_not_term = (1 - sure_terminations[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)])
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
                        sure_terminations[Y, X] = 1
                    sure_terminations[y1, x1] = 1
    return sure_terminations


def get_inner_vertices(sure_terminations, cnv8):
    # Initiate the vertices final matrix as a copy of the sure_terminations
    pad_vertices = deepcopy(sure_terminations)
    for neighbor_nb in [8, 7, 6, 5, 4]:
        # All pixels having neighbor_nb neighbor are potential vertices
        potential_vertices = np.zeros(sure_terminations.shape, dtype=np.uint8)

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
            # y3, x3 = 3,3
            # If, in the neighborhood of the 3, there is at least a 2 (in 8) that is 0 (in 4), and not a termination: the 3 is a node
            has_2_8neigh = cnv8.equal_neighbor_nb[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)] > 0  # 1
            # is_term = sure_terminations[y3, x3]
            # is_not_term = np.logical_not(sure_terminations[(y3-1):(y3+2), (x3-1):(x3+2)])
            has_2_8neigh_without_focal = has_2_8neigh.copy()
            has_2_8neigh_without_focal[1, 1] = 0
            # all_are_nodes = np.array_equal(has_2_8neigh_without_focal, pad_vertices[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)])
            node_but_not_term = pad_vertices[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)] * (1 - sure_terminations[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)])
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
        # potential_vertices = np.zeros(pad_skeleton.shape, dtype=np.uint8)
        # pad_vertices[np.logical_and(cnv4.equal_neighbor_nb == 2, cnv8.equal_neighbor_nb == 3)] = 1
    return pad_vertices


def get_branches_and_tips_coord(pad_vertices, pad_tips):
    pad_branches = pad_vertices - pad_tips
    branching_vertices_coord = np.transpose(np.array(np.nonzero(pad_branches)))
    tips_coord = np.transpose(np.array(np.nonzero(pad_tips)))
    return branching_vertices_coord, tips_coord


def find_closest_vertices(skeleton, all_vertices_coord, starting_vertices_coord):
    """ skeleton, all_vertices_coord, starting_vertices_coord = cropped_skeleton, cropped_branching_vertices_coord, connecting_vertices_coord
    For each vertex, find the nearest branching vertex along the skeleton.

    Parameters:
    - skeleton (2D np.ndarray): Binary skeleton image (0 and 1)
    - all_vertices_coord (tuple): (array_y, array_x) coordinates of the first vertices
    - starting_vertices_coord (tuple): (array_y, array_x) coordinates of the second vertices

    Returns:
    - dict: keys are tip coordinates, values are (branch_vertex_coords, geodesic_distance)
    """

    # Convert branching vertices to set for quick lookup
    branch_set = set(zip(all_vertices_coord[:, 0], all_vertices_coord[:, 1]))
    n = len(starting_vertices_coord[:, 0])
    connecting_vertices_coord = np.zeros((n, 2), np.uint32)
    edge_lengths = np.zeros(n, np.float64)
    all_path_pixels = []  # Will hold rows of (start_index, y, x)
    i = 0
    ne_i = 0
    for tip_y, tip_x in zip(starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]):
        # tip_y, tip_x = starting_vertices_coord[i, 0], starting_vertices_coord[i, 1]
        # ky = 5
        # ly = ky + 1
        # kx = 10
        # lx = kx + 1
        # a = skeleton[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]
        # a = pad_skeleton[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]
        # a = numbered_vertices[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]

        visited = np.zeros_like(skeleton, dtype=bool)
        parent = {}
        q = deque()

        q.append((tip_y, tip_x))
        visited[tip_y, tip_x] = True
        parent[(tip_y, tip_x)] = None
        found_vertex = None

        while q:
            r, c = q.popleft()

            # Check for branching vertex (ignore the starting tip itself)
            if (r, c) in branch_set and (r, c) != (tip_y, tip_x):
                found_vertex = (r, c)
                break  # stop at first encountered (shortest due to BFS)

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if (0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and
                    not visited[nr, nc] and skeleton[nr, nc] > 0):
                    visited[nr, nc] = True
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))

        if found_vertex:
            ne_i += 1
            fy, fx = found_vertex
            connecting_vertices_coord[i, :] = [fy, fx]

            # Reconstruct path from found_vertex back to tip
            path = []
            current = (fy, fx)
            while current is not None:
                path.append((i, *current))
                current = parent[current]

            # path.reverse()  # So path goes from starting tip to found vertex

            for _, y, x in path: # Exclude no vertices from the edge pixels path[1:-1]
                all_path_pixels.append((i, y, x))

            edge_lengths[i] = len(path) - 1  # exclude one node for length computation
        else:
            # connecting_vertices_coord[i, :] = None
            edge_lengths[i] = np.nan
        i += 1

    edges_pixel_coords = np.array(all_path_pixels, dtype=np.uint32)
    # example = np.nonzero(np.isnan(edge_lengths))[0]
    return connecting_vertices_coord, edge_lengths, edges_pixel_coords


def remove_tipped_edge_smaller_than_branch_width(pad_skeleton, connecting_vertices_coord, pad_distances, edge_lengths, tips_coord, branching_vertices_coord, edges_pixel_coords):

    # numbered_vertices = np.zeros(pad_skeleton.shape, dtype=np.uint32)
    # numbered_vertices[connecting_vertices_coord[:, 0], connecting_vertices_coord[:, 1]] = np.arange(1, connecting_vertices_coord.shape[0] + 1)
    # numbered_vertices[tips_coord[:, 0], tips_coord[:, 1]] = np.arange(1, tips_coord.shape[0] + 1)
    # numbered_vertices[branching_vertices_coord[:, 0], branching_vertices_coord[:, 1]] = np.arange(1, branching_vertices_coord.shape[0] + 1)
    # tip_y, tip_x = 164, 417
    # ky = 5
    # ly = ky + 1
    # kx = 10
    # lx = kx + 1
    # a = numbered_vertices[(tip_y - ky):(tip_y + ly), (tip_x - kx):(tip_x + lx)]
    # Identify edges that are smaller than the width of the branch it is attached to
    tipped_edges_to_remove = np.zeros(edge_lengths.shape[0], dtype=bool)
    connecting_vertices_to_remove = np.zeros(connecting_vertices_coord.shape[0], dtype=bool)
    branches_to_remove = np.zeros(branching_vertices_coord.shape[0], dtype=bool)
    new_edges_pixel_coords = []
    remaining_tipped_edges_nb = 0
    for i in range(len(edge_lengths)):
        Y, X  = connecting_vertices_coord[i, 0], connecting_vertices_coord[i, 1]
        edge_bool = edges_pixel_coords[:, 0] == i
        eY, eX = edges_pixel_coords[edge_bool, 1], edges_pixel_coords[edge_bool, 2]
        if pad_distances[(Y - 1): (Y + 2), (X - 1): (X + 2)].max() > edge_lengths[i]:
            # print(i)
            tipped_edges_to_remove[i] = True

            # # Remove the corresponding tip: already done in find_closest_vertices
            # pad_skeleton[tips_coord[:, 0][i], tips_coord[:, 1][i]] = 0

            # Remove the corresponding edge, but not the connecting vertex (hence [1:])
            # edge_bool = edges_pixel_coords[:, 0] == (i + 1)
            eY, eX = eY[1:], eX[1:]
            pad_skeleton[eY, eX] = 0
            # pad_skeleton[(Y - 1): (Y + 2), (X - 1): (X + 2)]
            # t = 0
            #a=  pad_skeleton[(eY[t] - 10): (eY[t] + 11), (eX[t] -20): (eX[t] + 6)]
            # nb, shape, stats, centr= cv2.connectedComponentsWithStats(pad_skeleton)
            # yt, xt = 625, 809
            # numbered_vertices[(yt - 5): yt + 6, (xt - 5): xt + 6]
            # numbered_vertices[eY, eX] = 0
            # nb, _ = cv2.connectedComponents(pad_skeleton)
            # if nb > 2:
            #     print(i)
            #     break

            pad_distances[eY, eX] = 0
            edges_pixel_coords = np.delete(edges_pixel_coords, edge_bool, 0)

            # check whether the connecting vertex remains a vertex of not
            pad_sub_skeleton = np.pad(pad_skeleton[(Y - 2): (Y + 3), (X - 2): (X + 3)], [(1, ), (1, )], mode='constant')
            sub_vertices, sub_tips = get_vertices_and_tips_from_skeleton(pad_sub_skeleton)
            # If there is no vertex at the center, remove it
            if sub_vertices[3, 3] == 0:
                connecting_vertices_to_remove[i] = True
                vertex_to_remove = np.nonzero(np.logical_and(branching_vertices_coord[:, 0] == Y, branching_vertices_coord[:, 1] == X))[0]
                branches_to_remove[vertex_to_remove] = True
        else:
            remaining_tipped_edges_nb += 1
            new_edges_pixel_coords.append(np.stack((np.repeat(remaining_tipped_edges_nb, len(eY)),eY, eX), axis=1))

    # Remove vertices connecting a tip through very small edges
    tips_coord = np.delete(tips_coord, tipped_edges_to_remove, 0)
    vertices_branching_tips = np.delete(connecting_vertices_coord, tipped_edges_to_remove, 0)
    # Remove duplicates
    duplicates_vertices_branching_tips = find_duplicates_coord(vertices_branching_tips)
    vertices_branching_tips = np.delete(vertices_branching_tips, duplicates_vertices_branching_tips, 0)

    # Create arrays to store edges and vertices labels
    # Name edges and labels according to their orders
    branching_vertices_coord = np.delete(branching_vertices_coord, branches_to_remove, 0)
    # Find common coordinates between two arrays;
    v_branching_tips_in_branching_v = find_common_coord(branching_vertices_coord, connecting_vertices_coord)
    remaining_vertices = np.delete(branching_vertices_coord, v_branching_tips_in_branching_v, 0)

    ordered_vertices_coord = np.vstack((tips_coord, vertices_branching_tips, remaining_vertices))
    numbered_vertices = np.zeros(pad_skeleton.shape, dtype=np.uint32)
    numbered_vertices[ordered_vertices_coord[:, 0], ordered_vertices_coord[:, 1]] = np.arange(1, ordered_vertices_coord.shape[0] + 1)

    # Name edges from 1 to the number of edges connecting tips and set the vertices labels from all tips to their connected vertices:
    edges_labels = np.zeros((tips_coord.shape[0], 3), dtype=np.uint32)
    edges_labels[:, 0] = np.arange(tips_coord.shape[0]) + 1
    edges_labels[:, 1] = np.arange(tips_coord.shape[0]) + 1
    edges_labels[:, 2] = np.arange(tips_coord.shape[0], tips_coord.shape[0] * 2) + 1

    # update edges_pixel_coords
    edges_pixel_coords = np.vstack(new_edges_pixel_coords)

    #### TEST ####
    # un, counts = np.unique(edges_labels[:, 1:], return_counts=True)
    # counts.sum() == vertices_branching_tips.shape[0] + tips_coord.shape[0]
    # len(np.unique(numbered_vertices)) - 1 == ordered_vertices_coord.shape[0]
    #### TEST ####

    # # Use this vertex labeling to order their coordinates. Their order falls into 3 groups
    # # The first vertices are the tips, the second are the ones that are connected to the tipped edges
    # vertices_branching_tips = np.delete(connecting_vertices_coord, connecting_vertices_to_remove, 0)
    # 
    # kept_tips_and_branching_vertices_coord = np.vstack((tips_coord, vertices_branching_tips))
    # 
    #     
    # # # Update the branching_vertices_coord
    # # branching_vertices_coord = np.delete(branching_vertices_coord, branches_to_remove, 0)
    # # numbered_vertices = np.zeros(pad_skeleton.shape, dtype=np.uint32)
    # # for i, coord in enumerate(tips_and_connected_vertices_coord):
    # #     numbered_vertices[coord[0], coord[1]] = i + 1
    # # for j, coord in enumerate(branching_vertices_coord):
    # #     if not (coord == tips_and_connected_vertices_coord[:, None]).all(-1).any():
    # #         numbered_vertices[coord[0], coord[1]] = j + i + 2
    # 
    # 
    # # The third vertices are those that are nor tips nor connected to tipped edges
    # # To get them, update the branching_vertices_coord
    # branching_vertices_coord = np.delete(branching_vertices_coord, branches_to_remove, 0)
    # # Find out those that are neither tips neither connected to an edge that is connected to a tip
    # other_vertices = np.logical_not((branching_vertices_coord[:, None] == kept_tips_and_branching_vertices_coord).all(-1).any(-1))
    # # vertex_coord = np.zeros((tips_and_connected_vertices_coord.shape[0] + other_vertices.sum(), 3), dtype=np.uint32)
    # # # Name vertices from 1 to the total number of vertices
    # # vertex_coord[:, 0] = np.arange(vertex_coord.shape[0])
    # ordered_vertex_coord = np.vstack((kept_tips_and_branching_vertices_coord, branching_vertices_coord[other_vertices]))
    # numbered_vertices = np.zeros(pad_skeleton.shape, dtype=np.uint32)
    # for i, coord in enumerate(ordered_vertex_coord):
    #     numbered_vertices[coord[0], coord[1]] = i + 1

    # Update the connecting vertices coord to make sure that they are
    # connecting_vertices_coord = np.delete(connecting_vertices_coord, connecting_vertices_to_remove, 0)
    return pad_skeleton, pad_distances, edges_labels, numbered_vertices, edges_pixel_coords, tips_coord, branching_vertices_coord, vertices_branching_tips


def identify_other_edges(pad_skeleton, edges_labels, numbered_vertices, edges_pixel_coords, tips_coord, branching_vertices_coord, vertices_branching_tips):
    # First, create another version of these arrays, where we remove every already detected edge and vertices
    cropped_skeleton = pad_skeleton.copy()
    cropped_skeleton[edges_pixel_coords[:, 1], edges_pixel_coords[:, 2]] = 0
    cropped_skeleton[vertices_branching_tips[:, 0], vertices_branching_tips[:, 1]] = 1
    # cropped_skeleton[tips_coord[:, 0], tips_coord[:, 1]] = 0
    cropped_numbered_vertices = numbered_vertices.copy()
    cropped_numbered_vertices[tips_coord[:, 0], tips_coord[:, 1]] = 0
    cropped_numbered_vertices[tips_coord[:, 0], tips_coord[:, 1]] = 0

    # We will start from the vertices connecting tips
    connecting_vertices_coord = vertices_branching_tips.copy()

    # branching_vertices_coord does not need to be updated yet, because it only contains verified branching vertices
    cropped_branching_vertices_coord = branching_vertices_coord.copy()

    edges_nb = edges_labels.shape[0]
    # Clearly differentiate the loop that make sure that we explore all connexions of the current connecting_vertices_coord
    # And the loop that update connecting_vertices_coord until we explored all vertices.
    while np.any(cropped_branching_vertices_coord):
        explored_connexions_per_vertex = 0 # the maximal edge number that can connect a vertex
        new_level_vertices = None
        new_connexions = True
        while explored_connexions_per_vertex < 4 and new_connexions:
            explored_connexions_per_vertex += 1
            # 1. Find the ith closest vertex to each focal vertex
            new_connecting_vertices_coord, new_edge_lengths, new_edges_pixel_coords = find_closest_vertices(
                cropped_skeleton, cropped_branching_vertices_coord, connecting_vertices_coord)
            print(np.isnan(new_edge_lengths).sum())
            if np.isnan(new_edge_lengths).sum() == new_edge_lengths.shape[0]:
                new_connexions = False
            else:
                # ###
                # example = np.nonzero(np.isnan(new_edge_lengths))[0]
                # np.nonzero(new_edge_lengths == 0)
                # example = np.nonzero(new_edge_lengths == 1)[0]
                # example = [50]
                # new_edges_pixel_coords[new_edges_pixel_coords[:, 0] == example[0], :]
                # new_connecting_vertices_coord[example[0], :]
                # connecting_vertices_coord[example[0], :]
                # ###
                no_more_connexion = np.isnan(new_edge_lengths)
                lost_connexions = connecting_vertices_coord[no_more_connexion, :]
                vertex_to_vertex_connexions = new_edge_lengths == 1
                vertex_to_vertex_connexions = connecting_vertices_coord[vertex_to_vertex_connexions, :]

                remaining_connexions = np.logical_not(no_more_connexion)
                new_connecting_vertices_coord = new_connecting_vertices_coord[remaining_connexions, :]
                connecting_vertices_coord = connecting_vertices_coord[remaining_connexions, :]

                #     Y, X = cropped_branching_vertices_coord[0, 0], cropped_branching_vertices_coord[0, 1]
                #     cropped_skeleton[(Y - 2): (Y + 3), (X - 2): (X + 3)]



                if new_level_vertices is None:
                    new_level_vertices = new_connecting_vertices_coord.copy()
                else:
                    new_level_vertices = np.vstack((new_level_vertices, new_connecting_vertices_coord))
                # Add the new edges labels
                start = numbered_vertices[connecting_vertices_coord[:, 0], connecting_vertices_coord[:, 1]]
                end = numbered_vertices[new_connecting_vertices_coord[:, 0], new_connecting_vertices_coord[:, 1]]
                new_edges_nb = len(end)
                new_edges = np.zeros((new_edges_nb, 3), dtype=np.uint32)
                new_edges[:, 0] = np.arange(edges_nb, edges_nb + new_edges_nb)
                new_edges[:, 1] = start
                new_edges[:, 2] = end
                edges_labels = np.vstack((edges_labels, new_edges))

                # Add the new edge pixel coord
                new_edges_pixel_coords[:, 0] += edges_nb
                edges_pixel_coords = np.vstack((edges_pixel_coords, new_edges_pixel_coords))

                # Remove the edge connecting these two vertices
                cropped_skeleton[new_edges_pixel_coords[:, 1], new_edges_pixel_coords[:, 2]] = 0
                # Add the first vertices:
                fvY, fvX = connecting_vertices_coord[:, 0], connecting_vertices_coord[:, 1]
                cropped_skeleton[fvY, fvX] = 1

                # # Remove the branching that are not connected to anything else than an edge (not to another vertex)
                # # PROBLEM HERE : some are connected to vertices
                # first_vertices = np.zeros_like(cropped_skeleton)
                # # Add and dilate the first vertices:
                # first_vertices[fvY, fvX] = 1
                # first_vertices[fvY, fvX - 1] = 1
                # first_vertices[fvY, fvX + 1] = 1
                # first_vertices[fvY - 1, fvX] = 1
                # first_vertices[fvY + 1, fvX] = 1
                # first_vertices[fvY - 1, fvX - 1] = 1
                # first_vertices[fvY + 1, fvX + 1] = 1
                # first_vertices[fvY - 1, fvX + 1] = 1
                # first_vertices[fvY + 1, fvX - 1] = 1
                # # # not_connected_anymore[new_connecting_vertices_coord[:, 0], new_connecting_vertices_coord[:, 1]] = 1
                # # brY, brX = cropped_branching_vertices_coord[:, 0], cropped_branching_vertices_coord[:, 1]
                # # not_connected_anymore[brY, brX] = 1
                # # not_connected_anymore = cv2.dilate(not_connected_anymore, square_33)
                # first_vertices *= cropped_skeleton
                # _, first_vertices_conn_comp, stats, centroids = cv2.connectedComponentsWithStats(first_vertices)
                # not_connected_ids = np.nonzero(stats[:, 4] == 1)[0]
                # not_connected_ids = np.nonzero(np.isin(first_vertices_conn_comp, not_connected_ids))
                # cropped_numbered_vertices[not_connected_ids[0], not_connected_ids[1]] = 0

                # Faster way:
                cropped_numbered_vertices[lost_connexions[:, 0], lost_connexions[:, 1]] = 0
                cropped_numbered_vertices[vertex_to_vertex_connexions[:, 0], vertex_to_vertex_connexions[:, 1]] = 0
                # What happens to those that are connected to a vertex?
                cropped_branching_vertices_coord = np.transpose(np.array(np.nonzero(cropped_numbered_vertices)))

                # Update the edge number
                edges_nb += new_edges_nb
        # Update the connecting_vertices_coord array

        connecting_vertices_coord = np.unique(new_level_vertices, axis=0) # new_level_vertices
        # a[edges_pixel_coords[:, 1],edges_pixel_coords[:, 2]]=1
        #show(a)

    # 3. Find another closest vertex to each focal vertex
    # 4. Remove all edges connecting these two vertices
    # 5. Find -if existing- another closest vertex to each focal vertex
    # 6. Do 4. and 5. four times
    # 7. Remove all edges, use the new_connecting_vertices as the first vertices

    return edges_labels, edges_pixel_coords

def get_numbered_edges_and_vertices(pad_skeleton, pad_vertices, pad_tips):
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
    """
    Remaining problems:
    1. when 3 or more nodes contour one edge,
    find those that are close together and pick one (the closest to the edge)
    2. When nodes appear along the edge,
    split the edge into two parts

    :param vertices:
    :param edges:
    :return:
    """
    nb_e, tempo_numbered_edges, tempo_numbered_vertices = get_numbered_edges_and_vertices(pad_vertices, pad_edges)
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


def add_central_vertex(numbered_vertices, numbered_edges, vertices_table, edges_labels, origin, network_contour):
    """
    Make links between the center of the origin and the leaves touching it
    :return:
    """
    leaves_im = np.zeros_like(numbered_vertices)
    leaf_id = np.nonzero(vertices_table[:, 3] == 1)[0] + 1
    leaves_coord = np.nonzero(np.isin(numbered_vertices, leaf_id))
    leaves_im[leaves_coord[0], leaves_coord[1]] = numbered_vertices[leaves_coord[0], leaves_coord[1]]
    # dil_origin = cv2.dilate(origin, square_33)
    dil_origin = cv2.dilate(origin, Ellipse((5, 5)).create().astype(np.uint8), iterations=30)
    leaves_im *= dil_origin

    # Get center coord
    _, _, _, origin_centroid = cv2.connectedComponentsWithStats(origin)
    origin_centroid = np.round(origin_centroid[1, :]).astype(np.uint64)
    # add this new vertex to vertices_table, and numbered_vertices
    new_vertex_label = vertices_table.shape[0] + 1
    numbered_vertices[origin_centroid[0], origin_centroid[1]] = new_vertex_label
    new_vertex = np.array((new_vertex_label, origin_centroid[0], origin_centroid[1], 0), dtype=np.uint64)
    vertices_table = np.vstack((vertices_table, new_vertex))

    # Get skeleton pixels connecting center:
    connecting_pixels = vertices_table[np.unique(leaves_im)[1:] - 1, :3]
    # Same as: np.transpose(np.array(np.nonzero(leaves_im), dtype=np.uint64))
    # Draw lines between these and the center
    line_coordinates = get_all_line_coordinates(origin_centroid, connecting_pixels[:, 1:])
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

def save_vertices_and_tips_image(pad_skeleton, pad_vertices, pad_sure_terminations, pathway):
    plt.figure(figsize=(15, 15))
    valued_skeleton = pad_skeleton.copy()
    valued_skeleton *= 20
    valued_skeleton[np.nonzero(pad_vertices)] = 200
    valued_skeleton[np.nonzero(pad_sure_terminations)] = 40
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
    sure_terminations = np.zeros(pad_skeleton.shape, dtype=np.uint8)
    sure_terminations[cnv8.equal_neighbor_nb == 1] = 1

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
            is_not_term = 1 - sure_terminations[y1, x1]
            # is_not_term = (1 - sure_terminations[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)])
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
                        sure_terminations[Y, X] = 1
                    # Otherwise, it is a termination
                    else:
                        sure_terminations[y1, x1] = 1
    # Initiate the vertices final matrix as a copy of the sure_terminations
    pad_vertices = deepcopy(sure_terminations)
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
            is_not_term = 1 - sure_terminations[y3,x3]
            # is_not_term = np.logical_not(sure_terminations[(y3-1):(y3+2), (x3-1):(x3+2)])

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
                pad_vertices[np.logical_and(node_i > 0, sure_terminations == 0)] = 0
                dil_node_i = cv2.dilate(node_i, square_33, iterations=1)
                dil_node_i *= pad_skeleton
                bary = np.round(np.mean(np.array(np.nonzero(dil_node_i)), 1)).astype(np.uint64)
                # pad_vertices[node_i == 1] = 0
                pad_vertices[bary[0], bary[1]] = 1
                # pad_vertices[np.logical_and(node_i == 1, sure_terminations == 0)] = 0
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

