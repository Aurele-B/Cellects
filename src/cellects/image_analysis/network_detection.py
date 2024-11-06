#!/usr/bin/env python3
"""
This script contains the class for detecting networks out of a grayscale image of Physarum polycephalum
"""

import cv2
import numpy as np
from numpy import nonzero, zeros_like, uint8, min, max, ptp
from cv2 import connectedComponents, CV_16U
from skimage import morphology
from scipy import ndimage
from collections import defaultdict
from cellects.image_analysis.image_segmentation import get_otsu_threshold
from cellects.image_analysis.progressively_add_distant_shapes import ProgressivelyAddDistantShapes
from cellects.image_analysis.shape_descriptors import ShapeDescriptors
from cellects.image_analysis.morphological_operations import cross_33, cc, CompareNeighborsWithValue


class NetworkDetection:
    def __init__(self, grayscale_image, binary_image, lighter_background):
        """
        :param grayscale_image: current grayscale image to analyze
        :type grayscale_image: uint8
        :param binary_image: current binary image to analyze
        :type binary_image: uint8
        :param lighter_background: True if the background of the image is lighter than the shape to detect
        :type lighter_background: bool
        """
        self.grayscale_image = grayscale_image
        self.binary_image = binary_image
        self.lighter_background = lighter_background

    def segment_locally(self, side_length=8, step=2, int_variation_thresh=20):
        """
        Segment small squares of the images to detect local intensity valleys
        This method segment the network locally using otsu thresholding on a rolling window
        :param side_length: The size of the window to detect the tubes of the network
        :type side_length: uint8
        :param step:
        :type step: uint8
        :return:
        """
        self.side_length = side_length
        self.step = step
        y, x = nonzero(self.binary_image)
        self.min_y = min(y)
        if (self.min_y - 20) >= 0:
            self.min_y -= 20
        else:
            self.min_y = 0
        self.max_y = max(y)
        if (self.max_y + 20) < self.binary_image.shape[0]:
            self.max_y += 20
        else:
            self.max_y = self.binary_image.shape[0] - 1
        self.min_x = min(x)
        if (self.min_x - 20) >= 0:
            self.min_x -= 20
        else:
            self.min_x = 0
        self.max_x = max(x)
        if (self.max_x + 20) < self.binary_image.shape[1]:
            self.max_x += 20
        else:
            self.max_x = self.binary_image.shape[1] - 1

        # y_windows = np.arange(0, self.max_y - self.min_y + 1, side_length)
        # x_windows = np.arange(0, self.max_x - self.min_x + 1, side_length)
        y_size = self.max_y - self.min_y + 1
        x_size = self.max_x - self.min_x + 1
        network = np.zeros((self.max_y - self.min_y, self.max_x - self.min_x), np.uint64)
        self.homogeneities = np.zeros((self.max_y - self.min_y, self.max_x - self.min_x), np.uint64)
        cropped_binary_image = self.binary_image[self.min_y:self.max_y, self.min_x:self.max_x]
        cropped_grayscale_image = self.grayscale_image[self.min_y:self.max_y, self.min_x:self.max_x]
        # will be more efficient if it only loops over a zoom on self.binary_image == 1
        for to_add in np.arange(0, side_length, step):
            y_windows = np.arange(0, y_size, side_length)
            x_windows = np.arange(0, x_size, side_length)
            y_windows += to_add
            x_windows += to_add
            for y_start in y_windows:
                # y_start = 4
                if y_start < network.shape[0]:
                    y_end = y_start + side_length
                    if y_end < network.shape[0]:
                        for x_start in x_windows:
                            if x_start < network.shape[1]:
                                x_end = x_start + side_length
                                if x_end < network.shape[1]:
                                    if np.any(cropped_binary_image[y_start:y_end, x_start:x_end]):
                                        potential_network = cropped_grayscale_image[y_start:y_end, x_start:x_end]
                                        if ptp(potential_network[nonzero(potential_network)]) < int_variation_thresh:
                                            self.homogeneities[y_start:y_end, x_start:x_end] += 1
                                        threshold = get_otsu_threshold(potential_network)
                                        if self.lighter_background:
                                            net_coord = np.nonzero(potential_network < threshold)
                                        else:
                                            net_coord = np.nonzero(potential_network > threshold)
                                        network[y_start + net_coord[0], x_start + net_coord[1]] += 1

        self.network = np.zeros(self.binary_image.shape, np.uint8)
        self.network[self.min_y:self.max_y, self.min_x:self.max_x] = (network >= (side_length // step)).astype(np.uint8)
        self.network[self.min_y:self.max_y, self.min_x:self.max_x][self.homogeneities >= (((self.side_length // self.step) // 2) + 1)] = 0

    def segment_globally(self):
        """
        ABANDONED METHOD: This method remove pixels whose intensity is too close from the average
        :return:
        """
        grayscale_network = self.grayscale_image * self.network
        average = grayscale_network[grayscale_network != 0].mean()

        nb, components = cv2.connectedComponents(self.network)
        network = np.zeros_like(self.network)
        for compo_i in np.arange(1, nb):
            coord = np.nonzero(components == compo_i)
            if self.lighter_background:
                if np.mean(self.grayscale_image[coord]) < 0.8 * average:
                    network[coord] = 1
            else:
                if np.mean(self.grayscale_image[coord]) > 1.25 * average:
                    network[coord] = 1
        self.network = network

    def selects_elongated_or_holed_shapes(self, hole_surface_ratio=0.1, eccentricity=0.65):
        """
        Remove shapes that are very circular and does not contain holes:
        This method only keep the elongates or holed shapes.
        An elongated shape as a strong eccentricity
        A holed shape contain a holes of a decent size
        :param hole_surface_ratio: ratio threshold for hole selection
        :type hole_surface_ratio: float64
        :param eccentricity: eccentricity threshold for elongation selection
        :type eccentricity: float64
        :return:
        """
        potential_network = cv2.morphologyEx(self.network, cv2.MORPH_OPEN, cross_33)
        potential_network = cv2.morphologyEx(potential_network, cv2.MORPH_CLOSE, cross_33)

        # Filter out component that does not have the shape of the part of a network
        nb, components = cv2.connectedComponents(potential_network)
        self.network = np.zeros_like(self.network)
        for compo_i in np.arange(1, nb):
            #compo_i = 1
            #compo_i += 1
            matrix_i = components == compo_i
            # Accept as part of the network, the shapes that contains large holes and, if they don't,
            # that are eccentric enough
            SD = ShapeDescriptors(matrix_i, ['area', 'total_hole_area', 'eccentricity'])
            add_to_the_network: bool = False
            if SD.descriptors['total_hole_area'] > hole_surface_ratio * SD.descriptors['area']:
                add_to_the_network = True
            else:
                if SD.descriptors['eccentricity'] > eccentricity:
                    add_to_the_network = True
            if add_to_the_network:
                matrix_i = np.nonzero(matrix_i)
                self.network[matrix_i[0], matrix_i[1]] = 1

    def add_pixels_to_the_network(self, pixels_to_add):
        """
        Make sure that the previous network is still in
        When needed, consider all area occupied by the origin as part of the network
        :return:
        """
        self.network[pixels_to_add] = 1

    def remove_pixels_from_the_network(self, pixels_to_remove, remove_homogeneities=True):
        """
        When needed, consider all area occupied by the origin as part of the network
        :return:
        """
        self.network[pixels_to_remove] = 0
        if remove_homogeneities:
            self.network[self.min_y:self.max_y, self.min_x:self.max_x][self.homogeneities >= (((self.side_length // self.step) // 2) + 1)] = 0

    def connect_network(self, maximal_distance_connection=50):
        """
        Force all component of the network to be connected
        :param maximal_distance_connection:
        :type maximal_distance_connection: uint64
        :return:
        """
        # Force connection of the disconnected parts of the network
        if np.any(self.network):
            ordered_image, stats, centers = cc(self.network)
            # if self.lighter_background:
            #     intensity_valley = 255 - self.grayscale_image.copy()
            # else:
            #     intensity_valley = self.grayscale_image.copy()
            # new
            pads = ProgressivelyAddDistantShapes(self.network, ordered_image==1, maximal_distance_connection)
            # If max_distance is non nul look for distant shapes
            # pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False, intensity_valley=intensity_valley)
            pads.connect_shapes(only_keep_connected_shapes=True, rank_connecting_pixels=False)
            potential_network = np.nonzero(pads.expanded_shape)
            self.network[potential_network[0], potential_network[1]] = 1

            # Remove all disconnected components remaining
            ordered_image, stats, centers = cc(self.network)
            self.network[ordered_image != 1] = 0


    #### Houssam ####
    def skeletonize(self):
        """
        Skeletonize the network
        :return:
        """
        self.skeleton = morphology.skeletonize(self.network)


    def detect_nodes(self):
        """
        Function written By Houssam Henni, adapted for Cellects by Aurèle Boussard
        :return:
        """
        # Nous calculons le nombre de voisins pour chaque pixel du squelette
        cnv = CompareNeighborsWithValue(self.skeleton, 8)
        cnv.is_equal(1, and_itself=True)
        neighbor_counts = cnv.equal_neighbor_nb

        # Identification des nœuds : pixels avec 1 ou plus de 2 voisins
        nodes = ((neighbor_counts == 1) | (neighbor_counts > 2)) & self.skeleton

        # Étiquetage des nœuds
        labeled_nodes, num_labels = ndimage.label(nodes, structure=np.ones((3, 3), dtype=np.uint8))

        # Positions des nœuds
        node_positions = ndimage.center_of_mass(nodes, labeled_nodes, range(1, num_labels + 1))
        # Conversion des positions en entiers
        node_positions = [tuple(map(int, pos)) for pos in node_positions]
        # Création d'un mapping label -> position
        label_to_position = {label: pos for label, pos in zip(range(1, num_labels + 1), node_positions)}

        return labeled_nodes, label_to_position


    def find_segments(self, labeled_nodes, label_to_position):
        """
        Function written By Houssam Henni, adapted for Cellects by Aurèle Boussard
        :return:
        """
        # Suppression des nœuds du squelette
        skeleton_wo_nodes = self.skeleton.copy()
        skeleton_wo_nodes[labeled_nodes > 0] = 0

        # Détection des segments (composantes connectées sans les nœuds)
        num_labels, labels = cv2.connectedComponents(skeleton_wo_nodes.astype(np.uint8))
        segments = []

        for label in range(1, num_labels):
            segment_mask = (labels == label)
            coords = np.column_stack(np.where(segment_mask))

            # Dilatation du segment pour trouver les nœuds adjacents
            dilated_segment = morphology.binary_dilation(segment_mask, morphology.disk(2))
            overlapping_nodes = labeled_nodes * dilated_segment
            node_labels = np.unique(overlapping_nodes[overlapping_nodes > 0])

            if len(node_labels) >= 2:
                # Si au moins deux nœuds sont connectés, on trouve les deux plus éloignés
                node_positions = [label_to_position[n_label] for n_label in node_labels]
                distances = np.sum((np.array(node_positions)[:, None] - np.array(node_positions)[None, :]) ** 2, axis=2)
                idx_max = np.unravel_index(np.argmax(distances), distances.shape)
                start_label = node_labels[idx_max[0]]
                end_label = node_labels[idx_max[1]]
                start_pos = label_to_position[start_label]
                end_pos = label_to_position[end_label]
                segments.append({
                    'start_label': start_label,
                    'end_label': end_label,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'coords': coords
                })
            elif len(node_labels) == 1:
                # Si un seul nœud est connecté, on trouve l'extrémité la plus éloignée
                start_label = node_labels[0]
                start_pos = label_to_position[start_label]
                distances = np.sum((coords - np.array(start_pos)) ** 2, axis=1)
                idx_max = np.argmax(distances)
                end_pos = tuple(coords[idx_max])
                segments.append({
                    'start_label': start_label,
                    'end_label': None,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'coords': coords
                })
        return segments

    def extract_node_degrees(self, segments):
        """
        Function written By Houssam Henni, adapted for Cellects by Aurèle Boussard
        :return:
        """
        node_degrees = defaultdict(int)
        for segment in segments:
            start_label = segment['start_label']
            end_label = segment['end_label']
            node_degrees[start_label] += 1
            if end_label is not None:
                node_degrees[end_label] += 1
        return node_degrees


    def get_segment_width(self, binary_image, segment, distance_map):
        """
        Function written By Houssam Henni, adapted for Cellects by Aurèle Boussard
        :return:
        """
        coords = segment['coords']
        distances = []
        for i in range(1, len(coords) - 1):
            y, x = coords[i]
            if distance_map[y, x] == 0:
                continue
            dy = coords[i + 1][0] - coords[i - 1][0]
            dx = coords[i + 1][1] - coords[i - 1][1]
            norm = np.hypot(dx, dy)
            if norm == 0:
                continue
            perp_dx = -dy / norm
            perp_dy = dx / norm
            length = distance_map[y, x] * 2
            if length < 1:
                continue
            r0 = y - perp_dy * length / 2
            c0 = x - perp_dx * length / 2
            r1 = y + perp_dy * length / 2
            c1 = x + perp_dx * length / 2
            line_length = int(np.hypot(r1 - r0, c1 - c0))
            if line_length == 0:
                continue
            line_coords = np.linspace(0, 1, line_length)
            rr = ((1 - line_coords) * r0 + line_coords * r1).astype(int)
            cc = ((1 - line_coords) * c0 + line_coords * c1).astype(int)
            valid_idx = (rr >= 0) & (rr < binary_image.shape[0]) & (cc >= 0) & (cc < binary_image.shape[1])
            rr = rr[valid_idx]
            cc = cc[valid_idx]
            if not np.all(binary_image[rr, cc]):
                continue
            width = len(rr)
            distances.append(width)
        if distances:
            largeurs = {
                'largeur_moyenne': np.mean(distances),
                'largeur_noeud_A': distances[0],
                'largeur_noeud_B': distances[-1],
                'largeur_milieu': distances[len(distances) // 2],
                'largeur_minimale': np.min(distances),
                'largeur_maximale': np.max(distances)
            }
            return largeurs
        else:
            return None

        #### Houssam ####