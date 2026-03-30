#!/usr/bin/env python3
"""
Graph extraction on a binary video.

This module uses network functions on binary images to extract and track graph in a video.

Functions
---------
extract_graph_dynamics
"""
import cv2
import logging
import numpy as np
from numpy.typing import NDArray
from cellects.image.morphological_operations import get_contours, keep_one_connected_component
from cellects.image.network_functions import EdgeIdentification, get_skeleton_and_widths, add_padding, remove_padding
from numba.typed import Dict as TDict
import pandas as pd


def extract_graph_dynamics(converted_video: NDArray, coord_network: NDArray, arena_label: int,
                            starting_time: int=0, origin: NDArray[np.uint8]=None, coord_pseudopods: NDArray=None):
    """
    Extracts dynamic graph data from video frames based on network dynamics.

    This function processes time-series binary network structures to extract evolving vertices and edges over time. It computes spatial relationships between networks and an origin point through image processing steps including contour detection, padding for alignment, skeleton extraction, and morphological analysis. Vertex and edge attributes like position, connectivity, width, intensity, and betweenness are compiled into tables saved as CSV files.

    Parameters
    ----------
    converted_video : NDArray
        3D video data array (t x y x) containing pixel intensities used for calculating edge intensity attributes during table generation.
    coord_network : NDArray[np.uint8]
        3D binary network mask array (t x y x) representing connectivity structures across time points.
    arena_label : int
        Unique identifier to prefix output filenames corresponding to specific experimental arenas.
    starting_time : int, optional (default=0)
        Time index within `coord_network` to begin processing from (exclusive of origin initialization).
    origin : NDArray[np.uint8], optional (default=None)
        Binary mask identifying the region of interest's central origin for spatial reference during network comparison.

    Returns
    -------
    None
    Saves two CSV files in working directory:
    1. `vertex_table{arena_label}_t{T}_y{Y}_x{X}.csv` - Vertex table with time, coordinates, and connectivity information
    2. `edge_table{arena_label}_t{T}_y{Y}_x{X}.csv` - Edge table containing attributes like length, width, intensity, and betweenness

    Notes
    ---
    Output CSVs use NumPy arrays converted to pandas DataFrames with columns:
    - Vertex table includes timestamps (t), coordinates (y,x), and connectivity flags.
    - Edge table contains betweenness centrality calculated during skeleton processing.
    Origin contours are spatially aligned through padding operations to maintain coordinate consistency across time points.
    """
    dims = converted_video.shape[:3]
    if coord_network.shape[1] == 0:
        vertex_table = np.empty((0, 7))
        edge_table = np.empty((0, 8))
    else:
        logging.info(f"Arena n°{arena_label}. Starting graph extraction.")
        # converted_video = self.converted_video; coord_network=self.coord_network; arena_label=1; starting_time=0; origin=self.origin
        if origin is not None:
            _, _, _, origin_centroid = cv2.connectedComponentsWithStats(origin)
            origin_centroid = np.round((origin_centroid[1, 1], origin_centroid[1, 0])).astype(np.int64)
            pad_origin_centroid = origin_centroid + 1
            origin_contours = get_contours(origin)
            pad_origin = add_padding([origin])[0]
        else:
            pad_origin_centroid = None
            pad_origin = None
            origin_contours = None
        vertex_table = None
        edge_table = None
        for t in np.arange(starting_time, dims[0]): # t=320   Y, X = 729, 554
            computed_network = np.zeros((dims[1], dims[2]), dtype=np.uint8)
            net_t = coord_network[1:, coord_network[0, :] == t]
            computed_network[net_t[0], net_t[1]] = 1
            if origin is not None:
                computed_network = computed_network * (1 - origin)
                computed_network = np.logical_or(origin_contours, computed_network).astype(np.uint8)
            else:
                computed_network = computed_network.astype(np.uint8)
            if computed_network.any():
                computed_network = keep_one_connected_component(computed_network)
                pad_network = add_padding([computed_network])[0]
                pad_skeleton, pad_distances, pad_origin_contours = get_skeleton_and_widths(pad_network, pad_origin,
                                                                                               pad_origin_centroid)
                edge_id = EdgeIdentification(pad_skeleton, pad_distances, t)
                edge_id.run_edge_identification()
                if pad_origin_contours is not None:
                    origin_contours = remove_padding([pad_origin_contours])[0]
                growing_areas = None
                if coord_pseudopods is not None:
                    growing_areas = coord_pseudopods[1:, coord_pseudopods[0, :] == t]
                edge_id.make_vertex_table(origin_contours, growing_areas)
                edge_id.make_edge_table(converted_video[t, ...])
                edge_id.vertex_table = np.hstack((np.repeat(t, edge_id.vertex_table.shape[0])[:, None], edge_id.vertex_table))
                edge_id.edge_table = np.hstack((np.repeat(t, edge_id.edge_table.shape[0])[:, None], edge_id.edge_table))
                if vertex_table is None:
                    vertex_table = edge_id.vertex_table.copy()
                    edge_table = edge_id.edge_table.copy()
                else:
                    vertex_table = np.vstack((vertex_table, edge_id.vertex_table))
                    edge_table = np.vstack((edge_table, edge_id.edge_table))

    vertex_table = pd.DataFrame(vertex_table, columns=["t", "y", "x", "vertex_id", "is_tip", "origin",
                                                       "vertex_connected"])
    edge_table = pd.DataFrame(edge_table,
                              columns=["t", "edge_id", "vertex1", "vertex2", "length", "average_width", "intensity",
                                       "betweenness_centrality"])
    vertex_table.to_csv(
        f"vertex_table{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.csv")
    edge_table.to_csv(
        f"edge_table{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.csv")
