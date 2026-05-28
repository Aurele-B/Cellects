#!/usr/bin/env python3
"""
Graph extraction on a binary video.

This module uses GraphTracking class on binary images to extract and track graph in a video.

Classes
-------
GraphTracking
"""
import cv2
import logging
import numpy as np
from numpy.typing import NDArray
from cellects.image.morphological_operations import get_contours, keep_one_connected_component
from cellects.image.network_functions import EdgeIdentification, get_skeleton_and_widths, ad_pad, un_pad
from numba.typed import Dict as TDict
import pandas as pd

class GraphTracking:
    """Extract dynamic graph data from binary video frames.

    The class processes time‑series binary network masks to build evolving
    vertex and edge tables.  It aligns networks with an optional origin
    contour, extracts skeletons, computes geometric and intensity attributes,
    and stores the results in CSV files whose names encode the arena label
    and video dimensions.

    Attributes
    ----------
    starting_time : int
        Frame index at which tracking starts (initially ``0``).
    coord_network : NDArray
        Binary mask of the network for each time point (shape ``t × y × x``).
    converted_video : NDArray
        Original video frames used to calculate edge intensity attributes
        (shape ``t × y × x``).
    coord_pseudopods : NDArray or None
        Optional mask of pseudopod regions; used to label growing areas.
    origin : NDArray or None
        Binary mask of the region of interest’s central origin.
    arena_label : int
        Identifier that prefixes all output file names.
    dims : tuple of int
        Dimensions of the video (``t, y, x``).
    pad_origin_centroid : np.ndarray or None
        Centroid of ``origin`` after padding (or ``None`` if no origin).
    pad_origin : np.ndarray or None
        Padded version of ``origin`` (or ``None`` if no origin).
    origin_contours : list or None
        Contours extracted from ``origin`` (or ``None`` if no origin).
    vertex_table : np.ndarray or None
        Accumulated vertex data; populated after tracking.
    edge_table : np.ndarray or None
        Accumulated edge data; populated after tracking.
    edge_pix_coord : np.ndarray or None
        Pixel coordinates belonging to each edge; populated after tracking.
    """
    def __init__(self, converted_video: NDArray, coord_network: NDArray, arena_label: int,
                           origin: NDArray[np.uint8]=None, coord_pseudopods: NDArray=None):
        """
        Initialize the arena graph extraction object.

        Parameters
        ----------
        converted_video : NDArray
            Video data that has been pre‑processed and converted to a suitable
            numeric format. Its shape is expected to contain at least three
            dimensions, where the first three are used to compute ``dims``.
        coord_network : NDArray
            Array describing the connectivity network of the arena; used later
            for graph construction.
        arena_label : int
            Identifier of the arena instance. Used for logging and tracking.
        origin : NDArray[np.uint8], optional
            Binary mask of the arena origin. When provided the centroid of the
            origin is computed, padded, and its contours are extracted. If
            ``None`` the related origin attributes remain ``None``.
        coord_pseudopods : NDArray, optional
            Optional coordinate data for pseudopod structures. Stored directly
            without further processing.

        Attributes
        ----------
        starting_time : int
            Timestamp (in whatever units the caller uses) marking the start of
            processing; initialized to ``0``.
        coord_network : NDArray
            Reference to the ``coord_network`` argument.
        converted_video : NDArray
            Reference to the ``converted_video`` argument.
        coord_pseudopods : NDArray or None
            Reference to the ``coord_pseudopods`` argument.
        origin : NDArray[np.uint8] or None
            Reference to the ``origin`` argument.
        arena_label : int
            Reference to the ``arena_label`` argument.
        dims : tuple of int
            Spatial dimensions of ``converted_video`` (the first three axes).
        pad_origin_centroid : tuple of int or None
            Centroid of ``origin`` (rounded and padded) when ``origin`` is
            provided; otherwise ``None``.
        origin_contours : list or None
            Contours extracted from ``origin`` when available; otherwise ``None``.
        pad_origin : NDArray or None
            Padded version of ``origin`` when provided; otherwise ``None``.
        vertex_table : None
            Placeholder for the vertex table populated after graph extraction.
        edge_table : None
            Placeholder for the edge table populated after graph extraction.

        """
        self.starting_time = 0
        self.coord_network = coord_network
        self.converted_video = converted_video
        self.coord_pseudopods = coord_pseudopods
        self.origin = origin
        self.arena_label = arena_label
        self.dims = converted_video.shape[:3]
        if self.origin is not None:
            _, _, _, origin_centroid = cv2.connectedComponentsWithStats(self.origin)
            origin_centroid = np.round((origin_centroid[1, 1], origin_centroid[1, 0])).astype(np.int64)
            self.pad_origin_centroid = origin_centroid + 1
            self.origin_contours = get_contours(self.origin)
            self.pad_origin = ad_pad(self.origin)
        else:
            self.pad_origin_centroid = None
            self.pad_origin = None
            self.origin_contours = None
        self.vertex_table = None
        self.edge_table = None
        logging.info(f"Arena n°{arena_label}. Starting graph extraction.")


    def frame_by_frame_tracking(self):
        """
        Perform frame‑by‑frame tracking of the network.

        When ``self.coord_network`` has no columns the method creates empty
        tables for vertices, edges and pixel coordinates.  Otherwise it iterates
        over the time dimension starting at ``self.starting_time`` and extracts
        the graph for each frame by calling :meth:`extract_graph`.

        Returns
        -------
        None
            The method updates the instance attributes in‑place and does not
            return a value.
        """
        if self.coord_network.shape[1] == 0:
            self.vertex_table = np.empty((0, 7))
            self.edge_table = np.empty((0, 8))
            self.edge_pix_coord = np.empty((0, 4))
        else:
            for t in np.arange(self.starting_time, self.dims[0]):
                computed_network = self.extract_graph(t)

    def extract_graph(self, t: int) -> NDArray[np.uint8]:
        """
        Compute the graph representation for a given time point ``t``.

        Parameters
        ----------
        t
            Index of the time frame to process.

        Returns
        -------
        binary_image
            A ``uint8`` array containing the skeleton (value ``1``) of a
            network.  The array has the original image dimensions ``(height, width)``.

        Notes
        -----
        * The network for time ``t`` is extracted from ``self.coord_network``.
        * If ``self.origin`` is defined, the origin mask contour is combined with the
          computed network.
        * Only the largest connected component is retained via
          ``keep_one_connected_component``.
        * Padding is added before skeletonisation and removed before the
          result is returned.
        * Vertex and edge tables are accumulated in the instance attributes
          ``vertex_table``, ``edge_table`` and ``edge_pix_coord``.
        * When ``self.coord_pseudopods`` is provided, its coordinates are used
          as growing areas during vertex identification.
        """
        computed_network = np.zeros((self.dims[1], self.dims[2]), dtype=np.uint8)
        net_t = self.coord_network[1:, self.coord_network[0, :] == t]
        computed_network[net_t[0], net_t[1]] = 1
        if self.origin is not None:
            computed_network = computed_network * (1 - self.origin)
            computed_network = np.logical_or(self.origin_contours, computed_network).astype(np.uint8)
        else:
            computed_network = computed_network.astype(np.uint8)
        if computed_network.any():
            computed_network = keep_one_connected_component(computed_network)
            pad_network = ad_pad(computed_network)
            pad_skeleton, pad_distances, pad_origin_contours = get_skeleton_and_widths(pad_network, self.pad_origin,
                                                                                       self.pad_origin_centroid)
            edge_id = EdgeIdentification(pad_skeleton, pad_distances, t)
            edge_id.run_edge_identification()
            if self.origin is not None:
                self.origin_contours = un_pad(pad_origin_contours)
            growing_areas = None
            if self.coord_pseudopods is not None:
                growing_areas = self.coord_pseudopods[1:, self.coord_pseudopods[0, :] == t]
            edge_id.make_vertex_table(self.origin_contours, growing_areas)
            edge_id.make_edge_table(self.converted_video[t, ...])
            pad_skeleton[edge_id.vertex_table[:, 0], edge_id.vertex_table[:, 1]] = 2

            edge_id.edge_pix_coord = np.hstack(
                (np.repeat(t, edge_id.edge_pix_coord.shape[0])[:, None], edge_id.edge_pix_coord))
            edge_id.vertex_table = np.hstack(
                (np.repeat(t, edge_id.vertex_table.shape[0])[:, None], edge_id.vertex_table))
            edge_id.edge_table = np.hstack(
                (np.repeat(t, edge_id.edge_table.shape[0])[:, None], edge_id.edge_table))
            if self.vertex_table is None:
                self.vertex_table = edge_id.vertex_table.copy()
                self.edge_table = edge_id.edge_table.copy()
                self.edge_pix_coord = edge_id.edge_pix_coord.copy()
            else:
                self.vertex_table = np.vstack((self.vertex_table, edge_id.vertex_table))
                self.edge_table = np.vstack((self.edge_table, edge_id.edge_table))
                self.edge_pix_coord = np.vstack((self.edge_pix_coord, edge_id.edge_pix_coord))
        return un_pad(pad_skeleton)
    
    def save_graph(self):
        """
        Save the graph data stored in the instance to CSV files.

        Parameters
        ----------
        self
            The instance containing the graph attributes.  It must provide the
            following attributes:

            - ``vertex_table``: iterable of vertex records.
            - ``edge_table``: iterable of edge records.
            - ``edge_pix_coord``: iterable of edge‑pixel coordinate records.
            - ``arena_label``: label used in the output filenames.
            - ``dims``: video dimensions (t, y, x) saved in the filenames to ease video reconstruction.

        Returns
        -------
        ``None``
            The method writes three CSV files and does not return a value.

        Notes
        -----
        The method converts the internal containers to :class:`pandas.DataFrame`
        objects with fixed column orders and then writes them to CSV files whose
        names embed ``arena_label`` and the three video dimensions from ``dims``:

        - ``vertices_coord{arena_label}_t{t}_y{y}_x{x}.csv``
        - ``edges_to_vertices{arena_label}_t{t}_y{y}_x{x}.csv``
        - ``edges_coord{arena_label}_t{t}_y{y}_x{x}.csv``
        """
        self.vertex_table = pd.DataFrame(self.vertex_table, columns=["t", "y", "x", "vertex_id", "is_tip", "origin",
                                                           "vertex_connected"])
        self.edge_table = pd.DataFrame(self.edge_table,
                                  columns=["t", "edge_id", "vertex1", "vertex2", "length", "average_width", "intensity",
                                           "betweenness_centrality"])
        self.edge_pix_coord = pd.DataFrame(self.edge_pix_coord, columns=["t", "y", "x", "edge_id"])
        self.vertex_table.to_csv(
            f"vertices_coord{self.arena_label}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.csv",
            index=False)
        self.edge_table.to_csv(
            f"edges_to_vertices{self.arena_label}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.csv",
            index=False)
        self.edge_pix_coord.to_csv(
            f"edges_coord{self.arena_label}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.csv",
            index=False)
