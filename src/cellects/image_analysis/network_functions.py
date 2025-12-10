#!/usr/bin/env python3
"""
Network detection and skeleton analysis for biological networks (such as Physarum polycephalum's) images.

This module provides tools for analyzing network structures in grayscale images of biological networks.
It implements vessel detection using Frangi/Sato filters, thresholding methods, and quality metrics to select optimal
network representations. Additional functionality includes pseudopod detection, skeletonization, loop removal,
edge identification, and network topology analysis through vertex/edge tracking.

Classes
-------
NetworkDetection : Detects vessels in images using multi-scale filters with parameter variations.
EdgeIdentification : Identifies edges between vertices in a skeletonized network structure.

Functions
---------
get_skeleton_and_widths: Computes medial axis skeleton and distance transforms for networks.
remove_small_loops: Eliminates small loops from skeletons while preserving topology.
get_neighbor_comparisons: Analyzes pixel connectivity patterns in skeletons.
get_vertices_and_tips_from_skeleton: Identifies junctions and endpoints in network skeletons.
merge_network_with_pseudopods: Combines detected network structures with identified pseudopods.

Notes
-----
Uses morphological operations for network refinement, including hole closing, component labeling,
and distance transform analysis. Implements both Otsu thresholding and rolling window segmentation
methods for image processing workflows.
"""
from cellects.image_analysis.morphological_operations import square_33, cross_33, rhombus_55, create_ellipse, image_borders, CompareNeighborsWithValue, get_contours, get_all_line_coordinates, close_holes, keep_one_connected_component, get_min_or_max_euclidean_pair
from cellects.utils.utilitarian import remove_coordinates, smallest_memory_array
from cellects.utils.formulas import *
from cellects.utils.load_display_save import *
from cellects.image_analysis.image_segmentation import generate_color_space_combination, rolling_window_segmentation, binary_quality_index, find_threshold_given_mask
from numba.typed import Dict as TDict
from skimage import morphology
from skimage.filters import frangi, sato, threshold_otsu
from collections import deque
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
import networkx as nx
import pandas as pd
from timeit import default_timer as timer

# 8-connectivity neighbors
neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]
neighbors_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]

def detect_network_dynamics(converted_video: NDArray, binary: NDArray[np.uint8], arena_label: int=1,
                            starting_time: int=0, visu: NDArray=None, origin: NDArray[np.uint8]=None,
                            smooth_segmentation_over_time: bool = True, detect_pseudopods: bool = True,
                            save_coord_network: bool = True, show_seg: bool = False):
    """
    Detects and tracks dynamic features (e.g., pseudopods) in a biological network over time from video data.

    Analyzes spatiotemporal dynamics of a network structure using binary masks and grayscale video data. Processes each frame to detect network components, optionally identifies pseudopods, applies temporal smoothing, and generates visualization overlays. Saves coordinate data for detected networks if enabled.

    Parameters
    ----------
    converted_video : NDArray
        Input video data array with shape (time x y x z) representing grayscale intensities.
    binary : NDArray[np.uint8]
        Binary mask array with shape (time x y x z) indicating filled regions in each frame.
    arena_label : int
        Unique identifier for the current processing arena/session to name saved output files.
    starting_time : int
        Zero-based index of the first frame to begin network detection and analysis from.
    visu : NDArray
        Visualization video array (time x y x z) with RGB channels for overlay rendering.
    origin : NDArray[np.uint8]
        Binary mask defining a central region of interest to exclude from network detection.
    smooth_segmentation_over_time : bool, optional (default=True)
        Flag indicating whether to apply temporal smoothing using adjacent frame data.
    detect_pseudopods : bool, optional (default=True)
        Determines if pseudopod regions should be detected and merged with the network.
    save_coord_network : bool, optional (default=True)
        Controls saving of detected network/pseudopod coordinates as NumPy arrays.
    show_seg : bool, optional (default=False)
        Enables real-time visualization display during processing.

    Returns
    -------
    NDArray[np.uint8]
        3D array containing detected network structures with shape (time x y x z). Uses:
        - `0` for background,
        - `1` for regular network components,
        - `2` for pseudopod regions when detect_pseudopods is True.

    Notes
    -----
    - Memory-intensive operations on large arrays may require system resources.
    - Temporal smoothing effectiveness depends on network dynamics consistency between frames.
    - Pseudopod detection requires sufficient contrast with the background in grayscale images.
    """
    logging.info(f"Arena n°{arena_label}. Starting network detection.")
    # converted_video = self.converted_video; binary=self.binary; arena_label=1; starting_time=0; visu=self.visu; origin=None; smooth_segmentation_over_time=True; detect_pseudopods=True;save_coord_network=True; show_seg=False
    dims = binary.shape
    pseudopod_min_size = 50
    if detect_pseudopods:
        pseudopod_vid = np.zeros_like(binary, dtype=bool)
    potential_network = np.zeros_like(binary, dtype=bool)
    network_dynamics = np.zeros_like(binary, dtype=np.uint8)
    do_convert = True
    if visu is None:
        do_convert = False
        visu = np.stack((converted_video, converted_video, converted_video), axis=3)
        greyscale = converted_video[-1, ...]
    else:
        greyscale = visu[-1, ...].mean(axis=-1)
    NetDet = NetworkDetection(greyscale, possibly_filled_pixels=binary[-1, ...],
                              origin_to_add=origin)
    NetDet.get_best_network_detection_method()
    if do_convert:
        NetDet.greyscale_image = converted_video[-1, ...]
    lighter_background = NetDet.greyscale_image[binary[-1, ...] > 0].mean() < NetDet.greyscale_image[
        binary[-1, ...] == 0].mean()

    for t in np.arange(starting_time, dims[0]):  # 20):#
        if do_convert:
            greyscale = visu[t, ...].mean(axis=-1)
        else:
            greyscale = converted_video[t, ...]
        NetDet_fast = NetworkDetection(greyscale, possibly_filled_pixels=binary[t, ...],
                                       origin_to_add=origin, best_result=NetDet.best_result)
        NetDet_fast.detect_network()
        NetDet_fast.greyscale_image = converted_video[t, ...]
        if detect_pseudopods:
            NetDet_fast.detect_pseudopods(lighter_background, pseudopod_min_size=pseudopod_min_size)
            NetDet_fast.merge_network_with_pseudopods()
            pseudopod_vid[t, ...] = NetDet_fast.pseudopods
        potential_network[t, ...] = NetDet_fast.complete_network
    if dims[0] == 1:
        network_dynamics = potential_network
    else:
        for t in np.arange(starting_time, dims[0]):  # 20):#
            if smooth_segmentation_over_time:
                if 2 <= t <= (dims[0] - 2):
                    computed_network = potential_network[(t - 2):(t + 3), :, :].sum(axis=0)
                    computed_network[computed_network == 1] = 0
                    computed_network[computed_network > 1] = 1
                else:
                    if t < 2:
                        computed_network = potential_network[:2, :, :].sum(axis=0)
                    else:
                        computed_network = potential_network[-2:, :, :].sum(axis=0)
                    computed_network[computed_network > 0] = 1
            else:
                computed_network = computed_network[t, :, :].copy()

            if origin is not None:
                computed_network = computed_network * (1 - origin)
                origin_contours = get_contours(origin)
                complete_network = np.logical_or(origin_contours, computed_network).astype(np.uint8)
            else:
                complete_network = computed_network
            complete_network = keep_one_connected_component(complete_network)

            if detect_pseudopods:
                # Make sure that removing pseudopods do not cut the network:
                without_pseudopods = complete_network * (1 - pseudopod_vid[t])
                only_connected_network = keep_one_connected_component(without_pseudopods)
                # # Option A: To add these cutting regions to the pseudopods do:
                pseudopods = (1 - only_connected_network) * complete_network
                pseudopod_vid[t] = pseudopods
            network_dynamics[t] = complete_network

            imtoshow = visu[t, ...]
            eroded_binary = cv2.erode(network_dynamics[t, ...], cross_33)
            net_coord = np.nonzero(network_dynamics[t, ...] - eroded_binary)
            imtoshow[net_coord[0], net_coord[1], :] = (34, 34, 158)
            if show_seg:
                cv2.imshow("", cv2.resize(imtoshow, (1000, 1000)))
                cv2.waitKey(1)
            else:
                visu[t, ...] = imtoshow
            if show_seg:
                cv2.destroyAllWindows()

    network_coord = smallest_memory_array(np.nonzero(network_dynamics), "uint")
    pseudopod_coord = None
    if detect_pseudopods:
        network_dynamics[pseudopod_vid > 0] = 2
        pseudopod_coord = smallest_memory_array(np.nonzero(pseudopod_vid), "uint")
        if save_coord_network:
            np.save(f"coord_pseudopods{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.npy", pseudopod_coord)
    if save_coord_network:
        np.save(f"coord_network{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.npy", network_coord)
    return network_coord, pseudopod_coord


class  NetworkDetection:
    """
    NetworkDetection

    Class for detecting vessels in images using Frangi and Sato filters with various parameter sets.
    It applies different thresholding methods, calculates quality metrics, and selects the best detection method.
    """
    def __init__(self, greyscale_image: NDArray[np.uint8], possibly_filled_pixels: NDArray[np.uint8]=None, add_rolling_window: bool=False, origin_to_add: NDArray[np.uint8]=None, best_result: dict=None):
        """
        Initialize the object with given parameters.

        Parameters
        ----------
        greyscale_image : NDArray[np.uint8]
            The input greyscale image.
        possibly_filled_pixels : NDArray[np.uint8], optional
            Image containing possibly filled pixels. Defaults to None.
        add_rolling_window : bool, optional
            Flag to add rolling window. Defaults to False.
        origin_to_add : NDArray[np.uint8], optional
            Origin to add. Defaults to None.
        best_result : dict, optional
            Best result dictionary. Defaults to None.
        """
        self.greyscale_image = greyscale_image
        if possibly_filled_pixels is None:
            self.possibly_filled_pixels = np.ones(self.greyscale_image.shape, dtype=np.uint8)
        else:
            self.possibly_filled_pixels = possibly_filled_pixels
        self.best_result = best_result
        self.add_rolling_window = add_rolling_window
        self.origin_to_add = origin_to_add
        self.frangi_beta = 1.
        self.frangi_gamma = 1.
        self.black_ridges = True

    def apply_frangi_variations(self) -> list:
        """
        Applies various Frangi filter variations with different sigma values and thresholding methods.

        This method applies the Frangi vesselness filter with multiple sets of sigma values
        to detect vessels at different scales. It applies both Otsu thresholding and rolling window
        segmentation to the filtered results and calculates binary quality indices.

        Returns
        -------
        results : list of dict
            A list containing dictionaries with the method name, binary result, quality index,
            filtered image, filter type, rolling window flag, and sigma values used.
        """
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
            quality_otsu = binary_quality_index(self.possibly_filled_pixels * binary_otsu)

            # Method 2: Rolling window thresholding

            # Store results
            results.append({
                'method': f'f_{sigmas}_thresh',
                'binary': binary_otsu,
                'quality': quality_otsu,
                'filtered': frangi_result,
                'filter': f'Frangi',
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
                    'filter': f'Frangi',
                    'rolling_window': True,
                    'sigmas': sigmas
                })

        return results


    def apply_sato_variations(self) -> list:
        """
        Apply various Sato filter variations to an image and store the results.

        This function applies different parameter sets for the Sato vesselness
        filter to an image, applies two thresholding methods (Otsu and rolling window),
        and stores the results. The function supports optional rolling window
        segmentation based on a configuration flag.

        Returns
        -------
        list of dict
            A list containing dictionaries with the results for each filter variation.
            Each dictionary includes method name, binary image, quality index,
            filtered result, filter type, rolling window flag, and sigma values.
        """
        results = []

        # Parameter variations for Frangi filter
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
            quality_otsu = binary_quality_index(self.possibly_filled_pixels * binary_otsu)


            # Store results
            results.append({
                'method': f's_{sigmas}_thresh',
                'binary': binary_otsu,
                'quality': quality_otsu,
                'filtered': sato_result,
                'filter': f'Sato',
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
                    'filter': f'Sato',
                    'rolling_window': True,
                    'sigmas': sigmas
                })

        return results

    def get_best_network_detection_method(self):
        """
        Get the best network detection method based on quality metrics.

        This function applies Frangi and Sato variations, combines their results,
        calculates quality metrics for each result, and selects the best method.

        Attributes
        ----------
        all_results : list of dicts
            Combined results from Frangi and Sato variations.
        quality_metrics : ndarray of float64
            Quality metrics for each detection result.
        best_idx : int
            Index of the best detection method based on quality metrics.
        best_result : dict
            The best detection result from all possible methods.
        incomplete_network : ndarray of bool
            Binary representation of the best detection result.

        Examples
        ----------
        >>> possibly_filled_pixels = np.zeros((9, 9), dtype=np.uint8)
        >>> possibly_filled_pixels[3:6, 3:6] = 1
        >>> possibly_filled_pixels[1:6, 3] = 1
        >>> possibly_filled_pixels[6:-1, 5] = 1
        >>> possibly_filled_pixels[4, 1:-1] = 1
        >>> greyscale_image = possibly_filled_pixels.copy()
        >>> greyscale_image[greyscale_image > 0] = np.random.randint(170, 255, possibly_filled_pixels.sum())
        >>> greyscale_image[greyscale_image == 0] = np.random.randint(0, 120, possibly_filled_pixels.size - possibly_filled_pixels.sum())
        >>> add_rolling_window=False
        >>> origin_to_add = np.zeros((9, 9), dtype=np.uint8)
        >>> origin_to_add[3:6, 3:6] = 1
        >>> NetDet = NetworkDetection(greyscale_image, possibly_filled_pixels, add_rolling_window, origin_to_add)
        >>> NetDet.get_best_network_detection_method()
        >>> print(NetDet.best_result['method'])
        >>> print(NetDet.best_result['binary'])
        >>> print(NetDet.best_result['quality'])
        >>> print(NetDet.best_result['filtered'])
        >>> print(NetDet.best_result['filter'])
        >>> print(NetDet.best_result['rolling_window'])
        >>> print(NetDet.best_result['sigmas'])
        bgr_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        """
        frangi_res = self.apply_frangi_variations()
        sato_res = self.apply_sato_variations()
        self.all_results = frangi_res + sato_res
        self.quality_metrics = np.array([result['quality'] for result in self.all_results])
        self.best_idx = np.argmax(self.quality_metrics)
        self.best_result = self.all_results[self.best_idx]
        self.incomplete_network = self.best_result['binary'] * self.possibly_filled_pixels


    def detect_network(self):
        """
        Process and detect network features in the greyscale image.

        This method applies a frangi or sato filter based on the best result and
        performs segmentation using either rolling window or Otsu's thresholding.
        The final network detection result is stored in `self.incomplete_network`.
        """
        if self.best_result['filter'] == 'Frangi':
            filtered_result = frangi(self.greyscale_image, sigmas=self.best_result['sigmas'], beta=self.frangi_beta, gamma=self.frangi_gamma, black_ridges=self.black_ridges)
        else:
            filtered_result = sato(self.greyscale_image, sigmas=self.best_result['sigmas'], black_ridges=self.black_ridges, mode='reflect')

        if self.best_result['rolling_window']:
            binary_image = rolling_window_segmentation(filtered_result, self.possibly_filled_pixels, patch_size=(10, 10))
        else:
            thresh_otsu = threshold_otsu(filtered_result)
            binary_image = filtered_result > thresh_otsu
        self.incomplete_network = binary_image * self.possibly_filled_pixels

    def change_greyscale(self, img: NDArray[np.uint8], first_dict: dict):
        """
        Change the image to greyscale using color space combinations.

        This function converts an input image to greyscale by generating
        and applying a combination of color spaces specified in the dictionary.
        The resulting greyscale image is stored as an attribute of the instance.

        Parameters
        ----------
        img : ndarray of uint8
            The input image to be converted to greyscale.
        """
        self.greyscale_image, g2, all_c_spaces, first_pc_vector  = generate_color_space_combination(img, list(first_dict.keys()), first_dict)

    def detect_pseudopods(self, lighter_background: bool, pseudopod_min_width: int=5, pseudopod_min_size: int=50, only_one_connected_component: bool=True):
        """
        Detect pseudopods in a binary image.

        Identify and process regions that resemble pseudopods based on width, size,
        and connectivity criteria. This function is used to detect and label areas
        that are indicative of pseudopod-like structures within a binary image.

        Parameters
        ----------
        lighter_background : bool
            Boolean flag to indicate if the background should be considered lighter.
        pseudopod_min_width : int, optional
            Minimum width for pseudopods to be considered valid. Default is 5.
        pseudopod_min_size : int, optional
            Minimum size for pseudopods to be considered valid. Default is 50.
        only_one_connected_component : bool, optional
            Flag to ensure only one connected component is kept. Default is True.

        Returns
        -------
        None

        Notes
        -----
        This function modifies internal attributes of the object, specifically setting `self.pseudopods` to an array indicating pseudopod regions.

        Examples
        --------
        >>> result = detect_pseudopods(True, 5, 50)
        >>> print(self.pseudopods)
        array([[0, 1, ..., 0],
               [0, 0, ..., 0],
               ...,
               [0, 1, ..., 0]], dtype=uint8)

        """

        closed_im = close_holes(self.possibly_filled_pixels)
        dist_trans = distance_transform_edt(closed_im)
        dist_trans = dist_trans.max() - dist_trans
        # Add dilatation of bracket of distances from medial_axis to the multiplication
        if lighter_background:
            grey = self.greyscale_image.max() - self.greyscale_image
        else:
            grey = self.greyscale_image
        if self.origin_to_add is not None:
            dist_trans_ori = distance_transform_edt(1 - self.origin_to_add)
            scored_im = dist_trans * dist_trans_ori * grey
        else:
            scored_im = (dist_trans**2) * grey
        scored_im = bracket_to_uint8_image_contrast(scored_im)
        thresh = threshold_otsu(scored_im)
        thresh = find_threshold_given_mask(scored_im, self.possibly_filled_pixels, min_threshold=thresh)
        high_int_in_periphery = (scored_im > thresh).astype(np.uint8) * self.possibly_filled_pixels

        _, pseudopod_widths = morphology.medial_axis(high_int_in_periphery, return_distance=True, rng=0)
        bin_im = pseudopod_widths >= pseudopod_min_width
        dil_bin_im = cv2.dilate(bin_im.astype(np.uint8), kernel=create_ellipse(7, 7).astype(np.uint8), iterations=1)
        bin_im = high_int_in_periphery * dil_bin_im
        nb, shapes, stats, centro = cv2.connectedComponentsWithStats(bin_im)
        true_pseudopods = np.nonzero(stats[:, 4] > pseudopod_min_size)[0][1:]
        true_pseudopods = np.isin(shapes, true_pseudopods)

        # Make sure that the tubes connecting two pseudopods belong to pseudopods if removing pseudopods cuts the network
        complete_network = np.logical_or(true_pseudopods, self.incomplete_network).astype(np.uint8)
        if only_one_connected_component:
            complete_network = keep_one_connected_component(complete_network)
            without_pseudopods = complete_network.copy()
            without_pseudopods[true_pseudopods] = 0
            only_connected_network = keep_one_connected_component(without_pseudopods)
            self.pseudopods = (1 - only_connected_network) * complete_network  * self.possibly_filled_pixels
        else:
            self.pseudopods = true_pseudopods.astype(np.uint8)

    def merge_network_with_pseudopods(self):
        """
        Merge the incomplete network with pseudopods.

        This method combines the incomplete network and pseudopods to form
        the complete network. The incomplete network is updated by subtracting
        areas where pseudopods are present.
        """
        self.complete_network = np.logical_or(self.incomplete_network, self.pseudopods).astype(np.uint8)
        self.incomplete_network *= (1 - self.pseudopods)


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
    logging.info(f"Arena n°{arena_label}. Starting graph extraction.")
    # converted_video = self.converted_video; coord_network=self.coord_network; arena_label=1; starting_time=0; origin=self.origin
    dims = converted_video.shape[:3]
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


def get_skeleton_and_widths(pad_network: NDArray[np.uint8], pad_origin: NDArray[np.uint8]=None, pad_origin_centroid: NDArray[np.int64]=None) -> Tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.uint8]]:
    """
    Get skeleton and widths from a network.

    This function computes the morphological skeleton of a network and calculates
    the distances to the closest zero pixel for each non-zero pixel using medial_axis.
    If pad_origin is provided, it adds a central contour. Finally, the function
    removes small loops and keeps only one connected component.

    Parameters
    ----------
    pad_network : ndarray of uint8
        The binary pad network image.
    pad_origin : ndarray of uint8, optional
        An array indicating the origin for adding central contour.
    pad_origin_centroid : ndarray, optional
        The centroid of the pad origin. Defaults to None.

    Returns
    -------
    out : tuple(ndarray of uint8, ndarray of uint8, ndarray of uint8)
        A tuple containing:
        - pad_skeleton: The skeletonized image.
        - pad_distances: The distances to the closest zero pixel.
        - pad_origin_contours: The contours of the central origin, or None if not
          used.

    Examples
    --------
    >>> pad_network = np.array([[0, 1], [1, 0]])
    >>> skeleton, distances, contours = get_skeleton_and_widths(pad_network)
    >>> print(skeleton)
    """
    pad_skeleton, pad_distances = morphology.medial_axis(pad_network, return_distance=True, rng=0)
    pad_skeleton = pad_skeleton.astype(np.uint8)
    if pad_origin is not None:
        pad_skeleton, pad_distances, pad_origin_contours = _add_central_contour(pad_skeleton, pad_distances, pad_origin, pad_network, pad_origin_centroid)
    else:
        pad_origin_contours = None
    pad_skeleton, pad_distances = remove_small_loops(pad_skeleton, pad_distances)
    pad_skeleton = keep_one_connected_component(pad_skeleton)
    pad_distances *= pad_skeleton
    return pad_skeleton, pad_distances, pad_origin_contours


def remove_small_loops(pad_skeleton: NDArray[np.uint8], pad_distances: NDArray[np.float64]=None):
    """
    Remove small loops from a skeletonized image.

    This function identifies and removes small loops in a skeletonized image, returning the modified skeleton.
    If distance information is provided, it updates that as well.

    Parameters
    ----------
    pad_skeleton : ndarray of uint8
        The skeletonized image with potential small loops.
    pad_distances : ndarray of float64, optional
        The distance map corresponding to the skeleton image. Default is `None`.

    Returns
    -------
    out : ndarray of uint8 or tuple(ndarray of uint8, ndarray of float64)
        If `pad_distances` is None, returns the modified skeleton. Otherwise,
        returns a tuple of the modified skeleton and updated distances.
    """
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    # potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)

    cnv_diag_0 = CompareNeighborsWithValue(pad_skeleton, 0)
    cnv_diag_0.is_equal(0, and_itself=True)

    cnv4_false = CompareNeighborsWithValue(pad_skeleton, 4)
    cnv4_false.is_equal(1, and_itself=False)

    loop_centers = np.logical_and((cnv4_false.equal_neighbor_nb == 4), cnv_diag_0.equal_neighbor_nb > 2).astype(np.uint8)

    surrounding = cv2.dilate(loop_centers, kernel=square_33)
    surrounding -= loop_centers
    surrounding = surrounding * cnv8.equal_neighbor_nb

    # Every 2 can be replaced by 0 if the loop center becomes 1
    filled_loops = pad_skeleton.copy()
    filled_loops[surrounding == 2] = 0
    filled_loops += loop_centers

    new_pad_skeleton = morphology.skeletonize(filled_loops, method='lee')

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


def get_neighbor_comparisons(pad_skeleton: NDArray[np.uint8]) -> Tuple[object, object]:
    """
    Get neighbor comparisons for a padded skeleton.

    This function creates two `CompareNeighborsWithValue` objects with different
    neighborhood sizes (4 and 8) and checks if the neighbors are equal to 1. It
    returns both comparison objects.

    Parameters
    ----------
    pad_skeleton : ndarray of uint8
        The input padded skeleton array.

    Returns
    -------
    out : tuple of CompareNeighborsWithValue, CompareNeighborsWithValue
        Two comparison objects for 4 and 8 neighbors.

    Examples
    --------
    >>> cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    """
    cnv4 = CompareNeighborsWithValue(pad_skeleton, 4)
    cnv4.is_equal(1, and_itself=True)
    cnv8 = CompareNeighborsWithValue(pad_skeleton, 8)
    cnv8.is_equal(1, and_itself=True)
    return cnv4, cnv8


def get_vertices_and_tips_from_skeleton(pad_skeleton: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Get vertices and tips from a padded skeleton.

    This function identifies the vertices and tips of a skeletonized image.
    Tips are endpoints of the skeleton while vertices include tips and points where three or more edges meet.

    Parameters
    ----------
    pad_skeleton : ndarray of uint8
        Input skeleton image that has been padded.

    Returns
    -------
    out : tuple (ndarray of uint8, ndarray of uint8)
        Tuple containing arrays of vertex points and tip points.
    """
    cnv4, cnv8 = get_neighbor_comparisons(pad_skeleton)
    potential_tips = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    pad_vertices, pad_tips = get_inner_vertices(pad_skeleton, potential_tips, cnv4, cnv8)
    return pad_vertices, pad_tips


def get_terminations_and_their_connected_nodes(pad_skeleton: NDArray[np.uint8], cnv4: object, cnv8: object) -> NDArray[np.uint8]:
    """
    Get terminations in a skeleton and their connected nodes.

    This function identifies termination points in a padded skeleton array
    based on pixel connectivity, marking them and their connected nodes.

    Parameters
    ----------
    pad_skeleton : ndarray of uint8
        The padded skeleton array where terminations are to be identified.
    cnv4 : object
        Convolution object with 4-connectivity for neighbor comparison.
    cnv8 : object
        Convolution object with 8-connectivity for neighbor comparison.

    Returns
    -------
    out : ndarray of uint8
        Array containing marked terminations and their connected nodes.

    Examples
    --------
    >>> result = get_terminations_and_their_connected_nodes(pad_skeleton, cnv4, cnv8)
    >>> print(result)
    """
    # All pixels having only one neighbor, and containing the value 1, are terminations for sure
    potential_tips = np.zeros(pad_skeleton.shape, dtype=np.uint8)
    potential_tips[cnv8.equal_neighbor_nb == 1] = 1
    # Add more terminations using 4-connectivity
    # If a pixel is 1 (in 4) and all its neighbors are neighbors (in 4), it is a termination

    coord1_4 = cnv4.equal_neighbor_nb == 1
    if np.any(coord1_4):
        coord1_4 = np.nonzero(coord1_4)
        for y1, x1 in zip(coord1_4[0], coord1_4[1]): # y1, x1 = 3,5
            # If, in the neighborhood of the 1 (in 4), all (in 8) its neighbors are 4-connected together, and none of them are terminations, the 1 is a termination
            is_4neigh = cnv4.equal_neighbor_nb[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] != 0
            all_4_connected = pad_skeleton[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] == is_4neigh
            is_not_term = 1 - potential_tips[y1, x1]
            if np.all(all_4_connected * is_not_term):
                is_4neigh[1, 1] = 0
                is_4neigh = np.pad(is_4neigh, [(1,), (1,)], mode='constant')
                cnv_4con = CompareNeighborsWithValue(is_4neigh, 4)
                cnv_4con.is_equal(1, and_itself=True)
                all_connected = (is_4neigh.sum() - (cnv_4con.equal_neighbor_nb > 0).sum()) == 0
                # If they are connected, it can be a termination
                if all_connected:
                    # If its closest neighbor is above 3 (in 8), this one is also a node
                    is_closest_above_3 = cnv8.equal_neighbor_nb[(y1 - 1):(y1 + 2), (x1 - 1):(x1 + 2)] * cross_33 > 3
                    if np.any(is_closest_above_3):
                        Y, X = np.nonzero(is_closest_above_3)
                        Y += y1 - 1
                        X += x1 - 1
                        potential_tips[Y, X] = 1
                    potential_tips[y1, x1] = 1
    return potential_tips


def get_inner_vertices(pad_skeleton: NDArray[np.uint8], potential_tips: NDArray[np.uint8], cnv4: object, cnv8: object) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]: # potential_tips=pad_tips
    """
    Get inner vertices from skeleton image.

    This function identifies and returns the inner vertices of a skeletonized image.
    It processes potential tips to determine which pixels should be considered as
    vertices based on their neighbor count and connectivity.

    Parameters
    ----------
    pad_skeleton : ndarray of uint8
        The padded skeleton image.
    potential_tips : ndarray of uint8, optional
        Potential tip points in the skeleton. Defaults to pad_tips.
    cnv4 : object
        Object for handling 4-connections.
    cnv8 : object
        Object for handling 8-connections.

    Returns
    -------
    out : tuple of ndarray of uint8, ndarray of uint8
        A tuple containing the final vertices matrix and the updated potential tips.

    Examples
    --------
    >>> pad_vertices, potential_tips = get_inner_vertices(pad_skeleton, potential_tips)
    >>> print(pad_vertices)
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
        potential_vertices *= (1 - dilated_previous_intersections)
        pad_vertices[np.nonzero(potential_vertices)] = 1

    # Having 3 neighbors is ambiguous
    with_3_neighbors = cnv8.equal_neighbor_nb == 3
    if np.any(with_3_neighbors):
        # We compare 8-connections with 4-connections
        # We loop over all 3 connected
        coord_3 = np.nonzero(with_3_neighbors)
        for y3, x3 in zip(coord_3[0], coord_3[1]): # y3, x3 = 3,7
            # If, in the neighborhood of the 3, there is at least a 2 (in 8) that is 0 (in 4), and not a termination: the 3 is a node
            has_2_8neigh = cnv8.equal_neighbor_nb[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)] > 0  # 1
            has_2_8neigh_without_focal = has_2_8neigh.copy()
            has_2_8neigh_without_focal[1, 1] = 0
            node_but_not_term = pad_vertices[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)] * (1 - potential_tips[(y3 - 1):(y3 + 2), (x3 - 1):(x3 + 2)])
            all_are_node_but_not_term = np.array_equal(has_2_8neigh_without_focal, node_but_not_term)
            if np.any(has_2_8neigh * (1 - all_are_node_but_not_term)):
                # At least 3 of the 8neigh are not connected:
                has_2_8neigh_without_focal = np.pad(has_2_8neigh_without_focal, [(1,), (1,)], mode='constant')
                cnv_8con = CompareNeighborsWithValue(has_2_8neigh_without_focal, 4)
                cnv_8con.is_equal(1, and_itself=True)
                disconnected_nb = has_2_8neigh_without_focal.sum() - (cnv_8con.equal_neighbor_nb > 0).sum()
                if disconnected_nb > 2:
                    pad_vertices[y3, x3] = 1
    # Now there may be too many vertices:
    # - Those that are 4-connected:
    nb, sh, st, ce = cv2.connectedComponentsWithStats(pad_vertices, connectivity=4)
    problematic_vertices = np.nonzero(st[:, 4] > 1)[0][1:]
    for prob_v in problematic_vertices:
        vertices_group = sh == prob_v
        # If there is a tip in the group, do
        if np.any(potential_tips[vertices_group]):
            # Change the most connected one from tip to vertex
            curr_neighbor_nb = cnv8.equal_neighbor_nb * vertices_group
            wrong_tip = np.nonzero(curr_neighbor_nb == curr_neighbor_nb.max())
            potential_tips[wrong_tip] = 0
        else:
            #  otherwise do:
            # Find the most 8-connected one, if its 4-connected neighbors have no more 8-connexions than 4-connexions + 1, they can be removed
            # Otherwise,
            # Find the most 4-connected one, and remove its 4 connected neighbors having only 1 or other 8-connexion

            c = zoom_on_nonzero(vertices_group)
            # 1. Find the most 8-connected one:
            sub_v_grp = vertices_group[c[0]:c[1], c[2]:c[3]]
            c8 = cnv8.equal_neighbor_nb[c[0]:c[1], c[2]:c[3]]
            vertices_group_8 = c8 * sub_v_grp
            max_8_con = vertices_group_8.max()
            most_8_con = np.nonzero(vertices_group_8 == max_8_con)
            # c4[(most_8_con[0][0] - 1):(most_8_con[0][0] + 2), (most_8_con[1][0] - 1):(most_8_con[1][0] + 2)]
            if len(most_8_con[0]) == 1:
                skel_copy = pad_skeleton[c[0]:c[1], c[2]:c[3]].copy()
                skel_copy[most_8_con] = 0
                sub_cnv8 = CompareNeighborsWithValue(skel_copy, 8)
                sub_cnv8.is_equal(1, and_itself=False)
                sub_cnv4 = CompareNeighborsWithValue(skel_copy, 4)
                sub_cnv4.is_equal(1, and_itself=False)
                v_to_remove = sub_v_grp * (sub_cnv8.equal_neighbor_nb <= sub_cnv4.equal_neighbor_nb + 1)
            else:
                c4 = cnv4.equal_neighbor_nb[c[0]:c[1], c[2]:c[3]]
                # 1. # Find the most 4-connected one:
                vertices_group_4 = c4 * sub_v_grp
                max_con = vertices_group_4.max()
                most_con = np.nonzero(vertices_group_4 == max_con)
                if len(most_con[0]) < sub_v_grp.sum():
                    # 2. Check its 4-connected neighbors and remove those having only 1 other 8-connexion
                    skel_copy = pad_skeleton[c[0]:c[1], c[2]:c[3]].copy()
                    skel_copy[most_con] = 0
                    skel_copy[most_con[0] - 1, most_con[1]] = 0
                    skel_copy[most_con[0] + 1, most_con[1]] = 0
                    skel_copy[most_con[0], most_con[1] - 1] = 0
                    skel_copy[most_con[0], most_con[1] + 1] = 0
                    sub_cnv8 = CompareNeighborsWithValue(skel_copy, 8)
                    sub_cnv8.is_equal(1, and_itself=False)
                    # There are:
                    v_to_remove = ((vertices_group_4 > 0) * sub_cnv8.equal_neighbor_nb) == 1
                else:
                    v_to_remove = np.zeros(sub_v_grp.shape, dtype=bool)
            pad_vertices[c[0]:c[1], c[2]:c[3]][v_to_remove] = 0

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


def get_branches_and_tips_coord(pad_vertices: NDArray[np.uint8], pad_tips: NDArray[np.uint8]) -> Tuple[NDArray, NDArray]:
    """
    Extracts the coordinates of branches and tips from vertices and tips binary images.

    This function calculates branch coordinates by subtracting
    tips from vertices. Then it finds and outputs the non-zero indices of branches and tips separatly.

    Parameters
    ----------
    pad_vertices : ndarray
        Array containing the vertices to be padded.
    pad_tips : ndarray
        Array containing the tips of the padding.

    Returns
    -------
    branch_v_coord : ndarray
        Coordinates of branches derived from subtracting tips from vertices.
    tips_coord : ndarray
        Coordinates of the tips.

    Examples
    --------
    >>> branch_v, tip_c = get_branches_and_tips_coord(pad_vertices, pad_tips)
    >>> branch_v
    >>> tip_c
    """
    pad_branches = pad_vertices - pad_tips
    branch_v_coord = np.transpose(np.array(np.nonzero(pad_branches)))
    tips_coord = np.transpose(np.array(np.nonzero(pad_tips)))
    return branch_v_coord, tips_coord


class EdgeIdentification:
    """Initialize the class with skeleton and distance arrays.

    This class is used to identify edges within a skeleton structure based on
    provided skeleton and distance arrays. It performs various operations to
    refine and label edges, ultimately producing a fully identified network.
    """
    def __init__(self, pad_skeleton: NDArray[np.uint8], pad_distances: NDArray[np.float64], t: int=0):
        """
        Initialize the class with skeleton and distance arrays.

        Parameters
        ----------
        pad_skeleton : ndarray of uint8
            Array representing the skeleton to pad.
        pad_distances : ndarray of float64
            Array representing distances corresponding to the skeleton.

        Attributes
        ----------
        remaining_vertices : None
            Remaining vertices. Initialized as `None`.
        vertices : None
            Vertices. Initialized as `None`.
        growing_vertices : None
            Growing vertices. Initialized as `None`.
        im_shape : tuple of ints
            Shape of the skeleton array.
        """
        self.pad_skeleton = pad_skeleton
        self.pad_distances = pad_distances
        self.t = t
        self.remaining_vertices = None
        self.vertices = None
        self.growing_vertices = None
        self.im_shape = pad_skeleton.shape

    def run_edge_identification(self):
        """
        Run the edge identification process.

        This method orchestrates a series of steps to identify and label edges
        within the graph structure. Each step handles a specific aspect of edge
        identification, ultimately leading to a clearer and more refined edge network.

        Steps involved:
        1. Get vertices and tips coordinates.
        2. Identify tipped edges.
        3. Remove tipped edges smaller than branch width.
        4. Label tipped edges and their vertices.
        5. Label edges connected with vertex clusters.
        6. Label edges connecting vertex clusters.
        7. Label edges from known vertices iteratively.
        8. Label edges looping on 1 vertex.
        9. Clear areas with 1 or 2 unidentified pixels.
        10. Clear edge duplicates.
        11. Clear vertices connecting 2 edges.
        """
        self.get_vertices_and_tips_coord()
        self.get_tipped_edges()
        self.remove_tipped_edge_smaller_than_branch_width()
        self.label_tipped_edges_and_their_vertices()
        self.check_vertex_existence()
        self.label_edges_connected_with_vertex_clusters()
        self.label_edges_connecting_vertex_clusters()
        self.label_edges_from_known_vertices_iteratively()
        self.label_edges_looping_on_1_vertex()
        self.clear_areas_of_1_or_2_unidentified_pixels()
        self.clear_edge_duplicates()
        self.clear_vertices_connecting_2_edges()

    def get_vertices_and_tips_coord(self):
        """Process skeleton data to extract non-tip vertices and tip coordinates.

        This method processes the skeleton stored in `self.pad_skeleton` by first
        extracting all vertices and tips. It then separates these into branch points
        (non-tip vertices) and specific tip coordinates using internal processing.

        Attributes
        ----------
        self.non_tip_vertices : array-like
            Coordinates of non-tip (branch) vertices.
        self.tips_coord : array-like
            Coordinates of identified tips in the skeleton.
        """
        pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(self.pad_skeleton)
        self.non_tip_vertices, self.tips_coord = get_branches_and_tips_coord(pad_vertices, pad_tips)

    def get_tipped_edges(self):
        """
        get_tipped_edges : method to extract skeleton edges connecting branching points and tips.

        Makes sure that there is only one connected component constituting the skeleton of the network and
        identifies all edges that are connected to a tip.

        Attributes
        ----------
        pad_skeleton : ndarray of bool, modified
            Boolean mask representing the pruned skeleton after isolating the largest connected component.
        vertices_branching_tips : ndarray of int, shape (N, 2)
            Coordinates of branching points that connect to tips in the skeleton structure.
        edge_lengths : ndarray of float, shape (M,)
            Lengths of edges connecting non-tip vertices to identified tip locations.
        edge_pix_coord : list of array of int
            Pixel coordinates for each edge path between connected skeleton elements.

        """
        self.pad_skeleton = keep_one_connected_component(self.pad_skeleton)
        self.vertices_branching_tips, self.edge_lengths, self.edge_pix_coord = _find_closest_vertices(self.pad_skeleton,
                                                                                        self.non_tip_vertices,
                                                                                        self.tips_coord[:, :2])

    def remove_tipped_edge_smaller_than_branch_width(self):
        """Remove very short edges from the skeleton.

        This method focuses on edges connecting tips. When too short, they are considered are noise and
        removed from the skeleton and distances matrices. These edges are considered too short when their length is
        smaller than the width of the nearest network branch (an information included in pad_distances).
        This method also updates internal data structures (skeleton, edge coordinates, vertex/tip positions)
        accordingly through pixel-wise analysis and connectivity checks.
        """
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
            if np.nanmax(self.pad_distances[(Y - 1): (Y + 2), (X - 1): (X + 2)]) >= self.edge_lengths[i]:
                tipped_edges_to_remove[i] = True
                # Remove the edge
                self.pad_skeleton[eY, eX] = 0
                # Remove the tip
                self.pad_skeleton[self.tips_coord[i, 0], self.tips_coord[i, 1]] = 0

                # Remove the coordinates corresponding to that edge
                self.edge_pix_coord = np.delete(self.edge_pix_coord, edge_bool, 0)

                # check whether the connecting vertex remains a vertex of not
                pad_sub_skeleton = np.pad(self.pad_skeleton[(Y - 2): (Y + 3), (X - 2): (X + 3)], [(1,), (1,)],
                                          mode='constant')
                sub_vertices, sub_tips = get_vertices_and_tips_from_skeleton(pad_sub_skeleton)
                # If the vertex does not connect at least 3 edges anymore, remove its vertex label
                if sub_vertices[3, 3] == 0:
                    vertex_to_remove = np.nonzero(np.logical_and(self.non_tip_vertices[:, 0] == Y, self.non_tip_vertices[:, 1] == X))[0]
                    branches_to_remove[vertex_to_remove] = True
                # If that pixel became a tip connected to another vertex remove it from the skeleton
                if sub_tips[3, 3]:
                    if sub_vertices[2:5, 2:5].sum() > 1:
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
            logging.error("Removing small tipped edges split the skeleton")
            # for i in range(2, nb):
            #     excedent = sh == i
            #     if (excedent).sum() == 1:
            #         self.pad_skeleton[excedent] = 0

        # Remove in distances the pixels removed in skeleton:
        self.pad_distances *= self.pad_skeleton

        # update edge_pix_coord
        if len(new_edge_pix_coord) > 0:
            self.edge_pix_coord = np.vstack(new_edge_pix_coord)

        # # Remove tips connected to very small edges
        # self.tips_coord = np.delete(self.tips_coord, tipped_edges_to_remove, 0)
        # # Add corresponding edge names
        # self.tips_coord = np.hstack((self.tips_coord, np.arange(1, len(self.tips_coord) + 1)[:, None]))

        # # Within all branching (non-tip) vertices, keep those that did not lose their vertex status because of the edge removal
        # self.non_tip_vertices = np.delete(self.non_tip_vertices, branches_to_remove, 0)

        # # Get the branching vertices who kept their typped edge
        # self.vertices_branching_tips = np.delete(self.vertices_branching_tips, tipped_edges_to_remove, 0)

        # Within all branching (non-tip) vertices, keep those that do not connect a tipped edge.
        # v_branching_tips_in_branching_v = find_common_coord(self.non_tip_vertices, self.vertices_branching_tips[:, :2])
        # self.remaining_vertices = np.delete(self.non_tip_vertices, v_branching_tips_in_branching_v, 0)
        # ordered_v_coord = np.vstack((self.tips_coord[:, :2], self.vertices_branching_tips[:, :2], self.remaining_vertices))

        # tips = self.tips_coord
        # branching_any_edge = self.non_tip_vertices
        # branching_typped_edges = self.vertices_branching_tips
        # branching_no_typped_edges = self.remaining_vertices

        self.get_vertices_and_tips_coord()
        self.get_tipped_edges()

    def label_tipped_edges_and_their_vertices(self):
        """Label edges connecting tip vertices to branching vertices and assign unique labels to all relevant vertices.

        Processes vertex coordinates by stacking tips, vertices branching from tips, and remaining non-tip vertices.
        Assigns unique sequential identifiers to these vertices in a new array. Constructs an array of edge-label information,
        where each row contains the edge label (starting at 1), corresponding tip label, and connected vertex label.

        Attributes
        ----------
        tip_number : int
            The number of tip coordinates available in `tips_coord`.

        ordered_v_coord : ndarray of float
            Stack of unique vertex coordinates ordered by: tips first, vertices branching tips second, non-tip vertices third.

        numbered_vertices : ndarray of uint32
            2D array where each coordinate position is labeled with a sequential integer (starting at 1) based on the order in `ordered_v_coord`.

        edges_labels : ndarray of uint32
            Array of shape (n_edges, 3). Each row contains:
            - Edge label (sequential from 1 to n_edges)
            - Label of the tip vertex for that edge.
            - Label of the vertex branching the tip.

        vertices_branching_tips : ndarray of float
            Unique coordinates of vertices directly connected to tips after removing duplicates.
        """
        self.tip_number = self.tips_coord.shape[0]

        # Stack vertex coordinates in that order: 1. Tips, 2. Vertices branching tips, 3. All remaining vertices
        ordered_v_coord = np.vstack((self.tips_coord[:, :2], self.vertices_branching_tips[:, :2], self.non_tip_vertices))
        ordered_v_coord = np.unique(ordered_v_coord, axis=0)

        # Create arrays to store edges and vertices labels
        self.numbered_vertices = np.zeros(self.im_shape, dtype=np.uint32)
        self.numbered_vertices[ordered_v_coord[:, 0], ordered_v_coord[:, 1]] = np.arange(1, ordered_v_coord.shape[0] + 1)
        self.vertices = None
        self.vertex_index_map = {}
        for idx, (y, x) in enumerate(ordered_v_coord):
            self.vertex_index_map[idx + 1] = tuple((np.uint32(y), np.uint32(x)))

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

    def check_vertex_existence(self):
        if self.tips_coord.shape[0] == 0 and self.non_tip_vertices.shape[0] == 0:
            loop_coord = np.nonzero(self.pad_skeleton)
            start = 1
            end = 1
            vertex_coord = loop_coord[0][0], loop_coord[1][0]
            self.numbered_vertices[vertex_coord[0], vertex_coord[1]] = 1
            self.vertex_index_map[1] = vertex_coord
            self.non_tip_vertices = np.array(vertex_coord)[None, :]
            new_edge_lengths = len(loop_coord[0]) - 1
            new_edge_pix_coord = np.transpose(np.vstack(((loop_coord[0][1:], loop_coord[1][1:], np.zeros(new_edge_lengths, dtype=np.int32)))))
            self.edge_pix_coord = np.zeros((0, 3), dtype=np.int32)
            self._update_edge_data(start, end, new_edge_lengths, new_edge_pix_coord)

    def label_edges_connected_with_vertex_clusters(self):
        """
        Identify edges connected to touching vertices by processing vertex clusters.

        This function processes the skeleton to identify edges connecting vertices
        that are part of touching clusters. It creates a cropped version of the skeleton
        by removing already detected edges and their tips, then iterates through vertex
        clusters to explore and identify nearby edges.
        """
        # I.1. Identify edges connected to touching vertices:
        # First, create another version of these arrays, where we remove every already detected edge and their tips
        cropped_skeleton = self.pad_skeleton.copy()
        cropped_skeleton[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 0
        cropped_skeleton[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 0

        # non_tip_vertices does not need to be updated yet, because it only contains verified branching vertices
        cropped_non_tip_vertices = self.non_tip_vertices.copy()

        self.new_level_vertices = None
        # The problem with vertex_to_vertex_connexion is that since they are not separated by zeros,
        # they always atract each other instead of exploring other paths.
        # To fix this, we loop over each vertex group to
        # 1. Add one edge per inter-vertex connexion inside the group
        # 2. Remove all except one, and loop as many time as necessary.
        # Inside that second loop, we explore and identify every edge nearby.
        # Find every vertex_to_vertex_connexion
        v_cluster_nb, self.v_cluster_lab, self.v_cluster_stats, vgc = cv2.connectedComponentsWithStats(
            (self.numbered_vertices > 0).astype(np.uint8), connectivity=8)
        if v_cluster_nb > 0:
            max_v_nb = np.max(self.v_cluster_stats[1:, 4])
            cropped_skeleton_list = []
            starting_vertices_list = []
            for v_nb in range(2, max_v_nb + 1):
                labels = np.nonzero(self.v_cluster_stats[:, 4] == v_nb)[0]
                coord_list = []
                for lab in labels:  # lab=labels[0]
                    coord_list.append(np.nonzero(self.v_cluster_lab == lab))
                for iter in range(v_nb):
                    for lab_ in range(labels.shape[0]): # lab=labels[0]
                        cs = cropped_skeleton.copy()
                        sv = []
                        v_c = coord_list[lab_]
                        # Save the current coordinate in the starting vertices array of this iteration
                        sv.append([v_c[0][iter], v_c[1][iter]])
                        # Remove one vertex coordinate to keep it from cs
                        v_y, v_x = np.delete(v_c[0], iter), np.delete(v_c[1], iter)
                        cs[v_y, v_x] = 0
                        cropped_skeleton_list.append(cs)
                        starting_vertices_list.append(np.array(sv))
            for cropped_skeleton, starting_vertices in zip(cropped_skeleton_list, starting_vertices_list):
                _, _ = self._identify_edges_connecting_a_vertex_list(cropped_skeleton, cropped_non_tip_vertices, starting_vertices)

    def label_edges_connecting_vertex_clusters(self):
        """
        Label edges connecting vertex clusters.

        This method identifies the connections between connected vertices within
        vertex clusters and labels these edges. It uses the previously found connected
        vertices, creates an image of the connections, and then identifies
        and labels the edges between these touching vertices.
        """
        # I.2. Identify the connexions between connected vertices:
        all_connected_vertices = np.nonzero(self.v_cluster_stats[:, 4] > 1)[0][1:]
        all_con_v_im = np.zeros_like(self.pad_skeleton)
        for v_group in all_connected_vertices:
            all_con_v_im[self.v_cluster_lab == v_group] = 1
        cropped_skeleton = all_con_v_im
        self.vertex_clusters_coord = np.transpose(np.array(np.nonzero(cropped_skeleton)))
        _, _ = self._identify_edges_connecting_a_vertex_list(cropped_skeleton, self.vertex_clusters_coord, self.vertex_clusters_coord)
        # self.edges_labels
        del self.v_cluster_stats
        del self.v_cluster_lab

    def label_edges_from_known_vertices_iteratively(self):
        """
        Label edges iteratively from known vertices.

        This method labels edges in an iterative process starting from
        known vertices. It handles the removal of detected edges and
        updates the skeleton accordingly, to avoid detecting edges twice.
        """
        # II/ Identify all remaining edges
        if self.new_level_vertices is not None:
            starting_vertices_coord = np.vstack((self.new_level_vertices[:, :2], self.vertices_branching_tips))
            starting_vertices_coord = np.unique(starting_vertices_coord, axis=0)
        else:
            # We start from the vertices connecting tips
            starting_vertices_coord = self.vertices_branching_tips.copy()
        # Remove the detected edges from cropped_skeleton:
        cropped_skeleton = self.pad_skeleton.copy()
        cropped_skeleton[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 0
        cropped_skeleton[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 0
        cropped_skeleton[self.vertex_clusters_coord[:, 0], self.vertex_clusters_coord[:, 1]] = 0

        # Reinitialize cropped_non_tip_vertices to browse all vertices except tips and groups
        cropped_non_tip_vertices = self.non_tip_vertices.copy()
        cropped_non_tip_vertices = remove_coordinates(cropped_non_tip_vertices, self.vertex_clusters_coord)
        del self.vertex_clusters_coord
        remaining_v = cropped_non_tip_vertices.shape[0] + 1
        while remaining_v > cropped_non_tip_vertices.shape[0]:
            remaining_v = cropped_non_tip_vertices.shape[0]
            cropped_skeleton, cropped_non_tip_vertices = self._identify_edges_connecting_a_vertex_list(cropped_skeleton, cropped_non_tip_vertices, starting_vertices_coord)
            if self.new_level_vertices is not None:
                starting_vertices_coord = np.unique(self.new_level_vertices[:, :2], axis=0)

    def label_edges_looping_on_1_vertex(self):
        """
        Identify and handle edges that form loops around a single vertex.
        This method processes the skeleton image to find looping edges and updates
        the edge data structure accordingly.
        """
        self.identified = np.zeros_like(self.numbered_vertices)
        self.identified[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 1
        self.identified[self.non_tip_vertices[:, 0], self.non_tip_vertices[:, 1]] = 1
        self.identified[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 1
        unidentified = (1 - self.identified) * self.pad_skeleton

        # Find out the remaining non-identified pixels
        nb, self.unidentified_shapes, self.unidentified_stats, ce = cv2.connectedComponentsWithStats(unidentified.astype(np.uint8))
        # Handle the cases where edges are loops over only one vertex
        looping_edges = np.nonzero(self.unidentified_stats[:, 4 ] > 2)[0][1:]
        for loop_i in looping_edges: # loop_i = looping_edges[0] loop_i=11 #  zoom_on_nonzero(unique_vertices_im, return_coord=False)
            edge_i = (self.unidentified_shapes == loop_i).astype(np.uint8)
            dil_edge_i = cv2.dilate(edge_i, square_33)
            unique_vertices_im = self.numbered_vertices.copy()
            unique_vertices_im[self.tips_coord[:, 0], self.tips_coord[:, 1]] = 0
            unique_vertices_im = dil_edge_i * unique_vertices_im
            unique_vertices = np.unique(unique_vertices_im)
            unique_vertices = unique_vertices[unique_vertices > 0]
            v_nb = len(unique_vertices)
            new_edge_lengths = edge_i.sum()
            new_edge_pix_coord = np.transpose(np.vstack((np.nonzero(edge_i))))
            new_edge_pix_coord = np.hstack((new_edge_pix_coord, np.repeat(1, new_edge_pix_coord.shape[0])[:, None]))
            if v_nb == 1:
                start, end = unique_vertices[0], unique_vertices[0]
                self._update_edge_data(start, end, new_edge_lengths, new_edge_pix_coord)
            elif v_nb == 2:
                # The edge loops around a group of connected vertices
                start, end = unique_vertices[0], unique_vertices[1]
                self._update_edge_data(start, end, new_edge_lengths, new_edge_pix_coord)
                # conn_v_nb, conn_v = cv2.connectedComponents((unique_vertices_im > 0).astype(np.uint8))
                # if len(unique_vertices) == 2 and conn_v_nb == 2:
            elif v_nb > 2: # The question is: How to choose two vertices so that they link all missing pixels?
                # 1. Find every edge pixel connected to these vertices
                vertex_connected_pixels = np.nonzero(cv2.dilate((unique_vertices_im > 0).astype(np.uint8), square_33) * edge_i)
                # 2. Find the most distant pair of these
                pix1, pix2 = get_min_or_max_euclidean_pair(vertex_connected_pixels, "max")
                # 3. The two best vertices are the two nearest to these two most distant edge pixels
                dist_to_pix1 = np.zeros(v_nb, np.float64)
                dist_to_pix2 = np.zeros(v_nb, np.float64)
                for _i, v_i in enumerate(unique_vertices):
                    v_coord = self.vertex_index_map[v_i]
                    dist_to_pix1[_i] = eudist(pix1, v_coord)
                    dist_to_pix2[_i] = eudist(pix2, v_coord)
                start, end = unique_vertices[np.argmin(dist_to_pix1)], unique_vertices[np.argmin(dist_to_pix2)]
                self._update_edge_data(start, end, new_edge_lengths, new_edge_pix_coord)
            else:
                logging.error(f"t={self.t}, One long edge is not identified: i={loop_i} of length={edge_i.sum()} close to {len(unique_vertices)} vertices.")
        self.identified[self.edge_pix_coord[:, 0], self.edge_pix_coord[:, 1]] = 1

    def clear_areas_of_1_or_2_unidentified_pixels(self):
        """Removes 1 or 2 pixel size non-identified areas from the skeleton.

        This function checks whether small non-identified areas (1 or 2 pixels)
        can be removed without breaking the skeleton structure. It performs
        a series of operations to ensure only safe removals are made, logging
        errors if the final skeleton is not fully connected or if some unidentified pixels remain.
        """
        # Check whether the 1 or 2 pixel size non-identified areas can be removed without breaking the skel
        one_pix = np.nonzero(self.unidentified_stats[:, 4 ] <= 2)[0] # == 1)[0]
        cutting_removal = []
        for pix_i in one_pix: #pix_i=one_pix[0]
            skel_copy = self.pad_skeleton.copy()
            y1, y2, x1, x2 = self.unidentified_stats[pix_i, 1], self.unidentified_stats[pix_i, 1] + self.unidentified_stats[pix_i, 3], self.unidentified_stats[pix_i, 0], self.unidentified_stats[pix_i, 0] + self.unidentified_stats[pix_i, 2]
            skel_copy[y1:y2, x1:x2][self.unidentified_shapes[y1:y2, x1:x2] == pix_i] = 0
            nb1, sh1 = cv2.connectedComponents(skel_copy.astype(np.uint8), connectivity=8)
            if nb1 > 2:
                cutting_removal.append(pix_i)
            else:
                self.pad_skeleton[y1:y2, x1:x2][self.unidentified_shapes[y1:y2, x1:x2] == pix_i] = 0
        if len(cutting_removal) > 0:
            logging.error(f"t={self.t}, These pixels break the skeleton when removed: {cutting_removal}")
        if (self.identified > 0).sum() != self.pad_skeleton.sum():
            logging.error(f"t={self.t}, Proportion of identified pixels in the skeleton: {(self.identified > 0).sum() / self.pad_skeleton.sum()}")
        self.pad_distances *= self.pad_skeleton
        del self.identified
        del self.unidentified_stats
        del self.unidentified_shapes


    def _identify_edges_connecting_a_vertex_list(self, cropped_skeleton: NDArray[np.uint8], cropped_non_tip_vertices: NDArray, starting_vertices_coord: NDArray) -> Tuple[NDArray[np.uint8], NDArray]:
        """Identify edges connecting a list of vertices within a cropped skeleton.

        This function iteratively connects the closest vertices from starting_vertices_coord to their nearest neighbors,
        updating the skeleton and removing already connected vertices until no new connections can be made or
        a maximum number of connections is reached.

        Parameters
        ----------
        cropped_skeleton : ndarray of uint8
            A binary skeleton image where skeletal pixels are marked as 1.
        cropped_non_tip_vertices : ndarray of int
            Coordinates of non-tip vertices in the cropped skeleton.
        starting_vertices_coord : ndarray of int
            Coordinates of vertices from which to find connections.

        Returns
        -------
        cropped_skeleton : ndarray of uint8
            Updated skeleton with edges marked as 0.
        cropped_non_tip_vertices : ndarray of int
            Updated list of non-tip vertices after removing those that have been connected.
        """
        explored_connexions_per_vertex = 0  # the maximal edge number that can connect a vertex
        new_connexions = True
        while new_connexions and explored_connexions_per_vertex < 5 and np.any(cropped_non_tip_vertices) and np.any(starting_vertices_coord):

            explored_connexions_per_vertex += 1
            # 1. Find the ith closest vertex to each focal vertex
            ending_vertices_coord, new_edge_lengths, new_edge_pix_coord = _find_closest_vertices(
                cropped_skeleton, cropped_non_tip_vertices, starting_vertices_coord)
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
                self._update_edge_data(start, end, new_edge_lengths, new_edge_pix_coord)

                no_new_connexion = np.logical_or(no_new_connexion, vertex_to_vertex_connexions)
                vertices_to_crop = starting_vertices_coord[no_new_connexion, :]

                # Remove non-connectable and connected_vertices from:
                cropped_non_tip_vertices = remove_coordinates(cropped_non_tip_vertices, vertices_to_crop)
                starting_vertices_coord = remove_coordinates(starting_vertices_coord, vertices_to_crop)

                if new_edge_pix_coord.shape[0] > 0:
                    # Update cropped_skeleton to not identify each edge more than once
                    cropped_skeleton[new_edge_pix_coord[:, 0], new_edge_pix_coord[:, 1]] = 0
                # And the starting vertices that cannot connect anymore
                cropped_skeleton[vertices_to_crop[:, 0], vertices_to_crop[:, 1]] = 0

                if self.new_level_vertices is None:
                    self.new_level_vertices = ending_vertices_coord[found_connexion, :].copy()
                else:
                    self.new_level_vertices = np.vstack((self.new_level_vertices, ending_vertices_coord[found_connexion, :]))

        return cropped_skeleton, cropped_non_tip_vertices

    def _update_edge_data(self, start, end, new_edge_lengths: NDArray, new_edge_pix_coord: NDArray):
        """
        Update edge data by expanding existing arrays with new edges.

        This method updates the internal edge data structures (lengths,
        labels, and pixel coordinates) by appending new edges.

        Parameters
        ----------
        start : int or ndarray of int
            The starting vertex label(s) for the new edges.
        end : int or ndarray of int
            The ending vertex label(s) for the new edges.
        new_edge_lengths : ndarray of float
            The lengths of the new edges to be added.
        new_edge_pix_coord : ndarray of float
            The pixel coordinates of the new edges.

        Attributes
        ----------
        edge_lengths : ndarray of float
            The lengths of all edges.
        edges_labels : ndarray of int
            The labels for each edge (start and end vertices).
        edge_pix_coord : ndarray of float
            The pixel coordinates for all edges.
        """
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

    def clear_edge_duplicates(self):
        """
        Remove duplicate edges by checking vertices and coordinates.

        This method identifies and removes duplicate edges based on their vertex labels
        and pixel coordinates. It scans through the edge attributes, compares them,
        and removes duplicates if they are found.
        """
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
        edges_to_remove = np.unique(edges_to_remove)
        for edge in edges_to_remove:
            edge_bool = self.edges_labels[:, 0] != edge
            self.edges_labels = self.edges_labels[edge_bool, :]
            self.edge_lengths = self.edge_lengths[edge_bool]
            self.edge_pix_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] != edge, :]


    def clear_vertices_connecting_2_edges(self):
        """
        Remove vertices connecting exactly two edges and update edge-related attributes.

        This method identifies vertices that are connected to exactly 2 edges,
        renames edges, updates edge lengths and vertex coordinates accordingly.
        It also removes the corresponding vertices from non-tip vertices list.
        """
        v_labels, v_counts = np.unique(self.edges_labels[:, 1:], return_counts=True)
        vertices2 = v_labels[v_counts == 2]
        for vertex2 in vertices2:  # vertex2 = vertices2[0]
            edge_indices = np.nonzero(self.edges_labels[:, 1:] == vertex2)[0]
            edge_names = [self.edges_labels[edge_indices[0], 0], self.edges_labels[edge_indices[1], 0]]
            v_names = np.concatenate((self.edges_labels[edge_indices[0], 1:], self.edges_labels[edge_indices[1], 1:]))
            v_names = v_names[v_names != vertex2]
            if len(v_names) > 0: # Otherwise it's a vertex between a normal edge and a loop
                kept_edge = int(self.edge_lengths[edge_indices[1]] >= self.edge_lengths[edge_indices[0]])
                # Rename the removed edge by the kept edge name in pix_coord:
                self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_names[1 - kept_edge], 2] = edge_names[kept_edge]
                # Add the removed edge length to the kept edge length (minus 2, corresponding to the removed vertex)
                self.edge_lengths[self.edges_labels[:, 0] == edge_names[kept_edge]] += self.edge_lengths[self.edges_labels[:, 0] == edge_names[1 - kept_edge]] - 1
                # Remove the corresponding edge length from the list
                self.edge_lengths = self.edge_lengths[self.edges_labels[:, 0] != edge_names[1 - kept_edge]]
                # Rename the vertex of the kept edge in edges_labels
                self.edges_labels[self.edges_labels[:, 0] == edge_names[kept_edge], 1:] = v_names[1 - kept_edge], v_names[kept_edge]
                # Remove the removed edge from the edges_labels array
                self.edges_labels = self.edges_labels[self.edges_labels[:, 0] != edge_names[1 - kept_edge], :]
                # vY, vX = np.nonzero(self.numbered_vertices == vertex2)
                # v_idx = np.nonzero(np.all(self.non_tip_vertices == [vY[0], vX[0]], axis=1))
                vY, vX = self.vertex_index_map[vertex2]
                v_idx = np.nonzero(np.all(self.non_tip_vertices == [vY, vX], axis=1))
                self.non_tip_vertices = np.delete(self.non_tip_vertices, v_idx, axis=0)
        # Sometimes, clearing vertices connecting 2 edges can create edge duplicates, so:
        self.clear_edge_duplicates()

    def _remove_padding(self):
        """
        Removes padding from various coordinate arrays.

        This method adjusts the coordinates of edge pixels, tips,
        and non-tip vertices by subtracting 1 from their x and y values.
        It also removes padding from the skeleton, distances, and vertices
        using the `remove_padding` function.
        """
        self.edge_pix_coord[:, :2] -= 1
        self.tips_coord[:, :2] -= 1
        self.non_tip_vertices[:, :2] -= 1
        del self.vertex_index_map
        self.skeleton, self.distances, self.vertices = remove_padding(
            [self.pad_skeleton, self.pad_distances, self.numbered_vertices])


    def make_vertex_table(self, origin_contours: NDArray[np.uint8]=None, growing_areas: NDArray=None):
        """
        Generate a table for the vertices.

        This method constructs and returns a 2D NumPy array holding information
        about all vertices. Each row corresponds to one vertex identified either
        by its coordinates in `self.tips_coord` or `self.non_tip_vertices`. The
        array includes additional information about each vertex, including whether
        they are food vertices, growing areas, and connected components.

        Parameters
        ----------
        origin_contours : ndarray of uint8, optional
            Binary map to identify food vertices. Default is `None`.
        growing_areas : ndarray, optional
            Binary map to identify growing regions. Default is `None`.

        Notes
        -----
            The method updates the instance attribute `self.vertex_table` with
            the generated vertex information.
        """
        if self.vertices is None:
            self._remove_padding()
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

        if growing_areas is not None and growing_areas.shape[1] > 0:
            # growing = np.unique(self.vertices * growing_areas)[1:]
            growing = np.unique(self.vertices[growing_areas[0], growing_areas[1]])
            growing = growing[growing > 0]
            if len(growing) > 0:
                growing = np.isin(self.vertex_table[:, 2], growing)
                self.vertex_table[growing, 4] = 2

        nb, sh, stats, cent = cv2.connectedComponentsWithStats((self.vertices > 0).astype(np.uint8))
        for i, v_i in enumerate(np.nonzero(stats[:, 4] > 1)[0][1:]):
            v_labs = self.vertices[sh == v_i]
            for v_lab in v_labs: # v_lab = v_labs[0]
                self.vertex_table[self.vertex_table[:, 2] == v_lab, 5] = 1


    def make_edge_table(self, greyscale: NDArray[np.uint8], compute_BC: bool=False):
        """
        Generate edge table with length and average intensity information.

        This method processes the vertex coordinates, calculates lengths
        between vertices for each edge, and computes average width and intensity
        along the edges. Additionally, it computes edge betweenness centrality
        for each vertex pair.

        Parameters
        ----------
        greyscale : ndarray of uint8
            Grayscale image.
        """
        if self.vertices is None:
            self._remove_padding()
        self.edge_table = np.zeros((self.edges_labels.shape[0], 7), float) # edge_id, vertex1, vertex2, length, average_width, int, BC
        self.edge_table[:, :3] = self.edges_labels[:, :]
        self.edge_table[:, 3] = self.edge_lengths
        for idx, edge_lab in enumerate(self.edges_labels[:, 0]):
            edge_coord = self.edge_pix_coord[self.edge_pix_coord[:, 2] == edge_lab, :]
            pix_widths = self.distances[edge_coord[:, 0], edge_coord[:, 1]]
            v_id = self.edges_labels[self.edges_labels[:, 0] == edge_lab, 1:][0]
            v1_coord = self.vertex_table[self.vertex_table[:, 2] == v_id[0], :2][0]#
            v2_coord = self.vertex_table[self.vertex_table[:, 2] == v_id[1], :2][0]#
            v1_width, v2_width = self.distances[v1_coord[0], v1_coord[1]], self.distances[v2_coord[0], v2_coord[1]]

            if not np.isnan(v1_width):
                pix_widths = np.append(pix_widths, v1_width)
            if not np.isnan(v2_width):
                pix_widths = np.append(pix_widths, v2_width)
            if pix_widths.size > 0:
                self.edge_table[idx, 4] = pix_widths.mean()
            else:
                self.edge_table[idx, 4] = np.nan
            pix_ints = greyscale[edge_coord[:, 0], edge_coord[:, 1]]
            v1_int, v2_int = greyscale[v1_coord[0], v1_coord[1]], greyscale[v2_coord[0], v2_coord[1]]
            pix_ints = np.append(pix_ints, (v1_int, v2_int))
            self.edge_table[idx, 5] = pix_ints.mean()

        if compute_BC:
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
                            logging.error(f"There still is two identical edges: {edge_lab} of len: {len(edge_coord)} linking vertices {v}")
                            break


def _find_closest_vertices(skeleton: NDArray[np.uint8], all_vertices_coord: NDArray, starting_vertices_coord: NDArray) -> Tuple[NDArray, NDArray[np.float64], NDArray[np.uint32]]:
    """
    Find the closest vertices in a skeleton graph.

    This function performs a breadth-first search (BFS) from each starting vertex to find the nearest branching
    vertex in a skeleton graph. It returns the coordinates of the ending vertices, edge lengths,
    and the coordinates of all pixels along each edge.

    Parameters
    ----------
    skeleton : ndarray of uint8
        The skeleton graph represented as a binary image.
    all_vertices_coord : ndarray
        Coordinates of all branching vertices in the skeleton.
    starting_vertices_coord : ndarray
        Coordinates of the starting vertices from which to search.

    Returns
    -------
    ending_vertices_coord : ndarray of uint32
        Coordinates of the ending vertices found for each starting vertex.
    edge_lengths : ndarray of float64
        Lengths of the edges from each starting vertex to its corresponding ending vertex.
    edges_coords : ndarray of uint32
        Coordinates of all pixels along each edge.

    Examples
    --------
    >>> skeleton=cropped_skeleton; all_vertices_coord=cropped_non_tip_vertices
    >>> skeleton = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]])
    >>> all_vertices_coord = np.array([[1, 1], [3, 1]])
    >>> starting_vertices_coord = np.array([[1, 1]])
    >>> ending_vertices_coord, edge_lengths, edges_coords = _find_closest_vertices(skeleton, all_vertices_coord, starting_vertices_coord)
    >>> print(ending_vertices_coord)
    [[3 1 1]]
    >>> print(edge_lengths)
    [2.]
    >>> print(edges_coords)
    [[2 1 1]]
    """

    # Convert branching vertices to set for quick lookup
    branch_set = set(zip(all_vertices_coord[:, 0], all_vertices_coord[:, 1]))
    n = starting_vertices_coord.shape[0]

    ending_vertices_coord = np.zeros((n, 3), np.int32)  # next_vertex_y, next_vertex_x, edge_id
    edge_lengths = np.zeros(n, np.float64)
    all_path_pixels = []  # Will hold rows of (y, x, edge_id)
    i = 0
    edge_i = 0
    for tip_y, tip_x in zip(starting_vertices_coord[:, 0], starting_vertices_coord[:, 1]):
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

        else:
            edge_lengths[i] = np.nan
        i += 1
    if len(all_path_pixels) > 0:
        edges_coords = np.array(all_path_pixels, dtype=np.uint32)
    else:
        edges_coords = np.zeros((0, 3), dtype=np.uint32)
    return ending_vertices_coord, edge_lengths, edges_coords

def ad_pad(arr: NDArray) -> NDArray:
    """
    Pad the input array with a single layer of zeros around its edges.

    Parameters
    ----------
    arr : ndarray
        The input array to pad. Must be at least 2-dimensional.

    Returns
    -------
    padded_arr : ndarray
        The output array with a single 0-padded layer around its edges.

    Notes
    -----
    This function uses NumPy's `pad` with mode='constant' to add a single layer
    of zeros around the edges of the input array.

    Examples
    --------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> ad_pad(arr)
    array([[0, 0, 0, 0],
       [0, 1, 2, 0],
       [0, 3, 4, 0],
       [0, 0, 0, 0]])
    """
    return np.pad(arr, [(1, ), (1, )], mode='constant')

def un_pad(arr: NDArray) -> NDArray:
    """
    Unpads a 2D NumPy array by removing the first and last row/column.

    Extended Description
    --------------------
    Reduces the size of a 2D array by removing the outermost rows and columns.
    Useful for trimming boundaries added during padding operations.

    Parameters
    ----------
    arr : ndarray
        Input 2D array to be unpadded. Shape (n,m) is expected.

    Returns
    -------
    ndarray
        Unpadded 2D array with shape (n-2, m-2).

    Examples
    --------
    >>> arr = np.array([[0, 0, 0],
    >>>                 [0, 4, 0],
    >>>                 [0, 0, 0]])
    >>> un_pad(arr)
    array([[4]])
    """
    return arr[1:-1, 1:-1]

def add_padding(array_list: list) -> list:
    """
    Add padding to each 2D array in a list.

    Parameters
    ----------
    array_list : list of ndarrays
        List of 2D NumPy arrays to be processed.

    Returns
    -------
    out : list of ndarrays
        List of 2D NumPy arrays with the padding removed.

    Examples
    --------
    >>> array_list = [np.array([[1, 2], [3, 4]])]
    >>> padded_list = add_padding(array_list)
    >>> print(padded_list[0])
    [[0 0 0]
     [0 1 2 0]
     [0 3 4 0]
     [0 0 0]]
    """
    new_array_list = []
    for arr in array_list:
        new_array_list.append(ad_pad(arr))
    return new_array_list


def remove_padding(array_list: list) -> list:
    """
    Remove padding from a list of 2D arrays.

    Parameters
    ----------
    array_list : list of ndarrays
        List of 2D NumPy arrays to be processed.

    Returns
    -------
    out : list of ndarrays
        List of 2D NumPy arrays with the padding removed.

    Examples
    --------
    >>> arr1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    >>> arr2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> remove_padding([arr1, arr2])
    [array([[1]]), array([[0]])]
    """
    new_array_list = []
    for arr in array_list:
        new_array_list.append(un_pad(arr))
    return new_array_list


def _add_central_contour(pad_skeleton: NDArray[np.uint8], pad_distances: NDArray[np.float64], pad_origin: NDArray[np.uint8], pad_network: NDArray[np.uint8], pad_origin_centroid: NDArray[np.int64]) -> Tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.uint8]]:
    """
    Add a central contour to the skeleton while preserving distances.

    This function modifies the input skeleton and distance arrays by adding a
    central contour around an initial shape, updating the skeleton to include this new contour, and
    preserving the distance information.

    Parameters
    ----------
    pad_skeleton : ndarray of uint8
        The initial skeleton.
    pad_distances : ndarray of float64
        The distance array corresponding to the input skeleton.
    pad_origin : ndarray of uint8
        Array representing origin points in the image.
    pad_network : ndarray of uint8
        Network structure array used to find contours in the skeleton.
    pad_origin_centroid : ndarray
        Centroids of origin points.

    Returns
    -------
    out : tuple(ndarray of uint8, ndarray of float64, ndarray of uint8)
        Tuple containing the new skeleton, updated distance array,
        and origin contours.
    """
    pad_net_contour = get_contours(pad_network)

    # Make a hole at the skeleton center and find the vertices connecting it
    holed_skeleton = pad_skeleton * (1 - pad_origin)
    pad_vertices, pad_tips = get_vertices_and_tips_from_skeleton(pad_skeleton)
    ite = 20
    dil_origin = cv2.dilate(pad_origin, rhombus_55, iterations=ite)
    im_border = 1 - image_borders(pad_network.shape)
    while np.any(dil_origin * im_border):
        ite -= 1
        dil_origin = cv2.dilate(pad_origin, rhombus_55, iterations=ite)
    pad_vertices *= dil_origin
    connecting_pixels = np.transpose(np.array(np.nonzero(pad_vertices)))

    skeleton_without_vertices = pad_skeleton.copy()
    skeleton_without_vertices[pad_vertices > 0] = 0

    # Previously was connected to the center of the shape.
    line_coordinates = get_all_line_coordinates(pad_origin_centroid, connecting_pixels)
    with_central_contour = holed_skeleton.copy()
    for vertex, new_edge in zip(connecting_pixels, line_coordinates): # nei = 65; new_edge=line_coordinates[nei]
        new_edge_im = np.zeros_like(pad_origin)
        new_edge_im[new_edge[:, 0], new_edge[:, 1]] = 1
        if not np.any(new_edge_im * pad_net_contour) and not np.any(new_edge_im * skeleton_without_vertices):# and not np.any(new_edge_im * holed_skeleton):
            with_central_contour[new_edge[:, 0], new_edge[:, 1]] = 1
    # Add dilated contour
    pad_origin_contours = get_contours(pad_origin)
    with_central_contour *= (1 - pad_origin)
    with_central_contour += pad_origin_contours
    if np.any(with_central_contour == 2):
        with_central_contour[with_central_contour > 0] = 1

    # show(dil_origin * with_central_contour)
    # Capture only the new contour and its neighborhood, get its skeleton and update the final skeleton
    new_contour = dil_origin * with_central_contour
    dil_im_border = cv2.dilate(im_border, cross_33, iterations=1)
    if not np.any(new_contour * dil_im_border):
        new_contour = cv2.morphologyEx(new_contour, cv2.MORPH_CLOSE, square_33)
    new_contour = morphology.medial_axis(new_contour, rng=0).astype(np.uint8)
    new_skeleton = with_central_contour * (1 - dil_origin)
    new_skeleton += new_contour
    new_pixels = np.logical_and(pad_distances == 0, new_skeleton == 1)
    new_pix_coord = np.transpose(np.array(np.nonzero(new_pixels)))
    dist_coord = np.transpose(np.array(np.nonzero(pad_distances)))

    dist_from_dist = cdist(new_pix_coord[:, :], dist_coord)
    for np_i, dist_i in enumerate(dist_from_dist): # dist_i=dist_from_dist[0]
        close_i = dist_i.argmin()
        pad_distances[new_pix_coord[np_i, 0], new_pix_coord[np_i, 1]] = pad_distances[dist_coord[close_i, 0], dist_coord[close_i, 1]]

    # Update distances
    pad_distances *= new_skeleton

    dil_pad_origin_contours = cv2.dilate(pad_origin_contours, cross_33, iterations=1)
    new_pad_origin_contours = dil_pad_origin_contours * new_skeleton
    new_pad_origin_contours += pad_origin
    new_pad_origin_contours[new_pad_origin_contours > 0] = 1
    new_pad_origin_contours = get_contours(new_pad_origin_contours)
    nb, sh = cv2.connectedComponents(new_pad_origin_contours)

    new_skeleton[new_pad_origin_contours > 0] = 1
    if nb > 2:
        new_pad_origin_contours = cv2.morphologyEx(new_pad_origin_contours, cv2.MORPH_CLOSE, square_33, iterations=1)
        nb, sh = cv2.connectedComponents(new_pad_origin_contours)
        current_contour_coord = np.argwhere(new_pad_origin_contours)
        cnv4, cnv8 = get_neighbor_comparisons(new_pad_origin_contours)
        potential_tips = get_terminations_and_their_connected_nodes(new_pad_origin_contours, cnv4, cnv8)
        tips_coord = np.transpose(np.array(np.nonzero(potential_tips)))
        ending_vertices_coord, edge_lengths, edges_coords = _find_closest_vertices(pad_origin, current_contour_coord, tips_coord)
        new_potentials = np.unique(edges_coords[:, 2])
        for new_pot in new_potentials:
            edge_coord = edges_coords[edges_coords[:, 2] == new_pot, :2]
            test = new_pad_origin_contours.copy()
            test[edge_coord[:, 0], edge_coord[:, 1]] = 1
            new_nb, sh = cv2.connectedComponents(test)
            if new_nb < nb:
                new_pad_origin_contours[edge_coord[:, 0], edge_coord[:, 1]] = 1

    pad_origin_contours = new_pad_origin_contours
    pad_distances[pad_origin_contours > 0] = np.nan

    return new_skeleton, pad_distances, pad_origin_contours

