#!/usr/bin/env python3
"""
Video network detection and skeleton analysis for biological networks (such as Physarum polycephalum's) images.

This module uses network functions on images to track networks on a video.

Functions
---------
detect_network_dynamics: detect the network on every image of a video and use their temporality to improve accuracy
"""
import cv2
import logging
import numpy as np
from numpy.typing import NDArray
from cellects.image.morphological_operations import cross_33, get_contours, keep_one_connected_component
from cellects.utils.utilitarian import smallest_memory_array
from cellects.io.save import write_h5
from cellects.image.network_functions import NetworkDetection
from numba.typed import Dict as TDict


def detect_network_dynamics(converted_video: NDArray, binary: NDArray[np.uint8], arena_label: int=1,
                            starting_time: int=0, visu: NDArray=None, origin: NDArray[np.uint8]=None,
                            smooth_segmentation_over_time: bool = True, morphological_closing: bool=True,
                            edge_max_width:int = 5, detect_pseudopods: bool = True, save_coord_network: bool = True,
                            show_seg: bool = False):
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
    morphological_closing : bool, optional (default=True)
        Flag indicating whether to apply morphological closing on binary images of the network.
    edge_max_width : int, optional
        Maximal width of network edges. Defaults to 5.
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
                              origin_to_add=origin, morphological_closing=morphological_closing)
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
        NetDet_fast = NetworkDetection(greyscale, possibly_filled_pixels=binary[t, ...], origin_to_add=origin,
                                       edge_max_width=edge_max_width, morphological_closing=morphological_closing,
                                       best_result=NetDet.best_result)
        NetDet_fast.detect_network()
        NetDet_fast.greyscale_image = converted_video[t, ...]
        if detect_pseudopods:
            NetDet_fast.detect_pseudopods(lighter_background, pseudopod_min_size=pseudopod_min_size)
            pseudopod_vid[t, ...] = NetDet_fast.pseudopods
        potential_network[t, ...] = NetDet_fast.complete_network
    del NetDet_fast
    del NetDet
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
                computed_network = potential_network[t, :, :].copy()

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
            write_h5(f"coord_pseudopods{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.h5", pseudopod_coord)
    if save_coord_network:
        write_h5(f"coord_network{arena_label}_t{dims[0]}_y{dims[1]}_x{dims[2]}.h5", network_coord)
    return network_coord, pseudopod_coord
