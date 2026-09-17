#!/usr/bin/env python3
"""
Video network detection and skeleton analysis for biological networks (such as Physarum polycephalum's) images.

This module uses network functions on images to track networks on a video.

Classes
-------
NetworkTracking: detect the network on every image of a video and use their temporality to improve accuracy
"""
import cv2
import logging
import numpy as np
from numpy.typing import NDArray
from cellects.image.morphological_operations import cross_33, get_contours, keep_one_connected_component
from cellects.utils.utilitarian import smallest_memory_array
from cellects.image.network_functions import NetworkDetection
from cellects.io.save import write_h5
from numba.typed import Dict as TDict


class NetworkTracking:
    def __init__(self, motion):
        self.motion = motion

    def init_tracking(self):
        """
        Summary
        -------
        Initialize tracking attributes, set up visualizations, and configure the
        network detection pipeline for the current motion data.

        Notes
        -----
        - Resets motion‑related buffers such as ``coord_network`` and
          ``pseudopod_coord``.
        - Determines the origin based on ``self.motion.vars['origin_state']``.
        - Allocates arrays (e.g., ``pseudopod_vid``, ``potential_network``,
          ``network_dynamics``) matching the shape of ``self.motion.binary``.
        - Creates a greyscale image from either ``self.motion.visu`` or
          ``self.motion.converted_video`` for further processing.
        - Instantiates :class:`NetworkDetection` with the greyscale frame,
          the binary mask of the last frame, the origin (if any), and the
          morphological closing flag.
        """
        self.motion.coord_network = None
        self.motion.pseudopod_coord = None
        self.motion.check_converted_video_type()
        if self.motion.vars['origin_state'] == "constant":
            self.origin = self.motion.origin
        else:
            self.origin = None
        self.dims = self.motion.binary.shape
        self.starting_time = 0
        self.edge_max_width = 5
        self.pseudopod_min_size = 50
        self.detect_pseudopods = True
        if self.detect_pseudopods:
            self.pseudopod_vid = np.zeros_like(self.motion.binary, dtype=bool)
        self.potential_network = np.zeros_like(self.motion.binary, dtype=bool)
        self.network_dynamics = np.zeros_like(self.motion.binary, dtype=np.uint8)
        self.do_convert = True
        if self.motion.visu is None:
            self.do_convert = False
            self.motion.visu = np.stack(
                (self.motion.converted_video, self.motion.converted_video, self.motion.converted_video),
                axis=3)
        self.find_network_detection_method()

    def find_network_detection_method(self):
        """
        Explore 3 sets of 3 frames in the video, exclude methods producing too much variation, keep the best score


        """
        max_surface_area_variation = .002
        if self.dims[0] < 10:
            if self.do_convert:
                greyscale = self.motion.visu[0, ...].mean(axis=-1)
            else:
                greyscale = self.motion.converted_video[0, ...]
            self.NetDet = NetworkDetection(greyscale, possibly_filled_pixels=self.motion.binary[0, ...],
                                           origin_to_add=self.origin,
                                           morphological_closing=self.motion.vars['morphological_closing'])
            self.NetDet.get_best_network_detection_method(include_images=False)
        else:
            if self.dims[0] < 20:
                frames = list(range(self.dims[0] - 9, self.dims[0]))
            else:
                tot = self.dims[0]
                deno = self.dims[0] // 6
                frames = [4 * deno - 3, 4 * deno - 2, 4 * deno - 1, 5 * deno - 1, 5 * deno, 5 * deno + 1, tot - 3, tot - 2, tot - 1]
            metric_table = np.zeros((len(frames) * 24, 5))
            metric_table[:, 2] = np.repeat(frames, 24)
            metric_table[:, 3] = np.tile(np.arange(24), len(frames))
            for f_i, frame in enumerate(frames):
                if self.do_convert:
                    greyscale = self.motion.visu[frame, ...].mean(axis=-1)
                else:
                    greyscale = self.motion.converted_video[frame, ...]
                self.NetDet = NetworkDetection(greyscale, possibly_filled_pixels=self.motion.binary[frame, ...],
                                               origin_to_add=self.origin,
                                               morphological_closing=self.motion.vars['morphological_closing'])
                self.NetDet.get_best_network_detection_method(include_images=False)
                start = f_i * len(self.NetDet.quality_metrics)
                end = (f_i + 1) * len(self.NetDet.quality_metrics)
                metric_table[start:end, :2] = self.NetDet.quality_metrics

            areas = metric_table[:, 1].reshape(3, 3, 24)
            area_variation = (areas.max(axis=1) - areas.min(axis=1)) / areas.mean(axis=1)
            valid_methods = np.all(area_variation <= max_surface_area_variation, axis=0)
            m_i = 0
            m_i_max = - np.log(max_surface_area_variation) / np.log(2)
            while valid_methods.sum() < 3 and m_i < m_i_max:
                max_surface_area_variation *= 2
                valid_methods = np.all(area_variation <= max_surface_area_variation, axis=0)
                m_i += 1

            scores = metric_table[:, 0].reshape(3, 3, 24)
            mean_scores = scores.mean(axis=(0, 1))
            if valid_methods.any():
                sorted_idx = list(np.argsort(mean_scores)[::-1])
                while not valid_methods[sorted_idx[0]]:
                    sorted_idx.pop(0)
                best_idx = sorted_idx[0]
            else:
                best_idx = np.argmax(mean_scores)
            self.NetDet.best_idx = best_idx
            self.NetDet.best_result = self.NetDet.all_results[self.NetDet.best_idx]
            # display_network_methods(self.NetDet)
        if self.do_convert:
            self.NetDet.change_greyscale(self.motion.converted_video[-1, ...])

    def frame_by_frame_tracking(self) -> NDArray[np.uint8]:
        """
        Summary
        -------
        Iterate network detection over video frames and return the network segmentation of the final
        frame.

        Returns
        -------
        complete_network: ndarray of uint8
            The segmentation network produced for the last processed frame.
        """
        for t in np.arange(self.starting_time, self.dims[0]):
            complete_network = self.segment_frame(t)
        return complete_network
            
    def segment_frame(self, t: int) -> NDArray[np.uint8]:
        """
        Segment a single frame into a network mask.

        Parameters
        ----------
        t: int
            Index of the frame to process.

        Returns
        -------
        complete_network: ndarray of uint8
            Binary network mask for frame ``t``.

        Notes
        -----
        The method updates internal buffers:
        ``self.potential_network`` stores the detected network,
        ``self.pseudopod_vid`` is filled when ``detect_pseudopods`` is ``True``,
        and ``NetDet_fast.greyscale_image`` receives the greyscale frame.
        """
        if self.do_convert:
            greyscale = self.motion.visu[t, ...].mean(axis=-1)
        else:
            greyscale = self.motion.converted_video[t, ...]
        NetDet_fast = NetworkDetection(greyscale, possibly_filled_pixels=self.motion.binary[t, ...],
                                       origin_to_add=self.origin,
                                       edge_max_width=self.edge_max_width,
                                       morphological_closing=self.motion.vars['morphological_closing'],
                                       best_result=self.NetDet.best_result)
        NetDet_fast.detect_network()
        if self.detect_pseudopods:
            if self.do_convert:
                NetDet_fast.change_greyscale(self.motion.converted_video[t, ...])
            NetDet_fast.detect_pseudopods(pseudopod_min_size=self.pseudopod_min_size)
            self.pseudopod_vid[t, ...] = NetDet_fast.pseudopods
        else:
            NetDet_fast.complete_network = NetDet_fast.incomplete_network
        self.potential_network[t, ...] = NetDet_fast.complete_network
        return NetDet_fast.complete_network
        
    def post_processing(self):
        """
        Apply post‑processing to all time steps from ``starting_time`` up to the
        last index of the time dimension.

        Returns
        -------
        None
            The method updates internal state; it does not return a value.

        Notes
        -----
        For each ``t`` the private method ``post_process`` is invoked to perform
        dynamic improvement of the segmentation.
        """
        for t in np.arange(self.starting_time, self.dims[0]):
            imtoshow = self.post_process(t)
            
    def post_process(self, t: int, shape_disappearance_supremum: int=500) -> NDArray[np.uint8]:
        """
        Post‑process a single time‑frame of the network and return a visualisation image.

        Parameters
        ----------
        t: int
            Index of the time‑frame to be processed. Must be a non‑negative integer
            within the range of the video sequence.
        shape_disappearance_supremum: int
            All connected components having an inferior surface area (in pixels), are allowed to disappear from one frame to the next.

        Returns
        -------
        imtoshow: ndarray of uint8
            Visualises the processed network overlayed on the original frame.

        Notes
        -----
        * If ``self.motion.vars['sliding_window_segmentation']`` is ``True``, a
          five‑frame sliding window centred on ``t`` is summed to obtain a provisional
          network.  Edge cases (``t < 2`` or ``t > self.dims[0] - 3``) use a truncated
          window.  The provisional network is binarised so that any pixel with a value
          greater than ``1`` becomes ``1`` and isolated single‑pixel detections are
          removed.

        * When ``self.origin`` is provided, its contour is merged with the provisional
          network.  The original shape is discarded and only its contour contributes
          to the final network.

        * ``keep_one_connected_component`` guarantees that the resulting binary mask
          contains a single 8‑connected component, removing stray islands.

        * If ``self.detect_pseudopods`` is ``True``, pseudopods from the current frame
          (``self.pseudopod_vid[t]``) are temporarily added back to the network before
          growth‑control checks.  The maximal allowed decrease in pixel count is
          computed from the previous frame (``self.network_dynamics[t‑1]``) scaled by
          ``1 + self.motion.vars['maximal_growth_factor']``.  Should the network fall
          below this threshold, missing pieces that belong to large connected
          components are reinstated.

        * After growth control, large growing regions that are not part of the
          main network are identified as pseudopods.  These are stored in
          ``self.pseudopod_vid[t]`` while ensuring that removal of pseudopods does not
          fragment the remaining network.

        * The visualisation image is created by eroding the final binary network with
          a 3×3 cross‑shaped structuring element (``cross_33``) and highlighting the
          resulting boundary pixels in a distinct colour.

        * This method mutates ``self.network_dynamics`` and, when enabled,
          ``self.pseudopod_vid`` as side‑effects.
        """
        if self.motion.vars['sliding_window_segmentation']:
            if 2 <= t <= (self.dims[0] - 2):
                computed_network = self.potential_network[(t - 2):(t + 3), :, :].sum(axis=0)
                computed_network[computed_network == 1] = 0
                computed_network[computed_network > 1] = 1
            else:
                if t < 2:
                    computed_network = self.potential_network[:2, :, :].sum(axis=0)
                else:
                    computed_network = self.potential_network[-2:, :, :].sum(axis=0)
                computed_network[computed_network > 0] = 1
        else:
            computed_network = self.potential_network[t, :, :].copy()

        complete_network = keep_one_connected_component(computed_network)

        # Impede large parts of the network to disappear from one frame to the next
        current_network = complete_network.copy()
        if t > self.starting_time:
            prev_network = self.network_dynamics[t - 1]
        else:
            prev_network = np.zeros_like(self.network_dynamics[t], dtype=np.uint8)
        # minimal_network_size = prev_network.sum() * allowed_shrinkage_rate
        # if current_network.sum() < minimal_network_size:
        mising_pieces = ((1 - current_network) * prev_network).astype(np.uint8)
        nb, sh, stats, centroids = cv2.connectedComponentsWithStats(mising_pieces)
        large_shapes = stats[1:, 4] > shape_disappearance_supremum
        if large_shapes.any():
            large_shapes = np.nonzero(large_shapes)[0] + 1
            for large_shape in large_shapes:
                complete_network[sh == large_shape] = 1

        # Discriminate large growing regions from the rest of the network
        if self.detect_pseudopods:
            complete_network = np.logical_or(complete_network, self.pseudopod_vid[t])
            complete_network = keep_one_connected_component(complete_network)
            # Make sure that removing pseudopods do not cut the network:
            without_pseudopods = complete_network * (1 - self.pseudopod_vid[t])
            only_connected_network = keep_one_connected_component(without_pseudopods)
            # # Option A: To add these cutting regions to the pseudopods do:
            pseudopods = (1 - only_connected_network) * complete_network
            self.pseudopod_vid[t] = pseudopods
        self.network_dynamics[t] = complete_network

        imtoshow = self.motion.visu[t, ...]
        eroded_binary = cv2.erode(self.network_dynamics[t, ...], cross_33)
        net_coord = np.nonzero(self.network_dynamics[t, ...] - eroded_binary)
        imtoshow[net_coord[0], net_coord[1], :] = (34, 34, 158)
        return imtoshow
    
    def save_network(self):
        """
        Save the coordinates of the network and, optionally, pseudopods to HDF5 files.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray or None)
            - ``network_coord``: Array of indices where ``self.network_dynamics`` is
              non‑zero, stored as an unsigned integer array.
            - ``pseudopod_coord``: Array of indices where ``self.pseudopod_vid`` is
              non‑zero (``uint``) if pseudopods are detected; otherwise ``None``.

        Notes
        -----
        * If ``self.detect_pseudopods`` is ``True``, cells identified as
          pseudopods are marked with the value ``2`` in ``self.network_dynamics``.
        * Coordinate arrays are written to HDF5 files when
          ``self.motion.vars['save_coord_network']`` is ``True``.  The filenames
          embed the arena identifier and the dimensions ``t``, ``y``, and ``x``.
        * The function relies on ``smallest_memory_array`` to cast the result of
          ``np.nonzero`` to the minimal unsigned‑integer representation.
        """
        network_coord = smallest_memory_array(np.nonzero(self.network_dynamics), "uint")
        pseudopod_coord = None
        if self.detect_pseudopods:
            self.network_dynamics[self.pseudopod_vid] = 2
            pseudopod_coord = smallest_memory_array(np.nonzero(self.pseudopod_vid), "uint")
            if self.motion.vars['save_coord_network']:
                write_h5(f"coord_pseudopods{self.motion.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.h5", pseudopod_coord)
        if self.motion.vars['save_coord_network']:
            write_h5(f"coord_network{self.motion.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.h5", network_coord)
        return network_coord, pseudopod_coord
    

