#!/usr/bin/env python3
"""Module for tracking oscillatory dynamics in processed video arenas.

The module defines :class:`OscillationsTracking`, which computes a signed
oscillation map for each frame of a pre‑processed video (``motion.
converted_video``).  The analysis normalises frames by their average
intensity, estimates gradients over an expected oscillation period, and
stores the result as an ``int8`` array where ``1`` denotes upward motion,
``-1`` downward motion, and ``0`` background.  Non‑specimen regions are masked using the
``motion.binary`` mask.

Classes
-------
OscillationsTracking : Detects oscillatory dynamics in a labeled arena from
processed video data.

Notes
-----
* Relies on :pymod:`numpy`, :pymod:`psutil`, and utilities from the
  ``cellects`` package.
* Memory‑intensive operations switch to a ``float16`` element‑wise loop when
  free RAM is insufficient.
* Uses ``numpy.gradient`` for gradient estimation.
"""
import numpy as np
from numpy.typing import NDArray
import logging
from cellects.image.morphological_operations import cc, CompareNeighborsWithValue
from cellects.io.save import write_h5
from cellects.utils.utilitarian import smallest_memory_array
from psutil import virtual_memory


class OscillationsTracking:
    """
    Detects oscillatory dynamics in a labeled arena from processed video data.

    The class analyses a pre‑processed video (stored in ``motion.converted_video``)
    to compute a signed oscillation map for each frame.  Memory‑intensive
    operations are performed in ``float16`` when the required RAM exceeds the
    available amount, and the resulting video can optionally be saved as
    coordinate arrays.

    Parameters
    ----------
    motion : object
        Container providing the processed video and associated metadata.
        Required attributes are ``converted_video``, ``vars``, ``binary`` and
        ``one_descriptor_per_arena`` which are accessed throughout the
        analysis.

    Attributes
    ----------
    motion : object
        Reference to the motion data passed at construction.
    starting_time : int
        Frame index at which tracking starts; set by :meth:`init_tracking`.
    dims : tuple of int
        Shape of the video array ``(n_frames, height, width[, channels])``.
    oscillations_video : numpy.ndarray
        Signed oscillation map (``int8``) for all frames; populated by
        :meth:`init_tracking`.

    Notes
    -----
    * Uses ``numpy.gradient`` to estimate intensity changes over the expected
      oscillation period.
    * When the estimated memory usage exceeds the free RAM (or when the user
      requests ``lose_accuracy_to_save_memory``) the computation falls back
      to a slower, element‑wise loop with ``float16`` storage to reduce the
      memory footprint.
    """
    def __init__(self, motion: object):
        """
        Summary
        -------
        Initialize the object with a motion analysis instance.

        Parameters
        ----------
        motion
            An instance of :class:`MotionAnalysis` that provides the motion data to be
            associated with this object.
        """
        self.motion = motion

    def init_tracking(self):
        """
        Summary
        -------
        Initialize oscillation tracking for the current arena.

        Returns
        -------
        None
            The method populates ``self.oscillations_video`` and updates internal
            attributes; it does not return a value.

        Notes
        -----
        * The method estimates the required memory for the full‑resolution
          oscillation video. If the estimate exceeds the available RAM (or if the
          ``lose_accuracy_to_save_memory`` flag is set), the computation is performed
          slice‑by‑slice using ``np.float16`` to reduce memory usage.
        * When sufficient memory is available, the oscillation video is computed in a
          single call to ``np.gradient`` for speed.
        * The gradient is taken over ``period_in_frame_nb`` frames, which is the
          expected oscillation period expressed in frame numbers.
        * The resulting gradient is rounded to three decimal places, converted to
          ``np.int8`` where positive values become ``1`` (upward motion), negative
          values become ``-1`` (downward motion), and zero remains ``0``.
        * Pixels for which ``self.motion.binary == 0`` are forced to ``0`` to mask
          out non‑specimen regions.
        """
        logging.info(f"Arena n°{self.motion.one_descriptor_per_arena['arena']}. Starting oscillation analysis.")
        self.starting_time = 0
        self.dims = self.motion.converted_video.shape
        if self.dims[0] == 1:
            self.oscillations_video = np.zeros(self.dims[:3], dtype=np.float64)
        else:
            self.oscillations_video = None
            period_in_frame_nb = int(self.motion.vars['expected_oscillation_period'] / self.motion.time_interval)
            if period_in_frame_nb < 2:
                period_in_frame_nb = 2
            necessary_memory = self.dims[0] * self.dims[1] * self.dims[2] * 64 * 4 * 1.16415e-10
            available_memory = virtual_memory().available / (1024 ** 3) - self.motion.vars['min_ram_free']
            if len(self.dims) == 4:
                self.motion.converted_video = self.motion.converted_video[:, :, :, 0]
            average_intensities = np.mean(self.motion.converted_video, (1, 2))
            if self.motion.vars['lose_accuracy_to_save_memory'] or (necessary_memory > available_memory):
                self.oscillations_video = np.zeros(self.dims[:3], dtype=np.float16)
                for cy in np.arange(self.dims[1]):
                    for cx in np.arange(self.dims[2]):
                        self.oscillations_video[:, cy, cx] = np.round(
                            np.gradient(self.motion.converted_video[:, cy, cx, ...] / average_intensities,
                                        period_in_frame_nb),
                            3).astype(np.float16)
            else:
                self.oscillations_video = np.gradient(self.motion.converted_video / average_intensities[:, None, None],
                                                 period_in_frame_nb, axis=0)
            self.oscillations_video = np.sign(self.oscillations_video)
            self.oscillations_video = self.oscillations_video.astype(np.int8)
            self.oscillations_video[self.motion.binary == 0] = 0
        
    def frame_by_frame_tracking(self) -> NDArray[np.uint8]:
        """
        Compute oscillations for each frame from ``starting_time`` to the end of the
        data set and return the result from the last processed frame.

        The method iterates over the time axis and calls
        :meth:`find_oscillations_in_frame` for each time step.

        Returns
        -------
        oscillations_image: ndarray of uint8
            The oscillations image computed for the last frame processed in the
            loop.
        """
        for t in np.arange(self.self.starting_time, self.dims[0]):
            oscillations_image = self.find_oscillations_in_frame(t)
        return oscillations_image
    
    def find_oscillations_in_frame(self, t: int) -> NDArray[np.uint8]:
        """
        Find oscillations in a single video frame.

        Parameters
        ----------
        t: int
            Index of the frame to process.

        Returns
        -------
        oscillations_image: ndarray of uint8
            Values are ``0`` for background, ``1`` for influx clusters, and ``2``
            for efflux clusters after neighbor‑count and size filtering.

        Notes
        -----
        * The function updates ``self.oscillations_video[t, :, :]`` in place.
        * A pixel is classified as influx (resp. efflux) only if it has at
          least four positive (resp. negative) 8‑connected neighbors.
        * Connected‑component labeling is performed via ``cc``.
        * Clusters smaller than
          ``self.motion.vars['minimal_oscillating_cluster_size']`` pixels are
          discarded as noise.
        """
        oscillations_image = np.zeros(self.dims[1:3], np.uint8)
        # Add in or ef if a pixel has at least 4 neighbor in or ef
        neigh_comp = CompareNeighborsWithValue(self.oscillations_video[t, :, :], connectivity=8, data_type=np.int8)
        neigh_comp.is_inf(0, and_itself=False)
        neigh_comp.is_sup(0, and_itself=False)
        # Not verified if influx is really influx (resp efflux)
        influx = neigh_comp.sup_neighbor_nb
        efflux = neigh_comp.inf_neighbor_nb

        # Only keep pixels having at least 4 positive (resp. negative) neighbors
        influx[influx <= 4] = 0
        efflux[efflux <= 4] = 0
        influx[influx > 4] = 1
        efflux[efflux > 4] = 1
        if np.any(influx) or np.any(efflux):
            influx, in_stats, in_centroids = cc(influx)
            efflux, ef_stats, ef_centroids = cc(efflux)
            # Only keep clusters larger than 'minimal_oscillating_cluster_size' pixels (smaller are considered as noise
            in_smalls = np.nonzero(in_stats[:, 4] < self.motion.vars['minimal_oscillating_cluster_size'])[0]
            if len(in_smalls) > 0:
                influx[np.isin(influx, in_smalls)] = 0
            ef_smalls = np.nonzero(ef_stats[:, 4] < self.motion.vars['minimal_oscillating_cluster_size'])[0]
            if len(ef_smalls) > 0:
                efflux[np.isin(efflux, ef_smalls)] = 0
            oscillations_image[influx > 0] = 1
            oscillations_image[efflux > 0] = 2
        self.oscillations_video[t, :, :] = oscillations_image
        return oscillations_image
    
    def save_oscillations(self):
        """
        Summary
        -------
        Zero out oscillation data prior to the starting time and, if enabled, write
        coordinate masks for thickening and slimming oscillations to HDF5 files.

        Extended Description
        --------------------
        When the motion variable ``save_coord_thickening_slimming`` is ``True``,
        two HDF5 files are created:
            * a file containing the coordinates where the video equals ``1`` (thickening
              oscillations);
            * a file containing the coordinates where the video equals ``2`` (slimming
              oscillations).

        Both files are named using the arena descriptor and the dimensions of the
        video (``t``, ``y``, ``x``).  The coordinate arrays are first reduced to the
        smallest possible memory footprint via ``smallest_memory_array`` before being
        written with ``write_h5``.

        Returns
        -------
        None
            The function performs its work in‑place and does not return a value.

        Raises
        ------
        IOError
            If writing either HDF5 file fails (e.g., due to filesystem permissions).

        Notes
        -----
        * The function assumes that ``self.oscillations_video`` is a NumPy array with
          shape ``(t, y, x)`` and integer values.
        """
        self.oscillations_video[:self.starting_time, :, :] = 0
        if self.motion.vars['save_coord_thickening_slimming']:
            write_h5(
                f"coord_thickening{self.motion.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.h5",
                smallest_memory_array(np.nonzero(self.oscillations_video == 1), "uint"))
            write_h5(
                f"coord_slimming{self.motion.one_descriptor_per_arena['arena']}_t{self.dims[0]}_y{self.dims[1]}_x{self.dims[2]}.h5",
                smallest_memory_array(np.nonzero(self.oscillations_video == 2), "uint"))
