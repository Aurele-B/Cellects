#!/usr/bin/env python3
"""
Generate synthetic videos of moving cells with noisy circular patterns.


Each cell is a square region filled with a noisy circular intensity pattern; cells are
initialised at random positions and then displaced outward from the
image centre over a configurable number of frames. The resulting video
is returned as a NumPy ``uint8`` array, optionally visualised frame‑by‑frame.

Functions
---------
moving_cells : Generate a synthetic video of moving square cells.
"""

import numpy as np
from numpy.typing import NDArray
from cellects.simulation.shapes import sim_noisy_circle, random_blob_coord
from cellects.simulation.coloring import colorize_mask
from cellects.display.image import show

def moving_cells(im_size: int=1000, cell_size: int=50, cell_nb: int=25, frame_nb: int=20, delta: int=12, display: bool=False) -> NDArray[np.uint8]:
    """
    Generate a synthetic video of moving square cells.

    The function creates ``cell_nb`` square cells with noisy circular
    intensity patterns, places them at random positions inside an
    ``im_size``×``im_size`` field, and moves each cell away from the
    image centre by ``delta`` pixels per frame for ``frame_nb``
    frames.  An optional visualization of each frame can be shown
    during generation.

    Parameters
    ----------
    im_size : int
        Width and height of each video frame (in pixels).  The video is
        square, so the total size per frame is ``(im_size, im_size)``.
    cell_size : int
        Side length of the square region that each cell occupies.
    cell_nb : int
        Number of cells to generate.
    frame_nb : int
        Number of frames in the output video.
    delta : int
        Approximate displacement (in pixels) of each cell per frame,
        measured along the unit vector pointing away from the image
        centre.
    display : bool
        If ``True``, displays each generated frame using the ``show``
        function.

    Returns
    -------
    video : NDArray[np.uint8]
        Array of shape ``(frame_nb, im_size, im_size)`` containing the
        generated video.  Each slice ``video[t]`` is a single‑frame
        ``uint8`` image.

    Notes
    -----
    * The random number generator is seeded with ``42`` for reproducible
      results.
    * Cells are moved using a vector normalised to unit length; the
      displacement is rounded to the nearest integer pixel.

    Examples
    --------
    >>> vid = moving_cells()
    >>> print(vid.shape)
    (20, 1000, 1000)
    >>> print(vid.dtype)
    uint8
    """
    np.random.seed(42)
    cy, cx = im_size // 2, im_size // 2
    moving_cells_video = np.zeros((frame_nb, im_size, im_size), dtype=np.uint8)
    cells = [sim_noisy_circle(cell_size) for cell in range(cell_nb)]
    coordinates = random_blob_coord(im_size // 2, cell_size, cell_nb).astype(np.int32) + im_size // 4
    for t_ in range(frame_nb):
        for c_, coord in enumerate(coordinates):
            moving_cells_video[t_, coord[0]:(coord[0] + cell_size), coord[1]:(coord[1] + cell_size)] = cells[c_]
        vect_from_centre = coordinates - [cy, cx]
        dist_from_centre = np.sqrt(vect_from_centre[:, 0] ** 2 + vect_from_centre[:, 1] ** 2)
        unit_dir_from_centre = np.zeros_like(vect_from_centre, float)
        np.divide(vect_from_centre[:, 0], dist_from_centre, out=unit_dir_from_centre[:, 0], where=dist_from_centre != 0)
        np.divide(vect_from_centre[:, 1], dist_from_centre, out=unit_dir_from_centre[:, 1], where=dist_from_centre != 0)
        coordinates += np.round(unit_dir_from_centre * delta).astype(np.int32)
        if display:
            show(moving_cells_video[t_])
    rgb_video = colorize_mask(moving_cells_video)
    return rgb_video