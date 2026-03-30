#!/usr/bin/env python3
"""
Generate colored visualizations from binary masks.

This module offers utilities that turns a 2‑D or 3‑D binary mask into
an RGB array. Supports 2‑D and 3‑D masks.

Functions
---------
colorize_mask : Convert a binary mask to a color RGB array.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def colorize_mask(mask, blob_to_back_diff: int=100, blob_extent: int=20, back_extent: int=20) -> NDArray[np.uint8]:
    """
    Generate an RGB image from a binary mask with random foreground and background colors.

    Parameters
    ----------
    mask
        Binary mask indicating foreground (`> 0`) and background (`== 0`). Can be 2‑D
        (height × width) or 3‑D (height × width × depth).
    blob_to_back_diff : int, optional
        Minimum absolute difference between foreground and background colors per
        channel. Default is ``100``.
    blob_extent : int, optional
        Width of the random color range for foreground (blob) pixels. The actual
        per‑channel variation is half of this value. Default is ``20``.
    back_extent : int, optional
        Width of the random color range for background pixels. The actual
        per‑channel variation is half of this value. Default is ``20``.*

    Returns
    -------
    rgb_from_mask : uint8
        RGB image with the same spatial dimensions as ``mask`` and an additional
        final color dimension of size three. Foreground pixels receive random colors
        within the computed ``blob`` ranges, while background pixels receive random
        colors within the computed ``back`` ranges.

    Notes
    -----
    * Random colors are sampled independently for each channel and each call, so
      invoking the function repeatedly with the same ``mask`` will yield different
      results.
    * The function supports both 2‑D and 3‑D masks; the output shape mirrors the input
      spatial dimensions and appends a channel axis.

    Examples
    --------
    >>> bin_mask = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    >>> rgb_im = colorize_mask(bin_mask)
    >>> rgb_im.shape
    (3, 3, 3)
    >>> # Foreground pixel (value > 0)
    >>> rgb_im[:, :, 0]
    array([[ 53, 162,  62],
           [162, 168,  60],
           [ 65,  58,  66]], dtype=uint8) # Higher values where there were ones on the mask
    """
    blob_diff = blob_extent // 2
    back_diff = back_extent // 2
    blob_c = []
    blob_c_min = []
    blob_c_max = []
    back_c = []
    back_c_min = []
    back_c_max = []
    for i in range(3):
        pot_bc = np.random.choice((np.random.randint(0, 255 - blob_to_back_diff), np.random.randint(blob_to_back_diff + 1, 256)))
        blob_c.append(pot_bc)
        if blob_c[i] <= blob_to_back_diff:
            # print(blob_c[i])
            back_c.append(np.random.randint(blob_c[i] + blob_to_back_diff, 256))
        elif blob_c[i] > 255 - blob_to_back_diff:
            back_c.append(np.random.randint(0 , blob_c[i] - blob_to_back_diff))
        else:
            back_c.append(np.random.choice((np.random.randint(blob_c[i] + blob_to_back_diff , 256), np.random.randint(0, blob_c[i] - blob_to_back_diff))))
        blob_c_min.append(np.max((blob_c[i] - blob_diff, 0)))
        blob_c_max.append(np.min((blob_c[i] + blob_diff, 255)))
        back_c_min.append(np.max((back_c[i] - back_diff, 0)))
        back_c_max.append(np.min((back_c[i] + back_diff, 255)))
        # print(f'back: {(back_c_min[i], back_c_max[i])}, blob: {(blob_c_min[i], blob_c_max[i])}')
    if len(mask.shape) == 2:
        rgb_from_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    elif len(mask.shape) == 3:
        rgb_from_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    else:
        raise ValueError('Invalid mask shape')
    for c_, (blob_min, blob_max, back_min, back_max) in enumerate(zip(blob_c_min, blob_c_max, back_c_min, back_c_max)):
        if len(mask.shape) == 2:
            rgb_from_mask[:, :, c_][mask > 0] = np.random.randint(blob_min, blob_max, mask.sum())
            rgb_from_mask[:, :, c_][mask == 0] = np.random.randint(back_min, back_max, mask.size - mask.sum())
        elif len(mask.shape) == 3:
            rgb_from_mask[:, :, :, c_][mask > 0] = np.random.randint(blob_min, blob_max, mask.sum())
            rgb_from_mask[:, :, :, c_][mask == 0] = np.random.randint(back_min, back_max, mask.size - mask.sum())
    return rgb_from_mask