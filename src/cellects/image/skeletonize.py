"""
Medial-axis and Lee skeletonization for 2-D and 3-D binary images.

This module provides two public algorithms:

- ``medial_axis`` computes the 2-D medial axis from a binary image.
- ``skeletonize`` computes a Lee skeleton of a 2-D or 3-D binary image.

Both functions accept array-like binary images. Nonzero values are treated
as foreground, and returned arrays use boolean dtype.

Functions
---------
medial_axis : Compute a 2-D medial axis and optional distance transform.
skeletonize : Compute a Lee skeleton of a 2-D or 3-D binary image.

Notes
-----
The medial-axis implementation targets scikit-image 0.26.0 with
``mask=None`` and ``rng=0``. The Lee implementation targets scikit-image
0.26.0's Lee method. Numba-compiled helpers are used for inner loops.
"""
from __future__ import annotations
import numpy as np
from cellects.utils.decorators import njit
from scipy.ndimage import (
    distance_transform_edt,
    generate_binary_structure,
    label,
)

_EIGHT_CONNECT = generate_binary_structure(2, 2)


def _pattern_of(index: int) -> np.ndarray:
    """Return the 3x3 boolean pattern represented by a 9-bit index."""
    return np.array(
        [
            [
                bool(index & (1 << 0)),
                bool(index & (1 << 1)),
                bool(index & (1 << 2)),
            ],
            [
                bool(index & (1 << 3)),
                bool(index & (1 << 4)),
                bool(index & (1 << 5)),
            ],
            [
                bool(index & (1 << 6)),
                bool(index & (1 << 7)),
                bool(index & (1 << 8)),
            ],
        ],
        dtype=bool,
    )


def _make_medial_axis_table() -> np.ndarray:
    """
    Build the 512-entry medial-axis lookup table.

    A foreground center is retained iff deleting it changes connectivity
    or the 3x3 neighborhood contains fewer than three foreground pixels.
    """
    table = np.empty(512, dtype=np.uint8)

    for index in range(512):
        pattern = _pattern_of(index)

        if not pattern[1, 1]:
            table[index] = 0
            continue

        before = label(
            pattern,
            structure=_EIGHT_CONNECT,
        )[1]

        without_center = pattern.copy()
        without_center[1, 1] = False

        after = label(
            without_center,
            structure=_EIGHT_CONNECT,
        )[1]

        table[index] = np.uint8(
            before != after or np.sum(pattern) < 3
        )

    return np.ascontiguousarray(table)


_MEDIAL_AXIS_TABLE = _make_medial_axis_table()


@njit(nogil=True, cache=True)
def _ma_corner_score(image):
    """
    Number of background pixels in each 3x3 neighborhood.

    This is equivalent to the cornerness lookup used by scikit-image.
    """
    rows, cols = image.shape
    score = np.empty((rows, cols), dtype=np.int16)

    for y in range(rows):
        for x in range(cols):
            foreground = 0

            if y > 0:
                if x > 0:
                    foreground += image[y - 1, x - 1]

                foreground += image[y - 1, x]

                if x + 1 < cols:
                    foreground += image[y - 1, x + 1]

            if x > 0:
                foreground += image[y, x - 1]

            foreground += image[y, x]

            if x + 1 < cols:
                foreground += image[y, x + 1]

            if y + 1 < rows:
                if x > 0:
                    foreground += image[y + 1, x - 1]

                foreground += image[y + 1, x]

                if x + 1 < cols:
                    foreground += image[y + 1, x + 1]

            score[y, x] = 9 - foreground

    return score


@njit(nogil=True, cache=True)
def _ma_skeletonize_loop(
    result,
    ii,
    jj,
    order,
    table,
):
    """
    Numba implementation of the medial-axis deletion loop.
    """
    rows, cols = result.shape

    for k in range(order.size):
        q = order[k]

        y = ii[q]
        x = jj[q]

        index = 16  # center pixel

        if y > 0:
            if x > 0 and result[y - 1, x - 1]:
                index += 1

            if result[y - 1, x]:
                index += 2

            if x + 1 < cols and result[y - 1, x + 1]:
                index += 4

        if x > 0 and result[y, x - 1]:
            index += 8

        if x + 1 < cols and result[y, x + 1]:
            index += 32

        if y + 1 < rows:
            if x > 0 and result[y + 1, x - 1]:
                index += 64

            if result[y + 1, x]:
                index += 128

            if x + 1 < cols and result[y + 1, x + 1]:
                index += 256

        result[y, x] = table[index]


def medial_axis(image, return_distance=False):
    """
    Summary
    -------
    Compute the 2-D medial axis of a binary image.

    Extended Description
    --------------------
    The medial axis is obtained by deleting foreground pixels whose removal
    does not change local connectivity and whose 3x3 neighborhood contains
    at least three foreground pixels. Pixels are visited in order of
    increasing distance to background, then cornerness, then a fixed random
    tie-breaker.

    Parameters
    ----------
    image : array_like
        2-D binary image. Nonzero values are foreground.
    return_distance : bool
        If ``True``, also return the Euclidean distance transform.
        The default is ``False``.

    Returns
    -------
    skeleton : ndarray of bool
        Medial axis with the same shape as `image`.
    distance : ndarray of float64
        Euclidean distance transform of `image`. This object is returned
        only when `return_distance` is ``True``. When `return_distance`
        is ``False``, only `skeleton` is returned.

    Raises
    ------
    ValueError
        If `image` is not a 2-D array.

    Notes
    -----
    This implementation targets scikit-image 0.26.0 with ``mask=None``
    and ``rng=0``. The random tie-breaker is deterministic because the
    random generator is seeded with 0. Numba-compiled helpers are used for
    the corner score and deletion loop.

    Examples
    --------
    >>> skeleton, distance = medial_axis(image, return_distance=True)
    >>> skeleton = medial_axis(image)
    >>> print(skeleton.shape)
    # (3, 5)
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("image must be a 2-D array")

    image = np.ascontiguousarray(
        image.astype(bool, copy=False)
    )

    distance_full = distance_transform_edt(image)

    if not image.any():
        result = np.zeros_like(image, dtype=bool)

        if return_distance:
            return result, distance_full

        return result

    corner_score = _ma_corner_score(image)

    ii, jj = np.nonzero(image)

    ii = np.ascontiguousarray(ii, dtype=np.intp)
    jj = np.ascontiguousarray(jj, dtype=np.intp)

    distance = np.ascontiguousarray(
        distance_full[image],
        dtype=np.float64,
    )

    n = ii.size

    # Fixed compatibility target: rng=0.
    rng = np.random.default_rng(0)

    tiebreaker = rng.permutation(
        np.arange(n)
    )

    # Primary: distance
    # Secondary: cornerness
    # Tertiary: random tiebreaker
    order = np.lexsort(
        (
            tiebreaker,
            corner_score[image],
            distance,
        )
    )

    order = np.ascontiguousarray(
        order,
        dtype=np.intp,
    )

    result = np.ascontiguousarray(
        image,
        dtype=np.uint8,
    )

    _ma_skeletonize_loop(
        result,
        ii,
        jj,
        order,
        _MEDIAL_AXIS_TABLE,
    )

    result = result.astype(bool)

    if return_distance:
        return result, distance_full

    return result


_LEE_EULER_LUT = np.array(
    [
         1, -1, -1,  1, -3, -1, -1,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,
        -3, -1,  3,  1,  1, -1,  3,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,

        -3,  3, -1,  1,  1,  3, -1,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,
         1,  3,  3,  1,  5,  3,  3,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,

        -7, -1, -1,  1, -3, -1, -1,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,
        -3, -1,  3,  1,  1, -1,  3,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,

        -3,  3, -1,  1,  1,  3, -1,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,
         1,  3,  3,  1,  5,  3,  3,  1,
        -1,  1,  1, -1,  3,  1,  1, -1,
    ],
    dtype=np.int32,
)

assert _LEE_EULER_LUT.size == 128

_LEE_LUT = np.zeros(256, dtype=np.int32)
_LEE_LUT[1::2] = _LEE_EULER_LUT
_LEE_LUT = np.ascontiguousarray(_LEE_LUT)


_LEE_OCTANT_INDICES = np.ascontiguousarray(
    np.array(
        [
            [2, 1, 11, 10, 5, 4, 14],
            [0, 9, 3, 12, 1, 10, 4],
            [8, 7, 17, 16, 5, 4, 14],
            [6, 15, 7, 16, 3, 12, 4],
            [20, 23, 19, 22, 11, 14, 10],
            [18, 21, 9, 12, 19, 22, 10],
            [26, 23, 17, 14, 25, 22, 16],
            [24, 25, 15, 16, 21, 22, 12],
        ],
        dtype=np.int64,
    )
)

_LEE_OCTREE_INDICES = np.ascontiguousarray(
    np.array(
        [
            [0, 1, 3, 4, 9, 10, 12],
            [1, 4, 10, 2, 5, 11, 13],
            [3, 4, 12, 6, 7, 14, 15],
            [4, 5, 13, 7, 15, 8, 16],
            [9, 10, 12, 17, 18, 20, 21],
            [10, 11, 13, 18, 21, 19, 22],
            [12, 14, 15, 20, 21, 23, 24],
            [13, 15, 16, 21, 22, 24, 25],
        ],
        dtype=np.int8,
    )
)


_LEE_OCTREE_NEXT = np.full(
    (8, 7, 3),
    -1,
    dtype=np.int8,
)


_LEE_OCTREE_NEXT[0] = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [2, -1, -1],
        [1, 2, 3],
        [4, -1, -1],
        [1, 4, 5],
        [2, 4, 6],
    ],
    dtype=np.int8,
)

_LEE_OCTREE_NEXT[1] = np.array(
    [
        [0, -1, -1],
        [0, 2, 3],
        [0, 4, 5],
        [-1, -1, -1],
        [3, -1, -1],
        [5, -1, -1],
        [3, 5, 7],
    ],
    dtype=np.int8,
)

_LEE_OCTREE_NEXT[2] = np.array(
    [
        [0, -1, -1],
        [0, 1, 3],
        [0, 4, 6],
        [-1, -1, -1],
        [3, -1, -1],
        [6, -1, -1],
        [3, 6, 7],
    ],
    dtype=np.int8,
)

_LEE_OCTREE_NEXT[3] = np.array(
    [
        [0, 1, 2],
        [1, -1, -1],
        [1, 5, 7],
        [2, -1, -1],
        [2, 6, 7],
        [-1, -1, -1],
        [7, -1, -1],
    ],
    dtype=np.int8,
)

_LEE_OCTREE_NEXT[4] = np.array(
    [
        [0, -1, -1],
        [0, 1, 5],
        [0, 2, 6],
        [-1, -1, -1],
        [5, -1, -1],
        [6, -1, -1],
        [5, 6, 7],
    ],
    dtype=np.int8,
)

_LEE_OCTREE_NEXT[5] = np.array(
    [
        [0, 1, 4],
        [1, -1, -1],
        [1, 3, 7],
        [4, -1, -1],
        [4, 6, 7],
        [-1, -1, -1],
        [7, -1, -1],
    ],
    dtype=np.int8,
)

_LEE_OCTREE_NEXT[6] = np.array(
    [
        [0, 2, 4],
        [2, -1, -1],
        [2, 3, 7],
        [4, -1, -1],
        [4, 5, 7],
        [-1, -1, -1],
        [7, -1, -1],
    ],
    dtype=np.int8,
)

_LEE_OCTREE_NEXT[7] = np.array(
    [
        [1, 3, 5],
        [2, 3, 6],
        [3, -1, -1],
        [4, 5, 6],
        [5, -1, -1],
        [6, -1, -1],
        [-1, -1, -1],
    ],
    dtype=np.int8,
)


# Mapping from the 26-voxel cube index to its starting octant.
#
# This used to be constructed with np.array() inside every simple-point
# test. It is now a single immutable module-level array.
_LEE_VOXEL_OCTANT = np.ascontiguousarray(
    np.array(
        [
            0, 0, 1, 0, 0, 1,
            2, 2, 3, 0, 0, 1,
            0, 1, 2, 2, 3, 4,
            4, 5, 4, 4, 5, 6,
            6, 7,
        ],
        dtype=np.int8,
    )
)


@njit(nogil=True, cache=True, inline="always")
def _lee_get_neighborhood(
    img,
    p,
    r,
    c,
    neigh,
):
    """
    Fill the 27-element Lee neighborhood.

    Returns the total number of foreground voxels, including the center.
    """
    s = 0

    v = img[p - 1, r - 1, c - 1]
    neigh[0] = v
    s += v

    v = img[p - 1, r, c - 1]
    neigh[1] = v
    s += v

    v = img[p - 1, r + 1, c - 1]
    neigh[2] = v
    s += v

    v = img[p - 1, r - 1, c]
    neigh[3] = v
    s += v

    v = img[p - 1, r, c]
    neigh[4] = v
    s += v

    v = img[p - 1, r + 1, c]
    neigh[5] = v
    s += v

    v = img[p - 1, r - 1, c + 1]
    neigh[6] = v
    s += v

    v = img[p - 1, r, c + 1]
    neigh[7] = v
    s += v

    v = img[p - 1, r + 1, c + 1]
    neigh[8] = v
    s += v

    v = img[p, r - 1, c - 1]
    neigh[9] = v
    s += v

    v = img[p, r, c - 1]
    neigh[10] = v
    s += v

    v = img[p, r + 1, c - 1]
    neigh[11] = v
    s += v

    v = img[p, r - 1, c]
    neigh[12] = v
    s += v

    v = img[p, r, c]
    neigh[13] = v
    s += v

    v = img[p, r + 1, c]
    neigh[14] = v
    s += v

    v = img[p, r - 1, c + 1]
    neigh[15] = v
    s += v

    v = img[p, r, c + 1]
    neigh[16] = v
    s += v

    v = img[p, r + 1, c + 1]
    neigh[17] = v
    s += v

    v = img[p + 1, r - 1, c - 1]
    neigh[18] = v
    s += v

    v = img[p + 1, r, c - 1]
    neigh[19] = v
    s += v

    v = img[p + 1, r + 1, c - 1]
    neigh[20] = v
    s += v

    v = img[p + 1, r - 1, c]
    neigh[21] = v
    s += v

    v = img[p + 1, r, c]
    neigh[22] = v
    s += v

    v = img[p + 1, r + 1, c]
    neigh[23] = v
    s += v

    v = img[p + 1, r - 1, c + 1]
    neigh[24] = v
    s += v

    v = img[p + 1, r, c + 1]
    neigh[25] = v
    s += v

    v = img[p + 1, r + 1, c + 1]
    neigh[26] = v
    s += v

    return s

@njit(nogil=True, cache=True, inline="always")
def _lee_is_endpoint(neigh):
    """
    Endpoint = center + exactly one foreground neighbor.

    Therefore total foreground count == 2.
    """
    s = 0

    for i in range(27):
        s += neigh[i]

    return s == 2


# ---------------------------------------------------------------------------
# Euler invariance
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, inline="always")
def _lee_is_euler_invariant(neigh):
    """
    Exact Euler-invariance test used by Lee thinning.
    """
    euler_char = 0

    for octant in range(8):
        n = 1

        for j in range(7):
            idx = _LEE_OCTANT_INDICES[octant, j]

            if neigh[idx] == 1:
                n |= 1 << (7 - j)

        euler_char += _LEE_LUT[n]

    return euler_char == 0


@njit(nogil=True, cache=True, inline="always")
def _lee_octree_label(
    octant,
    label_value,
    cube,
    stack_octant,
    stack_pos,
):
    """
    Iterative equivalent of the recursive octree_labeling().

    The stack arrays are supplied by the caller and reused.

    Children are pushed in reverse order so that they are visited in
    exactly the same order as the recursive implementation.
    """
    top = 0

    stack_octant[0] = octant
    stack_pos[0] = 0
    top = 1

    while top > 0:
        current_octant = int(stack_octant[top - 1])
        pos = int(stack_pos[top - 1])

        if pos >= 7:
            top -= 1
            continue

        # Advance this frame before descending.
        stack_pos[top - 1] = pos + 1

        idx = int(
            _LEE_OCTREE_INDICES[
                current_octant,
                pos,
            ]
        )

        if cube[idx] != 1:
            continue

        cube[idx] = label_value

        # Reverse push order because the stack is LIFO.
        for k in range(2, -1, -1):
            next_octant = int(
                _LEE_OCTREE_NEXT[
                    current_octant,
                    pos,
                    k,
                ]
            )

            if next_octant >= 0:
                stack_octant[top] = next_octant
                stack_pos[top] = 0
                top += 1


@njit(nogil=True, cache=True, inline="always")
def _lee_is_simple_point(
    neigh,
    cube,
    stack_octant,
    stack_pos,
):
    """
    N(v)-labeling simple-point test.

    All scratch storage is supplied by the caller.
    """
    # Remove center voxel (neigh[13]).
    for i in range(13):
        cube[i] = neigh[i]

    for i in range(13):
        cube[13 + i] = neigh[14 + i]

    label_value = 2

    for i in range(26):
        if cube[i] != 1:
            continue

        octant = int(
            _LEE_VOXEL_OCTANT[i]
        )

        _lee_octree_label(
            octant,
            label_value,
            cube,
            stack_octant,
            stack_pos,
        )

        label_value += 1

        # Two connected components are enough to establish that the
        # point is not simple.
        if label_value >= 4:
            return False

    return True


@njit(nogil=True, cache=True)
def _lee_find_candidates(
    img,
    border,
    candidates,
    neigh,
    cube,
    stack_octant,
    stack_pos,
):
    """
    Find Lee deletion candidates for one directional sweep.

    Scratch buffers are reused for every candidate.
    """
    rows = img.shape[1]
    cols = img.shape[2]

    count = 0

    for p in range(1, img.shape[0] - 1):
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):

                if img[p, r, c] != 1:
                    continue

                if border == 1:
                    is_border = (
                        img[p, r, c - 1] == 0
                    )

                elif border == 2:
                    is_border = (
                        img[p, r, c + 1] == 0
                    )

                elif border == 3:
                    is_border = (
                        img[p, r + 1, c] == 0
                    )

                elif border == 4:
                    is_border = (
                        img[p, r - 1, c] == 0
                    )

                elif border == 5:
                    is_border = (
                        img[p + 1, r, c] == 0
                    )

                else:
                    is_border = (
                        img[p - 1, r, c] == 0
                    )

                if not is_border:
                    continue

                foreground_count = _lee_get_neighborhood(
                    img,
                    p,
                    r,
                    c,
                    neigh,
                )

                # Endpoint.
                if foreground_count == 2:
                    continue

                if not _lee_is_euler_invariant(neigh):
                    continue

                if not _lee_is_simple_point(
                    neigh,
                    cube,
                    stack_octant,
                    stack_pos,
                ):
                    continue

                candidates[count, 0] = p
                candidates[count, 1] = r
                candidates[count, 2] = c

                count += 1

    return count


@njit(nogil=True, cache=True)
def _lee_compute_thin_image(img):
    """
    Numba implementation of Lee's thinning algorithm.

    Scratch storage is allocated once and reused for the complete sweep.
    """
    borders = np.array(
        [4, 3, 2, 1, 5, 6],
        dtype=np.int8,
    )

    if img.shape[0] == 3:
        num_borders = 4
    else:
        num_borders = 6

    max_candidates = (
        (img.shape[0] - 2)
        * (img.shape[1] - 2)
        * (img.shape[2] - 2)
    )

    candidates = np.empty(
        (max_candidates, 3),
        dtype=np.intp,
    )

    # ------------------------------------------------------------------
    # All of these are now allocated once rather than once per candidate.
    # ------------------------------------------------------------------

    neigh = np.empty(
        27,
        dtype=np.uint8,
    )

    cube = np.empty(
        26,
        dtype=np.uint8,
    )

    stack_octant = np.empty(
        64,
        dtype=np.int8,
    )

    stack_pos = np.empty(
        64,
        dtype=np.int8,
    )

    unchanged_borders = 0

    while unchanged_borders < num_borders:
        unchanged_borders = 0

        for j in range(num_borders):
            border = borders[j]

            count = _lee_find_candidates(
                img,
                border,
                candidates,
                neigh,
                cube,
                stack_octant,
                stack_pos,
            )

            no_change = True

            # Candidates must be rechecked sequentially because earlier
            # candidates in this same sweep may already have been deleted.
            for i in range(count):
                p = candidates[i, 0]
                r = candidates[i, 1]
                c = candidates[i, 2]

                if img[p, r, c] == 0:
                    continue

                _lee_get_neighborhood(
                    img,
                    p,
                    r,
                    c,
                    neigh,
                )

                if _lee_is_simple_point(
                    neigh,
                    cube,
                    stack_octant,
                    stack_pos,
                ):
                    img[p, r, c] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1

    return img


def skeletonize(image):
    """
    Summary
    -------
    Skeletonize a 2-D or 3-D binary image using Lee thinning.

    Extended Description
    --------------------
    2-D images are promoted to a one-voxel-thick 3-D volume before
    thinning. The input is padded with a one-voxel zero border, thinned,
    and the padding is removed before the result is returned.

    Parameters
    ----------
    image : array_like
        2-D or 3-D binary image. Nonzero values are foreground.

    Returns
    -------
    ndarray of bool
        Lee skeleton with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` has fewer than 2 dimensions or more than 3 dimensions.

    Notes
    -----
    This implementation targets scikit-image 0.26.0's Lee method. It uses
    Numba-compiled loops for directional sweeps, endpoint tests, Euler
    invariance tests, and N(v)-labeling simple-point tests.

    Examples
    --------
    >>> image2d = [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    >>> skeleton = skeletonize(image2d)
    >>> print(skeleton.shape)
    # (3, 4)

    >>> image3d = [
    >>>     [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
    >>>     [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
    >>>     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    >>> skeleton = skeletonize(image3d)
    >>> print(skeleton.shape)
    # (3, 3, 3)
    """
    image = np.asarray(image)

    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(
            "skeletonize can only handle 2D or 3D images; "
            f"got image.ndim = {image.ndim} instead."
        )

    image = np.ascontiguousarray(
        image.astype(bool, copy=False)
    )

    was_2d = image.ndim == 2

    if was_2d:
        image = image[np.newaxis, ...]

    image = np.pad(
        image,
        pad_width=1,
        mode="constant",
    )

    image = np.ascontiguousarray(
        image,
        dtype=np.uint8,
    )

    result = _lee_compute_thin_image(image)

    result = result[
        1:-1,
        1:-1,
        1:-1,
    ]

    if was_2d:
        result = result[0]

    return result.astype(bool)

