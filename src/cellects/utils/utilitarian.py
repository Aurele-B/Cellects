#!/usr/bin/env python3
"""
Utility module with array operations, path manipulation, and progress tracking.

This module provides performance-optimized utilities for numerical comparisons using Numba,
path string truncation, dictionary filtering, and iteration progress monitoring. It is designed
for applications requiring efficient data processing pipelines with both low-level optimization
and human-readable output formatting.

Classes
-------
PercentAndTimeTracker : Track iteration progress with time estimates and percentage completion

Functions
---------
greater_along_first_axis : Compare arrays element-wise along first axis (>) using Numba
less_along_first_axis    : Compare arrays element-wise along first axis (<) using Numba
translate_dict           : Convert standard dict to typed dict, filtering non-string values
reduce_path_len          : Truncate long path strings with ellipsis insertion
find_nearest             : Find array element closest to target value

Notes
-----
Numba-optimized functions (greater_along_first_axis and less_along_first_axis) require
input arrays of identical shape. String manipulation utilities include automatic type conversion.
The progress tracker records initialization time for potential performance analysis.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from timeit import default_timer
import time
from numba.typed import Dict
from numba import njit
from glob import glob
vectorized_len = np.vectorize(len)


@njit()
def greater_along_first_axis(array_in_1: NDArray, array_in_2: NDArray) -> NDArray[np.uint8]:
    """
    Compare two arrays along the first axis and store the result in a third array.

    This function performs a comparison between two input arrays
    along their first axis and stores the result in a third array. The comparison is
    made to determine which elements of each row of the first array are greater than
    the elements(s) corresponding to that row in the second array.

    Parameters
    ----------
    array_in_1 : ndarray
        First input array.
    array_in_2 : ndarray
        Second input array.

    Returns
    -------
    out : ndarray of uint8
        Boolean ndarray with same shape as input arrays,
        containing the result of element-wise comparison.

    Examples
    --------
    >>> array_in_1 = np.array([[2, 4], [5, 8]])
    >>> array_in_2 = np.array([3, 6])
    >>> array_out = greater_along_first_axis(array_in_1, array_in_2)
    >>> print(array_out)
    [[0 1]
     [0 1]]
    """
    array_out = np.zeros(array_in_1.shape, dtype=np.uint8)
    for i, value in enumerate(array_in_2):
        array_out[i, ...] = array_in_1[i, ...] > value
    return array_out


@njit()
def less_along_first_axis(array_in_1: NDArray, array_in_2: NDArray) -> NDArray[np.uint8]:
    """
    Compare two arrays along the first axis and store the result in a third array.

    This function performs a comparison between two input arrays
    along their first axis and stores the result in a third array. The comparison is
    made to determine which elements of each row of the first array are lesser than
    the elements(s) corresponding to that row in the second array.

    Parameters
    ----------
    array_in_1 : ndarray
        The first input array.
    array_in_2 : ndarray
        The second input array.

    Returns
    -------
    ndarray of uint8
        A boolean array where each element is `True` if the corresponding
        element in `array_in_1` is lesser than the corresponding element
        in `array_in_2`, and `False` otherwise.

    Examples
    --------
    >>> array_in_1 = np.array([[2, 4], [5, 8]])
    >>> array_in_2 = np.array([3, 6])
    >>> array_out = less_along_first_axis(array_in_1, array_in_2)
    >>> print(array_out)
    [[1 0]
     [1 0]]
    """
    array_out = np.zeros(array_in_1.shape, dtype=np.uint8)
    for i, value in enumerate(array_in_2):
        array_out[i, ...] = array_in_1[i, ...] < value
    return array_out


def translate_dict(old_dict: dict) -> Dict:
    """
    Translate a dictionary to a typed dictionary and filter out non-string values.

    Parameters
    ----------
    old_dict : dict
        The input dictionary that may contain non-string values

    Returns
    -------
    numba_dict : Dict
        A typed dictionary containing only the items from `old_dict` where the value is not a string

    Examples
    --------
    >>> result = translate_dict({'a': 1., 'b': 'string', 'c': 2.0})
    >>> print(result)
    {a: 1.0, c: 2.0}
    """
    numba_dict = Dict()
    for k, v in old_dict.items():
        if not isinstance(v, str):
            numba_dict[k] = v
    return numba_dict


def reduce_path_len(pathway: str, to_start: int, from_end: int) -> str:
    """
    Reduce the length of a given pathway string by truncating it from both ends.

    The function is used to shorten the `pathway` string if its length exceeds
    a calculated maximum size. If it does, the function truncates it from both ends,
    inserting an ellipsis ("...") in between.

    Parameters
    ----------
    pathway : str
        The pathway string to be reduced. If an integer is provided,
        it will be converted into a string.
    to_start : int
        Number of characters from the start to keep in the pathway string.
    from_end : int
        Number of characters from the end to keep in the pathway string.

    Returns
    -------
    str
        The reduced version of the `pathway` string. If truncation is not necessary,
        returns the original pathway string.

    Examples
    --------
    >>> reduce_path_len("example/complicated/path/to/resource", 8, 12)
    'example/.../to/resource'
    """
    if not isinstance(pathway, str):
        pathway = str(pathway)
    max_size = to_start + from_end + 3
    if len(pathway) > max_size:
        pathway = pathway[:to_start] + "..." + pathway[-from_end:]
    return pathway


def find_nearest(array: NDArray, value):
    """
    Find the element in an array that is closest to a given value.

    Parameters
    ----------
    array : array_like
        Input array. Can be any array-like data structure.
    value : int or float
        The value to find the closest element to.

    Returns
    -------
    :obj:`array` type
        The element in `array` that is closest to `value`.

    Examples
    --------
    >>> find_nearest([1, 2, 3, 4], 2.5)
    2
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class PercentAndTimeTracker:
    """
    Initialize a progress bar object to track and display the progress of an iteration.

    Parameters
    ----------
    total : int
        The total number of iterations.
    compute_with_elements_number : bool, optional
        If True, create an element vector. Default is False.
    core_number : int, optional
        The number of cores to use. Default is 1.

    Attributes
    ----------
    starting_time : float
        The time when the ProgressBar object is initialized.
    total : int
        The total number of iterations.
    current_step : int
        The current iteration step (initialized to 0).
    element_vector : numpy.ndarray, optional
        A vector of zeros with the same length as `total`, created if
        `compute_with_elements_number` is True.
    core_number : int
        The number of cores.

    Examples
    --------
    >>> p = PercentAndTimeTracker(10)
    >>> print(p.total)  # prints: 10
    >>> p = PercentAndTimeTracker(10, compute_with_elements_number=True)
    >>> print(p.element_vector)  # prints: [0 0 0 0 0 0 0 0 0 0]

    Notes
    -----
    Starting time is recorded for potential performance tracking.

    """
    def __init__(self, total: int, compute_with_elements_number: bool=False, core_number:int =1):
        """Initialize an instance of the class.

        This constructor sets up the initial attributes including
        a starting time, total value, current step, and an optional
        element vector if ``compute_with_elements_number`` is set to True.
        The core number can be specified, defaulting to 1.

        Parameters
        ----------
        total : int
            The total number of elements or steps.
        compute_with_elements_number : bool, optional
            If True, initialize an element vector of zeros. Defaults to False.
        core_number : int, optional
            The number of cores to use. Defaults to 1.

        Attributes
        ----------
        starting_time : float
            The time of instantiation.
        total : int
            The total number of elements or steps.
        current_step : int
            The current step in the process.
        element_vector : ndarray of int64, optional
            A vector initialized with zeros. Exists if ``compute_with_elements_number`` is True.
        core_number : int
            The number of cores to use.
        """
        self.starting_time = default_timer()
        self.total = total
        self.current_step = 0
        if compute_with_elements_number:
            self.element_vector = np.zeros(total, dtype=np.int64)
        self.core_number = core_number

    def get_progress(self, step=None, element_number=None):
        """
        Calculate and update the current progress, including elapsed time and estimated remaining time.

        This function updates the internal state of the object to reflect progress
        based on the current step and element number. It calculates elapsed time,
        estimates total time, and computes the estimated time of arrival (ETA).

        Parameters
        ----------
        step : int or None, optional
            The current step of the process. If ``None``, the internal counter is incremented.
        element_number : int or None, optional
            The current element number. If ``None``, no update is made to the element vector.

        Returns
        -------
        tuple
            A tuple containing:
            - `int`: The current progress percentage.
            - `str`: A string with the ETA and remaining time.

        Raises
        ------
        ValueError
            If ``step`` or ``element_number`` are invalid.

        Notes
        -----
        The function uses linear regression to estimate future progress values when the current step is sufficiently large.

        Examples
        --------
        >>> PercentAndTimeTracker(10, compute_with_elements_number=True).get_progress(9, 5)
        (0, ', wait to get a more accurate ETA...')
        """
        if step is not None:
            self.current_step = step
        if element_number is not None:
            self.element_vector[self.current_step] = element_number

        if self.current_step > 0:
            elapsed_time = default_timer() - self.starting_time
            if element_number is None or element_number == 0 or self.current_step < 15:
                if self.current_step < self.core_number:
                    current_prop = self.core_number / self.total
                else:
                    current_prop = (self.current_step + 1) / self.total
            else:
                x_mat = np.array([np.ones(self.current_step - 4), np.arange(5, self.current_step + 1)]).T
                coefs = np.linalg.lstsq(x_mat, self.element_vector[5:self.current_step + 1], rcond=-1)[0]
                self.element_vector = coefs[0] + (np.arange(self.total) * coefs[1])
                self.element_vector[self.element_vector < 0] = 0
                current_prop = self.element_vector[:self.current_step + 1].sum() / self.element_vector.sum()

            total_time = elapsed_time / current_prop
            current_prop = int(np.round(current_prop * 100))
            remaining_time_s = total_time - elapsed_time

            local_time = time.localtime()
            local_m = int(time.strftime("%M", local_time))
            local_h = int(time.strftime("%H", local_time))
            remaining_time_h = remaining_time_s // 3600
            reste_s = remaining_time_s % 3600
            reste_m = reste_s // 60
            # + str(int(np.floor(reste_s % 60))) + "S"
            hours = int(np.floor(remaining_time_h))
            minutes = int(np.floor(reste_m))

            if (local_m + minutes) < 60:
                eta_m = local_m + minutes
            else:
                eta_m = (local_m + minutes) % 60
                local_h += 1

            if (local_h + hours) < 24:
                output = current_prop, f", ETA {local_h + hours}:{eta_m} ({hours}:{minutes} left)"
            else:
                days = (local_h + hours) // 24
                eta_h = (local_h + hours) % 24
                eta_d = time.strftime("%m", local_time) + "/" + str(int(time.strftime("%d", local_time)) + days)
                output = current_prop, f", ETA {eta_d} {eta_h}:{eta_m} ({hours}:{minutes} left)"
            # return current_prop, str(local_h + hours) + ":" + str(local_m + minutes) + "(" + str()
        else:
            output = int(np.round(100 / self.total)), ", wait..."
        if step is None:
            self.current_step += 1
        if element_number is not None:
            if self.current_step < 50:
                output = int(0), ", wait to get a more accurate ETA..."
        return output


def insensitive_glob(pattern: str):
    """
    Generates a glob pattern that matches both lowercase and uppercase letters.

    Parameters
    ----------
    pattern : str
        The glob pattern to be made case-insensitive.

    Returns
    -------
    str
        A new glob pattern that will match both lowercase and uppercase letters.

    Examples
    --------
    >>> insensitive_glob('*.TXT')
    """
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob(''.join(map(either, pattern)))


def smallest_memory_array(array_object, array_type='uint') -> NDArray:
    """
    Convert input data to the smallest possible NumPy array type that can hold it.

    Parameters
    ----------
    array_object : ndarray or list of lists
        The input data to be converted.
    array_type : str, optional, default is 'uint'
        The type of NumPy data type to use ('uint').

    Returns
    -------
    ndarray
        A NumPy array of the smallest data type that can hold all values in `array_object`.

    Examples
    --------
    >>> import numpy as np
    >>> array = [[1, 2], [3, 4]]
    >>> smallest_memory_array(array)
    array([[1, 2],
           [3, 4]], dtype=np.uint8)

    >>> array = [[1000, 2000], [3000, 4000]]
    >>> smallest_memory_array(array)
    array([[1000, 2000],
           [3000, 4000]], dtype=uint16)

    >>> array = [[2**31, 2**32], [2**33, 2**34]]
    >>> smallest_memory_array(array)
    array([[         2147483648,          4294967296],
           [         8589934592,        17179869184]], dtype=uint64)
    """
    if isinstance(array_object, list):
        array_object = np.array(array_object)
    if isinstance(array_object, NDArray):
        value_max = array_object.max()
    else:

        if len(array_object[0]) > 0:
            value_max = np.max((array_object[0].max(), array_object[1].max()))
        else:
            value_max = 0

    if array_type == 'uint':
        if value_max <= np.iinfo(np.uint8).max:
            array_object = np.array(array_object, dtype=np.uint8)
        elif value_max <= np.iinfo(np.uint16).max:
            array_object = np.array(array_object, dtype=np.uint16)
        elif value_max <= np.iinfo(np.uint32).max:
            array_object = np.array(array_object, dtype=np.uint32)
        else:
            array_object = np.array(array_object, dtype=np.uint64)
    return array_object


def remove_coordinates(arr1: NDArray, arr2: NDArray) -> NDArray:
    """
    Remove coordinates from `arr1` that are present in `arr2`.

    Given two arrays of coordinates, remove rows from the first array
    that match any row in the second array.

    Parameters
    ----------
    arr1 : ndarray of shape (n, 2)
        Array containing coordinates to filter.
    arr2 : ndarray of shape (m, 2)
        Array containing coordinates to match for removal.

    Returns
    -------
    ndarray of shape (k, 2)
        Array with coordinates from `arr1` that are not in `arr2`.

    Examples
    --------
    >>> arr1 = np.arange(200).reshape(100, 2)
    >>> arr2 = np.array([[196, 197], [198, 199]])
    >>> remove_coordinates(arr1, arr2)
    array([[0, 0],
           [3, 4]])
    """
    if arr1.shape[1] != 2 or arr2.shape[1] != 2:
        raise ValueError("Both arrays must have shape (n, 2)")
    mask = ~np.isin(arr1, arr2).all(axis=1)
    return arr1[mask]


def find_threshold_given_mask(greyscale: NDArray[np.uint8], mask: np.uint8, min_threshold: np.uint8=0) -> np.uint8:
    """
    Find the optimal threshold value for a greyscale image given a mask.

    This function performs a binary search to find the optimal threshold
    that maximizes the separation between two regions defined by the mask.
    The search is bounded by a minimum threshold value.

    Parameters
    ----------
    greyscale : ndarray of uint8
        The greyscale image array.
    mask : ndarray of uint8
        The binary mask array where positive values define region A and zero values define region B.
    min_threshold : uint8, optional
        The minimum threshold value for the search. Defaults to 0.

    Returns
    -------
    out : uint8
        The optimal threshold value found.

    Examples
    --------
    >>> import numpy as np
    >>> greyscale = np.array([[255, 128, 54], [0, 64, 20]], dtype=np.uint8)
    >>> mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    >>> find_threshold_given_mask(greyscale, mask)
    54
    """
    region_a = greyscale[mask > 0]
    if len(region_a) == 0:
        return np.uint8(255)
    region_b = greyscale[mask == 0]
    if len(region_b) == 0:
        return min_threshold
    else:
        low = min_threshold
        high = 255
        best_thresh = low

        while 0 <= low <= high:
            mid = (low + high) // 2
            count_a, count_b = _get_counts_jit(mid, region_a, region_b)

            if count_a > count_b:
                # Try to find a lower threshold that still satisfies the condition
                best_thresh = mid
                high = mid - 1
            else:
                if count_a == 0 and count_b == 0:
                    best_thresh = greyscale.mean()
                    break
                # Need higher threshold
                low = mid + 1
    return best_thresh


@njit()
def _get_counts_jit(thresh: np.uint8, region_a: NDArray[np.uint8], region_b: NDArray[np.uint8]) -> Tuple[int, int]:
    """
    Get counts of values in two regions above a threshold using Just-In-Time compilation.

    Count the number of elements greater than `thresh` in both `region_a`
    and `region_b`, returning the counts as a tuple. This function utilizes
    Numba's JIT compilation for performance optimization.

    Parameters
    ----------
    thresh : uint8
        The threshold value to compare against.
    region_a : ndarray of uint8
        First region array containing values to be compared with `thresh`.
    region_b : ndarray of uint8
        Second region array containing values to be compared with `thresh`.

    Returns
    -------
    out : tuple of int, int
        A tuple containing the count of elements greater than `thresh` in
        `region_a` and `region_b`, respectively.

    Examples
    --------
    >>> import numpy as np
    >>> region_a = np.array([1, 250, 3], dtype=np.uint8)
    >>> region_b = np.array([4, 250, 6], dtype=np.uint8)
    >>> thresh = np.uint8(100)
    >>> get_counts_jit(thresh, region_a, region_b)
    (1, 1)
    """
    count_a = 0
    count_b = 0
    for val in region_a:
        if val > thresh:
            count_a += 1
    for val in region_b:
        if val > thresh:
            count_b += 1
    return count_a, count_b
