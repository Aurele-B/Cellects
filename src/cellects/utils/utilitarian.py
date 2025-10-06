
import numpy as np
from timeit import default_timer
import time
from numba.typed import Dict
from numba import njit
from glob import glob
vectorized_len = np.vectorize(len)


@njit()
def equal_along_first_axis(array_in_1, array_in_2):
    """
    Compare two arrays element-wise along the first axis and return a boolean array where elements are equal.

    This function checks if corresponding elements in two arrays along the first axis
    are equal, returning a boolean array of the same shape as `array_in_1`.

    Parameters
    ----------
    array_in_1 : ndarray
        First array to compare.
    array_in_2 : ndarray
        Second array to compare.

    Returns
    -------
    ndarray
        Boolean array indicating equality of elements along the first axis.
        Has the same shape as `array_in_1` and dtype of `np.bool_`.

    Raises
    ------
    ValueError
        If the shape of `array_in_1` and `array_in_2` do not match along all axes
        except the first.

    Examples
    --------
    >>> array_in_1 = np.array([[1, 2], [3, 4]])
    >>> array_in_2 = np.array([[1, 2], [0, 4]])
    >>> equal_along_first_axis(array_in_1, array_in_2)
    array([[ True,  True],
           [False,  True]])
    >>> equal_along_first_axis(array_in_1, array_in_2).dtype
    dtype('bool')
    """
    array_out = np.zeros_like(array_in_1)
    for i, value in enumerate(array_in_2):
        array_out[i, ...] = array_in_1[i, ...] == value
    return array_out


@njit()
def greater_along_first_axis(array_in_1, array_in_2):
    """
    Compare two arrays element-wise along the first axis and return a boolean array,
    where each element indicates whether the corresponding element in `array_in_1`
    is greater than in `array_in_2`.

    Parameters
    ----------
    array_in_1 : array_like
        The first input array.
    array_in_2 : array_like
        The second input array.

    Returns
    -------
    out : ndarray, bool
        A boolean array indicating where `array_in_1` elements are greater than
        corresponding `array_in_2` elements.

    Examples
    --------
    >>> array1 = np.array([[1, 2], [3, 4]])
    >>> array2 = np.array([[0, 1], [2, 3]])
    >>> result = greater_along_first_axis(array1, array2)
    >>> print(result)  # doctest: +NORMALIZE_WHITESPACE
    [[ True False]
     [ True False]]
    """
    array_out = np.zeros_like(array_in_1)
    for i, value in enumerate(array_in_2):
        array_out[i, ...] = array_in_1[i, ...] > value
    return array_out


@njit()
def less_along_first_axis(array_in_1, array_in_2):
    """
    Compare two arrays element-wise along the first axis, returning a boolean array.

    This function performs an element-wise comparison between two arrays along
    the first axis and returns a boolean array. The comparison is less than, i.e.,
    element-wise `array_in_1 < array_in_2`.

    Parameters
    ----------
    array_in_1 : numpy.ndarray
        The first input array.
    array_in_2 : numpy.ndarray
        The second input array.

    Returns
    -------
    numpy.ndarray[bool]
        A boolean array where each element is the result of the comparison
        `array_in_1[i, ...] < array_in_2[i, ...]` for all indices `i` along the first axis.

    Notes
    -----
    This function uses Numba's `@njit` decorator for performance.

    Examples
    --------
    >>> result = less_along_first_axis(np.array([[1, 2], [3, 4]]), np.array([2, 2]))
    >>> print(result)
    [[ True  True]
     [False False]]

    >>> result = less_along_first_axis(np.array([[5, -1], [-3, 0]]), np.array([4.9, 2]))
    >>> print(result)
    [[False False]
     [ True True]]
    """
    array_out = np.zeros_like(array_in_1)
    for i, value in enumerate(array_in_2):
        array_out[i, ...] = array_in_1[i, ...] < value
    return array_out


def translate_dict(old_dict):
    """
    Translate a dictionary to a typed dictionary and filter out non-string values.

    Parameters
    ----------
    old_dict : dict
        The input dictionary that may contain non-string values

    Returns
    -------
    dict
        A typed dictionary containing only the items from `old_dict` where the value is not a string

    Examples
    --------
    >>> result = translate_dict({'a': 1., 'b': 'string', 'c': 2.0})
    >>> print(result)
    DictType[unicode_type,float64]<iv=None>({a: 1.0, c: 2.0})
    """
    numba_dict = Dict()
    for k, v in old_dict.items():
        if not isinstance(v, str):
            numba_dict[k] = v
    return numba_dict

def reduce_path_len(pathway, to_start, from_end):
    """
    Reduce the length of a given pathway string by truncating it from both ends.

    The function is used to shorten the `pathway` string if its length exceeds
    a calculated maximum size. If it does, the function truncates it from both ends,
    inserting an ellipsis ("...") in between.

    Parameters
    ----------
    pathway : str or int
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


def find_nearest(array, value):
    """
    Find the element in an array that is closest to a given value.

    Parameters
    ----------
    array : array_like
        Input array. Can be any array-like data structure.
    value :
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
    def __init__(self, total, compute_with_elements_number=False, core_number=1):
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

        Notes
        -----
        Starting time is recorded for potential performance tracking.

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


def insensitive_glob(pattern):
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


def smallest_memory_array(array_object, array_type='uint'):
    """
    Convert the given array object to the smallest possible memory type.

    This function determines the optimal data type for an array
    based on its maximum value and converts it to that data type.

    Parameters
    ----------
    array_object : numpy.ndarray or list of lists
        The input array object which can be either a NumPy array or a list of lists.

    array_type : str, optional
        The type of data to which the input array should be converted. Should be either 'uint' (default) or any other NumPy data type.

    Returns
    -------
    numpy.ndarray
        The converted array object with the smallest possible memory type.

    Raises
    ------
    TypeError
        If the input `array_object` is not a NumPy array or list of lists.

    ValueError
        If the specified `array_type` is not supported by NumPy.

    Notes
    -----
    This function uses NumPy's `iinfo` to determine the information about the integer data types and finds the smallest
    data type that can store all values in the array without overflow.

    Examples
    --------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> result = smallest_memory_array(arr)
    >>> print(result.dtype)
    uint8

    >>> import numpy as np
    >>> arr = np.array([[1000, 2000], [3000, 4000]])
    >>> result = smallest_memory_array(arr)
    >>> print(result.dtype)
    uint16
    """
    if isinstance(array_object, np.ndarray):
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


def remove_coordinates(arr1, arr2):
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
    >>> arr1 = np.array([[0, 0], [1, 2], [3, 4]])
    >>> arr2 = np.array([[1, 2], [5, 6]])
    >>> remove_coordinates(arr1, arr2)
    array([[0, 0],
           [3, 4]])
    """
    # Convert to set of tuples
    coords_to_remove = set(map(tuple, arr2))
    return np.array([coord for coord in arr1 if tuple(coord) not in coords_to_remove])
