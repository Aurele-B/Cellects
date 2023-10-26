import logging
import os

from numpy import asarray, lib, round, arange, zeros_like, zeros, uint8, \
    dstack, row_stack, column_stack, quantile, sum, int64, linalg, floor, argmax, gradient, diff, sign, empty, float64, mean, pad, convolve, equal, where, array, ones

import pickle
from timeit import default_timer
import time
# from numba import njit, vectorize, cuda
# methods = ["get_" in i for i in elem_list] # grepl
from numba.typed import Dict
from numba import njit
#from pathlib import Path


@njit()
def greater_along_first_axis(array_in_1, array_in_2):
    array_out = zeros_like(array_in_1)
    for i, value in enumerate(array_in_2):
        array_out[i, ...] = array_in_1[i, ...] > value
    return array_out


@njit()
def less_along_first_axis(array_in_1, array_in_2):
    array_out = zeros_like(array_in_1)
    for i, value in enumerate(array_in_2):
        array_out[i, ...] = array_in_1[i, ...] < value
    return array_out


def translate_dict(old_dict):
    """
    Translate a usual python dictionary into a typed one
    :param old_dict: usual python dictionary
    :type old_dict: dict
    :return: typed dictionary
    :rtype: TDict
    """
    numba_dict = Dict()
    for k, v in old_dict.items():
        numba_dict[k] = v
    return numba_dict

def reduce_path_len(pathway, to_start, from_end):
    """
    pathway=Path(os.getcwd())
    reduce_path_len(pathway, 15, 4, 8)
    """
    max_size = to_start + from_end + 3
    if not isinstance(pathway, str):
        pathway = str(pathway)
    if len(pathway) > max_size:
        return pathway[:to_start] + "..." + pathway[-from_end:]
    else:
        return pathway


def find_nearest(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]


def rolling_window(a, window):
    """
    Efficient rolling statistics with NumPy
    Author: Erik Rigtorp
    https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    :param a:array
    :param window:length of the windows to assess within the array
    :return:The array containing each possible windows
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class PercentAndTimeTracker:
    def __init__(self, total, compute_with_elements_number=False, core_number=1):
        """
        Give the progress of a for looped process. Initialize before the for loop
        :param total: Total number of iteration
        :type total: uint64
        :param compute_with_elements_number: when False, uses the number of times get_progress() will be used
        when True, uses the number of elements specified in the get_progress() method
        :type compute_with_elements_number: bool
        :param core_number: The number of core used when multiprocessing
        :type core_number: uint16
        """
        self.starting_time = default_timer()
        self.total = total
        self.current_step = 0
        if compute_with_elements_number:
            self.element_vector = zeros(total, dtype=int64)
        self.core_number = core_number

    def get_progress(self, step=None, element_number=None):
        """
        This method gives the percentage of loop advance, the ETA and the remaining time.
        Call this method at each iteration
        :param step: if None, add 1, put the number of the current iteration instead
        :type step: uint64
        :param element_number: if None use the number of calls of get_progress() or the step.
        Otherwise, the remaining percentage and time and will be computed according to
        the current and previous element_number values
        :type element_number: uint64
        :return: Current percentage, ETA (remaining time)
        :rtype: str
        """
        if step is not None:
            self.current_step = step
        if element_number is not None:
            self.element_vector[self.current_step] = element_number

        if self.current_step > 0:
            elapsed_time = default_timer() - self.starting_time
            if element_number is None or self.current_step < 15:
                if self.current_step < self.core_number:
                    current_prop = self.core_number / self.total
                else:
                    current_prop = (self.current_step + 1) / self.total
            else:
                x_mat = array([ones(self.current_step - 4), arange(5, self.current_step + 1)]).T
                coefs = linalg.lstsq(x_mat, self.element_vector[5:self.current_step + 1], rcond=-1)[0]
                self.element_vector = coefs[0] + (arange(self.total) * coefs[1])
                self.element_vector[self.element_vector < 0] = 0
                current_prop = self.element_vector[:self.current_step + 1].sum() / self.element_vector.sum()

            total_time = elapsed_time / current_prop
            current_prop = int(round(current_prop * 100))
            remaining_time_s = total_time - elapsed_time

            local_time = time.localtime()
            local_m = int(time.strftime("%M", local_time))
            local_h = int(time.strftime("%H", local_time))
            remaining_time_h = remaining_time_s // 3600
            reste_s = remaining_time_s % 3600
            reste_m = reste_s // 60
            # + str(int(floor(reste_s % 60))) + "S"
            hours = int(floor(remaining_time_h))
            minutes = int(floor(reste_m))

            if (local_m + minutes) < 60:
                eta_m = local_m + minutes
            else:
                eta_m = (local_m + minutes) % 60
                local_h += 1

            if (local_h + hours) < 24:
                output = current_prop, f"{local_h + hours}:{eta_m} ({hours}:{minutes} left)"
            else:
                days = (local_h + hours) // 24
                eta_h = (local_h + hours) % 24
                eta_d = time.strftime("%m", local_time) + "/" + str(int(time.strftime("%d", local_time)) + days)
                output = current_prop, f"{eta_d} {eta_h}:{eta_m} ({hours}:{minutes} left)"
            # return current_prop, str(local_h + hours) + ":" + str(local_m + minutes) + "(" + str()
        else:
            output = 0, "loading, wait..."
        if step is None:
            self.current_step += 1
        return output

