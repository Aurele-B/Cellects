import os
from numba import njit as _real_njit

USE_NUMBA = os.getenv("USE_NUMBA", "1") == "1"

def njit(*args, **kwargs):
    """ numba.njit decorator that can be disabled. Useful for testing.
    """
    if USE_NUMBA:
        return _real_njit(*args, **kwargs)
    # test mode: return identity decorator
    def deco(func):
        return func
    return deco