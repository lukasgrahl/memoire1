from functools import wraps
import numpy as np

def skipna(func):
    """

    :param func:
    :return: changes on arr
    """
    @wraps(func)
    def wrapper(arr, *args, is_skipna: bool=True, **kwargs):
        _arr = arr.copy()
        if is_skipna:
            mask = np.isnan(_arr)
            _ = _arr[~mask].copy()
            _arr[~mask] = func(_, **kwargs)
            return _arr
        else:
            return func(_arr, *args, **kwargs)
    return wrapper
