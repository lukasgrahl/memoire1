import pandas as pd
from itertools import groupby
import datetime as dt
from src.wrappers import skipna
import os

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def str_time_format(x: str,
                    t_format: str = '%Y-%m-%d %H:%M:%S'):
    return dt.datetime.strptime(x, t_format)


def ser_time_format(ser: pd.Series,
                    t_format: str = "%Y-%m-%d"):
    return ser.apply(lambda x: str_time_format(str(x), t_format))

@skipna
def apply_func(arr, func, **kwargs):
    """
    *args:
    arr: array

    **kwargs:
    skipna: wether to skipna or note
    func: transformation function
    """
    return func(arr, **kwargs)


def printProgBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    perc = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {perc}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
    pass


def get_most_recent_mod_output(path: str, mod_name: str) -> str:
    mod_list = [item for item in os.listdir(path) if mod_name in item]
    if len(mod_list) == 0:
        raise KeyError('No model file found')
    else:
        return mod_list[-1]