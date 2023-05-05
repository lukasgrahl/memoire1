import os
import pandas as pd
import numpy as np
from itertools import groupby
import datetime as dt
from src.wrappers import skipna


def solve_updated_mod(mod, verbose: bool = True, **kwargs) -> (bool, object):
    """

    :param mod:
    :param verbose:
    :param kwargs: solver, default is cycle_reduction, 'gensys' also supported
    :return:
    """
    # solve for steady state
    mod.steady_state(verbose=verbose, **kwargs)
    is_solved = mod.steady_state_solved
    if not is_solved:
        return False, mod

    # solve model, capture np.LinAlgEr
    try:
        mod.solve_model(verbose=verbose, **kwargs)
    except np.linalg.LinAlgError:
        if verbose: print("LinAlg Erorr in solving")
        return False, mod

    # check blanchard kahn
    is_bk = mod.check_bk_condition(return_value='bool', verbose=verbose)

    return is_solved & is_bk, mod


# @jit(cache=True)
def sample_from_priors(priors: dict, mod_params: dict, shock_names: list) -> (dict, dict, dict):
    params = {k: v for k, v in zip(priors.keys(), [item.rvs() for item in priors.values()]) if k in mod_params}
    shocks = {k: v for k, v in zip(priors.keys(), [item.rvs() for item in priors.values()]) if k in shock_names}
    return params, shocks


# @njit
def get_arr_pdf_from_dist(dict_vals, dict_dists):
    # Get pdf from distribution for val
    return np.array([dict_dists[item].pdf(dict_vals[item]) for item in dict_vals.keys()])


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


def get_most_recent_mod_output(path: str, mod_name: str, return_all: bool = False) -> str:
    mod_list = [item for item in os.listdir(path) if mod_name in item]
    if len(mod_list) == 0:
        raise KeyError('No model file found')
    else:
        if return_all:
            return mod_list
        else:
            return mod_list[-1]


def get_confidence_interval(mu: np.array, cov: np.array, sigma=1.96):
    assert len(mu) == len(cov), 'mu and cov do not correspond in legnth'
    upper = mu + cov * 1.96
    lower = mu - cov * 1.96
    return upper, lower