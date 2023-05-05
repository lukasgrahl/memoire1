import numpy as np
import xarray as xr
from statsmodels.tsa.api import VAR
import pandas as pd

from config import seed
rng = np.random.default_rng(seed=seed)

def get_xarr_InferenceData(xarr_in: xr.Dataset):
    xarr = xarr_in.where(xarr_in.is_solved).copy()
    draw = int(xarr_in.n_runs_acc[0][0])

    xr_prior = xr.Dataset(
        {
            key: (
                ['chain', 'draw', 'a_dim'],
                np.concatenate(
                    [
                        np.array([0] * draw).reshape(draw, 1),
                        np.array(xarr.sel(parameter=key).prior).reshape(draw, 1),
                        np.array([0] * draw).reshape(draw, 1)
                    ],
                    axis=1
                ).reshape(1, draw, 3)
            )
            for key in xarr.parameter.values
        },
        coords={
            "chain": (["chain"], np.arange(1)),
            "draw": (["draw"], np.arange(draw)),
            "a_dim": (["a_dim"], ["x", "y", "z"])

        }
    )

    xr_post = xr.Dataset(
        {
            key: (
                ['chain', 'draw', 'a_dim'],
                np.concatenate(
                    [
                        np.array([0] * draw, dtype=float).reshape(draw, 1),
                        np.array(xarr.sel(parameter=key).posterior, dtype=float).reshape(draw, 1),
                        np.array([0] * draw, dtype=float).reshape(draw, 1)
                    ],
                    axis=1
                ).reshape(1, draw, 3)
            )
            for key in xarr.parameter.values
        },
        coords={
            "chain": (["chain"], np.arange(1)),
            "draw": (["draw"], np.arange(draw)),
            "a_dim": (["a_dim"], ["x", "y", "z"])

        }
    )

    xr_loglike = xr.Dataset(
        {
            'obs': (
                ['chain', 'draw', 'a_dim'],
                np.concatenate(
                    [
                        np.array([0.] * draw, dtype=float).reshape(draw, 1),
                        np.array(xarr.log_likelihood, dtype=float).reshape(draw, 1),
                        np.array([0.] * draw, dtype=float).reshape(draw, 1)
                    ],
                    axis=1
                ).reshape(1, draw, 3)
            )
        },
        coords={
            "chain": (["chain"], np.arange(1)),
            "draw": (["draw"], np.arange(draw)),
            "a_dim": (["a_dim"], ["x", "y", "z"])

        }
    )
    return xr_prior, xr_post, xr_loglike


def grid_serach_var(data: pd.DataFrame, p_max: int = 15, ic: str = 'aic'):
    ic_res = []
    for p in range(1, p_max + 1):
        mod = VAR(data)
        res = mod.fit(p)
        res.summary()

        if ic == 'aic':
            ic_res.append(res.aic)
        elif ic == 'bic':
            ic_res.append(res.bic)
    return ic_res


# Function that takes a single draw of parameters and forecasts n steps
def bvar_forecast(data, coords, lags, intercept, lag_coefs, noise, forecast=10):
    len_data = len(data)
    new_draws = np.zeros((data.shape[0]+forecast, data.shape[1]))
    # Fill the new array with the observed data
    new_draws[:len_data] = data[:]
    for i in range(forecast):
        ar_list = []
        for ind in range(0, len(coords['cross_vars'])):
            ar = np.sum(lag_coefs[:, ind] * new_draws[len_data+i-lags: len_data+i])
            ar_list.append(ar)
        mean = intercept + np.stack(ar_list)
        new_draws[len_data+i] = rng.normal(mean, noise)
    # Replace all observed data with nan, so they don't show when we plot it
    new_draws[:-forecast-1] = np.nan
    return new_draws