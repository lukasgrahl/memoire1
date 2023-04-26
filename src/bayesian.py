import numpy as np
import xarray as xr


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


import statsmodels.api as sm
from statsmodels.tsa.api import VAR

import pandas as pd

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