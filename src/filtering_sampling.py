import numpy as np
import pandas as pd
# from numba import jit, njit
from scipy import linalg

from filterpy.kalman import KalmanFilter
from filterpy.common import Saver
from src.utils import sample_from_priors, solve_updated_mod

import arviz as az

def get_Sigma(priors: dict, mod_params: dict, shock_names: list) -> (dict, dict, dict):
    params = {k: v for k, v in zip(priors.keys(), [item.std() for item in priors.values()]) if k in mod_params}
    shocks = {k: v for k, v in zip(priors.keys(), [item.std() for item in priors.values()]) if k in shock_names}
    return params, shocks

def run_kalman_filter_forecast(model, observed_vars: list, train_data: pd.DataFrame, test_data: pd.DataFrame, prior_dist: dict,
                               mod_params: list, parameters: dict, observation_noise: float = .01, update_filter:bool=True):
    shock_names = [x.base_name for x in model.shocks]
    state_variables = [x.base_name for x in model.variables]

    # update mod params with posterior mean
    model.free_param_dict.update(parameters)
    _, shocks = sample_from_priors(prior_dist, mod_params, shock_names)

    solved, model = solve_updated_mod(model, verbose=False)
    if not solved: raise ValueError('Model not solved')

    T, R = model.T.values, model.R.values
    H, Z, T, R, QN, zs = set_up_kalman_filter(R=R, T=T, observed_data=train_data[observed_vars].values,
                                              observed_vars=observed_vars,
                                              shock_names=shock_names, shocks_drawn_prior=shocks,
                                              state_variables=state_variables, H0=observation_noise)

    kfilter = KalmanFilter(len(state_variables), len(observed_vars))
    kfilter.F = T
    kfilter.Q = QN
    kfilter.H = Z
    kfilter.R = H

    # filtering
    saver = Saver(kfilter)
    mu, cov, _, _ = kfilter.batch_filter(zs, saver=saver)
    ll_train = saver.log_likelihood

    mu_df_train = pd.DataFrame(mu.reshape(len(mu), -1),
                               columns=[item.base_name for item in model.variables],
                               index=train_data.index)
    cov_df_train = pd.DataFrame(cov.diagonal(axis1=1, axis2=2), columns=[item.base_name for item in model.variables],
                                index=train_data.index)

    # forecasting
    mu, cov, ll_test = kalman_filter_forecast(kfilter, test_data[observed_vars].values, update_filter=update_filter)

    mu_df_test = pd.DataFrame(mu.reshape(len(mu), -1),
                              columns=[item.base_name for item in model.variables],
                              index=test_data.index)
    cov_df_test = pd.DataFrame(cov.diagonal(axis1=1, axis2=2), columns=[item.base_name for item in model.variables],
                               index=test_data.index)

    return mu_df_train, cov_df_train, ll_test, mu_df_test, cov_df_test, ll_train


def get_Q_H(shocks_dict: dict, observed_names: list, noise_diag):
    """

    :param shocks_dict: Dictionary containing shocks: value
    :param observed_names: Names of observed variables for the Kalman filter
    :param noise_diag: list of observation noise (distrust in data) for the observed variables
    :return:
    """
    Q = np.diag([v for v in shocks_dict.values()])
    assert len(noise_diag) == len(observed_names), \
        f"H matrix needs zdim x zdim, but is {len(noise_diag)} x {len(noise_diag)}"
    H = np.diag(noise_diag)
    return Q, H


# @njit
def set_up_kalman_filter(R: np.array, T: np.array, observed_data: np.array, shocks_drawn_prior: dict,
                         state_variables: list, observed_vars: list, shock_names: list, H0: float = 1e-8) \
        -> (np.array, np.array, np.array, np.array, np.array, np.array):
    """

    :param shock_names:
    :param observed_data:
    :param R: gEconpy Model.R, shock covariance matrix
    :param T: gEconpy Model.T, policy matrix
    :param shocks_drawn_prior: dictionary containing shock: parameter value
    :param state_variables: list of the models state variables
    :param observed_vars: list of observed variables
    :param train_data: data used to calibrate the filter
    :param H0: observation noise on all observed variables
    :return:
    """

    _ = [item for item in shock_names if item not in shocks_drawn_prior.keys()]
    assert len(_) == 0, f"The following shocks have no defined priors {_}"

    _ = [item for item in observed_vars if item not in state_variables]
    assert len(_) == 0, f"The following variables are not contained in the model specification {_}"

    xdim = len(state_variables)
    zdim = len(observed_vars)

    Q, H = get_Q_H(shocks_drawn_prior, observed_vars, [H0] * zdim)
    Z = np.array([[x == var for x in state_variables] for var in observed_vars], dtype='float64')
    T, R = T, R
    QN = R @ Q @ R.T
    zs = observed_data.copy()

    return H, Z, T, R, QN, zs


# @njit
def kalman_predict(x, P, T, QN) -> (np.array, np.array):
    # predict
    x_pred = T @ x
    P_pred = T @ P @ T.T + QN
    P_pred = (P_pred + P_pred.T) / 2
    return x_pred, P_pred


# @njit
def kalman_update(z, x, P, Z, H) -> (np.array, np.array, np.array, bool):
    # residual
    y = z - Z @ x

    # System Uncertainty
    PZT = P @ Z.T
    F = Z @ PZT + H

    # Kalman Gain
    try:
        F_chol = linalg.cholesky(F)
    except Exception as e:
        print(e)
        return None, None, None, False

    K = P @ Z.T @ F_chol

    # update x
    x_up = x + K @ y

    # update P
    I_KZ = np.eye(K.shape[0]) - K @ Z
    P_up = I_KZ @ P @ I_KZ.T + K @ H @ K.T
    P_up = .5 * (P_up + P_up.T)

    # get log-likelihood
    MVN_CONST = np.log(2.0 * np.pi)
    inner_term = linalg.solve_triangular(F_chol, linalg.solve_triangular(F_chol, y, lower=True), lower=True, trans=1)
    n = y.shape[0]
    ll = -0.5 * (n * MVN_CONST + (y.T @ inner_term).ravel()) - np.log(np.diag(F_chol)).sum()
    # ll = np.array([-100])

    return np.array(x_up), np.array(P_up), np.array(ll), True


def kalman_filter(R: np.array, T: np.array, state_variables: list, observed_vars: list, shock_names: list,
                  shocks_drawn_prior: dict, train_data: pd.DataFrame, x0: float = 0., P0: list = [.1]) \
        -> (np.array, np.array, np.array, bool):
    solved = None
    X_out, P_out, LL_out = [], [], []

    xdim = len(state_variables)
    zdim = len(observed_vars)

    observed_data = train_data[observed_vars].values.copy()

    H, Z, T, R, QN, zs = set_up_kalman_filter(R=R, T=T, observed_data=observed_data,
                                              shocks_drawn_prior=shocks_drawn_prior, state_variables=state_variables,
                                              observed_vars=observed_vars, shock_names=shock_names)
    x = np.zeros(xdim) + x0
    P = np.diag(P0 * xdim)

    for i, z in enumerate(zs):
        x, P = kalman_predict(x, P, T, QN)
        x, P, ll, solved = kalman_update(z, x, P, Z, H)

        if solved:
            X_out.append(x)
            P_out.append(P)
            LL_out.append(ll)
        else:
            return None, None, None, False

    return np.array(X_out), np.array(P), np.array(LL_out), solved


def kalman_filter_forecast(kfilter, data, update_filter:bool=True):
    mu, cov, ll = [], [], []
    zs = data.copy()
    for i, z in enumerate(zs):
        kfilter.predict()
        mu.append(kfilter.x)
        cov.append(kfilter.P)

        if update_filter:
            kfilter.update(z)
        ll.append(kfilter.log_likelihood)

    return np.array(mu), np.array(cov), np.array(ll)
