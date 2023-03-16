import numpy as np
from scipy import linalg

def kalman_filter(zs, x, P, T, QN, Z, H) -> (np.array, np.array, np.array):

    X_out, P_out, LL_out = [], [], []
    for i, z in enumerate(zs):
        x, P = kalman_predict(x, P, T, QN)
        x, P, ll = kalman_update(z, x, P, Z, H)

        X_out.append(x)
        P_out.append(P)
        LL_out.append(ll)

    return np.array(X_out), np.array(P), np.array(LL_out)

def kalman_predict(x, P, T, QN) -> (np.array, np.array):
    # predict
    x_pred = T @ x
    P_pred = T @ P @ T.T + QN
    P_pred = (P_pred + P_pred.T) / 2
    return x_pred, P_pred


def kalman_update(z, x, P, Z, H) -> (np.array, np.array, np.array):
    # residual
    y = z - Z @ x

    # System Uncertainty
    PZT = P @ Z.T
    F = Z @ PZT + H

    # Kalman Gain
    F_chol = linalg.cholesky(F)
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

    return np.array(x_up), np.array(P_up), np.array(ll)