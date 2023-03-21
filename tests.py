import pickle
from src.process_data import load_data
from src.plotting import plot_dfs
import numpy as np
import os
from settings import DATA_DIR
from filterpy.kalman import KalmanFilter
from filterpy.common import Saver
from src.filtering_sampling import sample_from_priors, set_up_kalman_filter, get_arr_pdf_from_dist
from config import fred_dict, recession_dict
import time
import seaborn as sns

if __name__ == "__main__":

    # df = load_data('raw_data.csv', DATA_DIR, fred_dict)
    # df2 = df.copy() * 1.1
    # plot_dfs([df, df2], sns.lineplot, fill_arr=recession_dict, legend=['a', 'b'])

    # mod_dict = np.load(os.path.join(DATA_DIR, 'mod_test_dict.npy'), allow_pickle=True)

    with open(os.path.join(DATA_DIR, 'mod_test_dict.pickle'), 'rb') as f:
        mod_dict = pickle.load(f)

    prior_dist, mod_params, shock_names, T, R, train,\
        observed_vars, state_variables, old_prior = mod_dict.values()

    start = time.time()

    new_prior, shocks = sample_from_priors(prior_dist, mod_params, shock_names)

    H, Z, T, R, QN, zs = set_up_kalman_filter(R=R, T=T, observed_data=train, observed_vars=observed_vars,
                                              shock_names=shock_names, shocks_drawn_prior=shocks, state_variables=state_variables)

    kfilter = KalmanFilter(len(state_variables), len(observed_vars))
    kfilter.F = T
    kfilter.Q = QN
    kfilter.H = Z
    kfilter.R = H

    # run Kalman filter
    saver = Saver(kfilter)
    mu, cov, _, _ = kfilter.batch_filter(zs, saver=saver)
    ll = saver.log_likelihood
    new_loglike = np.sum(ll)

    old_loglike = -100

    ratio = ((new_loglike * get_arr_pdf_from_dist(new_prior, prior_dist)) / (
                old_loglike * get_arr_pdf_from_dist(old_prior, prior_dist))).mean()

    print(time.time() - start)