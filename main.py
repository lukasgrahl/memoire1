from config import fred_end, fred_start, fred_dict, plt_config
from settings import DATA_DIR

from src.process_data import get_fred_data, get_recs_dict, get_yf_ticker_data
from src.plotting import plot_dfs

import pandas as pd

import os
import numpy as np

import seaborn as sns
import matplotlib as plt

if __name__ == "__main__":

    # set plt config
    plt.rcParams.update(plt_config)

    # get data from fred
    df = get_fred_data(fred_dict, fred_start, fred_end)
    recessions = get_recs_dict(df['recs'])
    # save recession dict
    np.save(os.path.join(DATA_DIR, 'recessions_periods.npy'), recessions)

    # get yfinance data
    wti = get_yf_ticker_data(['CL=F'], fred_start, fred_end)
    wti.rename(columns={'CL=F': 's'}, inplace=True)
    wti['quarter'] = pd.PeriodIndex(wti.index, freq='Q')
    wti = wti.drop_duplicates('quarter', keep='first')

    df['quarter'] = pd.PeriodIndex(df.index, freq='Q')
    df.reset_index(names='date', inplace=True)

    df = pd.merge(df, wti, on='quarter', how='left')
    df.drop('quarter', axis=1, inplace=True)

    # save data to DATA_DIR
    df.to_csv(os.path.join(DATA_DIR, 'raw_data.csv'))

    pass
