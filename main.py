from config import fred_end, fred_start, fred_dict, plt_config
from settings import DATA_DIR

from src.process_data import get_fred_data, get_recs_dict
from src.plotting import plot_dfs

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

    # plot data
    # plot_dfs([df], sns.lineplot, cols=2, fill_arr=recessions)

    # save data to DATA_DIR
    df.reset_index(names='date').to_csv(os.path.join(DATA_DIR, 'raw_data.csv'))

    pass
