import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils import all_equal


def plot_dfs(dfs: pd.DataFrame,
             plotfunc,
             cols: int = 3,
             figsize: tuple = (8, 3),
             fill_arr=None,
             **kwargs):
    # plotfunc kwargs

    if type(dfs) != list:
        dfs = [dfs]

    for df in dfs:
        assert type(df) == pd.core.frame.DataFrame, "dfs list item is not a pandas data frame"

    # assert all_equal([item.columns for item in dfs]) == True, "df columns have to be the same"

    # set rows based on max dimensions
    rows = int(round(max([item.shape[1] for item in dfs]) / cols))

    start = min([item.index.min() for item in dfs])
    end = max([item.index.max() for item in dfs])

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize[0], figsize[1] * rows))
    fig.autofmt_xdate(rotation=45)

    for df in dfs:

        for i, col in enumerate(df.columns):
            _axr = int(np.floor(i / cols))
            _axc = int(round((i / cols - np.floor(i / cols)) * cols))

            plotfunc(df[col], ax=ax[_axr, _axc], **kwargs)
            ax[_axr, _axc].set_title(df[col].name)

            if fill_arr is not None:
                # only inlcude relevant recessions
                for t in fill_arr:
                    if t[1] < start:
                        continue
                    if t[0] < start:
                        t[0] = start
                    if t[0] > end:
                        continue
                    if t[1] > end:
                        t[1] = end
                    # plot recessions
                    ax[_axr, _axc].axvspan(t[0], t[1], alpha=.1, color='red')

    fig.tight_layout()

    plt.show()
    pass