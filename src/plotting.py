import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils import all_equal

from src.utils import all_equal, get_confidence_interval

from src.utils import all_equal


def plot_dfs(dfs_data: list,
             dfs_cov: list,
             plotfunc, cols: int = 3, figsize: tuple = (14, 4), fill_arr=None,
             legend: list = None, recs_lable='Recessions', conf_int: np.array = None, **kwargs):
    # if single data frame put into list
    if type(dfs_data) != list: dfs_data = [dfs_data]
    if type(dfs_cov) != list: dfs_cov = [dfs_cov]

    # check legend sanity
    if legend is not None: assert len(legend) == len(dfs_data), "No sufficient legend titles supplied"

    # check df sanity
    if dfs_cov is not None:
        for mu, cov in [[x, y] for x in dfs_data for y in dfs_cov]:
            _ = [item for item in mu.columns if item not in cov.columns]
            assert len(_) == 0, f"{_} not contained in dfs_cov"
            assert type(mu) == type(cov) == pd.core.frame.DataFrame, "dfs list item is not a pandas data frame"
    else:
        for df in dfs_data:
            assert type(df) == pd.core.frame.DataFrame, "dfs list item is not a pandas data frame"

    assert all_equal([list(item.columns) for item in dfs_data]) == True, "df columns have to be the same"

    # set rows based on max dimensions
    rows = int(np.ceil(max([item.shape[1] for item in dfs_data]) / cols))
    start = min([item.index.min() for item in dfs_data])
    end = max([item.index.max() for item in dfs_data])

    # initatiate figure
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize[0], figsize[1] * rows))

    for id, df in enumerate(dfs_data):

        for i, col in enumerate(df.columns):
            _axr = int(np.floor(i / cols))
            _axc = int(round((i / cols - np.floor(i / cols)) * cols))

            # excluding axcol value, in case of one dimensional axis
            if len(ax.shape) == 1:
                _ax = ax[_axc]
            else:
                _ax = ax[_axr, _axc]

            # plot confidence intervals
            if dfs_cov is not None:
                upper, lower = get_confidence_interval(df[col], cov[col])
                _ax.fill_between(df[col].index, upper, lower, color='b', alpha=.1)

            # plot graph
            if legend is not None:
                plotfunc(df[col], ax=_ax, label=legend[id], **kwargs)
            else:
                plotfunc(df[col], ax=_ax, **kwargs)
            _ax.set_title(col)

            if fill_arr is not None and id == len(dfs_data) - 1:
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
                    _ax.axvspan(t[0], t[1], alpha=.1, color='red')

    # if legend is not None: plt.legend()
    fig.tight_layout()
    plt.show()
    pass


