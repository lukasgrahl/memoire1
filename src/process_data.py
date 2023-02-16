import pandas as pd
import numpy as np

from pandas_datareader import fred
from src.utils import str_time_format
from statsmodels.tsa.filters.hp_filter import hpfilter

import os


def load_data(filename: str,
              path: str,
              file_dict: dict,
              index_col: str = 'date',
              time_freq: str = 'QS'):
    # only support csv atm

    df = pd.read_csv(os.path.join(path, filename), index_col=index_col)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    df.index = pd.to_datetime(df.index)
    df = df.asfreq(time_freq)

    for col in df.columns:
        df[col].name = file_dict[col][1]

    return df


def get_fred_data(fred_dict: dict,
                  start: str,
                  end: str,
                  freq: str = "QS"):
    df = pd.DataFrame()
    df.index = pd.date_range(start, end, freq=freq)

    for col, sym in fred_dict.items():
        data = fred.FredReader(symbols=sym[0], start=start, end=end).read()
        data.columns = [col]
        data[col].name = sym[1]

        data.index = pd.Series(data.index).apply(lambda x: str_time_format(str(x), '%Y-%m-%d %H:%M:%S'))

        df = df.join(data)
    return df


def get_recs_dict(ser: pd.Series):
    # create recession dict
    recs = ser > 35

    beg = recs & (recs != recs.shift(1))
    end = recs & (recs != recs.shift(-1))

    return np.array([beg[beg == True].index, end[end == True].index]).transpose()

from statsmodels.tsa.stattools import adfuller
def ser_adf(ser: pd.Series, maxlag: int=10, p_level: float=.05):
    """
    The null hypothesis of the Augmented Dickey-Fuller states there is a unit root, hence data is non-stationary.
    """
    test = adfuller(ser, maxlag=maxlag)
    print("Augmented Dickey-Fuller Test: H0 -> unit root")
    print(f"{'-'*20} {ser.name} {'-'*20}")
    print(f" p-val: {test[1]},  reject: {test[1] <= p_level}")
    # print(test)
    print('\n')
    pass

def get_seasonal_hp(ser: pd.Series, lamb: float = 1600.0, **kwargs):
    """
    returns: cylce or trend based on "return_trend"
    """
    cycle, trend = hpfilter(ser, lamb=lamb, **kwargs)
    return cycle, trend
