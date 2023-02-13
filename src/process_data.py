import pandas as pd
import numpy as np

from pandas_datareader import fred
from src.utils import str_time_format


def get_fred_data(fred_dict: dict,
                  start: str,
                  end: str,
                  freq: str = "QS"):
    df = pd.DataFrame()
    df.index = pd.date_range(start, end, freq=freq)

    for col, sym in fred_dict.items():
        data = fred.FredReader(symbols=sym, start=start, end=end).read()
        data.columns = [col]

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



from statsmodels.tsa.filters.hp_filter import hpfilter


def get_seasonal_hp(ser: pd.Series, lamb: float = 1600.0, **kwargs):
    """
    returns: cylce or trend based on "return_trend"
    """
    cycle, trend = hpfilter(ser, lamb=lamb, **kwargs)
    return cycle, trend
