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
