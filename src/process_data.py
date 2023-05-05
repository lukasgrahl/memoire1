import os
import pandas as pd
import numpy as np

from pandas_datareader import fred
from statsmodels.tsa.filters.hp_filter import hpfilter
import yfinance as yf
from itertools import chain

from src.wrappers import skipna
from src.utils import str_time_format
from datetime import datetime


def get_yf_ticker_data(tickers: list, start: str, end: str, price_kind: list = ['Adj Close']) -> pd.DataFrame:
    """
    Pull data from yahoo finance based of yfinance tickers
    :param tickers: list of yfinance ticker
    :param start: start date in '%Y-%m-%d' format
    :param end: end date in '%Y-%m-%d' format
    :param price_kind: choose between Open, Close, Low, High, Volume
    :return: df of tickers and price_kind
    """

    df_prices = pd.DataFrame()

    df_prices.index = pd.date_range(start, periods=(
            str_time_format(end, '%m/%d/%Y') - str_time_format(start, '%m/%d/%Y')).days)

    for item in tickers:
        data = yf.download(item,
                           datetime.strftime(str_time_format(start, '%m/%d/%Y'), '%Y-%m-%d'),
                           datetime.strftime(str_time_format(end, '%m/%d/%Y'), '%Y-%m-%d'))
        data = data.drop('Close', axis=1)
        data.columns = list([f'{item}_{x}' for x in data.columns])
        df_prices = df_prices.join(data)

    # get closing price
    cols = [*chain.from_iterable([[item for item in df_prices.columns if price in item] for price in price_kind])]
    df_c = df_prices[cols].copy()
    if price_kind == ['Adj Close']: df_c.columns = [item[:-len(price_kind[0]) - 1] for item in df_c.columns]
    df_c.dropna(inplace=True)

    return df_c


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
        try:
            df[col].name = file_dict[col][1]
        except Exception as e:
            print(f'Error occured {e}, file_dict may be incomplete')

    return df


def get_recs_dict(ser: pd.Series):
    # create recession dict
    recs = ser > 35

    beg = recs & (recs != recs.shift(1))
    end = recs & (recs != recs.shift(-1))

    return np.array([beg[beg == True].index, end[end == True].index]).transpose()


from statsmodels.tsa.stattools import adfuller


@skipna
def ser_adf(ser: pd.Series, maxlag: int = 10, p_level: float = .05):
    """
    The null hypothesis of the Augmented Dickey-Fuller states there is a unit root, hence data is non-stationary.
    """
    _ = ser.copy()
    test = adfuller(_, maxlag=maxlag)
    print("Augmented Dickey-Fuller Test: H0 -> unit root")
    print(f"{'-' * 20} {ser.name} {'-' * 20}")
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