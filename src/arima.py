from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np


def get_arma_model(train_ser: pd.Series, test_ser: pd.Series, order=(2, 1)):
    col = train_ser.name
    arima = ARIMA(train_ser.values, order=(order[0], 0, order[1]))
    arima_res = arima.fit()

    # arima_res.summary()
    mu_df = pd.DataFrame(arima_res.predict(), index=train_ser.index).rename(columns={0: col})

    forecasts = []
    for i in range(0, len(test_ser)):
        forecasts.append(arima_res.forecast(1))
        arima_res = arima_res.apply(endog=[test_ser.iloc[i]])
    forecasts = pd.DataFrame(forecasts, columns=[col], index=test_ser.index)

    return mu_df, forecasts
