import pandas as pd
import numpy as np

class PolyDetrend:

    def __init__(self,
                 ser: pd.Series,
                 poly_order: int):
        self.data = ser.copy()
        self.poly_order = poly_order
        self._x = range(0, len(self.data))

        self.model = None

        self._get_model()
        pass

    def _get_model(self,
                   **kwargs):
        self.model = np.polyfit(self._x, self.data.values, self.poly_order, **kwargs)
        pass

    def get_detrend(self, **kwargs):
        return self.data - np.polyval(self.model, self._x, **kwargs)

    def predict_detrend(self, data: pd.Series, **kwargs):
        return data - np.polyval(self.model, range(0, len(data)), **kwargs)
    pass