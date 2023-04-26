import pandas as pd
import numpy as np

import sys
import time
import threading

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False

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