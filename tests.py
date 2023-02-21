import numpy as np

from src.process_data import load_data, ser_adf
from src.utils import apply_func

from settings import DATA_DIR
from config import fred_dict

if __name__ == "__main__":

    df = load_data('raw_data.csv', DATA_DIR, fred_dict)

    arr = df.w.copy()
    arr.iloc[-2] = np.nan

    apply_func(arr, func=np.log)
    ser_adf(arr)
