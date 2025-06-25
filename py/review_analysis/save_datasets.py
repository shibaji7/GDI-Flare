import sys
sys.path.extend(["py/", "py/review_analysis/"])

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from loguru import logger # type: ignore
import datetime as dt # type: ignore
import numpy as np # type: ignore

import os
from fetch import Radar

def save_dataset(rad, dates):
    """
    Save the dataset for a given radar and dates to a CSV file.
    
    Parameters:
    r (Radar): Radar object containing the data.
    dates (list): List of datetime objects representing the dates.
    filename (str): The name of the file to save the dataset.
    """
    filename = f"dataset/{dates[0].strftime('%Y%m%d')}.{rad}.Z.fitacf.csv"
    if os.path.exists(filename):
        logger.info(f"File {filename} already exists. Skipping save.")
        return
    r = Radar(rad, dates, type="fitacf")
    frame = r.df.copy()
    frame = frame[
        (frame["time"] >= dates[0].replace(hour=11))
        & (frame["time"] <= dates[0].replace(hour=13))
        & (np.abs(frame["v"]) >= 50.)
        & (np.abs(frame["v"]) <= 1000.)
        & (frame["gflg"] == 0)
    ]
    
    frame.to_csv(filename, index=False)
    logger.info(f"Dataset saved to {filename}")
    return

if __name__ == "__main__":
    o = pd.read_csv("dataset/dates.csv", parse_dates=["dates"])
    o.run = o.run.astype(bool)
    for i, row in o.iterrows():
        if row.run:
            try:
                save_dataset("sas", [row.dates, row.dates + dt.timedelta(days=1)])
            except Exception as e:
                logger.error(f"Error processing {row.dates}: {e}")
        if i==24: break
