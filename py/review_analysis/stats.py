import sys
sys.path.extend(["py/", "py/review_analysis/"])

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from loguru import logger # type: ignore
import datetime as dt # type: ignore
import numpy as np # type: ignore

import os
from fetch import Radar
import plots_analysis as pa

def retset_frame(r, dates):
    frame = r.df.copy()
    frame = frame[
        (frame["time"] >= dates[0].replace(hour=11))
        & (frame["time"] <= dates[0].replace(hour=13))
        & (np.abs(frame["v"]) >= 50.)
        & (np.abs(frame["v"]) <= 1000.)
        & (frame["gflg"]==0)
    ]
    frame["srange"] = frame["slist"] * 45 + 180
    setattr(r, "frame", frame)
    return r

def get_rAug(rad):
    import glob
    files = glob.glob(f"dataset/201708*{rad}.fitacf.csv")
    files.sort()
    dates = [dt.datetime.strptime(f.split("/")[-1].split(".")[0], "%Y%m%d") for f in files]
    print(f"Files found: {files}, dates: {dates}")
    rAug = pd.concat([
        retset_frame(
            Radar(rad, [d, d+dt.timedelta(1)], type="fitacf"),
            [d, d+dt.timedelta(1)]
        ).frame
        for d in dates
    ])
    return rAug

def plot_histograms(rad):
    rSep = retset_frame(
        Radar(rad, [dt.datetime(2017,9,6), dt.datetime(2017,9,7),]),
        [dt.datetime(2017,9,6), dt.datetime(2017,9,7),]
    )
    rAugFrame = get_rAug("sas")
    # rAug = retset_frame(
    #     Radar(rad, [dt.datetime(2017,8,30), dt.datetime(2017,8,31),]),
    #     [dt.datetime(2017,8,30), dt.datetime(2017,8,31),]
    # )
    pa.plot_histograms_fig6(
        rSep.frame, rAugFrame, bins=100,
        filename=f"figures/Figure06.png"
    )
    return

def get_data_for_radar(rad, dates, ):
    r = Radar(rad, dates)
    frame = r.df.copy()
    frame = frame[
        (frame["time"] >= dates[0].replace(hour=11))
        & (frame["time"] <= dates[0].replace(hour=13))
        & (np.abs(frame["v"]) >= 50.)
        & (np.abs(frame["v"]) <= 1000.)
        & (frame["gflg"]==0)
    ]
    setattr(r, "frame", frame)
    logger.info(f"Data for {rad} on {dates}:\n{frame.head()}")
    pa.plot_histograms(
        frame, column="v", bins=50,
        
        filename=f"figures/histogram_{rad}.png", abs_value=True,
        color="red", ls="-", lw=0.4,
    )
    return

def check_events(dates):
    fname = f"database/figures/{dates[0].strftime('%Y%m%d')}.png"

    if not os.path.exists(fname):
        from fetch import SolarDataset
        sd = SolarDataset(
            dates, dataset=[2]
        )
        omni = sd.omni.copy()

        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(["science", "ieee"])
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]

        import matplotlib.dates as mdates
        from matplotlib.dates import DateFormatter
        plt.rcParams["text.usetex"] = False

        fig = plt.figure(figsize=(6, 3), dpi=240)
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
        ax.plot(omni.time, omni.AE, ls="-", color="b", lw=0.8)
        ax.set_ylabel("AE", fontdict=dict(color="b"))
        ax.set_ylim(0, 1000)
        ax.set_xlim(omni.time.iloc[0], omni.time.iloc[-1])
        ax.set_xlabel("Time, UT")
        ax = ax.twinx()
        ax.set_ylim(-50, 50)
        ax.plot(omni.time, omni.SymH, ls="-", color="r", lw=0.8)
        ax.set_ylabel("SymH", fontdict=dict(color="r"))
        fig.savefig(
            f"database/figures/{dates[0].strftime('%Y%m%d')}.png", 
            bbox_inches="tight"
        )
        import csv
        with open("dataset/events.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows([[
                omni.time.iloc[0], 
                omni.AE.max(),
                omni.SymH.iloc[np.abs(omni.SymH).argmax()]
            ]])
    return

if __name__ == "__main__":
    # Example usage
    rad = "sas"
    dates = [
        dt.datetime(2017,9,6), dt.datetime(2017,9,7),
    ]
    # get_data_for_radar(rad, dates)
    # check_events(dates)

    # for d in range(90):
    #     dates = [
    #         dt.datetime(2017,7,1)+dt.timedelta(d),
    #         dt.datetime(2017,7,1)+dt.timedelta(d+1),
    #     ]
    #     check_events(dates)
    plot_histograms(rad)