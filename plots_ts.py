import matplotlib.pyplot as plt

import mplstyle
import matplotlib as mpl
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import numpy as np
from scipy.ndimage import gaussian_filter as GF
from radar import Radar
from pysolar.solar import get_altitude
from goes import FlareTS

from plots import RangeTimePlot as RTI


def to_datetime(date):
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return dt.datetime.utcfromtimestamp(timestamp)

def parse_omni():
    o = []
    with open("database/omni.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = list(filter(None, line.replace("\n", "").split(" "))) 
            o.append({
                "date": dt.datetime(int(line[0]), 1, 1) +\
                    dt.timedelta(int(line[1])-1) + dt.timedelta(hours=int(line[2])) +\
                    dt.timedelta(minutes=int(line[3])),
                "Bx": float(line[4]), "By": float(line[5]), "Bz": float(line[6]),
                "Vx": float(line[7]), "Vy": float(line[8]), "Vz": float(line[9]),
                "Pd": float(line[10]), "Fd": float(line[11]), "AE": float(line[12]),
                "AL": float(line[13]), "AU": float(line[14]), "SYMH": float(line[15]), 
                "ASYH": float(line[16])
            })
    o = pd.DataFrame.from_records(o)
    return o

def plot_figure1():
    omni = parse_omni()
    ft = FlareTS([dt.datetime(2017,9,6), dt.datetime(2017,9,7)], flare_info={})

    fig = plt.figure(figsize=(8, 18), dpi=240)
    ax = fig.add_subplot(611)
    ft.plot_TS_from_axes(ax, xlabel="")
    ax.text(0.1,0.9,"(II.a)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=1.2, label="12:02 UT")
    ax.text(0.95,1.05,"Flare Class: X9.3 / 6 September 2017",ha="right",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})

    ax = fig.add_subplot(612)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.Bz, "ko", ms=1, ls="None")
    ax.set_ylim(-5, 5)
    ax.set_xlim([dt.datetime(2017, 9, 6), dt.datetime(2017, 9, 7)])
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=1.2)
    ax.set_ylabel(r"IMF, $B_z$ (nT)", fontdict={"size":15, "fontweight": "bold", "color":"k"})
    ax.text(0.1,0.9,"(II.b)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    ax = ax.twinx()
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=15)
    ax.set_ylim(-5, 5)
    ax.plot(omni.date, omni.By, "bo", ms=1, ls="None", label=r"$B_y$")
    ax.set_ylabel(r"IMF, $B_y$ (nT)", fontdict={"size":15, "fontweight": "bold", "color":"b"})
    ax.set_xticklabels([])
    
    ax = fig.add_subplot(613)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=12)
    ax.plot(omni.date, omni.Pd, "ko", ls="None", ms=0.8)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.set_ylabel("Proton Density (\#/cc)", fontdict={"size":15, "fontweight": "bold"})
    ax.text(0.1,0.9,"(II.c)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    ax.set_ylim(0, 12)
    ax.set_xticklabels([])
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=0.6)
    ax = ax.twinx()
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.Fd, "bo", ls="None", ms=0.8)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.set_ylabel("Dynamic pressure (nPa)", fontdict={"size":15, "fontweight": "bold", "color":"b"})
    ax.set_ylim(0, 4)
    ax.set_xticklabels([])
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=1.2, label="12:02 UT")

    ax = fig.add_subplot(614)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.AU, "r-", lw=0.8, label="AU")
    ax.plot(omni.date, omni.AL, "b-", lw=0.8, label="AL")
    ax.plot(omni.date, omni.AE, "k-", lw=0.8, label="AE")
    ax.legend(loc=1, fontsize=12)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.set_ylabel("AE/AL/AU (nT)", fontdict={"size":15, "fontweight": "bold"})
    ax.text(0.1,0.9,"(II.d)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    ax.set_ylim(-500, 1000)
    ax.set_xticklabels([])
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=1.2, label="12:02 UT")

    ax = fig.add_subplot(615)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.SYMH, "k", lw=0.8)
    ax.set_ylabel("SYM-H (nT)", fontdict={"size":15, "fontweight": "bold"})
    ax.set_ylim(-50, 50)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=1.2, label="12:02 UT")
    ax = ax.twinx()
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.ASYH, "b", lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("ASY-H (nT)", fontdict={"size":15, "fontweight": "bold", "color":"b"})
    ax.set_xlabel("Time (UT)", fontdict={"size":15, "fontweight": "bold"})
    ax.text(0.1,0.9,"(II.e)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    ax.set_xticklabels([])
    
    ax = fig.add_subplot(616)
    dates=[
            dt.datetime(2017,9,6),
            dt.datetime(2017,9,7)
        ]
    rad = Radar("sas", dates)
    o = rad.df
    o["gate"] = np.copy(o.slist)
    o.slist = (o.slist*o.rsep) + o.frang
    o["unique_tfreq"] = o.tfreq.apply(lambda x: int(x/0.5)*0.5)
    o = o[o.unique_tfreq.isin([10.5])]
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.text(0.1,0.9,"(II.f)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    rti = RTI(rad.fov, 3000, dates, "", 1)
    ax = rti.addParamPlot(
        o, 7, "", add_gflg=True, ax=ax, p_max=300, p_min=-300
    )
    ax.set_ylabel("Slant Range (km)", fontdict={"size":15, "fontweight": "bold"})
    ax.set_xlabel("Time (UT)", fontdict={"size":15, "fontweight": "bold"})
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=1.2, label="12:02 UT")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))

    fig.savefig("dataset/RTI.png", bbox_inches="tight")
    return

def plot_figure2():
    tags = ["(a)", "(b)", "(c)"]
    dates = [
        dt.datetime(2017,9,6),
        dt.datetime(2017,9,7)
    ]
    fig = plt.figure(figsize=(8, 9), dpi=240)
    for i, rad in enumerate(["sas","pgr","kod"]):
        ax = fig.add_subplot(311+i)
        rad = Radar(rad, dates)
        rti = RTI(rad.fov, 3000, dates, "", 1)
        o = rad.df
        o["gate"] = np.copy(o.slist)
        o.slist = (o.slist*o.rsep) + o.frang
        o["unique_tfreq"] = o.tfreq.apply(lambda x: int(x/0.5)*0.5)
        o = o[o.unique_tfreq.isin([10.5])]
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
        ax.text(0.1,0.9,tags[i],ha="left",va="center",transform=ax.transAxes, 
                fontdict={"size":15, "fontweight": "bold"})
        ax = rti.addParamPlot(
            o, 7, "", add_gflg=True, ax=ax, p_max=300, p_min=-300
        )
        ax.set_ylabel("Slant Range (km)", fontdict={"size":15, "fontweight": "bold"})
        ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="r", ls="-", lw=1.2, label="12:02 UT")
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.set_xlabel("Time (UT)", fontdict={"size":15, "fontweight": "bold"})
    fig.savefig("dataset/IS_RTI.png", bbox_inches="tight")
    return

if __name__ == "__main__":
    plot_figure1()
    plot_figure2()