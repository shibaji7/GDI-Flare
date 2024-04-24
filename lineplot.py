import matplotlib.pyplot as plt
import mplstyle
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]

import matplotlib as mpl
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import numpy as np
from scipy.ndimage import gaussian_filter as GF
from radar import Radar
from pysolar.solar import get_altitude
from gps import GPS1X1, Gardient

def to_datetime(date):
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return dt.datetime.utcfromtimestamp(timestamp)


def plot_TEC_TS():
    mpl.rcParams.update({"xtick.labelsize": 15, "ytick.labelsize":15, "font.size":15})
    fig = plt.figure(figsize=(8, 12), dpi=200)
    axes = [fig.add_subplot(511+i) for i in range(4)]
    labels = ["(a)","(b)","(c)","(d)","(e)"]
    for i,ax in enumerate(axes):
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 20)))
        ax.axvline(dt.datetime(2017,9,6,12,2), color="r", ls="-", lw=0.9)
        ax.axvline(dt.datetime(2017,9,6,11,56), color="m", ls="--", lw=0.6)
        ax.text(0.05, 0.95, labels[i], ha="left", va="center", transform=ax.transAxes)
    axes[3].set_xlabel("Time [UT]", fontdict={"size":15, "fontweight": "bold"})
    return fig, axes

def generate_gradients(dates, g, ds_counter=1, clat=55, grid_lat=5, clon=-95, grid_lon=5):
    du = g.frames[g.__fetch_key__(ds_counter)]["data"].copy()
    du = du[
        (du.TIME>=dates[0])
        & (du.TIME<dates[1])
    ]
    times = du.TIME.unique()
    dxZ, dyZ, Z = [], [], []
    for d in times:
        gx = Gardient(du[du.TIME == d], dlat=grid_lat, dlon=grid_lon)
        gx.parse_matrix()
        gx.grad2D_by_np()
        i, j = (
            gx.lon_cell.tolist().index(clon),
            gx.lat_cell.tolist().index(clat)
        )
        dxZ.append(gx.dxZ[j,i])
        dyZ.append(gx.dyZ[j,i])
        Z.append(gx.Z[j,i])
    return np.array(dxZ)*grid_lon, np.array(dyZ)*grid_lat, Z, times

def fetch_velocity(radars, dates, times, gdlat_lim=[57.5, 62.5], glong_lim=[-107.5, -102.5]):
    from scipy.stats import median_abs_deviation
    data = radars[dates[0]].df.copy()
    data = data[
        (data.time>=dates[0]) &
        (data.time<dates[1]) &
        (data.slist<=75) &
        (data.slist>7) &
        (data.gdlat<=gdlat_lim[1]) &
        (data.gdlat>=gdlat_lim[0]) &
        (data.glong<=glong_lim[1]) &
        (data.glong>=glong_lim[0])
    ]
    data.time = data.time.apply(lambda x: x.replace(microsecond=0))
    t0, v, vstd, vtop, vbot = dates[0], [], [], [], []
    qu, ql = 0.60, 0.40
    for t in times:
        o = data[
            (data.time<=t) &
            (data.time>=t0)
        ]
        t0 = t
        m = (np.quantile(o.v, qu) + np.quantile(o.v, ql))/2
        v.append(m)
        vtop.append(np.quantile(o.v, qu)-m)
        vbot.append(m-np.quantile(o.v, ql))
        print((np.quantile(o.v, qu)+np.quantile(o.v, ql))/2, np.quantile(o.v, qu), np.quantile(o.v, ql))
    o = pd.DataFrame()
    o["mid"], o["top"], o["bot"] = v, vtop, vbot
    return np.array(v), np.array(vtop), np.array(vbot)

def plot_lines():
    g1 = GPS1X1(
        "database/gps170906g.003.txt.gz", 
        "TXT.GZ"
    )
    g2 = GPS1X1(
        "database/gps170830g.002.txt.gz", 
        "TXT.GZ"
    )
    dates=[
        dt.datetime(2017,9,6,11),
        dt.datetime(2017,9,6,13)
    ]
    
    dxZ, dyZ, Z, times = generate_gradients(
        dates, g1, clat=60, grid_lat=5, 
        clon=-105, grid_lon=5, ds_counter=1
    )
    dxZ0, dyZ0, Z0, times0 = generate_gradients(
        [dates[0]-dt.timedelta(7), dates[1]-dt.timedelta(7)], 
        g2, clat=60, grid_lat=5, 
        clon=-105, grid_lon=5, ds_counter=1
    )
    radars = {}
    for ddates in [dates, [dates[0]-dt.timedelta(7), dates[1]-dt.timedelta(7)]]:
        rad_data = Radar("sas", ddates)
        radars[ddates[0]] = rad_data
    SZA = []
    for d in times:
        sza = 90.-get_altitude(60, -105, d.replace(tzinfo=dt.timezone.utc))
        if (sza > 85.) & (sza < 120.): sza += np.rad2deg(np.arccos(6371/(6371+300)))
        SZA.append(sza)
    ZA = np.zeros_like(SZA)
    ZA[np.array(SZA)>110] = 1.
    ZA[np.array(SZA)<=110] = 0.
    zaI = (ZA==0).argmax(axis=0)

    fig, axes = plot_TEC_TS()
    ax = axes[0]
    ax.plot(times, Z, "ko", ms=2.5, ls="None", label=r"06 Sep")
    ax.plot(times, Z0, "ro", ms=2.5, ls="None", label=r"30 Aug")
    ax.set_ylabel(
        "$n'_0$", 
        fontdict={"size":15, "fontweight": "bold"}
    )
    ax.text(0.05, 1.05, r"$\lambda,\phi=%d^\circ,%d^\circ$"%(60, -105), 
            ha="left", va="center", transform=ax.transAxes,
            fontdict={"size":15}
            )
    ax.axvspan(dates[0], times[zaI], color="gray", alpha=0.4)
    ax.legend(loc=4, fontsize="15")
    ax.set_xlim(dates)
    ax.set_ylim(1, 8)
    ax.set_xticks([])

    ax = axes[1]
    ax.plot(times, dxZ, "ko", ms=2.5, ls="None")
    ax.plot(times, dxZ0, "ro", ms=2.5, ls="None")
    ax.set_ylabel(
        r"$\frac{\partial n'_0}{\partial \phi}$", 
        fontdict={"size":15, "fontweight": "bold"}
    )
    ax.set_xlim(dates)
    ax.axvspan(dates[0], times[zaI], color="gray", alpha=0.4)
    ax.set_ylim(0, 2)
    ax.set_xticks([])

    # ax = axes[2]
    # ax.plot(times, dxZ/Z, "ko", ms=2.5, ls="None")
    # ax.plot(times, dxZ0/Z0, "ro", ms=2.5, ls="None")
    # ax.set_ylabel(
    #     r"$\frac{1}{n'_0}\frac{\partial n'_0}{\partial \phi}$ [TECu/$^\circ$]", 
    #     fontdict={"size":15, "fontweight": "bold"}
    # )
    # ax.set_xlim(dates)
    # ax.axvspan(dates[0], times[zaI], color="gray", alpha=0.4)
    # ax.set_ylim(0, 0.5)
    # ax.set_xticks([])

    ax = axes[2]
    v, vtop, vbot = fetch_velocity(radars, dates, times)
    #ax.plot(times, v, "ko", ms=2.5, ls="None")
    ax.errorbar(times, v, 
                yerr=np.array([vbot.ravel(), vtop.ravel()]),
                fmt="o", ms=2.5, ls="None", color="k")
    v, vtop, vbot = fetch_velocity(radars, 
                            [dates[0]-dt.timedelta(7), dates[1]-dt.timedelta(7)],
                            times0)
    #ax.plot(times, v, "ro", ms=2.5, ls="None")
    ax.errorbar(times, v, yerr=np.array([vbot.ravel(), vtop.ravel()]),
                fmt="o", ms=2.5, ls="None", color="r")
    ax.set_ylabel(
        "$V'_0$ [m/s]",
        fontdict={"size":15, "fontweight": "bold"}
    )
    ax.axvspan(dates[0], times[zaI], color="gray", alpha=0.4)
    ax.set_xlim(dates)
    ax.set_ylim(-500, 500)
    ax.set_xticks([])

    ax = axes[3]
    eta = dxZ/Z
    v, vtop, vbot = fetch_velocity(radars, dates, times)
    ax.errorbar(times, np.abs(v)*eta, yerr=np.array([np.abs(vbot.ravel())*eta*.5, 0.5*np.abs(vtop.ravel())*eta]),
                fmt="o", ms=2.5, ls="None", color="k")
    #ax.plot(times, abs(v)*, "ko", ms=2.5, ls="None")
    v, vtop, vbot = fetch_velocity(radars, 
                            [dates[0]-dt.timedelta(7), dates[1]-dt.timedelta(7)],
                            times0)
    eta = dxZ0/Z0
    ax.errorbar(times, np.abs(v)*eta, yerr=np.array([np.abs(vbot.ravel())*eta*.5, 0.5*np.abs(vtop.ravel())*eta]),
                fmt="o", ms=2.5, ls="None", color="r")
    ax.set_ylabel(
        r"$\gamma'$", 
        fontdict={"size":15, "fontweight": "bold"}
    )
    ax.set_xlim(dates)
    ax.axvspan(dates[0], times[zaI], color="gray", alpha=0.4)
    ax.set_ylim(-10, 100)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.savefig(f"figures/TS.png", bbox_inches="tight")
    return

import os
os.mkdir("figures/")
plot_lines()