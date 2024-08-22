import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt

def create_dtec_error_distribution(dtec, ptec, txt, fname="gps.dtec_error_dist.png"):
    setsize(10)
    fig = plt.figure(figsize=(8, 4), dpi=300)
    ax = fig.add_subplot(121)
    ax.hist(
        dtec, bins=100, density=False, 
        histtype="step", color="r", ls="-", 
        lw=0.7
    )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1e4)
    ax.set_ylabel("Count(bins)")
    ax.set_xlabel("dTEC, TECu")
    ax.text(0.05, 1.05, txt, ha="left", va="center", transform=ax.transAxes)

    ax = fig.add_subplot(122)
    ax.hist(
        ptec, bins=100, density=False, 
        histtype="step", color="r", ls="-", 
        lw=0.7
    )
    ax.set_ylabel("Count(bins)")
    ax.set_xlabel("\% dTEC")
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_ylim(1, 1e4)
    fig.savefig(f"figures/{fname}", bbox_inches="tight")
    return

def create_eiscat_line_plot(eiscat, fname, size=10):
    setsize(size)
    import matplotlib
    import numpy as np
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(211)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    import utils
    X, Y, Z = utils.get_gridded_parameters(eiscat.yf, "Time", "Alt", "Ne", rounding=True)
    # im = ax.pcolormesh(
    #         X, Y, Z.T, lw=0.01, edgecolors="None", cmap="plasma",
    #         norm=matplotlib.colors.LogNorm(vmax=1e12, vmin=1e10),
    #         shading='gouraud'
    #         #vmin=12e9, vmax=1e12
    # )
    im = ax.scatter(
        eiscat.yf.Time, eiscat.yf.Alt, c=eiscat.yf.Ne, s=10, marker="s", alpha=0.7,
        norm=matplotlib.colors.LogNorm(vmax=3e11, vmin=1e9),
        edgecolors="None", cmap="gist_rainbow", lw=0.01
    )
    pos = ax.get_position()
    cpos = [
        pos.x1 + 0.025,
        pos.y0 + 0.0125,
        0.015,
        pos.height * 0.9,
    ]  # this list defines (left, bottom, width, height
    cax = fig.add_axes(cpos)
    cb = fig.colorbar(im, ax=ax, cax=cax)
    cb.set_label(r"$N_e$, $m^{-3}$")
    ax.set_ylabel("Height, km")
    ax.axvline(dt.datetime(2017,9,6,12,2), color="k", ls="-", lw=1.2)
    ax.set_ylim(70, 400)
    ax.set_xlim(dt.datetime(2017,9,6,11), dt.datetime(2017,9,6,15))

    ax = fig.add_subplot(212)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    eiscat._get_at_specifif_height_(ax, 100, color="r")
    eiscat._get_at_specifif_height_(ax, 150, color="b")
    eiscat._get_at_specifif_height_(ax, 200, color="k")
    ax.set_xlim(dt.datetime(2017,9,6,11), dt.datetime(2017,9,6,15))
    ax.axvline(dt.datetime(2017,9,6,12,2), color="k", ls="-", lw=1.2)
    ax.set_xlabel("Time, UT")
    ax.set_ylabel(r"$N_e$, $\times 10^{9}$ $m^{-3}$")
    ax.legend(loc=1)
    ax.set_ylim(50, 300)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.savefig(f"figures/{fname}", bbox_inches="tight")
    return

def setsize(size=12):
    import matplotlib as mpl

    import matplotlib.pyplot as plt
    plt.style.use(["science", "ieee"])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]

    mpl.rcParams.update(
        {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
    )
    return


def plot_figure2(radars, dates, rads=["sas","pgr","kod"], tags=["(a)", "(b)", "(c)"]):
    import numpy as np
    from rti import RangeTimePlot
    setsize(12)
    dates = [
        dt.datetime(2017,9,6,11),
        dt.datetime(2017,9,6,17)
    ]
    fig = plt.figure(figsize=(8, 9), dpi=240)
    beams = [7, 7, 10]
    for i, rad in enumerate(rads):
        ax = fig.add_subplot(311+i)
        radr = radars[rad]
        rti = RangeTimePlot(3000, dates, "", 1, fov=radr.fov)
        o = radr.df
        o["gate"] = np.copy(o.slist)
        o.slist = (o.slist*o.rsep) + o.frang
        o["unique_tfreq"] = o.tfreq.apply(lambda x: int(x/0.5)*0.5)
        o = o[o.unique_tfreq.isin([10.5])]
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
        ax.text(0.01,0.9,tags[i] +"\t"+ rad.upper() + fr" / {beams[i]} / $f_0\sim$10.5 MHz",ha="left",va="center",transform=ax.transAxes,)
        ax = rti.addParamPlot(
            o, beams[i], "", xlabel="", add_gflg=True, ax=ax, p_max=300, p_min=-300
        )
        ax.set_ylabel("Slant Range (km)")
        ax.axvline(dt.datetime(2017, 9, 6, 12, 2), color="k", ls="-", lw=1.2, label="12:02 UT")
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.set_xlabel("Time (UT)")
    fig.savefig("figures/IS_RTI.png", bbox_inches="tight")
    return