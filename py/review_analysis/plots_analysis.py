import matplotlib.pyplot as plt
import scienceplots
import numpy as np
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]
plt.rcParams.update(
    {
        "text.usetex": False,
    }
)
from scipy.ndimage import gaussian_filter

import pandas as pd

def plot_histograms(
        df:pd.DataFrame, column:str, 
        bins:int=500, title:str=r"$\mathcal{D}$(|v|, log_{10}$\tau_l$)", 
        xlabel:str=r"Velocity ($|v|$), m/s", ylabel:str=r"$log_{10}\tau_l$, $s^{-1}$",
        filename:str="histogram.png",
        abs_value:bool=False, color="red", ls="-", lw=0.4
    ):
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(111)
    # ax.hist(
    #     df[column].abs() if abs_value else df[column], bins=bins, 
    #     density=False, color=color, ls=ls, lw=lw,
    #     alpha=0.7, histtype="step"
    # )
    x, y = (
        df[column].abs() if abs_value else df[column], 
        np.log10(df["tau_l"]), 
    )
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist_smooth = gaussian_filter(hist, sigma=1.0) * 1e2
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)
    cf = ax.contourf(
        X, Y, hist_smooth.T, levels=5, cmap='GnBu'
    )
    cbar_ax = fig.add_axes([1.01, 0.15, 0.03, 0.6])
    cb = fig.colorbar(cf, cax=cbar_ax, label="Counts")
    ax.set_title(title)
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=5, colors='black', 
        linewidths=0.5
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.05, 0.95, "(b)", ha="left", va="center", transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)

    from scipy.stats import percentileofscore
    # print(xedges, yedges)
    point = (
        np.argmin(np.abs(xedges-500)),
        np.argmin(np.abs(yedges-2))
    )
    value = hist[point[0], point[1]]
    print(point, value)
    print(np.median(percentileofscore(hist, value, kind='weak')))
    return

def add_histogram(x, y, ax, fig, cbar=False, bins:int=500, title="", xlabel="", ylabel="", txt="", vmin=1e2, vmax=2500):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist_smooth = gaussian_filter(hist, sigma=1.0) * 1e2
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)
    cf = ax.pcolormesh(
        X, Y, hist_smooth.T, levels=5, cmap='GnBu',
        clim=[vmin, vmax]  # Set color limits for the contour fill
    )
    if cbar:
        pos = ax.get_position()
        cpos = [pos.x1 + pos.width * 0.5, pos.y0 + pos.height*.1,
                0.01, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb = fig.colorbar(cf, cax,
                   spacing="uniform",
                   orientation="vertical")
        cb.set_label("Counts")
        # cbar_ax = fig.add_axes([1.01, 0.15, 0.03, 0.6])
        # cb = fig.colorbar(cf, ax=ax, cax=cbar_ax, label="Counts")
        # cb.set_clim(500, 2500)
    ax.set_title(title)
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=5, colors='black', 
        linewidths=0.5
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.05, 0.95, txt, ha="left", va="center", transform=ax.transAxes)
    return

def plot_histograms_fig6(
    dfsep:pd.DataFrame, dfaug:pd.DataFrame,
    bins:int=500, filename:str="histogram.png",
):
    fig = plt.figure(figsize=(3*3, 3*2), dpi=1000)
    axes = [fig.add_subplot(231 + f) for f in range(6)]

    # Velocity/tau histogram
    add_histogram(
        np.log10(dfsep["tau_l"]), dfsep["v"].abs(), axes[0], fig, cbar=False, bins=bins,
        title=r"$\mathcal{D}$(log$_{10}\tau_l$, |v|)",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}\tau_l$, $s^{-1}$",
        txt="(a) 06 September 2017",
    )
    # Velocity/range histogram
    add_histogram(
        np.log10(dfsep["srange"]), dfsep["v"].abs(), axes[1], fig, cbar=False, bins=bins,
        title=r"$\mathcal{D}$(log$_{10}S_r$, |v|)",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}S_r$, km",
        txt="(b)",
    )
    # Range/tau histogram
    add_histogram(
        np.log10(dfsep["srange"]), np.log10(dfsep["tau_l"]), axes[2], fig, cbar=True, bins=bins,
        title=r"$\mathcal{D}$(log$_{10}S_r$, log$_{10}\tau_l$)",
        ylabel=r"$log_{10}\tau_l$, $s^{-1}$", xlabel=r"$log_{10}S_r$, km",
        txt="(c)",
    )


    # Velocity/tau histogram
    add_histogram(
        np.log10(dfaug["tau_l"]), dfaug["v"].abs(), axes[3], fig, cbar=False, bins=bins,
        title="",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}\tau_l$, $s^{-1}$",
        txt="(d) August 2017",
    )
    # Velocity/range histogram
    add_histogram(
        np.log10(dfaug["srange"]), dfaug["v"].abs(), axes[4], fig, cbar=False, bins=bins,
        title="",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}S_r$, km",
        txt="(e)",
    )
    # Range/tau histogram
    add_histogram(
        np.log10(dfaug["srange"]), np.log10(dfaug["tau_l"]), axes[5], fig, cbar=True, bins=bins,
        title="", ylabel=r"$log_{10}\tau_l$, $s^{-1}$", xlabel=r"$log_{10}S_r$, km",
        txt="(f)",
    )
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    return