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
        np.log10(df["w_l"]), 
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