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
        X, Y, hist_smooth.T, levels=5, cmap="GnBu"
    )
    cbar_ax = fig.add_axes([1.01, 0.15, 0.03, 0.6])
    cb = fig.colorbar(cf, cax=cbar_ax, label="Counts")
    ax.set_title(title)
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=5, colors="black", 
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
    print(np.median(percentileofscore(hist, value, kind="weak")))
    return

def add_histogram(
    x, y, ax, fig, cbar=False, bins:int=500, 
    title="", xlabel="", ylabel="", txt="",  xline=None, yline=None,
    xline_locs=None, yline_locs=None, ylims=None, xlims=None,
):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    hist_smooth = gaussian_filter(hist, sigma=1.0)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)
    
    # hist_smooth[hist_smooth<np.mean(hist_smooth)-1*np.std(hist_smooth)] = np.nan
    hist_smooth = 6 * (hist_smooth-hist_smooth.min()) / (hist_smooth.max()-hist_smooth.min())
    cf = ax.pcolormesh(
        X, Y, np.log10(hist_smooth.T), cmap="GnBu",
        #levels=5, 
        #clim=[vmin, vmax]  # Set color limits for the contour fill
    )
    ax.set_xlim(xlims if xlims else (xedges[0], xedges[-1]))
    ax.set_ylim(ylims if ylims else (yedges[0], yedges[-1]))
    if cbar:
        pos = ax.get_position()
        cpos = [pos.x1 + pos.width * 0.3, pos.y0 + pos.height*.1,
                0.01, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb = fig.colorbar(cf, cax,
                   spacing="uniform",
                   orientation="vertical")
        cb.set_label("Density")
        # cbar_ax = fig.add_axes([1.01, 0.15, 0.03, 0.6])
        # cb = fig.colorbar(cf, ax=ax, cax=cbar_ax, label="Counts")
        # cb.set_clim(500, 2500)
    ax.set_title(title)
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=[2, 3, 5, 6], colors="black", 
        linewidths=0.2
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xline:
        ax.axvline(x=xline, color="m", ls="--", lw=0.5)
        ax.text(
            xline_locs[0], xline_locs[1], xline_locs[3], 
            ha="left", va="center", rotation=xline_locs[2],
            fontdict={"color": "m"}
        )
    if yline:
        ax.axhline(y=yline, color="r", ls="--", lw=0.5)
        ax.text(
            yline_locs[0], yline_locs[1], yline_locs[3], 
            ha="left", va="center", rotation=yline_locs[2],
            fontdict={"color": "r"}
        )
    ax.text(0.05, 0.95, txt, ha="left", va="center", transform=ax.transAxes)
    return

def plot_histograms_fig6(
    dfsep:pd.DataFrame, dfaug:pd.DataFrame,
    bins:int=500, filename:str="histogram.png",
):
    fig = plt.figure(figsize=(3*2, 3*2), dpi=1000)
    axes = [fig.add_subplot(221 + f) for f in range(4)]

    # Velocity/tau histogram
    add_histogram(
        np.log10(dfsep["tau_l"]), dfsep["v"].abs(), axes[0], fig, cbar=False, bins=bins,
        title=r"$\mathcal{D}$(log$_{10}\tau_l$, |v|)",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}\tau_l$, $s^{-1}$",
        txt="(a) 06 September 2017", xline=1.5, yline=400, 
        xline_locs=(1.7, 700, 90, fr"$\tau_l$ = {np.round(10**1.5, 2)} s$^{-1}$"),
        yline_locs=(3.5, 450, 0, r"$|v|$ = 400 m/s"), xlims=(1.2, 3.5), ylims=(0, 600),
    )
    # Velocity/range histogram
    add_histogram(
        np.log10(dfsep["srange"]), dfsep["v"].abs(), axes[1], fig, cbar=False, bins=bins,
        title=r"$\mathcal{D}$(log$_{10}S_r$, |v|)",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}S_r$, km",
        txt="(b)", xline=3.22, yline=400, 
        xline_locs=(3.25, 700, 90, fr"$S_r$ = {np.round(10**3.22, 2)} km"),
        yline_locs=(2.5, 450, 0, r"$|v|$ = 400 m/s"), xlims=(1.2, 3.5), ylims=(0, 600),
    )
    # # Range/tau histogram
    # add_histogram(
    #     np.log10(dfsep["srange"]), np.log10(dfsep["tau_l"]), axes[2], fig, cbar=False, bins=bins,
    #     title=r"$\mathcal{D}$(log$_{10}S_r$, log$_{10}\tau_l$)",
    #     ylabel=r"$log_{10}\tau_l$, $s^{-1}$", xlabel=r"$log_{10}S_r$, km",
    #     txt="(c)", xline=3.22, yline=400, 
    #     xline_locs=(3.25, 700, 90, fr"$S_r$ = {np.round(10**3.22, 2)} km"),
    #     yline_locs=(2.5, 450, 0, r"$|v|$ = 400 m/s"),
    # )


    # Velocity/tau histogram
    add_histogram(
        np.log10(dfaug["tau_l"]), dfaug["v"].abs(), axes[2], fig, cbar=False, bins=bins,
        title="",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}\tau_l$, $s^{-1}$",
        txt="(c) August 2017", xlims=(1.2, 3.5), ylims=(0, 600),
    )
    # Velocity/range histogram
    add_histogram(
        np.log10(dfaug["srange"]), dfaug["v"].abs(), axes[3], fig, cbar=True, bins=bins,
        title="",
        ylabel=r"Velocity ($|v|$), m/s", xlabel=r"$log_{10}S_r$, km",
        txt="(d)", xlims=(1.2, 3.5), ylims=(0, 600),
    )
    # # Range/tau histogram
    # add_histogram(
    #     np.log10(dfaug["srange"]), np.log10(dfaug["tau_l"]), axes[5], fig, cbar=True, bins=bins,
    #     title="", ylabel=r"$log_{10}\tau_l$, $s^{-1}$", xlabel=r"$log_{10}S_r$, km",
    #     txt="(f)"
    # )
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    return

def plot_histograms_fig06(
    dfsep:pd.DataFrame, dfaug:pd.DataFrame,
    bins:int=500, filename:str="histogram.png",
):
    fig = plt.figure(figsize=(3*2, 3*2), dpi=1000)
    axes = [fig.add_subplot(221 + f) for f in range(4)]

    ax = axes[0]
    x, y = (
        np.log10(dfsep["tau_l"]), dfsep["v"].abs()
    )
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    hist_smooth = gaussian_filter(hist, sigma=1.0)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)
    hist_smooth = 6 * (hist_smooth-hist_smooth.min()) /\
             (hist_smooth.max()-hist_smooth.min())
    hist_smooth[:, yedges[:-1]<50] *= 0.1
    cf = ax.contourf(
        X, Y, hist_smooth.T, cmap="GnBu",
        levels=5, 
    )
    ax.set_ylim([0, 600])
    ax.set_xlim([0, 3])
    ax.set_title(r"$\mathcal{D}$($\tau_l$, |v|)")
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=[0.25, 0.5, 0.75, 1], colors="black", 
        linewidths=0.2
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([])
    ax.set_ylabel("Velocity ($|v|$), m/s")
    ax.text(0.05, 0.95, "(a)", ha="left", va="top", transform=ax.transAxes)
    ax.axhline(y=400, color="r", ls="--", lw=0.5)
    ax.text(
        2.2, 425, "|v|=400 m/s", 
        ha="left", va="center", rotation=0,
        fontdict={"color": "r"}
    )
    ax.axvline(1.5, color="m", ls="--", lw=0.5)
    ax.text(
        1.6, 475, r"$\tau_l$=31.2 ms", 
        ha="left", va="center", rotation=90,
        fontdict={"color": "m"}
    )
    

    ax = axes[1]
    x, y = (
        np.log10(dfsep["srange"]), dfsep["v"].abs()
    )
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    hist_smooth = gaussian_filter(hist, sigma=1.0)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)
    hist_smooth = 6 * (hist_smooth-hist_smooth.min()) /\
             (hist_smooth.max()-hist_smooth.min())
    hist_smooth[:, yedges[:-1]<50] *= 0.1
    cf = ax.contourf(
        X, Y, hist_smooth.T, cmap="GnBu",
        levels=5, 
    )
    ax.set_ylim([0, 600])
    ax.set_xlim([2.8, 3.4])
    ax.set_title(r"$\mathcal{D}$($S_r$, |v|)")
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=[0.25, 0.5, 0.75, 1], colors="black", 
        linewidths=0.2
    )
    ax.set_yticklabels([])
    ax.set_xticks([2.8, 3, 3.2, 3.4])
    ax.set_xticklabels([])
    # ax.set_ylabel("Velocity ($|v|$), m/s")
    ax.text(1.1, 0.95, "06 September 2017", ha="right", va="top", transform=ax.transAxes, rotation=90)
    ax.text(0.05, 0.95, "(b)", ha="left", va="top", transform=ax.transAxes)
    ax.axhline(y=400, color="r", ls="--", lw=0.5)
    ax.text(
        2.9, 425, "|v|=400 m/s", 
        ha="left", va="center", rotation=0,
        fontdict={"color": "r"}
    )
    ax.axvline(3.23, color="m", ls="--", lw=0.5)
    ax.text(
        3.25, 475, fr"$S_r$={np.round(10**3.23, 1)} km", 
        ha="left", va="center", rotation=90,
        fontdict={"color": "m"}
    )

    pos = ax.get_position()
    cpos = [pos.x1 + pos.width * 0.1, pos.y0 + pos.height*.1,
                0.01, pos.height * 0.8]                # this list defines (left, bottom, width, height
    cax = fig.add_axes(cpos)
    cb = fig.colorbar(cf, cax,
                spacing="uniform",
                orientation="vertical")
    cb.set_label("Density")


    ax = axes[2]
    x, y = (
        np.log10(dfaug["tau_l"]), dfaug["v"].abs()
    )
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    hist_smooth = gaussian_filter(hist, sigma=1.0)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)
    hist_smooth = 6 * (hist_smooth-hist_smooth.min()) /\
             (hist_smooth.max()-hist_smooth.min())
    cf = ax.contourf(
        X, Y, hist_smooth.T, cmap="GnBu",
        levels=5, 
    )
    ax.set_ylim([0, 600])
    ax.set_xlim([0, 3])
    ax.set_title("")
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=[0.75, 1, 2, 4, 5], colors="black", 
        linewidths=0.2
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([1, 10, 100, 1000])
    ax.set_ylabel("Velocity ($|v|$), m/s")
    ax.set_xlabel(r"$\tau_l$, $ms$")
    ax.text(0.05, 0.95, "(c)", ha="left", va="top", transform=ax.transAxes)

    ax = axes[3]
    x, y = (
        np.log10(dfaug["srange"]), dfaug["v"].abs()
    )
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    hist_smooth = gaussian_filter(hist, sigma=1.0)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)
    hist_smooth = 6 * (hist_smooth-hist_smooth.min()) /\
             (hist_smooth.max()-hist_smooth.min())
    cf = ax.contourf(
        X, Y, hist_smooth.T, cmap="GnBu",
        levels=[0, 0.1, 0.2, 0.3, 1], 
    )
    ax.set_ylim([0, 600])
    ax.set_xlim([2.8, 3.4])
    ax.contour(
        X, Y, hist_smooth.T, 
        levels=[0, 0.1, 0.2, 0.3, 1], colors="black", 
        linewidths=0.2
    )
    ax.set_yticklabels([])
    ax.set_xticks([2.8, 3, 3.2, 3.4])
    ax.set_xticklabels(np.round([10**2.8, 10**3, 10**3.2, 10**3.4], 1))
    ax.set_xlabel(r"$S_r$, km")
    ax.text(1.1, 0.95, "August 2017", ha="right", va="top", transform=ax.transAxes, rotation=90)
    ax.text(0.05, 0.95, "(d)", ha="left", va="top", transform=ax.transAxes)
    
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    fig.savefig(filename, dpi=1000, bbox_inches="tight")
    return