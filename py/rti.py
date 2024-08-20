import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import numpy as np

def get_gridded_parameters(q, xparam="time", yparam="slist", zparam="v", round=False):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if round:
        plotParamDF[xparam] = np.array(plotParamDF[xparam]).astype(int)
        plotParamDF[yparam] = np.array(plotParamDF[yparam]).astype(int)
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).agg(np.nanmean).reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( index=xparam, columns=yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """
    def __init__(self, nrang, unique_times, fig_title, num_subplots=3, fov=None):
        self.fov = fov
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=240) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.95, ha="left", fontweight="bold", fontsize=18)
        return
    
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, xlabel="Time (UT)",
             ylabel="Range gate", zparam="v", label="Velocity (m/s)", add_gflg=False, sza_th=108,
             cmap = plt.cm.jet_r, ax=None):
        ax = ax if ax else self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam)
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
        ax.set_xlabel(xlabel, fontdict={"size":15, "fontweight": "bold"})
        ax.set_xlim(self.unique_times)
        ax.set_ylim([180, self.nrang])
        ax.set_ylabel(ylabel, fontdict={"size":15, "fontweight": "bold"})
        ax.set_title(title, loc="right", fontdict={"fontweight": "bold"})
        if add_gflg:
            Xg, Yg, Zg = get_gridded_parameters(df, xparam="time", yparam="slist", zparam="gflg")
            Zx = np.ma.masked_where(Zg==0, Zg)
            ax.pcolormesh(Xg, Yg, Zx.T, lw=0.01, edgecolors="None", cmap="gray",
                        vmax=2, vmin=0, shading="nearest")
            Z = np.ma.masked_where(Zg==1, Z)
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest")
        else:
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest")
        self._add_colorbar(im, ax, cmap, label=label)
        self.overlay_sza(ax, df.time.unique(), beam, [0, np.max(df.gate)], 
                df.rsep.iloc[0], df.frang.iloc[0], sza_th)
        return ax

    def overlay_sza(self, ax, times, beam, gate_range, rsep, frang, th=108):
        R = 6378.1
        from pysolar.solar import get_altitude
        gates = np.arange(gate_range[0], gate_range[1])
        SZA = np.zeros((len(times), len(gates)))
        for i, d in enumerate(times):
            d = d.to_pydatetime().replace(tzinfo=dt.timezone.utc)#dt.datetime.utcfromtimestamp(d.astype(dt.datetime) * 1e-9).replace(tzinfo=dt.timezone.utc)
            for j, g in enumerate(gates):
                gdlat, glong = self.fov[0][g, beam],self.fov[1][g, beam]
                sza = 90.-get_altitude(gdlat, glong, d)
                if (sza > 85.) & (sza < 120.): sza += np.rad2deg(np.arccos(R/(R+300)))
                SZA[i,j] = sza
        ZA = np.zeros_like(SZA)
        ZA[SZA>th] = 1.
        ZA[SZA<=th] = 0.
        times, gates = np.meshgrid(times, frang + (rsep*gates))
        ax.pcolormesh(times.T, gates.T, ZA, lw=0.01, edgecolors="None", cmap="gray_r",
                        vmax=2, vmin=0, shading="nearest", alpha=0.3)
        return

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=15)
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")

    def close(self):
        self.fig.clf()
        plt.close()

    def _add_colorbar(self, im, ax, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        fig = ax.get_figure()
        pos = ax.get_position()
        cpos = [pos.x1 + pos.width * 0.01, pos.y0 + pos.height*.1,
                0.01, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb2 = fig.colorbar(im, cax,
                   spacing="uniform",
                   orientation="vertical", 
                   cmap=colormap)
        cb2.set_label(label)
        return