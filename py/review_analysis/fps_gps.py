import matplotlib.pyplot as plt
import sys
sys.path.extend([".", "py/", "py/review_analysis/"])
import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]

from fan import Fan
import xarray as xr
import numpy as np
import datetime as dt
from scipy.ndimage import median_filter
from radar import Radar


def grid_by_latlon_cell(
    q, xcell, ycell, dx, dy,
    xparam="glon", yparam="gdlat", zparam="tec"
):
    paramDF = q[ [xparam, yparam, zparam] ]
    X, Y  = np.meshgrid( xcell, ycell )
    Z = np.zeros_like(X)*np.nan
    for i, x in enumerate(xcell):
        for j, y in enumerate(ycell):
            df = paramDF[
                (paramDF[xparam]>=x) &
                (paramDF[xparam]<x+dx) &
                (paramDF[yparam]>=y) &
                (paramDF[yparam]<y+dy)
            ]
            if len(df) > 0: Z[j, i] = np.nanmean(df[zparam])
    Z = np.ma.masked_invalid(Z)
    return X,Y,Z

def get_gridded_parameters(q, xparam="time", yparam="slist", zparam="v", round=False):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if round:
        plotParamDF[xparam] = np.array(plotParamDF[xparam]).astype(int)
        plotParamDF[yparam] = np.array(plotParamDF[yparam]).astype(int)
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).agg(np.nanmean).reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot(index=xparam, columns=yparam)
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

class Gardient(object):

    def __init__(
        self, tec_df, xparam="glon", 
        yparam="gdlat", zparam="tec",
        lat_range=[0, 90], 
        lon_range=[-150, -70], 
        dlat=3, dlon=3
    ):
        self.tec_df = tec_df
        self.xparam = xparam
        self.yparam = yparam
        self.zparam = zparam
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.dlat = dlat
        self.dlon = dlon
        self.__setup__()
        return
    
    def __setup__(self):
        self.lon_cell = np.arange(self.lon_range[0], self.lon_range[1], self.dlon)
        self.lat_cell = np.arange(self.lat_range[0], self.lat_range[1], self.dlat)
        return

    def parse_matrix(self, method="grid_by_latlon"):
        if method == "grid":
            self.X, self.Y, self.Z = get_gridded_parameters(
                self.tec_df, xparam=self.xparam, 
                yparam=self.yparam, zparam=self.zparam
            )
        elif method == "grid_by_latlon":
            self.X, self.Y, self.Z = grid_by_latlon_cell(
                self.tec_df, self.lon_cell, self.lat_cell, self.dlon, self.dlat,
                xparam=self.xparam, yparam=self.yparam, zparam=self.zparam
            )
        return

    def grad2D_by_np(self):
        #Z = weighted_2d_filter(self.Z)
        self.dxZ, self.dyZ = np.gradient(self.Z.T, self.lon_cell, self.lat_cell)
        return

    def lay_plot(self, f, ax, add_q=True, scale=1.5, length=0.5):
        dxZ, dyZ = self.dxZ, self.dyZ
        xyz = f.proj.transform_points(f.geo, self.X, self.Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        ax.scatter(x, y, color="k", s=0.05)
        ql = ax.quiver(
            x,
            y,
            dxZ.T, 
            np.zeros_like(dxZ.T),
            scale=scale,
            headaxislength=0,
            linewidth=0.6,
            scale_units="inches",
            transform=f.proj,
        )
        if add_q:
            ax.quiverkey(
                ql,
                0.15,
                1.05,
                length,
                r"$\nabla_{\phi}n'_0$:"+str(1.5) +r" TECu/$^{\circ}$",
                labelpos="E",
                transform=ax.transAxes,
                color="r",
                fontproperties={"size": 6},
            )
        return

def plot_gps_fig3():
    dSep = xr.load_dataset("database/gps170906g.003.nc")
    f = Fan(
        ["sas"],
        dt.datetime(2017, 9, 6, 11, 50),
        fig_title="",
        nrows=1, ncols=1,
        coord="geo",
        central_longitude=-100.0, central_latitude=40.0,
        extent=[-130, -60, 20, 80],
        plt_lats=np.arange(20, 80, 10),
        sup_title=False, mark_lon=-100
    )
    ax = f.add_axes(True, True)
    dates = [dt.datetime.utcfromtimestamp(d) for d in dSep["timestamps"].values]
    gdlat, glon = (dSep["gdlat"].values, dSep["glon"].values)
    tec = dSep["tec"].values
    idate = dates.index(dt.datetime(2017, 9, 6, 11, 50))
    T = tec[idate, :, :]
    glon, gdlat = np.meshgrid(glon, gdlat)
    xyz = f.proj.transform_points(f.geo, glon, gdlat)
    x, y = xyz[:, :, 0], xyz[:, :, 1]
    im = ax.pcolor(
        x, y, T,
        cmap="Spectral",
        vmin=0,
        vmax=8,
        transform=f.proj,
        alpha=0.8
    )
    ax._add_colorbar(im, label="TEC [TECu]")
    d = dSep.to_dataframe().reset_index()
    d["time"] = d.timestamps.apply(lambda x: dt.datetime.utcfromtimestamp(x))
    du = d[d.time==dt.datetime(2017, 9, 6, 11, 50)]
    g = Gardient(du, dlat=3, dlon=3)
    g.parse_matrix()
    g.grad2D_by_np()
    g.lay_plot(f, ax, add_q=True)
    f.fig.savefig("figures/Figure03.png")
    f.close()
    return

def plot_gps_figS04():
    dSep = xr.load_dataset("database/gps170906g.003.nc")
    f = Fan(
        ["sas"],
        dt.datetime(2017, 9, 6, 11, 50),
        fig_title="",
        nrows=1, ncols=3,
        coord="geo",
        central_longitude=-100.0, central_latitude=40.0,
        extent=[-130, -60, 20, 80],
        plt_lats=np.arange(20, 80, 10),
        sup_title=False, mark_lon=-100
    )
    ax = f.add_axes(True, True)
    dates = [dt.datetime.utcfromtimestamp(d) for d in dSep["timestamps"].values]
    gdlat, glon = (dSep["gdlat"].values, dSep["glon"].values)
    tec = dSep["tec"].values
    idate = dates.index(dt.datetime(2017, 9, 6, 11, 50))
    T = tec[idate, :, :]
    glon, gdlat = np.meshgrid(glon, gdlat)
    xyz = f.proj.transform_points(f.geo, glon, gdlat)
    x, y = xyz[:, :, 0], xyz[:, :, 1]
    im = ax.pcolor(
        x, y, T,
        cmap="Spectral",
        vmin=0,
        vmax=8,
        transform=f.proj,
        alpha=0.8
    )
    d = dSep.to_dataframe().reset_index()
    d["time"] = d.timestamps.apply(lambda x: dt.datetime.utcfromtimestamp(x))
    du = d[d.time==dt.datetime(2017, 9, 6, 11, 50)]
    g = Gardient(du, dlat=1, dlon=1)
    g.parse_matrix()
    g.grad2D_by_np()
    g.lay_plot(f, ax, add_q=True, scale=3, length=0.5)
    ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = f.add_axes(False, False)
    im = ax.pcolor(
        x, y, T,
        cmap="Spectral",
        vmin=0,
        vmax=8,
        transform=f.proj,
        alpha=0.8
    )
    g = Gardient(du, dlat=3, dlon=3)
    g.parse_matrix()
    g.grad2D_by_np()
    g.lay_plot(f, ax, add_q=False, scale=3, length=0.5)
    ax.text(0.05, 0.95, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = f.add_axes(False, False)
    im = ax.pcolor(
        x, y, T,
        cmap="Spectral",
        vmin=0,
        vmax=8,
        transform=f.proj,
        alpha=0.8
    )
    ax._add_colorbar(im, label="TEC [TECu]")
    g = Gardient(du, dlat=5, dlon=5)
    g.parse_matrix()
    g.grad2D_by_np()
    g.lay_plot(f, ax, add_q=False,  scale=3, length=0.5)
    ax.text(0.05, 0.95, "(c)", transform=ax.transAxes, fontsize=10, fontweight="bold")

    f.fig.savefig("figures/FigureS04.png")
    f.close()
    return


def get_data_for_date(
    rad="sas", 
    dates = [
        dt.datetime(2017,9,6,11),
        dt.datetime(2017,9,6,17)
    ],  
    tec_file="database/gps170906g.003.nc"
):
    radar = Radar(rad, dates)
    dTec = xr.load_dataset(tec_file)
    return radar, dTec

def plot_gps_fig7():
    radar, dSep = get_data_for_date()
    f = Fan(
        ["sas"],
        dt.datetime(2017, 9, 6, 11, 50),
        fig_title="",
        nrows=2, ncols=2,
        coord="geo",
        central_longitude=-100.0, central_latitude=40.0,
        extent=[-130, -60, 20, 80],
        plt_lats=np.arange(20, 80, 10),
        sup_title=False, mark_lon=-100
    )
    ax = f.add_axes(True, True)
    f.add_circle(ax, 60, -105, width=3, height=3)
    d = dSep.to_dataframe().reset_index()
    d["time"] = d.timestamps.apply(lambda x: dt.datetime.utcfromtimestamp(x))
    du = d[d.time==dt.datetime(2017, 9, 6, 11, 50)]
    g = Gardient(du, dlat=3, dlon=3)
    g.parse_matrix()
    g.grad2D_by_np()
    g.lay_plot(f, ax, add_q=True)

    radar, dSep = get_data_for_date(
        dates=[
            dt.datetime(2017,8,30,11),
            dt.datetime(2017,9,30,17)
        ]
    )

    f.fig.savefig("figures/Figure07.png")
    f.close()
    return

if __name__ == "__main__":
    # plot_gps_fig3()
    # plot_gps_figS04()
    plot_gps_fig7()