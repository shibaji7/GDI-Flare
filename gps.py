import os
import datetime as dt
from loguru import logger
import gzip
import pandas as pd
import numpy as np
from plot import CartoBase, grid_by_latlon_cell, get_gridded_parameters, plot_TEC_TS
from radar import Radar


def weighted_2d_filter(X, tau=0.05):
    W = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    Y = np.zeros_like(X)*np.nan
    for i in range(1, X.shape[0]-2):
        for j in range(1, X.shape[1]-2):
            v, wi = [], 0
            for x, w in zip(X[i-1:i+2, j-1:j+2].ravel(), W.ravel()):
                if not np.isnan(x):
                    v.extend([x]*w)
                    wi+=w
            #if wi/np.sum(W) > tau:
            Y[i,j] = np.nanmedian(v)
    return np.ma.masked_invalid(Y)

class Gardient(object):

    def __init__(
        self, tec_df, xparam="GLON", 
        yparam="GDLAT", zparam="TEC",
        lat_range=[0, 90], 
        lon_range=[-150, -70], 
        dlat=4, dlon=4
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

class GPS(object):

    def __init__(self, lat_range, lon_range):
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.frames = {}
        self.ds_counter = 0
        return

    def __fetch_key__(self, ds_counter=None):
        if ds_counter is not None:
            C = ds_counter
        else:
            C = self.ds_counter
            self.ds_counter += 1
        return f"DS{'%03d'%C}"

    def __fetch_file__(self):
        return

    def filter_by_date(self, dates, ds_counter):
        o = self.frames[self.__fetch_key__(ds_counter)].copy()
        o = o[
            (o.TIME>=dates[0])
            & (o.TIME<=dates[1])
        ]
        self.frames[self.__fetch_key__()] = {
            "data": pd.DataFrame.from_records(o),
            "parent": self.__fetch_key__(ds_counter)
        }
        return

    def filter_by_latlon(self, ds_counter=0, lat_range=None, lon_range=None):
        lat_range = lat_range if lat_range else self.lat_range
        lon_range = lon_range if lon_range else self.lon_range
        o = self.frames[self.__fetch_key__(ds_counter)].copy()
        print(o.head())
        o = o[
            (o.GDLAT>=lat_range[0])
            & (o.GDLAT<=lat_range[1])
            & (o.GLON>=lon_range[0])
            & (o.GLON<=lon_range[1])
        ]
        self.frames[self.__fetch_key__()] = {
            "data": pd.DataFrame.from_records(o),
            "parent": self.__fetch_key__(ds_counter)
        }
        return
    
    def __check_exists__(self, fname, ds_counter=None):
        if os.path.exists(fname):
            logger.info(f"Loading file: {fname}")
            o = pd.read_csv(fname, parse_dates=["TIME"])
            self.frames[self.__fetch_key__(ds_counter)] = o
            ret = True
        else: ret = False
        return ret
        
import os

class GPS1X1(GPS):

    def __init__(
        self, 
        fname, 
        ftype=None, 
        lat_range=[0, 90], 
        lon_range=[-150, -70]
    ):
        ftype = ftype if ftype else "TXT.GZ"
        super().__init__(lat_range, lon_range)
        self.fname = fname
        self.ftype = ftype
        self.raw_file = self.fname.replace(ftype.lower(), "csv")
        self.__fetch_file__()
        self.filter_by_latlon()
        return

    def __fetch_file__(self):
        logger.info(f"Check file: {self.raw_file}")
        if not self.__check_exists__(self.raw_file):
            fc, o = "", []
            logger.info(f"Loading file: {self.fname}")
            if self.ftype == "TXT.GZ":
                os.system(f"gzip -k -d {self.fname}")
            with open(self.fname.replace(".gz","")) as f:
                fc = f.readlines()
            header = list(filter(None, fc[0].replace("\n","").split(" ")))
            for line in fc[1:]:
                line = list(filter(None, line.replace("\n","").split(" ")))
                if len(line) > 0:
                    l2d = dict(zip(header, line))
                    l2d["TIME"] = dt.datetime(
                        int(line[0]), int(line[1]), int(line[2]),
                        int(line[3]), int(line[4]), int(line[5])
                    )
                    o.append(l2d)
            o = pd.DataFrame.from_records(o)
            self.frames[self.__fetch_key__()] = {
                "data": pd.DataFrame.from_records(o),
                "parent": None
            }
            o.to_csv(self.raw_file, index=False, header=True)
        logger.info(f"File cotent header:\n {self.frames[self.__fetch_key__(0)].head()}")
        return
    
    def summary(
        self, fname, date, ds_counter, 
        range=[-150, -70, 20, 90],
        rads=[],
    ):
        du = self.frames[self.__fetch_key__(ds_counter)]["data"].copy()
        du = du[du.TIME==date]
        X, Y, Z = get_gridded_parameters(du, xparam="GLON", yparam="GDLAT", zparam="TEC")
        cb = CartoBase(
            date, xPanels=1, yPanels=1,
            range=range,
            basetag=0,
            ytitlehandle=0.95,
        )
        cb.add_TEC(X, Y, Z)
        for rad in rads:
            cb.overlay_radar(rad)
            cb.overlay_fov(rad)
        cb.save(fname)
        return

    def gradient(
        self, fname, date, ds_counter, 
        range=[-150, -70, 20, 90],
        rads=[],
        component=None,
        sub=False,
    ):
        self.base_time = 0
        du = self.frames[self.__fetch_key__(ds_counter)]["data"].copy()
        du = du[du.TIME==date]
        g = Gardient(du)
        g.parse_matrix()
        g.grad2D_by_np()
        cb = CartoBase(
            date, xPanels=1, yPanels=1,
            range=range,
            basetag=0,
            ytitlehandle=0.95,
        )
        dxZ, dyZ = g.dxZ, g.dyZ
        if component:
            dyZ = np.zeros_like(dyZ) if component=="east" else dyZ
            dxZ = np.zeros_like(dxZ) if component=="north" else dxZ
        if sub:
            if self.base_time == 0:
                self.base_dyZ, self.base_dxZ = dyZ, dxZ
                self.base_time += 1
            else:
                dxZ = dxZ - self.base_dxZ
                dyZ = dyZ - self.base_dyZ
        cb.add_TEC_gradient(g.X, g.Y, dxZ, dyZ, True, 2, 2)
        X, Y, Z = get_gridded_parameters(du, xparam="GLON", yparam="GDLAT", zparam="TEC")
        cb.add_TEC(X, Y, Z, alpha=0.3)
        for rad in rads:
            cb.overlay_radar(rad)
            cb.overlay_fov(rad)
        cb.save(fname)
        return

    def overlay(
        self, fname, date, ds_counter, 
        range=[-150, -70, 20, 90],
        rads=[],
        component=None,
        sub=False,
    ):
        self.base_time = 0
        du = self.frames[self.__fetch_key__(ds_counter)]["data"].copy()
        du = du[du.TIME==date]
        g = Gardient(du)
        g.parse_matrix()
        g.grad2D_by_np()
        cb = CartoBase(
            date, xPanels=1, yPanels=1,
            range=range,
            basetag=0,
            ytitlehandle=0.95,
        )
        dxZ, dyZ = g.dxZ, g.dyZ
        if component:
            dyZ = np.zeros_like(dyZ) if component=="east" else dyZ
            dxZ = np.zeros_like(dxZ) if component=="north" else dxZ
        if sub:
            if self.base_time == 0:
                self.base_dyZ, self.base_dxZ = dyZ, dxZ
                self.base_time += 1
            else:
                dxZ = dxZ - self.base_dxZ
                dyZ = dyZ - self.base_dyZ
        cb.add_TEC_gradient(g.X, g.Y, dxZ, dyZ, True, 2, 2)
        X, Y, Z = get_gridded_parameters(du, xparam="GLON", yparam="GDLAT", zparam="TEC")
        cb.add_TEC(X, Y, Z, alpha=0.3, cmap="Greys", vlim=[0, 8])
        for rad in rads:
            cb.overlay_radar(rad)
            cb.overlay_fov(rad)
            cb.ovrlay_radar_data(rad, cbar=True, cbarh=False)
        cb.save(fname)
        return

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

def gradient_TS(dates, g, g0, clat=65, grid_lat=5, clon=-95, grid_lon=5, ds_counter=1):
    dxZ, dyZ, Z, times = generate_gradients(
        dates, g, clat=clat, grid_lat=grid_lat, 
        clon=clon, grid_lon=grid_lon, ds_counter=ds_counter
    )
    print([dates[0]-dt.timedelta(7), dates[1]-dt.timedelta(7)])
    dxZ0, dyZ0, Z0, times0 = generate_gradients(
        [dates[0]-dt.timedelta(7), dates[1]-dt.timedelta(7)], 
        g0, clat=clat, grid_lat=grid_lat, 
        clon=clon, grid_lon=grid_lon, ds_counter=ds_counter
    )

    fig, axes = plot_TEC_TS()
    ax = axes[0]
    print(times, Z0, Z)
    ax.plot(times, Z, "ko", ms=0.8, ls="None", label=r"06 Sep")
    ax.plot(times, Z0, "ro", ms=0.8, ls="None", label=r"05 Sep")
    ax.set_ylabel(
        "$n_0$ [TECu]", 
        fontdict={"size":15, "fontweight": "bold"}
    )
    ax.text(0.9, 0.9, r"$\lambda,\phi=%d,%d$"%(clat, clon), ha="right", va="center", transform=ax.transAxes)
    ax.legend(loc=0)

    ax = axes[1]
    ax.plot(times, dxZ, "ko", ms=0.8, ls="None", label=r"06 Sep")
    ax.plot(times, dxZ0, "ro", ms=0.8, ls="None", label=r"05 Sep")
    ax.set_ylabel(
        r"$\nabla_{\phi} n_0$ [TECu/$^\circ$]", 
        fontdict={"size":15, "fontweight": "bold"}
    )
    ax.legend(loc=0)

    # ax = axes[2]
    # ax.plot(times, dyZ, "ko", ms=0.8, ls="None", label=r"06 Sep")
    # ax.plot(times, dyZ0, "ro", ms=0.8, ls="None", label=r"05 Sep")
    # ax.set_ylabel(
    #     r"$\nabla_{\lambda} n_0$ [TECu/$^\circ$]", 
    #     fontdict={"size":15, "fontweight": "bold"}
    # )
    # ax.legend(loc=0)

    ax = axes[2]
    ax.plot(times, 300*dxZ/Z, "ko", ms=0.8, ls="None", label=r"06 Sep")
    ax.plot(times, 10*dxZ0/Z0, "ro", ms=0.8, ls="None", label=r"05 Sep")
    ax.set_ylabel(
        r"Growth Rate [$\gamma=\frac{V_0}{n_0}\frac{\partial n_0}{\partial\phi}$]", 
        fontdict={"size":15, "fontweight": "bold"}
    )
    ax.legend(loc=0)
    fig.savefig(f"figures/TS{dates[0].strftime('%d')}.png", bbox_inches="tight")
    return

if __name__ == "__main__":
    # Alternte date - 2017-08-30
    radars = []
    dates=[
        dt.datetime(2017,9,6,11),
        dt.datetime(2017,9,6,13)
    ]
    for r in ["sas"]:
        rad_data = Radar(r, dates)
        radars.append(rad_data)
    radars = []
    # Running code for 6Sep
    g = GPS1X1(
        "database/gps170906g.003.txt.gz", 
        "TXT.GZ"
    )
    g0 = GPS1X1(
        "database/gps170830g.002.txt.gz", 
        "TXT.GZ"
    )
    import os
    os.mkdir("figures/06Sep")
    #gradient_TS(dates, g, g0, clat=60, grid_lat=5, clon=-105, grid_lon=5)
    d = dt.datetime(2017,9,6,11,47,30)
    for i in range(10):
        # g.summary(f"figures/06Sep/TEC.{d.strftime('%H-%M')}.png", d, 1)
        # g.gradient(f"figures/06Sep/grad.{d.strftime('%H-%M')}.png", d, 1)
        # g.gradient(
        #     f"figures/06Sep/grad.east,{d.strftime('%H-%M')}.png", d, 1,
        #     component="east"
        # )
        # g.gradient(
        #     f"figures/06Sep/grad.east,sub,{d.strftime('%H-%M')}.png", d, 1,
        #     component="east", sub=True
        # )
        g.overlay(
            f"figures/06Sep/grad.east,sub,{d.strftime('%H-%M')}.png", d, 1,
            rads=radars, component="east", sub=True
        )
        d += dt.timedelta(minutes=5)
    # Running code for 5Sep
    # g = GPS1X1(
    #     "database/gps170905g.002.txt.gz", 
    #     "TXT.GZ"
    # )
    # d = dt.datetime(2017,9,5,11,47,30)
    # for i in range(10):
    #     g.summary(f"figures/05Sep/TEC.{d.strftime('%H-%M')}.png", d, 1)
    #     g.gradient(f"figures/05Sep/grad.{d.strftime('%H-%M')}.png", d, 1)
    #     d += dt.timedelta(minutes=5)