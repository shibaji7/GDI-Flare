import matplotlib.pyplot as plt
#import mplstyle
import scienceplots
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
from gps import GPS1X1, Gardient, weighted_2d_filter
from matplotlib.patches import Circle, Ellipse, Rectangle

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import LineString, MultiLineString, Polygon, mapping
from descartes import PolygonPatch
from matplotlib.patches import Circle, Ellipse

def boxcar(Z, tau=0.4):
    W = np.array([
        [1,1,1],
        [1,2,1],
        [1,1,1],
    ])
    K = np.zeros_like(Z)*np.nan
    for i in range(1, Z.shape[0]-2):
        for j in range(1, Z.shape[1]-2):
            weight, value = 0, 0
            w = W.ravel()
            print(Z[i-1:i+2, j-1:j+2].ravel())
            for z, w in zip(Z[i-1:i+2, j-1:j+2].ravel(), W.ravel()):
                if not np.isnan(z):
                    weight += w
                    value += w*z
            if weight/np.sum(W) >= tau:
                print(weight)
                K[i,j] = np.median(value) / weight
    return K

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
    z = plotParamDF[zparam].values
    #z = boxcar(z)
    Z = np.ma.masked_where(np.isnan(z), z)
    return X,Y,Z


class CartoBase(object):
    """
    This class holds cartobase code for the
    SD, SMag, and GPS TEC dataset.
    """

    def __init__(
            self, date, xPanels=1, yPanels=1, 
            range=[-140, -70, 40, 90], basetag=0, 
            ytitlehandle=0.95, dtau=2, terminator=True, title=None
        ):
        self.date = date
        self.xPanels = xPanels
        self.yPanels = yPanels
        self.range = range
        self.basetag = basetag
        self.dtau = dtau
        self._num_subplots_created = 0
        self.terminator = terminator
        self.title = title
        self.fig = plt.figure(figsize=(3*yPanels, 3*xPanels), dpi=1000) # Size for website
        mpl.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize":10, "font.size":10})
        self.ytitlehandle = ytitlehandle
        self.proj = {
            "to": ccrs.Orthographic(-110, 60),
            #"to": ccrs.NorthPolarStereo(-90),
            "from": ccrs.PlateCarree(),
        }
        return

    def add_circle(self, lat=60, lon=-105, width=3, height=3):
        width = width / np.cos(np.deg2rad(lat))
        self.ax.add_patch(Rectangle(
            xy=[lon-width/2, lat-height/2], width=width, height=height,
            color='red', alpha=0.3, 
            transform=self.proj["from"], zorder=30
        ))
        self.ax.scatter([lon], [lat], s=2, marker="o",
            color="red", zorder=3, transform=self.proj["from"], lw=0.8, alpha=0.4)
        return

    def _add_axis(self, draw_labels=True):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(
            self.xPanels, self.yPanels, 
            self._num_subplots_created,
            projection=self.proj["to"],
        )
        ax.tick_params(axis="both", labelsize=15)
        if self.terminator: ax.add_feature(Nightshade(self.date, alpha=0.3))
        ax.set_global()
        ax.coastlines(color="k", alpha=0.5, lw=0.5)
        gl = ax.gridlines(crs=self.proj["from"], linewidth=0.3, 
            color="k", alpha=0.5, linestyle="--", draw_labels=draw_labels)
        gl.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,20))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if self.range: ax.set_extent(self.range)
        if self.title is None:
            tag = "(%s) "%chr(96 + self._num_subplots_created + self.basetag)
            tag += self.date.strftime("%H:%M - ") +\
                (self.date+dt.timedelta(minutes=self.dtau)).strftime("%H:%M UT")
            ax.text(0.05, 0.9, tag, 
                ha="left", va="bottom", transform=ax.transAxes, fontdict={"size":10})
        else:
            ax.text(0.05, 0.9, self.title, ha="left", va="bottom", transform=ax.transAxes, fontdict={"size":10})
        
        # if self.basetag in [0, 2]:
        #     txt = self.date.strftime("%d %b %Y") + "\n" + "Coord: Geo"
        #     ax.text(0.05, 1.1, txt, 
        #         ha="left", va="bottom", transform=ax.transAxes,
        #         fontdict={"size":10, "weight":"bold"})
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def _fetch_axis(self, draw_labels=True, add_tag=False):
        if not hasattr(self, "ax"): 
            self.ax = self._add_axis(draw_labels)
        if add_tag:
            self.add_circle()
        return

    def add_radars(self, radars, draw_labels=True, cbar=False):
        self._fetch_axis(draw_labels)
        for r in radars.keys():
            rad = radars[r]
            self.overlay_radar(rad)
            self.overlay_fov(rad)
            self.ovrlay_radar_data(rad, cbar=cbar)
        return
    
    def overlay_radar(
        self, rad, marker="D", zorder=2, markerColor="k", 
        markerSize=2, fontSize="small", font_color="darkblue", xOffset=-5, 
        yOffset=-1.5, annotate=True,
    ):
        """ Adding the radar location """
        lat, lon = rad.hdw.geographic.lat, rad.hdw.geographic.lon
        self.ax.scatter([lon], [lat], s=markerSize, marker=marker,
            color=markerColor, zorder=zorder, transform=self.proj["from"], lw=0.8, alpha=0.4)
        nearby_rad = [["adw", "kod", "cve", "fhe", "wal", "gbr", "pyk", "aze", "sys"],
                    ["ade", "ksr", "cvw", "fhw", "bks", "sch", "sto", "azw", "sye"]]
        if annotate:
            rad = rad.rad
            if rad in nearby_rad[0]: xOff, ha = -5 if not xOffset else -xOffset, -2
            elif rad in nearby_rad[1]: xOff, ha = 5 if not xOffset else xOffset, -2
            else: xOff, ha = xOffset, -1
            x, y = self.proj["to"].transform_point(lon+xOff, lat+ha, src_crs=self.proj["from"])
            self.ax.text(x, y, rad.upper(), ha="center", va="center", transform=self.proj["to"],
                        fontdict={"color":font_color, "size":8}, alpha=0.8)
        return

    def overlay_fov(
        self, rad, maxGate=75, rangeLimits=None, beamLimits=None,
        model="IS", fov_dir="front", fovColor=None, fovAlpha=0.2,
        fovObj=None, zorder=1, lineColor="k", lineWidth=0.5, ls="-"
    ):
        """ Overlay radar FoV """
        from numpy import transpose, ones, concatenate, vstack, shape
        hdw = rad.hdw
        sgate = 0
        egate = hdw.gates if not maxGate else maxGate
        ebeam = hdw.beams
        if beamLimits is not None: sbeam, ebeam = beamLimits[0], beamLimits[1]
        else: sbeam = 0
        latFull, lonFull = rad.fov[0].T, rad.fov[1].T
        xyz = self.proj["to"].transform_points(self.proj["from"], lonFull, latFull)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        contour_x = concatenate((x[sbeam, sgate:egate], x[sbeam:ebeam, egate],
                    x[ebeam, egate:sgate:-1],
                    x[ebeam:sbeam:-1, sgate]))
        contour_y = concatenate((y[sbeam, sgate:egate], y[sbeam:ebeam, egate],
                y[ebeam, egate:sgate:-1],
                y[ebeam:sbeam:-1, sgate]))
        self.ax.plot(contour_x, contour_y, color=lineColor, 
            zorder=zorder, linewidth=lineWidth, ls=ls, alpha=1.0)
        if fovColor:
            contour = transpose(vstack((contour_x, contour_y)))
            polygon = Polygon(contour)
            patch = PolygonPatch(
                polygon,
                facecolor=fovColor,
                edgecolor=fovColor,
                alpha=fovAlpha,
                zorder=zorder,
            )
            self.ax.add_patch(patch)
        return
    
    def ovrlay_radar_data(self, rad, maxGate=75, cbar=False, cbarh=False):
        data = rad.df.copy()
        data = data[
            (data.time>=self.date) &
            (data.time<self.date+dt.timedelta(minutes=self.dtau)) &
            (data.slist<=maxGate) &
            (data.slist>7)
        ]
        kwargs = {"rad": rad}
        # add a function to create GLAT/GLON in Data
        data = data.apply(self.convert_to_latlon, axis=1, **kwargs)
        # Grid based on GLAT/GLON
        X, Y, Z = get_gridded_parameters(data, "glon", "glat", "v")
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        im = self.ax.scatter(
            x, y, c=Z.T,
            cmap="jet_r",
            vmin=-300,
            vmax=300,
            transform=self.proj["to"],
            alpha=0.6,
            s=1.2
        )
        if cbar: 
            if cbarh:
                self._add_hcolorbar(im, label="Velocity [m/s]")
            else:
                self._add_colorbar(im, label="Velocity [m/s]")
        return
    
    def convert_to_latlon(self, row, rad):
        row["glat"], row["glon"] = (
            rad.fov[0].T[row["bmnum"], row["slist"]],
            rad.fov[1].T[row["bmnum"], row["slist"]],
        )
        return row
    
    def _add_hcolorbar(self, im, colormap="jet_r", label=""):
        """Add a colorbar to the right of an axis."""
        pos = self.ax.get_position()
        cpos = [
            pos.x0 + 0.3 * pos.width,
            pos.y0 - 0.15 * pos.height,
            pos.width * 0.5,
            0.02,
        ]  # this list defines (left, bottom, width, height)
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(
            im,
            cax=cax,
            cmap=colormap,
            spacing="uniform",
            orientation="horizontal",
        )
        cb2.set_label(label)
        return

    def _add_colorbar(self, im, colormap="jet_r", label="", dx=0.05):
        """Add a colorbar to the right of an axis."""
        pos = self.ax.get_position()
        cpos = [
            pos.x1 + dx,
            pos.y0 + 0.2 * pos.height,
            0.02,
            pos.height * 0.5,
        ]  # this list defines (left, bottom, width, height)
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(
            im,
            cax=cax,
            cmap=colormap,
            spacing="uniform",
            orientation="vertical",
        )
        cb2.set_label(label)
        return
    
    def fetch_GPS(
        self, g, ds_counter,
        component=None, date=None,
    ):
        date = date if date else self.date
        du = g.frames[g.__fetch_key__(ds_counter)]["data"].copy()
        print("---->", du.head(), "\n",du.columns)
        du = du[
            (du.TIME>=date-dt.timedelta(minutes=2.5)) &
            (du.TIME<=date+dt.timedelta(minutes=2.5)) 
        ]
        grd = Gardient(du, dlat=3, dlon=3)
        grd.parse_matrix()
        grd.grad2D_by_np()

        X, Y, Z = get_gridded_parameters(du, xparam="GLON", yparam="GDLAT", zparam="TEC")

        dxZ, dyZ = grd.dxZ, grd.dyZ
        if component:
            dyZ = np.zeros_like(dyZ) if component=="east" else dyZ
            dxZ = np.zeros_like(dxZ) if component=="north" else dxZ
        dxZ = weighted_2d_filter(dxZ)
        return grd, dyZ, dxZ, X, Y, Z
    
    def add_gps(
        self, g, ds_counter,
        component=None,
        tag=False,
        base_date=None,
        draw_labels=False,
    ):
        self._fetch_axis(draw_labels)
        grd, dyZ, dxZ, _, _, _ = self.fetch_GPS(g, ds_counter, component)
        _, dyZ0, dxZ0, _, _, _ = self.fetch_GPS(g, ds_counter, component, 
                                       base_date)
        #dxZ = dxZ - dxZ0
        #dyZ = dyZ - dyZ0
        self.add_TEC_gradient(grd.X, grd.Y, dxZ, dyZ, tag, 0.5, 1.5)
        return
    
    def add_TEC_gradient(self, X, Y, dxZ, dyZ, tag=False, lenx=3., scale=3.):
        # Plot based on transcript
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        self.ax.scatter(x, y, color="k", s=0.05)
        ql = self.ax.quiver(
            x,
            y,
            dxZ.T,
            dyZ.T,
            scale=scale,
            headaxislength=0,
            linewidth=0.6,
            scale_units="inches",
        )
        if tag:
            self.ax.quiverkey(
                ql,
                0.85,
                1.05,
                lenx,
                r"$\nabla_{\phi}n'_0$:"+str(lenx),
                labelpos="N",
                transform=self.proj["from"],
                color="k",
                fontproperties={"size": 8},
            )
        return
    
    def add_TEC_gradient_diff(self, X, Y, dxZ, dyZ, tag=False, lenx=3., scale=3.):
        # Plot based on transcript
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        self.ax.scatter(x, y, color="k", s=0.05)
        xZ = np.ma.masked_array(dxZ, mask=(dxZ<=0))
        ql = self.ax.quiver(
            x,
            y,
            xZ.T,
            dyZ.T,
            scale=scale,
            headaxislength=0,
            linewidth=0.6,
            scale_units="inches",
            color="b"
        )
        xZ = np.ma.masked_array(dxZ, mask=(dxZ>0))
        self.ax.quiver(
            x,
            y,
            xZ.T,
            dyZ.T,
            scale=scale,
            headaxislength=0,
            linewidth=0.6,
            scale_units="inches",
            color="r"
        )
        if tag:
            self.ax.quiverkey(
                ql,
                0.85,
                1.05,
                lenx,
                r"$\nabla_{\phi}n'_0$:"+str(lenx),
                labelpos="N",
                transform=self.proj["from"],
                color="k",
                fontproperties={"size": 15},
            )
        return

    def add_TEC(self, g, ds_counter, alpha=0.6, cmap="Greys", vlim=[0, 8], cbar=False):
        self._fetch_axis(False)
        _, _, _, X, Y, Z = self.fetch_GPS(g, ds_counter, None)
        # Plot based on transcript
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        im = self.ax.pcolor(
            x, y, Z.T,
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
            transform=self.proj["to"],
            alpha=alpha
        )
        if cbar:
            self._add_colorbar(im, cmap, label="TEC [TECu]")
        return

def plot_difference(file, date_pairs):
    d0, d1 = date_pairs
    print(d0, d1)
    g = GPS1X1(file, "TXT.GZ")    
    cb = CartoBase(
            d0,                 
            xPanels=1, yPanels=1,
            basetag=0, ytitlehandle=0.95,
            terminator=False,
            title=r"$\partial_{TEC}^{@"+date_pairs[1].strftime('%H:%M')+"}$-" + 
            r"$\partial_{TEC}^{@"+date_pairs[0].strftime('%H:%M')+"}$" 
    )
    grd0, dyZ0, dxZ0, _, _, _ = cb.fetch_GPS(g, 1, "east")
    _, dyZ1, dxZ1, _, _, _ = cb.fetch_GPS(g, 1, "east", date=d1)
    cb._fetch_axis(False)
    cb.add_TEC_gradient_diff(grd0.X, grd0.Y, dxZ1-dxZ0, dyZ1-dyZ0, True, 0.2, 0.5)
    cb.save(f"figures/fov_diff_data.%s.png"%d0.strftime("%d%H%M"))
    return

def plot_fov_data(month, day, file, plot_cicle=False, ini_tag=0):
    g = GPS1X1(
        file, 
        "TXT.GZ"
    )    
    dates = [
        dt.datetime(2017,month,day,11),
        dt.datetime(2017,month,day,17)
    ]
    radars = {}
    for r in ["sas"]:
        rad_data = Radar(r, dates)
        radars[r] = rad_data
    dates = [
        dt.datetime(2017, month, day, 11, 52),
        #dt.datetime(2017, month, day, 12, 2),
        #dt.datetime(2017, month, day, 12, 10),
        dt.datetime(2017, month, day, 12, 22)
    ]
    for i, d in enumerate(dates):
        cb = CartoBase(
            d,                 
            xPanels=1, yPanels=1,
            basetag=i+ini_tag, ytitlehandle=0.95,
        )
        cb.add_radars(radars, draw_labels=False, cbar=(plot_cicle and i==0)) 
        cb.add_gps(
            g, 1, 
            component="east",
            tag=(i==0),
            base_date=dates[0] - dt.timedelta(minutes=5)
        )
        # cb.add_TEC(
        #     g, 1,
        #     cmap="Greys", vlim=[0, 8],
        #     cbar=(plot_cicle and i==0),
        # )
        # if i==0 and plot_cicle:
        #     cb.add_circle(45, -105)
        #     cb.add_circle(60, -105, width=8, height=5)
        cb.save(f"figures/fov_data.%s.png"%d.strftime("%d%H%M"))
    return
plot_fov_data(9, 6, "database/gps170906g.003.txt.gz", False)
# plot_fov_data(8, 30, "database/gps170830g.002.txt.gz", True, 2)

# plot_difference(
#     "database/gps170906g.003.txt.gz", 
#     (
#         dt.datetime(2017, 9, 6, 11, 52),
#         dt.datetime(2017, 9, 6, 12, 22)
#     )
# )

# plot_difference(
#     "database/gps170830g.002.txt.gz", 
#     (
#         dt.datetime(2017, 8, 30, 11, 52),
#         dt.datetime(2017, 8, 30, 12, 22)
#     )
# )