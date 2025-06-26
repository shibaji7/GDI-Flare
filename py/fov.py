import matplotlib.pyplot as plt

#import mplstyle
import matplotlib as mpl
import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import numpy as np
from scipy.ndimage import gaussian_filter as GF
from fetch import Radar
from matplotlib.patches import Circle, Ellipse, Rectangle

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import LineString, MultiLineString, Polygon, mapping
from descartes import PolygonPatch


class CartoBase(object):
    """
    This class holds cartobase code for the
    SD, SMag, and GPS TEC dataset.
    """

    def __init__(self, date, xPanels=1, yPanels=1, range=[-150, -70, 40, 90], basetag=0, 
                 ytitlehandle=1):
        self.date = date
        self.xPanels = xPanels
        self.yPanels = yPanels
        self.range = range
        self.basetag = basetag
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(6*yPanels, 6*xPanels), dpi=1000) # Size for website
        mpl.rcParams.update({"xtick.labelsize": 15, "ytick.labelsize":15, "font.size":15})
        self.ytitlehandle = ytitlehandle
        self.proj = {
            "to": ccrs.Orthographic(-110, 60),
            #"to": ccrs.NorthPolarStereo(-90),
            "from": ccrs.PlateCarree(),
        }
        return

    def _add_axis(self, draw_labels=True, tag=False):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(
            self.xPanels, self.yPanels, 
            self._num_subplots_created,
            projection=self.proj["to"],
        )
        ax.tick_params(axis="both", labelsize=15)
        ax.add_feature(Nightshade(self.date, alpha=0.3))
        ax.set_global()
        ax.coastlines(color="k", alpha=0.5, lw=0.5)
        gl = ax.gridlines(crs=self.proj["from"], linewidth=0.3, 
            color="k", alpha=0.5, linestyle="--", draw_labels=draw_labels)
        gl.xlocator = mticker.FixedLocator(np.arange(-180,180,20))
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,10))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if self.range: ax.set_extent(self.range)
        tag = chr(96 + self._num_subplots_created + self.basetag)
        ax.text(0.5, 1.2, self.date.strftime("%d %b %Y, %H:%M UT"), 
            ha="center", va="bottom", transform=ax.transAxes)
        # plt.suptitle(self.date.strftime("%d %b %Y, %H:%M UT"), 
        #     x=0.5, y=self.ytitlehandle, ha="center", va="bottom", fontweight="bold", fontsize=15)
        if tag:
            ax.text(0.05, 0.9, "(a)", 
                ha="left", va="center", transform=ax.transAxes)
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def _fetch_axis(self, draw_labels=True):
        if not hasattr(self, "ax"): self.ax = self._add_axis(draw_labels)
        return

    def add_radars(self, radars, beamLimits, draw_labels=True):
        self._fetch_axis(draw_labels)
        for b, r in zip(beamLimits, radars.keys()):
            rad = radars[r]
            self.overlay_radar(rad)
            self.overlay_fov(rad)
            self.overlay_fov(
                    rad,
                    beamLimits=[b, b+1],
                    lineColor="b",
                    fovAlpha=0.2,
            )
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
                        fontdict={"color":font_color, "size":fontSize}, alpha=0.8)
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
        # if fovColor:
        #     contour = transpose(vstack((contour_x, contour_y)))
        #     print(contour)
        #     polygon = Polygon(contour)
        #     patch = PolygonPatch(
        #         polygon,
        #         facecolor=fovColor,
        #         edgecolor=fovColor,
        #         alpha=fovAlpha,
        #         zorder=zorder,
        #     )
        #     self.ax.add_patch(patch)
        return
    
    def add_circle(self, lat, lon, width=3, height=3):
        width = width / np.cos(np.deg2rad(lat))
        self.ax.add_patch(Rectangle(
            xy=[lon-width/2, lat-height/2], width=width, height=height,
            color='red', alpha=0.3, 
            transform=self.proj["from"], zorder=30
        ))
        self.ax.scatter([lon], [lat], s=2, marker="o",
            color="red", zorder=3, transform=self.proj["from"], lw=0.8, alpha=0.4)
        return
    

def plot_fov():
    dates = [
        dt.datetime(2017,9,6,11),
        dt.datetime(2017,9,6,17)
    ]
    radars = {}
    for r in ["sas", "pgr", "kod"]:
        rad_data = Radar(r, dates)
        radars[r] = rad_data
    d = dt.datetime(2017, 9, 6, 12, 2)
    cb = CartoBase(
        d,                 
        xPanels=1, yPanels=1,
        basetag=0, ytitlehandle=0.95,
    )
    cb.add_radars(radars, beamLimits=[7, 7, 10], draw_labels=True)
    cb.add_circle(60, -105, width=3, height=3) 
    cb.save(f"figures/Figure02.png")
    return
plot_fov()