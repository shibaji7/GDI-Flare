import matplotlib.pyplot as plt
import sys
sys.path.append(".")
import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]
from map_plots import CartoBase
import xarray as xr
import datetime as dt
import numpy as np
from scipy.ndimage import median_filter

dSep = xr.load_dataset("database/gps170906g.003.nc")
cb = CartoBase(
    dt.datetime(2017,9,6,11,50),                 
    xPanels=1, yPanels=1,
    basetag=0, ytitlehandle=0.95,
    terminator=True,
    range=[-130, -80, 20, 60],
    title=""
)
cb._fetch_axis(True)
dates = [dt.datetime.utcfromtimestamp(d) for d in dSep["timestamps"].values]
gdlat, glon = (dSep["gdlat"].values, dSep["glon"].values)
tec = dSep["tec"].values
idate = dates.index(dt.datetime(2017, 9, 6, 11, 50))
T = tec[idate, :, :]
glon, gdlat = np.meshgrid(glon, gdlat)
xyz = cb.proj["to"].transform_points(cb.proj["from"], glon, gdlat)
x, y = xyz[:, :, 0], xyz[:, :, 1]
im = cb.ax.pcolor(
    x, y, T,
    cmap="Spectral",
    vmin=0,
    vmax=8,
    transform=cb.proj["to"],
    alpha=0.8
)
cb._add_colorbar(im, "Spectral", label="TEC [TECu]", dx=0.15)
dyT = np.gradient(T, axis=1)
cb.ax.scatter(x, y, color="k", s=1e-3)
dyTm = median_filter(dyT, size=3, mode="constant", cval=np.nan)
ql = cb.ax.quiver(
    x,
    y,
    dyTm, 
    np.zeros_like(dyT),
    scale=4,
    headaxislength=0,
    linewidth=0.6,
    scale_units="inches",
)
cb.ax.quiverkey(
    ql,
    0.9,
    1.1,
    0.5,
    r"$\nabla_{\phi}n'_0$:"+str(3),
    labelpos="N",
    transform=cb.proj["from"],
    color="k",
    fontproperties={"size": 8},
)
cb.ax.text(
    -0.2, 0.95, "Coord: Geo", 
    ha="left", va="top", transform=cb.ax.transAxes,
    fontdict={"size":10, "weight":"bold", "color": "b"},
    rotation=90,
)
cb.ax.text(
    -0.1, 1.15, "11:50 UT, 06 September 2017", 
    ha="left", va="bottom", transform=cb.ax.transAxes,
    fontdict={"size":10, "weight":"bold", "color": "b"},
)
cb.save("figures/Figure03.png")
cb.close()

if __name__ == "__main__":
    pass