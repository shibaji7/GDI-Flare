#!/usr/bin/env python

"""cartoUtils.py: utility module for Costom Carto py geoaxes to plot data on aacgmv2 coordinates."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import copy
import datetime as dt

import aacgmv2
import cartopy
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pydarn
# import eutils as utils
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.projections import register_projection
from shapely.geometry import LineString, MultiLineString, Polygon, mapping
from descartes import PolygonPatch
# import eutils as utils
from matplotlib.colors import LinearSegmentedColormap

RedBlackBlue = LinearSegmentedColormap.from_list(
    "RedBlackBlue",
    [
                (0.0, "#FF0000"),  # reddish
                (0.25, "#8B0000"),  # blackish red
                (0.5, "#000000"),  # black
                (0.75, "#00008B"),  # blackish blue
                (1.0, "#131DE3E8"),  # black
            ],
)


from scipy.interpolate import griddata

def setsize(size=12):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scienceplots
#    import mplstyle
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    mpl.rcParams.update(
        {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
    )
    return


def get_gridded_parameters(q, xparam="time", yparam="slist", zparam="v", round=False, a_filter=True):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if round:
        plotParamDF[xparam] = np.round(plotParamDF[xparam], 2)
        plotParamDF[yparam] = np.round(plotParamDF[yparam], 2)
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot(index=xparam, columns=yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    if a_filter:
        Z = apply_2d_filter(Z)
    return X,Y,Z

def apply_2d_filter(data, weights=np.array([[1,2,1],[2,5,2],[1,2,1]])):
    """
    Apply a 2D filter with a 3x3 weight matrix, rejecting a cell if 
    at least 4 cells in the 3x3 neighborhood are masked.

    Parameters:
    - data: 2D numpy masked array
    - weights: 3x3 numpy array (weight matrix)

    Returns:
    - filtered_data: 2D numpy masked array with the filter applied
    """
    # Ensure the input is a masked array
    filtered_data = np.ma.empty_like(data)

    rows, cols = data.shape
    kernel_size = 3
    offset = kernel_size // 2

    for i in range(rows):
        for j in range(cols):
            # Extract the 3x3 neighborhood
            r_start, r_end = max(0, i - offset), min(rows, i + offset + 1)
            c_start, c_end = max(0, j - offset), min(cols, j + offset + 1)
            neighborhood = data[r_start:r_end, c_start:c_end]

            # Check if enough cells are covered
            if np.ma.count(neighborhood) >= 5:  # At least 5 valid cells
                valid_weights = weights[:neighborhood.shape[0], :neighborhood.shape[1]]
                # print(valid_weights)
                # filtered_value = np.sum(neighborhood * valid_weights) / np.sum(valid_weights)
                filtered_data[i, j] = weighted_median(neighborhood.ravel(),  valid_weights.ravel())
            else:
                # Mask the cell if less than 5 valid cells
                filtered_data[i, j] = np.ma.masked

    return filtered_data

def weighted_median(data, weights):
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cum_weights = np.cumsum(sorted_weights)
    median_index = np.searchsorted(cum_weights, 0.5 * cum_weights[-1])
    # Calculate the weighted median
    if cum_weights[median_index - 1] == 0.5 * cum_weights[-1]:
        return 0.5 * (sorted_data[median_index - 1] + sorted_data[median_index])
    else:
        return sorted_data[median_index]

def smooth_2d_interpolate(glon, glat, of, grid_size=360, method='cubic'):
    """
    Interpolates and smooths a 2D array with possible NaNs using 1D longitude and latitude arrays.

    Parameters:
        glon (1D array): Longitudes.
        glat (1D array): Latitudes.
        of (2D array): Data array (shape: (len(glat), len(glon))), may contain NaNs.
        grid_size (int): Resolution of the output grid.
        method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        glon_grid (2D array): Meshgrid of interpolated longitudes.
        glat_grid (2D array): Meshgrid of interpolated latitudes.
        of_interp (2D array): Interpolated and smoothed data.
    """
    glon_mesh, glat_mesh = np.meshgrid(glon, glat)
    points = np.column_stack((glon_mesh.ravel(), glat_mesh.ravel()))
    values = of.ravel()

    # Mask out NaNs
    mask = ~np.isnan(values)
    points_valid = points[mask]
    values_valid = values[mask]

    # New grid for interpolation
    glon_new = np.linspace(np.min(glon), np.max(glon), grid_size)
    glat_new = np.linspace(np.min(glat), np.max(glat), grid_size)
    glon_grid, glat_grid = np.meshgrid(glon_new, glat_new)

    # Interpolate
    of_interp = griddata(points_valid, values_valid, (glon_grid, glat_grid), method=method)

    return glon_new, glat_new, glon_grid, glat_grid, of_interp

def load_eclipse_datasets(date, loc="database/December2021/December2021/"):
    import xarray as xr
    d = xr.open_dataset(loc + f"{date.strftime('%Y%m%d%H%M%S')}_150km_193_1.nc")
    return d

class SDCarto(GeoAxes):
    name = "SDCarto"

    def __init__(self, *args, **kwargs):
        if "map_projection" in kwargs:
            map_projection = kwargs.pop("map_projection")
        else:
            map_projection = cartopy.crs.NorthPolarStereo()
            print(
                "map_projection keyword not set, setting it to cartopy.crs.NorthPolarStereo()"
            )
        # first check if datetime keyword is given!
        # it should be since we need it for aacgm
        if "plot_date" in kwargs:
            self.plot_date = kwargs.pop("plot_date")
        else:
            raise TypeError(
                "need to provide a date using 'plot_date' keyword for aacgmv2 plotting"
            )
        # Now work with the coords!
        supported_coords = ["geo", "aacgmv2", "aacgmv2_mlt"]
        if "coords" in kwargs:
            self.coords = kwargs.pop("coords")
            if self.coords not in supported_coords:
                err_str = "coordinates not supported, choose from : "
                for _n, _sc in enumerate(supported_coords):
                    if _n + 1 != len(supported_coords):
                        err_str += _sc + ", "
                    else:
                        err_str += _sc
                raise TypeError(err_str)
        else:
            self.coords = "geo"
            print("coords keyword not set, setting it to aacgmv2")
        # finally, initialize te GeoAxes object
        super().__init__(map_projection=map_projection, *args, **kwargs)
        return

    def add_square_grid(self, box_lon_min, box_lat_min, radius):
        import matplotlib.patches as mpatches
        
        rect = mpatches.Circle((box_lon_min, box_lat_min), radius,
                              linewidth=1, edgecolor='k', facecolor='none',
                              transform=cartopy.crs.PlateCarree())
        self.add_patch(rect)
        return

    def overaly_coast_lakes(self, resolution="50m", color="black", **kwargs):
        """
        Overlay AACGM coastlines and lakes
        """
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        # overaly coastlines
        feature = cartopy.feature.NaturalEarthFeature(
            "physical", "coastline", resolution, **kwargs
        )
        self.add_feature(cartopy.feature.COASTLINE, **kwargs)
        self.add_feature(cartopy.feature.LAKES, **kwargs)

    def coastlines(self, resolution="50m", color="black", **kwargs):
        # details!
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        feature = cartopy.feature.NaturalEarthFeature(
            "physical", "coastline", resolution, **kwargs
        )
        return self.add_feature(feature, **kwargs)

    def add_feature(self, feature, **kwargs):
        # Now we"ll set facecolor as None because aacgm doesn"t close
        # continents near equator and it turns into a problem
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"
        if "facecolor" in kwargs:
            print(
                "manually setting facecolor keyword to none as aacgm fails for fill! want to know why?? think about equator!"
            )
        kwargs["facecolor"] = "none"
        if self.coords == "geo":
            super().add_feature(feature, **kwargs)
        else:
            aacgm_geom = self.get_aacgm_geom(feature)
            aacgm_feature = cartopy.feature.ShapelyFeature(
                aacgm_geom, cartopy.crs.Geodetic(), **kwargs
            )
            super().add_feature(aacgm_feature, **kwargs)

    def get_aacgm_geom(self, feature, out_height=300.0):
        new_i = []
        # cartopy.feature.COASTLINE
        for _n, i in enumerate(feature.geometries()):
            aa = mapping(i)
            mag_list = []
            geo_coords = aa["coordinates"]
            for _ni, _list in enumerate(geo_coords):
                mlon_check_jump_list = []
                split_mag_list = None
                if len(_list) == 1:
                    _loop_list = _list[0]
                else:
                    _loop_list = _list
                for _ngc, _gc in enumerate(_loop_list):
                    _mc = aacgmv2.get_aacgm_coord(
                        _gc[1], _gc[0], out_height, self.plot_date
                    )
                    if numpy.isnan(_mc[0]):
                        continue
                    mlon_check_jump_list.append(_mc[1])
                    if self.coords == "aacgmv2":
                        mag_list.append((_mc[1], _mc[0]))
                    else:
                        if _mc[2] * 15.0 > 180.0:
                            mag_list.append((_mc[2] * 15.0 - 360.0, _mc[0]))
                        else:
                            mag_list.append((_mc[2] * 15.0, _mc[0]))
                # check for unwanted jumps
                mlon_check_jump_list = numpy.array(mlon_check_jump_list)

                jump_arr = numpy.diff(mlon_check_jump_list)
                bad_inds = numpy.where(numpy.abs(jump_arr) > 10.0)[0]
                # delete the range of bad values
                # This is further complicated because
                # in some locations mlon jumps from -177 to +178
                # and this causes jumps in the maps! To deal with
                # this we"ll split arrays of such jumps
                # (these jumps typically have just one bad ind )
                # and make them into two seperate entities (LineStrings)
                # so that shapely will treat them as two seperate boundaries!
                if len(bad_inds) > 0:
                    if len(bad_inds) > 1:
                        mag_list = [
                            i
                            for j, i in enumerate(mag_list)
                            if j - 1 not in numpy.arange(bad_inds[0], bad_inds[1])
                        ]
                    else:
                        split_mag_list = mag_list[bad_inds[0] + 1 :]
                        mag_list = mag_list[: bad_inds[0] + 1]
                mag_coords = tuple(mag_list)
                if len(mag_list) > 1:
                    new_i.append(mag_coords)
                if split_mag_list is not None:
                    #             print(split_mag_list)
                    if len(split_mag_list) > 1:
                        new_i.append(tuple(split_mag_list))

        aacgm_coast = MultiLineString(new_i)
        return aacgm_coast

    def mark_latitudes(self, lat_arr, lon_location=-90, **kwargs):
        """
        mark the latitudes
        Write down the latitudes on the map for labeling!
        we are using this because cartopy doesn"t have a
        label by default for non-rectangular projections!
        """
        if isinstance(lat_arr, list):
            lat_arr = numpy.array(lat_arr)
        else:
            if not isinstance(lat_arr, numpy.ndarray):
                raise TypeError("lat_arr must either be a list or numpy array")
        # make an array of lon_location
        lon_location_arr = numpy.full(lat_arr.shape, lon_location)
        proj_xyz = self.projection.transform_points(
            cartopy.crs.PlateCarree(), lon_location_arr, lat_arr
        )
        # plot the lats now!
        out_extent_lats = False
        for _np, _pro in enumerate(proj_xyz[..., :2].tolist()):
            # check if lats are out of extent! if so ignore them
            lat_lim = self.get_extent(crs=cartopy.crs.Geodetic())[2::]
            if (lat_arr[_np] >= min(lat_lim)) and (lat_arr[_np] <= max(lat_lim)):
                self.text(
                    _pro[0],
                    _pro[1],
                    r"$%s^{\circ}$" % str(lat_arr[_np]),
                    **kwargs,
                    alpha=0.5,
                )
            else:
                out_extent_lats = True
        if out_extent_lats:
            print("some lats were out of extent ignored them")

    def mark_longitudes(
        self, lon_arr=numpy.arange(-180, 180, 60), 
        tx=cartopy.crs.PlateCarree(), **kwargs
    ):
        """
        mark the longitudes
        Write down the longitudes on the map for labeling!
        we are using this because cartopy doesn"t have a
        label by default for non-rectangular projections!
        This is also trickier compared to latitudes!
        """
        if isinstance(lon_arr, list):
            lon_arr = numpy.array(lon_arr)
        else:
            if not isinstance(lon_arr, numpy.ndarray):
                raise TypeError("lat_arr must either be a list or numpy array")
        plot_extent = self.get_extent(cartopy.crs.Geodetic())
        lat = plot_extent[2] if plot_extent[3] <= plot_extent[2] else plot_extent[3]
        lat -= 20.
        for t in lon_arr:
            marker_text = r"$%s^{\circ}$" % str(t)
            x, y = self.projection.transform_point(t, lat, src_crs=tx)
            self.text(
                x, y,
                marker_text,
                ha="center",
                va="center",
                **kwargs,
                alpha=0.5,
            )
        return

    def to_aagcm(self, lat, lon):
        if "aacgmv2" in self.coords:
            lat, lon, mlt = aacgmv2.get_aacgm_coord(lat, lon, 300, self.plot_date)
            if self.coords == "aacgmv2_mlt":
                lon = mlt * 15
        return lat, lon

    def to_aagcms(self, lats, lons):
        mlats, mlons = np.zeros_like(lats), np.zeros_like(lats)
        if "aacgmv2" in self.coords:
            for i in range(lats.shape[0]):
                mlats[i, :], mlons[i, :], mlt = aacgmv2.get_aacgm_coord_arr(
                    lats[i, :], lons[i, :], 300, self.plot_date
                )
                if self.coords == "aacgmv2_mlt":
                    mlons[i, :] = mlt * 15
        else:
            mlats, mlons = lats, lons
        return mlats, mlons

    def overlay_instument(
        self,
        name, lat, lon,
        tx=cartopy.crs.PlateCarree(),
        marker="D",
        zorder=5,
        markerColor="m",
        markerSize=2,
        fontSize=6,
        font_color="m",
        xOffset=5,
        yOffset=-1.5,
    ):
        """Adding the radar location"""
        if "aacgm" in self.coords:
            lat, lon = self.to_aagcm(lat, lon)
        self.scatter(
            [lon],
            [lat],
            s=markerSize,
            marker=marker,
            color=markerColor,
            zorder=zorder,
            transform=tx,
            lw=0.8,
            alpha=0.4,
        )
        lat, lon = lat + yOffset, lon + xOffset
        if "aacgm" in self.coords:
            lat, lon = self.to_aagcm(lat, lon)
        x, y = self.projection.transform_point(lon, lat, src_crs=tx)
        self.text(
            x,
            y,
            name.upper(),
            ha="center",
            va="center",
            transform=self.projection,
            fontdict={"color": font_color, "size": fontSize},
            alpha=0.8,
        )
        return

    def overlay_radar(
        self,
        rad,
        tx=cartopy.crs.PlateCarree(),
        marker="o",
        zorder=2,
        markerColor="k",
        markerSize=2,
        fontSize=6,
        font_color="k",
        xOffset=5,
        yOffset=-1.5,
        annotate=True,
    ):
        """Adding the radar location"""
        hdw = pydarn.read_hdw_file(rad)
        lat, lon = hdw.geographic.lat, hdw.geographic.lon
        if "aacgm" in self.coords:
            lat, lon = self.to_aagcm(lat, lon)
        self.scatter(
            [lon],
            [lat],
            s=markerSize,
            marker=marker,
            color=markerColor,
            zorder=zorder,
            transform=tx,
            lw=0.8,
            alpha=0.4,
        )
        nearby_rad = [
            ["adw", "kod", "cve", "fhe", "wal", "gbr", "pyk", "aze", "sys"],
            ["ade", "ksr", "cvw", "fhw", "bks", "sch", "sto", "azw", "sye"],
        ]
        if annotate:
            rad = hdw.abbrev
            if rad in nearby_rad[0]:
                xOff, yOff = (
                    1.5 if not xOffset else xOffset,
                    0 if not yOffset else yOffset,
                )
            elif rad in nearby_rad[1]:
                xOff, yOff = (
                    -1.5 if not xOffset else -xOffset,
                    1 if not yOffset else yOffset,
                )
            else:
                xOff, yOff = xOffset, yOffset
            lat, lon = hdw.geographic.lat + yOff, hdw.geographic.lon + xOff
            if "aacgm" in self.coords:
                lat, lon = self.to_aagcm(lat, lon)
            x, y = self.projection.transform_point(lon, lat, src_crs=tx)
            self.text(
                x,
                y,
                rad.upper(),
                ha="center",
                va="center",
                transform=self.projection,
                fontdict={"color": font_color, "size": fontSize},
                alpha=0.8,
            )
        return

    def overlay_fov(
        self,
        rad,
        tx=cartopy.crs.PlateCarree(),
        maxGate=75,
        beamLimits=None,
        fovColor=None,
        fovAlpha=0.2,
        zorder=3,
        lineColor="r",
        lineWidth=0.8,
        ls="-",
        model="IS",
        fov_dir="front",
    ):
        """Overlay radar FoV"""
        hdw = pydarn.read_hdw_file(rad)
        latFull, lonFull = pydarn.Coords.GEOGRAPHIC(hdw.stid)
        latFull, lonFull = latFull.T, lonFull.T
        self.maxGate = maxGate
        lcolor = lineColor
        from numpy import concatenate, ones, shape, transpose, vstack

        sgate = 0
        egate = hdw.gates if not maxGate else maxGate
        ebeam = hdw.beams
        if beamLimits is not None:
            sbeam, ebeam = beamLimits[0], beamLimits[1]
        else:
            sbeam = 0
        if "aacgm" in self.coords:
            latFull, lonFull = self.to_aagcms(latFull, lonFull)
            
        xyz = self.projection.transform_points(tx, lonFull, latFull)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        contour_x = concatenate(
            (
                x[sbeam, :egate],
                x[sbeam:ebeam, egate],
                x[ebeam, egate-1::-1],
                x[ebeam::-1, sgate],
            )
        )
        contour_y = concatenate(
            (
                y[sbeam, :egate],
                y[sbeam:ebeam, egate],
                y[ebeam, egate-1::-1],
                y[ebeam::-1, sgate],
            )
        )
        self.plot(
            contour_x,
            contour_y,
            color=lcolor,
            zorder=zorder,
            linewidth=lineWidth,
            ls=ls,
            alpha=0.7,
        )
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
            self.add_patch(patch)
        if sbeam+1 == ebeam:
            xloc, yloc = (
                np.mean(x[sbeam:ebeam+1, egate+6]), 
                np.mean(y[sbeam:ebeam+1, egate+6])
            )
            self.text(
                xloc, yloc,
                sbeam, ha="center", va="center",
                fontdict=dict(color="r", fontweight="normal", size=8),
            )
            self.plot(
                np.mean(x[sbeam:ebeam+1, sgate:egate], axis=0),
                np.mean(y[sbeam:ebeam+1, sgate:egate], axis=0),
                color="r",
                zorder=zorder,
                linewidth=lineWidth,
                ls=ls,
                alpha=0.6,
            )
        return

    def overlay_data(
        self,
        rad,
        df,
        tx,
        fm=cartopy.crs.Geodetic(),
        p_max=300,
        p_min=-300,
        p_name="v",
        label="Velovcit, m/s",
        cmap=RedBlackBlue,
        cbar=False,
        maxGate=None,
        scan_time=None,
        model="IS",
        fov_dir="front",
        **kwargs,

    ):
        """Overlay radar Data"""
        if maxGate or hasattr(self, "maxGate"):
            maxGate = maxGate if maxGate else self.maxGate
            df = df[(df.slist <= maxGate)]
        
        print("maxGate>>>>>>>>>>>>>", maxGate)
        if len(df) > 0:
            # TODO
            hdw = pydarn.read_hdw_file(rad)
            lats, lons = pydarn.Coords.GEOGRAPHIC(hdw.stid, center=True)
            print(lats.shape, lons.shape)
            lats, lons = lats.T, lons.T
            Xb, Yg, Px = get_gridded_parameters(
                df, xparam="bmnum", yparam="slist", zparam=p_name
            )
            Xb, Yg = Xb.astype(int), Yg.astype(int)
            lons, lats = lons[Xb.ravel(), Yg.ravel()].reshape(Xb.shape), lats[
                Xb.ravel(), Yg.ravel()
            ].reshape(Xb.shape)
            # lons, lats = ((lons+hdw.geographic.lon)/2, (lats+hdw.geographic.lat)/2)
            XYZ = tx.transform_points(fm, lons, lats)
            Px = np.ma.masked_invalid(Px)
            print(XYZ.shape, Px.shape)
            # im = self.scatter(
            #     XYZ[:, :, 0],
            #     XYZ[:, :, 1],
            #     c=Px.T,
            #     transform=tx,
            #     cmap=cmap,
            #     vmax=p_max,
            #     vmin=p_min,
            #     s=0.1,
            #     marker="s",
            #     alpha=1.,
            #     zorder=7,
            #     **kwargs,
            # )
            im = self.pcolormesh(
                XYZ[:, :, 0],
                XYZ[:, :, 1],
                Px.T,
                vmax=p_max,
                vmin=p_min,
                transform=tx,
                cmap=cmap,
                zorder=2
            )
            if cbar:
                self._add_colorbar(im, label=label)
        return

    def _add_colorbar(self, im, label=""):
        """
        Add a colorbar to the right of an axis.
        """
        setsize(8)
        fig = self.get_figure()
        cpos = [1.04, 0.1, 0.025, 0.8]
        cax = self.inset_axes(cpos, transform=self.transAxes)
        cb = fig.colorbar(im, ax=self, cax=cax)
        cb.set_label(label)
        setsize(8)
        return

    def overlay_tec(
        self,
        lats,
        lons,
        tec,
        tx,
        fm=cartopy.crs.Geodetic(),
        p_max=0.15,
        p_min=-0.15,
        label="dTEC [TECu]",
        cmap=plt.cm.Greys,
        hcbar=True,
        **kwargs,
    ):
        """Overlay TEC Data"""
        XYZ = tx.transform_points(fm, lons, lats)
        im = self.scatter(
            XYZ[:, 0],
            XYZ[:, 1],
            c=tec,
            transform=tx,
            cmap=cmap,
            vmax=p_max,
            vmin=p_min,
            s=0.1,
            marker="o",
            alpha=1.0,
            **kwargs,
        )
        if hcbar:
            self._add_hcolorbar(im, label)
        return

    def _add_hcolorbar(self, im, label=""):
        """Add a colorbar to the right of an axis."""
        fig = self.get_figure()
        pos = self.get_position()
        cpos = [
            pos.x0 + 0.3 * pos.width,
            pos.y0 - 0.6 * pos.height,
            pos.width * 0.5,
            0.02,
        ]  # this list defines (left, bottom, width, height)
        cax = self.inset_axes(cpos, transform=self.transAxes)
        cb = fig.colorbar(
            im,
            ax=self,
            cax=cax,
            spacing="uniform",
            orientation="horizontal",
        )
        cb.set_label(label)
        return

    

# Now register the projection with matplotlib so the user can select
# it.
register_projection(SDCarto)
