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


import aacgmv2
import cartopy
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pydarn
from cartopy.mpl.geoaxes import GeoAxes
from descartes import PolygonPatch
from matplotlib.projections import register_projection
from shapely.geometry import LineString, MultiLineString, Polygon, mapping
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
import datetime as dt

def setsize(size=6):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # plt.style.use("dark_background")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    mpl.rcParams.update(
        {
            "xtick.labelsize": size,
            "ytick.labelsize": size,
            "font.size": size,
            "text.color": "k",
        }
    )
    return


class CartoBase(GeoAxes):
    name = "CartoBase"

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

    def overaly_coast_lakes(self, resolution="50m", color="k", **kwargs):
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

    def mark_longitudes(self, lon_arr=numpy.arange(-180, 180, 60), **kwargs):
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
        # get the boundaries
        [x1, y1], [x2, y2] = self.viewLim.get_points()
        bound_lim_arr = []
        right_bound = LineString(([-x1, y1], [x2, y2]))
        top_bound = LineString(([x1, -y1], [x2, y2]))
        bottom_bound = LineString(([x1, y1], [x2, -y2]))
        left_bound = LineString(([x1, y1], [-x2, y2]))
        plot_outline = MultiLineString(
            [right_bound, top_bound, bottom_bound, left_bound]
        )
        # get the plot extent, we"ll get an intersection
        # to locate the ticks!
        plot_extent = self.get_extent(cartopy.crs.Geodetic())
        line_constructor = lambda t, n, b: numpy.vstack(
            (numpy.zeros(n) + t, numpy.linspace(b[2], b[3], n))
        ).T
        for t in lon_arr[:-1]:
            try:
                xy = line_constructor(t, 30, plot_extent)
                # print(xy)
                proj_xyz = self.projection.transform_points(
                    cartopy.crs.Geodetic(), xy[:, 0], xy[:, 1]
                )
                xyt = proj_xyz[..., :2]
                ls = LineString(xyt.tolist())
                locs = plot_outline.intersection(ls)
                if not locs:
                    continue
                # we need to get the alignment right
                # so get the boundary closest to the label
                # and plot it!
                closest_bound = min(
                    [
                        right_bound.distance(locs),
                        top_bound.distance(locs),
                        bottom_bound.distance(locs),
                        left_bound.distance(locs),
                    ]
                )
                if closest_bound == right_bound.distance(locs):
                    ha = "left"
                    va = "top"
                elif closest_bound == top_bound.distance(locs):
                    ha = "left"
                    va = "bottom"
                elif closest_bound == bottom_bound.distance(locs):
                    ha = "left"
                    va = "top"
                else:
                    ha = "right"
                    va = "top"
                if self.coords == "aacgmv2_mlt":
                    marker_text = str(int(t / 15.0))
                else:
                    marker_text = r"$%s^{\circ}$" % str(t)
                self.text(
                    locs.bounds[0] + 0.02 * locs.bounds[0],
                    locs.bounds[1] + 0.02 * locs.bounds[1],
                    marker_text,
                    ha=ha,
                    va=va,
                    **kwargs,
                    alpha=0.5,
                )
            except:
                pass
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

    def add_ground_station(
        self,
        stn,
        lat,
        lon,
        tx=cartopy.crs.PlateCarree(),
        marker="o",
        zorder=2,
        markerColor="k",
        markerSize=3,
        fontSize=4,
        font_color="k",
        xOffset=8,
        yOffset=-1,
    ):
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
        x, y = self.projection.transform_point(lon, lat, src_crs=tx)
        self.text(
            x,
            y,
            stn.upper(),
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
        fontSize=4,
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

    def lay_fov(
        self,
        lats,
        lons,
    ):
        return

    def overlay_fov(
        self,
        rad,
        tx=cartopy.crs.PlateCarree(),
        maxGate=70,
        beamLimits=None,
        fovColor=None,
        fovAlpha=0.2,
        zorder=1,
        lineColor="k",
        lineWidth=0.4,
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
        from numpy import concatenate, transpose, vstack

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
                x[sbeam, sgate:egate],
                x[sbeam:ebeam, egate],
                x[ebeam, egate:sgate:-1],
                x[ebeam:sbeam:-1, sgate],
            )
        )
        contour_y = concatenate(
            (
                y[sbeam, sgate:egate],
                y[sbeam:ebeam, egate],
                y[ebeam, egate:sgate:-1],
                y[ebeam:sbeam:-1, sgate],
            )
        )
        self.plot(
            contour_x,
            contour_y,
            color=lcolor,
            zorder=zorder,
            linewidth=lineWidth,
            ls=ls,
            alpha=0.6,
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
        if sbeam + 1 == ebeam:
            xloc, yloc = (
                np.mean(x[sbeam : ebeam + 1, egate]),
                np.mean(y[sbeam : ebeam + 1, egate]),
            )
            # self.text(
            #     xloc, yloc,
            #     sbeam, ha="center", va="center",
            #     fontdict=dict(color="r", size=5)
            # )
            # self.plot(
            #     np.mean(x[sbeam:ebeam+1, sgate:egate], axis=0),
            #     np.mean(y[sbeam:ebeam+1, sgate:egate], axis=0),
            #     color="r",
            #     zorder=zorder,
            #     linewidth=lineWidth,
            #     ls=ls,
            #     alpha=0.6,
            # )
        return

    def overlay_data(
        self,
        rad,
        df,
        tx,
        fm=cartopy.crs.Geodetic(),
        p_max=33,
        p_min=0,
        p_name="p_l",
        label="Power [dB]",
        cmap=plt.cm.plasma,
        cbar=True,
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
        if len(df) > 0:
            # TODO
            hdw = pydarn.read_hdw_file(rad)
            lats, lons = pydarn.Coords.GEOGRAPHIC(hdw.stid)
            lats, lons = lats.T, lons.T
            Xb, Yg, Px = utils.get_gridded_parameters(
                df, xparam="bmnum", yparam="slist", zparam=p_name
            )
            Xb, Yg = Xb.astype(int), Yg.astype(int)
            lons, lats = lons[Xb.ravel(), Yg.ravel()].reshape(Xb.shape), lats[
                Xb.ravel(), Yg.ravel()
            ].reshape(Xb.shape)
            lons, lats = (
                (lons + hdw.geographic.lon) / 2,
                (lats + hdw.geographic.lat) / 2,
            )
            XYZ = tx.transform_points(fm, lons, lats)
            Px = np.ma.masked_invalid(Px)
            # im = self.scatter(
            #     XYZ[:, :, 0],
            #     XYZ[:, :, 1],
            #     c=Px.T,
            #     transform=tx,
            #     cmap=cmap,
            #     vmax=p_max,
            #     vmin=p_min,
            #     s=0.3,
            #     marker="o",
            #     alpha=0.9,
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
                zorder=2,
            )
            if cbar:
                self._add_colorbar(im, label=label)
        return

    def _add_colorbar(self, im, label="", plims=[]):
        """
        Add a colorbar to the right of an axis.
        """
        # utils.setsize()
        fig = self.get_figure()
        cpos = [1.04, 0.1, 0.025, 0.8]
        cax = self.inset_axes(cpos, transform=self.transAxes)
        cb = fig.colorbar(im, ax=self, cax=cax)
        cb.set_label(label, color="k", fontsize=8)
        if len(plims) == 2:
            cb.set_ticks(np.linspace(plims[0], plims[1], 5))
            cb.ax.tick_params(which="both", colors="k", labelsize=8)
            cb.outline.set_edgecolor("k")
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
        label="TEC [TECu]",
        cmap=plt.cm.Spectral,
        cbar=True,
    ):
        """Overlay TEC Data"""
        XYZ = tx.transform_points(fm, lons, lats)
        im = self.pcolormesh(
            XYZ[:, :, 0],
            XYZ[:, :, 1],
            tec,
            vmax=p_max,
            vmin=p_min,
            transform=tx,
            cmap=cmap,
        )
        self.scatter(
            XYZ[:, :, 0].ravel(), 
            XYZ[:, :, 1].ravel(), 
            s=[
                0.01 if x >0 else 0 
                for x in tec.ravel()
            ], 
            color="k"
        )
        if cbar:
            setsize(10)
            self._add_colorbar(im, label=label, plims=[p_min, p_max])
        return

    def add_tec_grad(
        self,
        lats,
        lons,
        dxZ, dyZ,
        tx,
        fm=cartopy.crs.Geodetic(),
        scale=3., lenx=3., tag=False,
    ):
        XYZ = tx.transform_points(fm, lons, lats)
        # print(lons.min(), lats.min(), lons.max(), lats.max())
        ql = self.quiver(
            XYZ[:, :, 0], XYZ[:, :, 1],
            dxZ,
            dyZ*0,
            scale=scale,
            headaxislength=0,
            linewidth=0.6,
            scale_units="inches",
        )
        if tag:
            self.quiverkey(
                ql,
                0.85,
                1.05,
                lenx,
                r"$\nabla_{\phi}n'_0$:"+str(lenx),
                labelpos="S",
                transform=tx,
                color="k",
                fontproperties={"size": 8},
            )
        return

# Now register the projection with matplotlib so the user can select
# it.
register_projection(CartoBase)






class CartoDataOverlay(object):

    def __init__(
        self,
        date,
        fig_title=None,
        nrows=3,
        ncols=2,
        coord="geo",
        cb=True,
        central_longitude=-90.0,
        central_latitude=45.0,
        extent=[-130, -50, 10, 70],
        plt_lats=np.arange(10, 70, 10),
        txt_size=8,
    ):
        setsize(txt_size)
        self.cb = cb
        self.date = date
        self.nrows, self.ncols = nrows, ncols
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(3 * ncols, 3 * nrows), dpi=300)
        self.coord = coord
        self.central_longitude = central_longitude
        self.central_latitude = central_latitude
        self.plt_lats = plt_lats
        self.extent = extent
        self.fig_title = fig_title
        return

    def save(self, filepath):
        self.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def date_string(self, label_style="web"):
        # Set the date and time formats
        dfmt = "%d %b %Y" if label_style == "web" else "%d %b %Y,"
        tfmt = "%H:%M"
        stime = self.date
        date_str = "{:{dd} {tt}} UT".format(stime, dd=dfmt, tt=tfmt)
        return date_str

    def add_axes(self):
        """
        Instatitate figure and axes labels
        """
        self._num_subplots_created += 1
        # proj = cartopy.crs.NorthPolarStereo(central_longitude=self.central_longitude)
        proj = cartopy.crs.PlateCarree(central_longitude=self.central_longitude)
        ax = self.fig.add_subplot(
            100 * self.nrows + 10 * self.ncols + self._num_subplots_created,
            projection="CartoBase",
            map_projection=proj,
            coords=self.coord,
            plot_date=self.date,
        )
        ax.overaly_coast_lakes(lw=0.4, alpha=0.4)
        plt_lons = np.arange(-180, 181, 15)
        mark_lons = np.arange(self.extent[0], self.extent[1], 15)
        plt_lats = self.plt_lats
        ax.set_extent(self.extent, crs=cartopy.crs.PlateCarree())
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.2)
        gl.xlocator = mticker.FixedLocator(plt_lons)
        gl.ylocator = mticker.FixedLocator(plt_lats)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.n_steps = 90
        ax.mark_latitudes(plt_lats, fontsize="xx-small", color="k")
        ax.mark_longitudes(mark_lons, fontsize="xx-small", color="k")
        self.proj = proj
        self.geo = cartopy.crs.Geodetic()
        if self._num_subplots_created == 1:
            ax.text(
                -0.02,
                0.99,
                "Coord: Geo",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize="x-small",
                rotation=90,
            )
            ax.text(
                0.05,
                1.05,
                (
                    f"{self.date_string()} / {self.fig_title}"
                    if self.fig_title
                    else f"{self.date_string()}"
                ),
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontsize="x-small",
            )
        return ax

    def create_night_shade(
        self, date, 
        gdlats=np.arange(0,90), 
        glons=np.arange(-180,180), 
        ax=None
    ):
        import pysolar
        ax = ax if ax else self.add_axes()
        date = date.replace(tzinfo=dt.timezone.utc)
        SZA = np.zeros(shape=(len(gdlats), len(glons)))
        for i, glat in enumerate(gdlats):
            for j, glon in enumerate(glons):
                SZA[i,j] = 90 - pysolar.solar.get_altitude(glat, glon, date)
        gdlats, glons = np.meshgrid(gdlats, glons)
        XYZ = self.proj.transform_points(self.geo, glons, gdlats)
        ax.contourf(
            XYZ[:, :, 0],
            XYZ[:, :, 1],
            np.ma.masked_invalid(SZA.T),
            levels=[0, 30, 90,100, 110, 150, 180],
            transform=self.proj,
            cmap="gray_r",
        )
        from cartopy.feature.nightshade import Nightshade
        ax.add_feature(Nightshade(date, alpha = 0.9))
        return ax
    
    def add_tec_dataset(
        self,
        tec, lat, lon,
        ax=None,
        p_max=20,
        p_min=0,
        label="TEC [TECu]",
        cbar=True,
        date=None
    ):
        date = date if date else self.date
        ax = ax if ax else self.add_axes()
        self.create_night_shade(date=date, ax=ax)
        ax.overlay_tec(
            lat,
            lon,
            tec,
            tx=self.proj,
            p_max=p_max,
            p_min=p_min,
            label=label,
            cbar=cbar,
        )
        return ax
    
    def lay_tec_grad(
        self,
        gdlats, glons, dxZ, dyZ,
        ax=None,
        lenx=1.5, scale=0.5, tag=False,
    ):
        ax = ax if ax else self.add_axes()
        ax.add_tec_grad(
            gdlats,
            glons,
            dxZ, dyZ,
            self.proj,
            lenx=lenx, scale=scale, tag=tag
        )
        return
