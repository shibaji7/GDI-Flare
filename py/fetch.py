import os
import pandas as pd # type: ignore
import pydarn # type: ignore
import glob
import bz2
from loguru import logger # type: ignore
import datetime as dt
import numpy as np
from tqdm import tqdm # type: ignore
from scipy import constants as C
import xarray as xr # type: ignore

from sunpy import timeseries as ts # type: ignore
from sunpy.net import Fido # type: ignore
from sunpy.net import attrs as a # type: ignore

from plots import create_eiscat_line_plot
import utils

os.environ["OMNIDATA_PATH"] = "/home/chakras4/OMNI/"

class SolarDataset(object):
    """
    This class is dedicated to plot GOES, FISM, and OMNI data
    from the repo using SunPy
    """

    def __init__(self, dates, dataset=[1, 2, 3]):
        """
        Parameters
        ----------
        dates: list of datetime object for start and end of TS
        """
        self.dates = dates
        self.dfs = {}
        if 1 in dataset: self.__load_FISM__()
        if 2 in dataset: self._load_omni_()
        if 3 in dataset: self.__loadGOES__()
        return
    
    def _load_omni_(self, res=1):
        import pyomnidata
        logger.info(f"OMNIDATA_PATH: {os.environ['OMNIDATA_PATH']}")
        pyomnidata.UpdateLocalData()
        self.omni = pd.DataFrame(
            pyomnidata.GetOMNI(self.dates[0].year,Res=res)
        )
        self.omni["time"] = self.omni.apply(
            lambda r: (
                dt.datetime(
                    int(str(r.Date)[:4]), 
                    int(str(r.Date)[4:6]),
                    int(str(r.Date)[6:].replace(".0","")) 
                ) 
                + dt.timedelta(hours=r.ut)
            ), 
            axis=1
        )
        self.omni = self.omni[
            (self.omni.time>=self.dates[0])
            & (self.omni.time<=self.dates[1])
        ]
        return
    
    def __load_FISM__(self):
        year, doy = (
            self.dates[0].year,
            self.dates[0].timetuple().tm_yday
        )
        url = f"https://lasp.colorado.edu/eve/data_access/eve_data/fism/flare_hr_data/{year}/"
        fname = f"FISM_60sec_{year}{doy}_v02_01.sav"
        link = url + fname
        filepath = "database/"+fname
        if not os.path.exists(filepath):
            os.system(f"wget -O {filepath} {link}")
        from scipy.io import readsav
        self.fism = readsav(filepath)
        return
    
    def get_fism_spectrum_by_time(self, date):
        o = pd.DataFrame()
        i = int((date-self.dates[0]).total_seconds()/60)
        o["wv"], o["ir"] = (
            self.fism["wavelength"],
            self.fism["irradiance"][i,:]
        )
        return o

    def __loadGOES__(self):
        """
        Load GOES data from remote/local repository
        """
        self.flare = {}
        self.dfs["goes"], self.goes, self.flareHEK = pd.DataFrame(), [], None
        result = Fido.search(
            a.Time(
                self.dates[0].strftime("%Y-%m-%d %H:%M"),
                self.dates[1].strftime("%Y-%m-%d %H:%M"),
            ),
            a.Instrument("XRS") | a.hek.FL & (a.hek.FRM.Name == "SWPC"),
        )
        if len(result) > 0:
            logger.info(f"Fetching GOES ...")
            tmpfiles = Fido.fetch(result, progress=False)
            for tf in tmpfiles:
                if "avg1m" in tf:
                    self.goes.append(ts.TimeSeries(tf))
                    self.dfs["goes"] = pd.concat(
                        [self.dfs["goes"], self.goes[-1].to_dataframe()]
                    )
            if len(self.dfs["goes"]) > 0:
                self.dfs["goes"].index.name = "time"
                self.dfs["goes"] = self.dfs["goes"].reset_index()
                self.dfs["goes"] = self.dfs["goes"][
                    (self.dfs["goes"].time >= self.dates[0])
                    & (self.dfs["goes"].time <= self.dates[1])
                ]
            # Retrieve HEKTable from the Fido result and then load
            hek_results = result["hek"]
            if len(hek_results) > 0:
                self.flare = hek_results[
                    "event_starttime",
                    "event_peaktime",
                    "event_endtime",
                    "fl_goescls",
                    "ar_noaanum",
                ]
        self.dfs["goes"].drop_duplicates(subset="time", inplace=True)
        return
    
class GPS1deg(object):

    def __init__(
        self, dates,
    ):
        import glob
        import xarray as xr
        self.dates = dates
        file = glob.glob(
            f"database/*{self.dates[0].strftime('%m%d')}*.nc"
        )[0]
        logger.info(f"GPS file: {file}")
        self.ds = xr.open_dataset(file)
        self.load_dataset()
        self.check_dataset_errorspan()
        return
    
    def load_dataset(self):
        self.ddates = [
            dt.datetime.fromtimestamp(s)
            for s in self.ds["timestamps"].values
        ]
        self.lats = self.ds["gdlat"].values.tolist()
        self.lons = self.ds["glon"].values.tolist()
        return
    
    def check_dataset_errorspan(self):
        import plots
        date_range = [
            self.dates[0].replace(hour=11), 
            self.dates[1].replace(day=self.dates[0].day, hour=13)
        ]
        lat_range, lon_range = (
            [30, 60],
            [-80, -120]
        )
        time_index_range = [
            self.ddates.index(date_range[0]),
            self.ddates.index(date_range[1])
        ]
        lat_index_range = [
            self.lats.index(lat_range[0]),
            self.lats.index(lat_range[1])
        ]
        lon_index_range = [
            self.lons.index(lon_range[0]),
            self.lons.index(lon_range[1])
        ]
        dtec_3D_chunck = self.ds["dtec"].values[
            time_index_range[0]:time_index_range[1],
            lat_index_range[0]:lat_index_range[1],
            lon_index_range[1]:lon_index_range[0],
        ]
        tec_3D_chunck = self.ds["tec"].values[
            time_index_range[0]:time_index_range[1],
            lat_index_range[0]:lat_index_range[1],
            lon_index_range[1]:lon_index_range[0],
        ]
        logger.info(f"Logging dTEC, error, {np.nanmax(dtec_3D_chunck)}/{np.nanmin(dtec_3D_chunck)}")
        txt = f"{date_range[0].strftime('%H')}-{date_range[1].strftime('%H')} UT, {date_range[0].strftime('%d %b %Y UT ')}/ " +\
            fr"$\theta$=[{lat_range[0]}$^\circ$,{lat_range[1]}$^\circ$] / " +\
            fr"$\phi$=[{lon_range[0]}$^\circ$,{lon_range[1]}$^\circ$] / " +\
            f"C={np.count_nonzero(~np.isnan(dtec_3D_chunck.ravel()))}" 
        plots.create_dtec_error_distribution(
            dtec_3D_chunck.ravel(), 
            (100.*dtec_3D_chunck/tec_3D_chunck).ravel(),
            txt, fname=f"gps.dtec_error_dist_{self.dates[0].strftime('%d')}.png"
        )
        return

class Eiscat(object):
    
    def __init__(
        self, dates, 
        from_file = "database/MAD6400_2017-09-06_bella_60@vhf.txt",
        to_file = "dataset/EISCAT_Tromso_vhf.csv"
    ):
        if not os.path.exists(to_file):
            o = []
            with open(from_file, "r") as f:
                lines = f.readlines()
                header = list(filter(None, lines[0].replace("\n", "").split(" ")))
                for line in lines[1:]:
                    line = list(filter(None, line.replace("\n", "").split(" ")))
                    o.append(dict(zip(header, line)))
            o = pd.DataFrame.from_records(o)
            o = o.astype(
                {
                    "GDALT": "float32",
                    "NE": "float64",
                    "DNE": "float64",
                    "TR": "float64",
                    "DTR": "float64",
                    "UT1_UNIX": "float64",
                    "UT2_UNIX": "float64",
                    "AZM": "float32",
                    "ELM": "float32",
                }
            )
            o = o[["GDALT", "NE", "DNE", "TR", "DTR", "UT1_UNIX", "AZM", "ELM"]]
            o["TIME"] = o.UT1_UNIX.apply(lambda x: dt.datetime.utcfromtimestamp(x))
            o["COR_NE"] = o.NE * (1+o.TR)
            o.to_csv(to_file, float_format="%g", header=True, index=False)
        else:
            o = pd.read_csv(to_file, parse_dates=["TIME"])
        self.yf = o.copy()
        create_eiscat_line_plot(self, "eiscat.png")
        return
    
    def _get_at_specifif_height_(self, ax=None, h=100, color="g", ms=1.2, multiplier=1e-10):
        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        
        o = self.yf[
            (self.yf.GDALT >= h-5)
            & (self.yf.GDALT <= h+5)
        ]
        o.drop_duplicates(subset="TIME", keep="last", inplace=True,)
        o.COR_NE = smooth(o.COR_NE, 7)
        
        df = o[
            (o.TIME >= dt.datetime(2017,9,6,12,1,))
            & (o.TIME <= dt.datetime(2017,9,6,12,3))
        ]
        pcnt, absolute = (
            np.round((df.COR_NE.tolist()[0]-o.COR_NE.tolist()[0])/o.COR_NE.tolist()[0], 2),
            np.round((df.COR_NE.tolist()[0]-o.COR_NE.tolist()[0]), 2)
        )
        if ax:
            ax.plot(
                o.TIME, o.COR_NE*multiplier, marker=".",
                color=color, ls="None", ms=ms,
                label=fr"$h={h}$ km, " + r"[$\theta_{N_e}$=%.1f, $\delta_{N_e}$=%.1f$\times 10^{10}$]"%(pcnt, absolute*1e-9)
            )
        return o

class Radar(object):

    def __init__(self, rad, dates, clean=False, type="fitacf3",):
        logger.info(f"Initialize radar: {rad}")
        self.rad = rad
        self.dates = dates
        self.clean = clean
        self.type = type
        tqdm.pandas()
        self.__setup__()
        self.__fetch_data__()
        self.calculate_decay_rate()
        fname = f"dataset/{self.dates[0].strftime('%Y%m%d')}.{self.rad}.{self.type}.csv"
        self.df[
            [
                "v", "slist", "bmnum", "w_l",
                "elv", "phi0", "time", "tfreq",
                "scan", "gdlat", "glong", "sza"
            ]
        ].to_csv(fname, float_format="%g", index=False, header=True)
        return
    
    def __setup__(self):
        logger.info(f"Setup radar: {self.rad}")
        self.files = glob.glob(
            #f"/sd-data/{self.dates[0].year}/{self.type}/{self.rad}/{self.dates[0].strftime('%Y%m%d')}*{self.rad}.*"
            f"fitacf_location/{self.dates[0].strftime('%Y%m%d')}*{self.rad}.*{self.type}*"
        ) 
        self.files.sort()
        self.hdw = pydarn.read_hdw_file(self.rad)
        self.fov = pydarn.Coords.GEOGRAPHIC(self.hdw.stid)
        #self.fov = (fov.latFull.T, fov.lonFull.T)
        logger.info(f"Files: {len(self.files)}")
        return

    def get_glatlon(self, row):
        from pysolar.solar import get_altitude
        bm, gate = row["bmnum"], row["slist"]
        row["gdlat"], row["glong"] = self.fov[0][gate, bm], self.fov[1][gate, bm]
        date = row["time"].to_pydatetime().replace(tzinfo=dt.timezone.utc)
        row["sza"] = 90.-get_altitude(row["gdlat"], row["glong"], date)
        return row

    def __fetch_data__(self):
        if self.clean: os.remove(f"database/{self.dates[0].strftime('%Y%m%d')}.{self.rad}.{self.type}.csv")
        if os.path.exists(f"database/{self.dates[0].strftime('%Y%m%d')}.{self.rad}.{self.type}.csv"):
            logger.info(f"Reading file: {self.dates[0].strftime('%Y%m%d')}.{self.rad}.{self.type}.csv")
            self.df = pd.read_csv(f"database/{self.dates[0].strftime('%Y%m%d')}.{self.rad}.{self.type}.csv", parse_dates=["time"])
        else:
            records = []
            for f in self.files:
                logger.info(f"Reading file: {f}")
                with bz2.open(f) as fp:
                    reader = pydarn.SuperDARNRead(fp.read(), True)
                    records += reader.read_fitacf()
            if len(records)>0:
                self.__tocsv__(records)
        self.df.tfreq = np.round(np.array(self.df.tfreq)/1e3, 1)
        return

    def __tocsv__(self, records):
        time, v, slist, p_l, frang, scan, beam,\
            w_l, gflg, elv, phi0, tfreq, rsep, v_e = (
            [], [], [],
            [], [], [],
            [], [], [],
            [], [], [],
            [], [],
        )
        for r in records:
            if "v" in r.keys():
                t = dt.datetime(
                    r["time.yr"], 
                    r["time.mo"],
                    r["time.dy"],
                    r["time.hr"],
                    r["time.mt"],
                    r["time.sc"],
                    r["time.us"],
                )
                time.extend([t]*len(r["v"]))
                tfreq.extend([r["tfreq"]]*len(r["v"]))
                rsep.extend([r["rsep"]]*len(r["v"]))
                frang.extend([r["frang"]]*len(r["v"]))
                scan.extend([r["scan"]]*len(r["v"]))
                beam.extend([r["bmnum"]]*len(r["v"]))
                v.extend(r["v"])
                v_e.extend(r["v_e"])
                gflg.extend(r["gflg"])
                slist.extend(r["slist"])
                p_l.extend(r["p_l"])
                w_l.extend(r["w_l"])
                if "elv" in r.keys(): elv.extend(r["elv"])
                if "phi0" in r.keys(): phi0.extend(r["phi0"])                
            
        self.df = pd.DataFrame()
        self.df["v"] = v
        self.df["v_e"] = v_e
        self.df["gflg"] = gflg
        self.df["slist"] = slist
        self.df["bmnum"] = beam
        self.df["p_l"] = p_l
        self.df["w_l"] = w_l
        if len(elv) > 0: self.df["elv"] = elv
        if len(phi0) > 0: self.df["phi0"] = phi0
        self.df["time"] = time
        self.df["tfreq"] = tfreq
        self.df["scan"] = scan
        self.df["rsep"] = rsep
        self.df["frang"] = frang

        if self.dates:
            self.df = self.df[
                (self.df.time>=self.dates[0]) & 
                (self.df.time<=self.dates[1])
            ]
        self.df = self.df.apply(self.get_glatlon, axis=1)
        self.df.to_csv(f"database/{self.dates[0].strftime('%Y%m%d')}.{self.rad}.{self.type}.csv", index=False, header=True)
        return

    def caclulate_elevation_angle(self, psi_obs, bmnum, tfreq, tdiff=None):
        """
        Parameters
        ----------
        psi_obs: Phase difference in Obs.
        hwd: Radar Hardware configuration
        tdiff: External TDiff
        """
        from scipy import constants as scicon

        tdiff = self.hdw.tdiff if tdiff is None else tdiff
        logger.info(f" Tdiff : {tdiff}")   

        C = scicon.c
    
        # calculate the values that don't change if this hasn't already been done. 
        X, Y, Z = (
            self.hdw.interferometer_offset[0], 
            self.hdw.interferometer_offset[1], 
            self.hdw.interferometer_offset[2]
        )
        d = np.sqrt(X**2+Y**2+Z**2)
        
        # SGS: 20180926
        #
        # There is still some question as to exactly what the phidiff parameter in
        # the hdw.dat files means. The note in the hdw.dat files, presumably written
        # by Ray is:
        # 12) Phase sign (Cabling errors can lead to a 180 degree shift of the
        #     interferometry phase measurement. +1 indicates that the sign is
        #     correct, -1 indicates that it must be flipped.)
        # The _only_ hdw.dat file that has this value set to -1 is GBR during the
        # time period: 19870508 - 19921203
        #
        # To my knowlege there is no data available prior to 1993, so dealing with
        # this parameter is no longer necessary. For this reason I am simply
        # removing it from this algorithm.
    
        sgn = -1 if Y < 0 else 1
        boff = (self.hdw.beams / 2.0) - 0.5
        phi0 = self.hdw.beam_separation * (bmnum - boff) * np.pi / 180.0
        cp0  = np.cos(phi0)
        sp0  = np.sin(phi0)
        
        # Phase delay [radians] due to electrical path difference.                
        #   If the path length (cable and electronics) to the interferometer is   
        #  shorter than that to the main antenna array, then the time for the    
        #  to transit the interferometer electrical path is shorter: tdiff < 0   
        psi_ele = -2.0 * np.pi * tfreq * tdiff * 1.0e-3
        
        # Determine elevation angle (a0) where psi (phase difference) is maximum; 
        #   which occurs when k and d are anti-parallel. Using calculus of        
        #  variations to compute the value: d(psi)/d(a) = 0                      
        a0 = np.arcsin(sgn * Z * cp0 / np.sqrt(Y**2 + Z**2))
        
        # Note: we are assuming that negative elevation angles are unphysical.    
        #  The act of setting a0 = 0 _only_ has the effect to change psi_max     
        #  (which is used to compute the correct number of 2pi factors and map   
        #  the observed phase to the actual phase. The _only_ elevation angles   
        #  that are affected are the small range from [-a0, 0]. Instead of these 
        #  being mapped to negative elevation they are mapped to very small      
        #  range just below the maximum.                                         

        # Note that it is possible in some cases with sloping ground that extends 
        #  far in front of the radar, that negative elevation angles might exist.
        #  However, since elevation angles near the maximum "share" this phase   
        #  [-pi,pi] it is perhaps more likely that the higher elevation angles   
        #  are actually what is being observed.                                  

        #In either case, one must decide which angle to chose (just as with all  
        #  the aliased angles). Here we decide (unless the keyword 'negative' is 
        #  set) that negative elevation angles are unphysical and map them to    
        #  the upper end.    
        a0[a0 < 0.] = 0.
        ca0 = np.cos(a0)
        sa0 = np.sin(a0)
        
        # maximum phase = psi_ele + psi_geo(a0)
        psi_max = psi_ele + 2.0 * np.pi * tfreq *\
                    1.0e3 / C * (X * sp0 + Y * np.sqrt(ca0*ca0 - sp0*sp0) + Z * sa0)
        
        # compute the number of 2pi factors necessary to map to correct region
        dpsi = (psi_max - psi_obs)
        n2pi = np.floor(dpsi / (2.0 * np.pi)) if Y > 0 else np.ceil(dpsi / (2.0 * np.pi))
        
        # map observed phase to correct extended phase
        d2pi = n2pi * 2.0 * np.pi
        psi_obs += d2pi
        
        # now solve for the elevation angle: alpha
        E = (psi_obs / (2.0*np.pi*tfreq*1.0e3) + tdiff*1e-6) * C - X * sp0
        alpha = np.arcsin((E*Z + np.sqrt(E*E * Z*Z - (Y*Y + Z*Z)*(E*E - Y*Y*cp0*cp0))) / (Y*Y + Z*Z))
        return (180.0 * alpha / np.pi)
    
    def recalculate_elv_angle(self, tdiff=None):
        R = 6371.
        self.df["elv"] = self.caclulate_elevation_angle(
            self.df.phi0, self.df.bmnum, self.df.tfreq*1e3, tdiff
        )
        self.df["gate"] = np.copy(self.df.slist)
        self.df.slist = (self.df.slist*self.df.rsep) + self.df.frang 
        self.df["vheight_2p"] = (
            (
                R**2 + self.df.slist**2 + (
                    2*self.df.slist*R*np.sin(np.deg2rad(self.df.elv))
                )
            )**0.5 - 
            R
        )        
        self.df["vheight_Ch"] = self.df.slist.apply(lambda x: utils.chisham(x))
        return

    def calculate_decay_rate(self):
        logger.info(f"Calculate Decay")
        f, w_l = np.array(self.df.tfreq)*1e6, np.array(self.df.w_l)
        k = 2*np.pi*f/C.c
        self.df["tau_l"] = 1.e3/(k*w_l)
        return
        
    def load_VHF_file(self):
        file_names = [
            "database/MAD6301_2017-09-06_bella_60@vhf (1).nc"
        ]
        import xarray as xr
        return

if __name__ == "__main__":
    # dates = [
    #     dt.datetime(2017,9,6), dt.datetime(2017,9,7),
    # ]
    # GPS1deg(dates)
    # Eiscat(dates)
    # Radar("sas", dates, type="fitacf")
    # Radar("kod", dates)
    # Radar("pgr", dates)
    # SolarDataset(dates)
    # dates = [
    #     dt.datetime(2017,8,30), dt.datetime(2017,8,31)
    # ]
    # Radar("sas", dates)
    # Radar("kod", dates)
    # Radar("pgr", dates)
    # SolarDataset(dates)
    for d in range(1):
        dates = [
            dt.datetime(2017,8,1)+dt.timedelta(d), 
            dt.datetime(2017,8,2)+dt.timedelta(d+1),
        ]
        Radar("sas", dates, type="fitacf")

    