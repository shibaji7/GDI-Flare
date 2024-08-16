import os
import pandas as pd
import pydarn 
import glob
import bz2
from loguru import logger
import datetime as dt
import numpy as np
from tqdm import tqdm
from scipy import constants as C

from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a

os.environ["OMNIDATA_PATH"] = "/home/shibaji/omni/"

class SolarDataset(object):
    """
    This class is dedicated to plot GOES, FISM, and OMNI data
    from the repo using SunPy
    """

    def __init__(self, dates):
        """
        Parameters
        ----------
        dates: list of datetime object for start and end of TS
        """
        self.dates = dates
        self.dfs = {}
        self.__load_FISM__()
        self._load_omni_()
        self.__loadGOES__()
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
                self.goes.append(ts.TimeSeries(tf))
                self.dfs["goes"] = pd.concat(
                    [self.dfs["goes"], self.goes[-1].to_dataframe()]
                )
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
        return
    
class Eiscat(object):
    
    def __init__(self, dates, files = ["database/MAD6301_2017-09-06_bella_60@vhf.nc"]):
        import xarray as xr
        self.dates = dates
        self.dataset = xr.open_dataset(files[0])
        self._unwrap_()
        return
    
    def _unwrap_(self):
        self.dates = [
            dt.datetime.fromtimestamp(s,tz=dt.UTC)
            for s in self.dataset["timestamps"].values
        ]
        self.range = self.dataset["range"].values
        self.azm = self.dataset["azm"].values
        self.elm = self.dataset["elm"].values
        self.tfreq = self.dataset["tfreq"].values
        self.pop = self.dataset["pop"].values
        self.height = self.range / np.cos(np.deg2rad(self.range))
        return

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
            f"/sd-data/{self.dates[0].year}/{self.type}/{self.rad}/{self.dates[0].strftime('%Y%m%d')}*{self.rad}.*"
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
            w_l, gflg, elv, phi0, tfreq, rsep = (
            [], [], [],
            [], [], [],
            [], [], [],
            [], [], [],
            [],
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
                gflg.extend(r["gflg"])
                slist.extend(r["slist"])
                p_l.extend(r["p_l"])
                w_l.extend(r["w_l"])
                if "elv" in r.keys(): elv.extend(r["elv"])
                if "phi0" in r.keys(): phi0.extend(r["phi0"])                
            
        self.df = pd.DataFrame()
        self.df["v"] = v
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

    def recalculate_elv_angle(self, XOff=0, YOff=100, ZOff=0):
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
    dates = [
        dt.datetime(2017,9,6), dt.datetime(2017,9,7),
    ]
    Eiscat(dates)
    # Radar("sas", dates)
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