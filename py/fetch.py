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
import xarray as xr

from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a

from plots import create_eiscat_line_plot

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
        txt = f"{date_range[0].strftime('%H')}-{date_range[1].strftime('%H')}, {date_range[0].strftime('%d %b %Y UT ')}/ " +\
            fr"$\theta$=[{lat_range[0]}$^\circ$,{lat_range[1]}$^\circ$] / " +\
            fr"$\phi$=[{lon_range[0]}$^\circ$,{lon_range[1]}$^\circ$]"
        plots.create_dtec_error_distribution(
            dtec_3D_chunck.ravel(), 
            (100.*dtec_3D_chunck/tec_3D_chunck).ravel(),
            txt
        )
        return

class Eiscat(object):
    
    def __init__(
        self, dates, 
        from_file = "database/MAD6400_2017-09-06_bella_60@vhf.nc",
        to_file = "dataset/MAD6301_2017-09-06_bella_60@vhf.txt"
    ):
        self.dates = dates
        # Only load these parameters along with time
        keys = [
            "gdalt", "ne", 
            # "ti", "tr", "co", "vo", "po+",
            # "dne", "dtr", "dco", "dvo", "dpo+",
        ]
        ds = xr.open_dataset(from_file, engine="h5netcdf")
        # Initalize all the parameters
        yf = dict(zip([None]*len(keys), keys))
        for k in keys:
            yf[k] = list()
        yf["time"] = list()
        # Load these parameters
        L = len(ds["range"].values) # Range
        for i, s in enumerate(ds["timestamps"].values):
            t = dt.datetime.fromtimestamp(s)
            for _, k in enumerate(keys):
                yf[k].extend(ds[k].values[i, :].tolist())
            yf["time"].extend([t] * L)
        # print(len(yf["time"]), len(yf["gdalt"]) , len(yf["ne"]))
        self.yf = pd.DataFrame.from_dict(yf)
        self.yf.time = self.yf.time.apply(
            lambda x: x.replace(
                minute=5*int(x.minute/5),
                second=0, microsecond=0
            )
        )
        self.yf.dropna(inplace=True)
        # print(yf.head())
        # print(ds["gdalt"],ds["ne"] )
        # print(list(ds.keys()))
        # if not os.path.exists(to_file):
        #     records = []
        #     with open(from_file, "r") as f:
        #         lines = f.readlines()
        #         header = list(filter(None, lines[0].replace("\n", "").split(" ")))
        #         for line in lines[1:]:
        #             records.append(
        #                 dict(
        #                     zip(
        #                         header, 
        #                         [   
        #                             float(x)
        #                             for x in list(filter(None, line.replace("\n", "").split(" ")))
        #                         ]
        #                     )
        #                 )
        #             )
        #     self.records = pd.DataFrame.from_records(records)
        #     self.records["DATE"] = self.records.UT1_UNIX.apply(
        #         lambda x: dt.datetime.fromtimestamp(x,tz=dt.UTC)
        #     )
        #     self.records["HEIGHT"] = np.int8( 
        #         self.records["RANGE"] * np.sin(np.deg2rad(30.))
        #     )
        #     self.records.to_csv(
        #         to_file, float_format="%g", 
        #         header=True, index=False
        #     )
        # else:
        #     self.records = pd.read_csv(to_file, parse_dates=["DATE"])

        # self.records.DATE = self.records.DATE.apply(
        #     lambda x: x.replace(second=0, minute=5*int(x.minute/5))
        # )
        # print(self.records.DATE.unique())
        create_eiscat_line_plot(self, "eiscat.png")
        return
    
    def _get_at_specifif_height_(self, ax, h=100, color="g", ms=0.6, multiplier=1e-9):
        o = self.records[
            (self.records.HEIGHT == 10.*int(h/10))
        ]
        o.drop_duplicates(subset="DATE", keep="last", inplace=True,)
        o = o[o.POP>=0]
        o.set_index("DATE",inplace=True)
        f = o.reindex(
            pd.date_range(
                start=o.index.min(),
                end=o.index.max(),
                freq="40s"
            )
        ).interpolate(method="cubic")
        o.reset_index(inplace=True)
        f.index.name = "DATE"
        f.reset_index(inplace=True)
        mul = "{%d}"%int(np.log10(multiplier))
        df = f[
            f.DATE == dt.datetime(2017,9,6,12,1,20,tzinfo=dt.timezone.utc)
        ]
        pcnt, absolute = (
            np.round((df.POP.tolist()[0]-f.POP.tolist()[0])/f.POP.tolist()[0], 2),
            np.round((df.POP.tolist()[0]-f.POP.tolist()[0]), 2)
        )
        print(pcnt, absolute*1e-10)
        ax.plot(
            f.DATE, f.POP*multiplier, marker=".", 
            color=color, ls="None", ms=ms,
            label=fr"$h={h}$ km, ($\times 10^{mul}$) []"
        )
        return f

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
    # GPS1deg(dates)
    # Eiscat(dates)
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