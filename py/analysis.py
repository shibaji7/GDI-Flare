import datetime as dt
import sys
sys.path.append("py")
import numpy as np
from rti import RangeTimePlot
from fetch import SolarDataset, Radar, GPS1deg
from plots import plot_figure2

def compare_quiet_versus_event_day():
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    import plots
    import matplotlib.pyplot as plt

    plots.setsize(10)
    dates = [dt.datetime(2017,8,28), dt.datetime(2017,8,29)]
    quiet_data = SolarDataset( dates )
    print(quiet_data.omni.head())
    fig = plt.figure(figsize=(16, 9), dpi=300)
    
    ax = fig.add_subplot(321)
    ax.text(
        0.1, 1.05, "30 Aug, 2017",
        ha="left", va="center",
        transform=ax.transAxes
    )
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.set_ylabel("Irradiance, $W/m^2$")
    ax.semilogy(
        quiet_data.dfs["goes"].time, quiet_data.dfs["goes"].xrsb, 
        color="r", ls="-", lw=0.6
    )
    ax.set_ylim(1e-7, 1e-2)
    ax.set_xlim(dates)

    ax = fig.add_subplot(323)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.plot(
        quiet_data.omni.time, quiet_data.omni.BzGSM, 
        color="r", ls="-", lw=0.8, label=r"$B_z$"
    )
    ax.plot(
        quiet_data.omni.time, quiet_data.omni.ByGSM, 
        color="g", ls="-", lw=0.8, label=r"$B_y$"
    )
    ax.plot(
        quiet_data.omni.time, quiet_data.omni.BxGSE, 
        color="k", ls="-", lw=0.8, label=r"$B_x$"
    )
    ax.legend(loc=1)
    ax.set_ylabel("IMF, nT")
    ax.set_ylim(-10, 10)
    ax.set_xlim(dates)
    
    ax = fig.add_subplot(325)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.set_xlim([dt.datetime(2017,8,30), dt.datetime(2017,8,31)])
    ax.plot(
        quiet_data.omni.time, quiet_data.omni.ProtonDensity/10, 
        color="k", ls="-", lw=0.8, label=r"$n\times 10$, /cc"
    )
    ax.plot(
        quiet_data.omni.time, quiet_data.omni.FlowPressure, 
        color="b", ls="-", lw=0.8, label=r"$P_{dyn}$"
    )
    ax.plot(
        quiet_data.omni.time, quiet_data.omni.FlowSpeed/100, 
        color="m", ls="-", lw=0.8, label=r"$V\times 10^{2}$, km/s"
    )
    ax.legend(loc=1)
    ax.set_ylabel("SW Params")
    ax.set_xlabel("Time, UT")
    ax.set_ylim(0, 8)
    ax.set_xlim(dates)
    

    
    event_data = SolarDataset(
        [dt.datetime(2017,9,6), dt.datetime(2017,9,7)]
    )
    ax = fig.add_subplot(322)
    ax.text(
        0.1, 1.05, "6 Sep, 2017",
        ha="left", va="center",
        transform=ax.transAxes
    )
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.set_ylabel("Irradiance, $W/m^2$")
    ax.semilogy(
        event_data.dfs["goes"].time, event_data.dfs["goes"].xrsb, 
        color="r", ls="-", lw=0.6
    )
    ax.set_ylim(1e-7, 1e-2)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])

    ax = fig.add_subplot(324)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.plot(
        event_data.omni.time, event_data.omni.BzGSM, 
        color="r", ls="-", lw=0.8, label=r"$B_z$"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.ByGSM, 
        color="g", ls="-", lw=0.8, label=r"$B_y$"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.BxGSE, 
        color="k", ls="-", lw=0.8, label=r"$B_x$"
    )
    ax.legend(loc=1)
    ax.set_ylabel("IMF, nT")
    ax.set_ylim(-10, 10)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    
    ax = fig.add_subplot(326)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.plot(
        event_data.omni.time, event_data.omni.ProtonDensity/10, 
        color="k", ls="-", lw=0.8, label=r"$n\times 10$, /cc"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.FlowPressure, 
        color="b", ls="-", lw=0.8, label=r"$P_{dyn}$"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.FlowSpeed/100, 
        color="m", ls="-", lw=0.8, label=r"$V\times 10^{2}$, km/s"
    )
    ax.legend(loc=1)
    ax.set_ylabel("SW Params")
    ax.set_xlabel("Time, UT")
    ax.set_ylim(0, 8)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])


    fig.savefig("figures/compare_days.png", bbox_inches="tight")
    return

def create_RTI_figure():
    dates = [dt.datetime(2017,9,6), dt.datetime(2017,9,7)]
    rads = ["sas", "pgr", "kod"]
    radars = {}
    for rad in rads:
        radars[rad] = Radar(rad, dates)
    
    plot_figure2(radars, dates, rads)
    return
    
def create_GPS_error_list():
    dates = [
        dt.datetime(2017,9,6),
        dt.datetime(2017,9,7)
    ]
    GPS1deg(dates)
    dates = [
        dt.datetime(2017,8,30),
        dt.datetime(2017,8,31)
    ]
    GPS1deg(dates)
    return

def create_RTI_plot():
    dates = [dt.datetime(2017,9,6), dt.datetime(2017,9,7)]
    radar = Radar("sas", dates)
    event_data = SolarDataset(dates)
    
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    import plots
    import matplotlib.pyplot as plt
    plots.setsize(18)
    fig = plt.figure(figsize=(8, 4*6), dpi=300)


    ax = fig.add_subplot(611)
    ax.text(
        0.1, 1.05, "6 Sep, 2017",
        ha="left", va="center",
        transform=ax.transAxes
    )
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.set_ylabel("Irradiance, $W/m^2$")
    ax.semilogy(
        event_data.dfs["goes"].time, event_data.dfs["goes"].xrsb, 
        color="r", ls="-", lw=0.9, label="X-ray [0.1-0.8 nm]"
    )
    ax.semilogy(
        event_data.dfs["goes"].time, event_data.dfs["goes"].xrsa, 
        color="b", ls="-", lw=0.9, label="X-ray [0.05-0.4 nm]"
    )
    ax.text(0.1, 0.9, "(a)", ha="left", va="center", transform=ax.transAxes)
    ax.set_ylim(1e-7, 1e-2)
    ax.legend(loc=1)
    ax.axvline(dt.datetime(2017, 9, 6, 11, 56), ls="--", lw=0.4, color="r")
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), ls="--", lw=0.4, color="k")
    ax.set_xlim(dates)
    ax.set_xticks([])

    ax = fig.add_subplot(612)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    # ax.axvline(dt.datetime(2017, 9, 6, 11, 56), ls="--", lw=0.4, color="r")
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), ls="--", lw=0.4, color="k")
    ax.plot(
        event_data.omni.time, event_data.omni.BzGSM, 
        color="r", ls="-", lw=0.9, label=r"$B_z$"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.ByGSM, 
        color="g", ls="-", lw=0.9, label=r"$B_y$"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.BxGSE, 
        color="k", ls="-", lw=0.9, label=r"$B_x$"
    )
    ax.text(0.1, 0.9, "(b)", ha="left", va="center", transform=ax.transAxes)
    ax.legend(loc=1)
    ax.set_ylabel("IMF, nT")
    ax.set_xticks([])
    ax.set_ylim(-10, 10)
    ax.set_xlim(dates)

    ax = fig.add_subplot(613)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.set_xlim([dt.datetime(2017,8,30), dt.datetime(2017,8,31)])
    ax.plot(
        event_data.omni.time, event_data.omni.ProtonDensity/10, 
        color="k", ls="-", lw=0.9, label=r"$n\times 10$, /cc"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.FlowPressure, 
        color="b", ls="-", lw=0.9, label=r"$P_{dyn}$"
    )
    ax.plot(
        event_data.omni.time, event_data.omni.FlowSpeed/100, 
        color="m", ls="-", lw=0.9, label=r"$V\times 10^{2}$, km/s"
    )
    ax.text(0.1, 0.9, "(c)", ha="left", va="center", transform=ax.transAxes)
    # ax.axvline(dt.datetime(2017, 9, 6, 11, 56), ls="--", lw=0.4, color="r")
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), ls="--", lw=0.4, color="k")
    ax.legend(loc=1)
    ax.set_ylabel("SW Params")
    ax.set_xticks([])
    ax.set_ylim(0, 8)
    ax.set_xlim(dates)

    ax = fig.add_subplot(614)
    ax.plot(event_data.omni.time, event_data.omni.SymH, "k", lw=0.9)
    ax.set_ylabel("SYM-H (nT)")
    ax.text(0.1, 0.9, "(d)", ha="left", va="center", transform=ax.transAxes)
    # ax.axvline(dt.datetime(2017, 9, 6, 11, 56), ls="--", lw=0.4, color="r")
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), ls="--", lw=0.4, color="k")
    ax.set_ylim(-50, 50)
    ax = ax.twinx()
    ax.plot(event_data.omni.time, event_data.omni.AsyH, "b", lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("ASY-H (nT)", fontdict={"color":"b"})
    ax.set_xticks([])
    ax.set_xlim(dates)

    ax = fig.add_subplot(615)
    ax.plot(event_data.omni.time, event_data.omni.AU, "r-", lw=0.8, label="AU")
    ax.plot(event_data.omni.time, event_data.omni.AL, "b-", lw=0.8, label="AL")
    ax.plot(event_data.omni.time, event_data.omni.AE, "k-", lw=0.8, label="AE")
    ax.legend(loc=1)
    ax.text(0.1, 0.9, "(e)", ha="left", va="center", transform=ax.transAxes)
    # ax.axvline(dt.datetime(2017, 9, 6, 11, 56), ls="--", lw=0.4, color="r")
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), ls="--", lw=0.4, color="k")
    ax.set_ylim(-500, 1000)
    ax.set_ylabel("AE/AL/AU (nT)")
    ax.set_xticks([])
    ax.set_xlim(dates)


    ax = fig.add_subplot(616)
    rti = RangeTimePlot(3000, dates, "", 1, fov=radar.fov)
    o = radar.df.copy()
    o["gate"] = np.copy(o.slist)
    o.slist = (o.slist*o.rsep) + o.frang
    o["unique_tfreq"] = o.tfreq.apply(lambda x: int(x/0.5)*0.5)
    o = o[o.unique_tfreq.isin([10.5])]
    rti.addParamPlot(
        o, 7, r"SAS / 7 / $f_0\sim$10.5 MHz", p_max=300, p_min=-300, xlabel="Time (UT)", 
        add_gflg=True, ylabel="Slant Range (km)", zparam="v", label="Velocity (m/s)",
        overlay_sza=True, ax=ax
    )
    ax.text(0.1, 0.9, "(f)", ha="left", va="center", transform=ax.transAxes)
    # ax.axvline(dt.datetime(2017, 9, 6, 11, 56), ls="--", lw=0.4, color="r")
    ax.axvline(dt.datetime(2017, 9, 6, 12, 2), ls="--", lw=0.4, color="k")
    fig.savefig("figures/RTI.png", bbox_inches="tight")
    print(event_data.omni.columns)
    return

def create_elev_angle_analysis(rad="sas", tdiff=None, beam=7):
    import plots
    import matplotlib.pyplot as plt
    
    
    plots.setsize(12)
    dates = [dt.datetime(2017,9,6,11), dt.datetime(2017,9,6,17)]
    radar = Radar(rad, dates)
    radar.recalculate_elv_angle(tdiff=tdiff)    
    radar.df["unique_tfreq"] = radar.df.tfreq.apply(lambda x: int(x/0.5)*0.5)
    radar.df = radar.df[radar.df.unique_tfreq.isin([10.5])]   
    rti = RangeTimePlot(3000, dates, "", 3)
    ax = rti.addParamPlot(
        radar.df, beam, "", p_max=45, p_min=0, xlabel="", add_gflg=False,
        ylabel="Range gate", zparam="elv", label="Elevation (deg)",
        overlay_sza=False
    )
    ax.text(0.01, 1.05, f"{rad.upper()} / {beam}", ha="left", va="center", transform=ax.transAxes)
    rti.addParamPlot(
        radar.df, beam, "", p_max=1000, p_min=0, xlabel="", add_gflg=False,
        ylabel="Range gate", zparam="vheight_2p", label=r"$V_h$ [Chisham Model] (km)",
        overlay_sza=False, cmap = plt.cm.Spectral_r, 
    )
    rti.addParamPlot(
        radar.df, beam, "", p_max=100, p_min=0, xlabel="Time (UT)", add_gflg=False,
        ylabel="Range gate", zparam="tau_l", label=r"$\tau_l$",
        overlay_sza=False, cmap = plt.cm.jet, 
    )
    rti.save(f"figures/height_analysis.{rad}.png")
    rti.close()
    return

if __name__ == "__main__":
    # for rad, tdiff, b in zip(["sas", "pgr", "kod"], [10, 5, 18], [7, 7, 10]):
    #     create_elev_angle_analysis(rad, tdiff*1e-3, b)
        
    # create_GPS_error_list()
    # create_RTI_figure()
    # compare_quiet_versus_event_day()
    create_RTI_plot()