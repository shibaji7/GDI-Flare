

def gerenate_fov_plot(
        rads, radars, date, cfg,
    ):
    import cartopy
    from fan import Fan
    import numpy as np
    import datetime as dt

    

    lons = np.arange(-180, 180, 30)
    lats = np.arange(-180, 180, 30)
    proj = cartopy.crs.Orthographic(-120, 55)
    extent = [ -140, -60, 30, 90 ]
    fan = Fan(
        rads[0],
        date,
        fig_title=f"",
    )
    fan.setup(
        lons,
        lats,
        extent=extent,
        proj=proj,
        lay_eclipse=None,
    )
    ax = fan.add_axes()
    
    for i, rad in enumerate(rads):
        f = radars[i].df
        f = f[
            (f.time>=date)
            & (f.time<date+dt.timedelta(minutes=2))
            & (f.slist>7)
        ]
        fan.generate_fov(rad, f, ax=ax)
    fan.save(f"tmp/{date.strftime('%Y%m%d.%H%M')}.png")
    return

if __name__ == "__main__":
    import json
    from types import SimpleNamespace
    import datetime as dt

    fname = "rt2d.json"
    with open(fname, "r") as f:
        cfg = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
    date_range = [dt.datetime(2024, 5, 9), dt.datetime(2024, 5, 12)]
    min_mul = 30
    dates = [
        date_range[0]+dt.timedelta(minutes=i*min_mul)
        for i in range(int((date_range[1]-date_range[0]).total_seconds()/(min_mul*60)))
    ]
    rads = ["bks", "fhe", "fhw", "cvw", "cve", "kod", "pgr", "sas", "gbr"]
    from radar import Radar

    radars = [
        Radar(
            rad, dates, cfg,
        )
        for rad in rads
    ]
    for date in dates:
        gerenate_fov_plot(rads, radars, date, cfg)
        #break