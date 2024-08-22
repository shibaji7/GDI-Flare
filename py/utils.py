import numpy as np


def chisham(target_range, **kwargs):
    """
    Mapping ionospheric backscatter measured by the SuperDARN HF
    radars – Part 1: A new empirical virtual height model by
    G. Chisham 2008 (https://doi.org/10.5194/angeo-26-823-2008)
    Parameters
    ----------
    target_range: float
        is the range from radar to the target (echos)
        sometimes known as slant range [km]
    kwargs: is only needed to avoid key item errors
    Returns
    -------
    altered target_range (slant range) [km]
    """
    # Model constants
    A_const = (108.974, 384.416, 1098.28)
    B_const = (0.0191271, -0.178640, -0.354557)
    C_const = (6.68283e-5, 1.81405e-4, 9.39961e-5)

    # determine which region of ionosphere the gate
    if target_range < 115:
        return (target_range / 115.0) * 112.0
    elif target_range < 787.5:
        return A_const[0] + B_const[0] * target_range + C_const[0] *\
                 target_range**2
    elif target_range <= 2137.5:
        return A_const[1] + B_const[1] * target_range + C_const[1] *\
                 target_range**2
    else:
        return A_const[2] + B_const[2] * target_range + C_const[2] *\
                 target_range**2

def get_gridded_parameters(
    q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=True
):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[[xparam, yparam, zparam]]
    if rounding:
        if "Time" not in xparam:
            plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    plotParamDF = plotParamDF.groupby([xparam, yparam]).mean().reset_index()
    plotParamDF = plotParamDF[[xparam, yparam, zparam]].pivot(
        index=xparam, columns=yparam
    )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y = np.meshgrid(x, y)
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
        np.isnan(plotParamDF[zparam].values), plotParamDF[zparam].values
    )
    return X, Y, Z