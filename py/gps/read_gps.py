import datetime as dt
from loguru import logger
import xarray as xr
import numpy as np

def gaussian_kernel(size, sigma):
    """
    Generates a 2D Gaussian kernel as a NumPy array with integer values.

    Args:
      size: The size of the square kernel (e.g., 11 for an 11x11 kernel).
      sigma: The standard deviation of the Gaussian.

    Returns:
      A NumPy array with integer values representing the Gaussian kernel.
    """

    center = size // 2
    kernel = np.zeros((size, size), dtype=float)  # Start with float

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            exponent = - (x**2 + y**2) / (2 * sigma**2)
            kernel[i, j] = np.exp(exponent)
    # Normalize and convert to int
    kernel_sum = np.sum(kernel)
    normalized_kernel = (kernel / kernel_sum)
    integer_kernel = np.round(normalized_kernel * (10000) ).astype(int) # Scale and convert to integers.
    return integer_kernel

def weighted_2d_filter(
        X, 
        W=np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]), 
        gridsize=3,
        tau=0.
    ):
    grid_center_left = (int(gridsize/2), int(gridsize/2))
    grid_center_right = (int(gridsize/2)+1, int(gridsize/2)+1)
    Y = np.zeros_like(X)*np.nan
    for i in range(grid_center_left[0], X.shape[0]-grid_center_right[0]):
        for j in range(grid_center_left[1], X.shape[1]-grid_center_right[1]):
            v, wi = [], 0
            for x, w in zip(X[i-1:i+2, j-1:j+2].ravel(), W.ravel()):
                if not np.isnan(x):
                    v.extend([x]*w)
                    wi+=w
            if wi/np.sum(W) > tau:
                Y[i,j] = np.nanmedian(v)
    print(np.nanmin(Y), np.nanmax(Y))
    return np.ma.masked_invalid(Y)

class GPS:
    def __init__(self, date, file):
        self.date = date
        self.file = file
        self.read_file()
        return
    
    def read_file(self):
        if self.file.endswith(".nc"):
            self.read_ncfiles()
        elif self.file.endswith(".h5") or self.file.endswith(".hdf5"):
            self.read_h5files()
        else:
            logger.info(f"Not implemented extension: {self.file}")
        return
    
    def read_ncfiles(self):
        logger.info(f"Get datasets from: {self.file}")
        self.ds = xr.open_dataset(self.file, engine="h5netcdf")
        return

    def parse_time_loc(self):
        if not hasattr(self, "ds"): self.read_ncfiles()
        self.datetime = [
            dt.datetime.fromtimestamp(s)
            for s in self.ds.timestamps.values
        ]
        self.gdlat = self.ds.gdlat.values
        self.glon = self.ds.glon.values
        return

    def get_index_by_time(self, x):
        if not hasattr(self, "datetime"): self.parse_time_loc()
        return self.datetime.index(x)
    
    def get_index_by_glat(self, x):
        if not hasattr(self, "gdlat"): self.parse_time_loc()
        return np.argmin(np.abs(self.gdlat - x))

    def get_index_by_glon(self, x):
        if not hasattr(self, "glon"): self.parse_time_loc()
        return np.argmin(np.abs(self.glon - x))

    def read_h5files(self):
        return

class VTEC(GPS):

    def __init__(self, date, nc_file):
        super().__init__(date, nc_file)
        return

    def compute_gradient(
        self, key="tec",
        t = dt.datetime(2017, 9, 6, 11, 45),
        range=[-150, -50, 0, 90], 
        gridsize=3, sigma=3,
        tau=0
    ):
        t_index = self.get_index_by_time(t)
        i_left, l_right = (
            self.get_index_by_glat(range[2]),
            self.get_index_by_glat(range[3])
        )
        j_left, j_right = (
            self.get_index_by_glon(range[0]),
            self.get_index_by_glon(range[1])
        )
        Z = self.ds.variables[key].values[t_index, i_left:l_right, j_left:j_right]
        logger.info("Running median filter ....")
        weights = gaussian_kernel(gridsize, sigma)
        Z = weighted_2d_filter(Z, weights, gridsize, tau)
        # Z = gaussian_filter(Z, 4)
        print(Z.shape)
        gdlat, glon = np.meshgrid(
            self.gdlat[i_left:l_right], 
            self.glon[j_left:j_right]
        )
        raw = dict(
            Z=Z,
            gdlat=gdlat,
            glon=glon,
        )

        # Compute Gradient
        lon_cell = np.arange(range[0], range[1],)
        lat_cell = np.arange(range[2], range[3],)[:-1]
        dxZ, dyZ = np.gradient(Z.T, self.glon[j_left:j_right], self.gdlat[i_left:l_right], )
        dxZ = weighted_2d_filter(dxZ)
        dxZ[dxZ>=1]= 1
        print(dxZ.min(), dxZ.max())
        # gdlat, glon = np.meshgrid(lat_cell, lon_cell)
        space_grad = dict(
            dxZ=dxZ[::2,::2], dyZ=dyZ[::2,::2],
            gdlat=gdlat[::2,::2], glon=glon[::2,::2]
        )
        return dict(raw=raw, space_grad=space_grad,)

    
    

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    plt.style.use(["science", "ieee"])
    sys.path.append("py/gps/*")
    t = dt.datetime(2017, 9, 6, 12, 10)
    tec = VTEC(t, "database/gps170906g.003.nc")
    data = tec.compute_gradient(gridsize=5,tau=0.2)
    from map import CartoDataOverlay
    cb = CartoDataOverlay(t)
    ax = cb.add_tec_dataset(
        data["raw"]["Z"].T, 
        data["raw"]["gdlat"], 
        data["raw"]["glon"]
    )
    cb.lay_tec_grad(
        data["space_grad"]["gdlat"], 
        data["space_grad"]["glon"],
        data["space_grad"]["dxZ"], 
        data["space_grad"]["dyZ"], 
        ax=ax, tag=True, scale=3, lenx=0.5
    )
    cb.save("figures/Figure2.png")
    cb.close()