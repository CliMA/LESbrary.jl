import os
import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import xarray as xr

from datetime import datetime
from dask.diagnostics import ProgressBar


def open_sose_2d_datasets(dir):
    logging.info("Opening SOSE 2D datasets...")
    mld          = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_MLD.nc"),       chunks={'XC': 100, 'YC': 100})
    tau_x        = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_oceTAUX.nc"),   chunks={'XG': 100, 'YC': 100})
    tau_y        = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_oceTAUY.nc"),   chunks={'XC': 100, 'YG': 100})
    surf_S_flux  = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_surfSflx.nc"),  chunks={'XC': 100, 'YC': 100})
    surf_T_flux  = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_surfTflx.nc"),  chunks={'XC': 100, 'YC': 100})
    surf_FW_flux = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_daily_oceFWflx.nc"), chunks={'XC': 100, 'YC': 100})
    Qnet         = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_daily_oceQnet.nc"),  chunks={'XC': 100, 'YC': 100})
    Qsw          = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_daily_oceQsw.nc"),   chunks={'XC': 100, 'YC': 100})

    return xr.merge([mld, tau_x, tau_y, surf_S_flux, surf_T_flux, surf_FW_flux, Qnet, Qsw])

def open_sose_3d_datasets(dir):
    logging.info("Opening SOSE 3D datasets...")
    u = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Uvel.nc"),  chunks={'XG': 10, 'YC': 10, 'time': 10})
    v = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Vvel.nc"),  chunks={'XC': 10, 'YG': 10, 'time': 10})
    w = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Wvel.nc"),  chunks={'XC': 10, 'YC': 10, 'time': 10})
    T = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Theta.nc"), chunks={'XC': 10, 'YC': 10, 'time': 10})
    S = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Salt.nc"),  chunks={'XC': 10, 'YC': 10, 'time': 10})

    return xr.merge([u, v, w, T, S])

def get_times(ds):
    ts = ds.time.values
    # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    ts = [datetime.utcfromtimestamp((dt - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")) for dt in ts]
    return ts

def get_time_series(ds, var, lat, lon):
    logging.info(f"Getting time series of {var} at (lat, lon) = ({lat}, {lon})...")
    with ProgressBar():
        if var in ["oceTAUX"]:
            time_series = ds[var].sel(XG=lon, YC=lat, method="nearest").values
        elif var in ["oceTAUY"]:
            time_series = ds[var].sel(XC=lon, YG=lat, method="nearest").values
        else:
            time_series = ds[var].sel(XC=lon, YC=lat, method="nearest").values
    return time_series

