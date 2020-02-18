import os
import logging
logging.getLogger().setLevel(logging.INFO)

import xarray as xr
from dask.diagnostics import ProgressBar


def open_sose_2d_datasets(dir):
    mld   = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_MLD.nc"),     chunks={'XC': 100, 'YC': 100})
    tau_x = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_oceTAUX.nc"), chunks={'XG': 100, 'YC': 100})
    tau_y = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_oceTAUY.nc"), chunks={'XC': 100, 'YG': 100})

    surf_S_flux = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_surfSflx.nc"),   chunks={'XC': 100, 'YC': 100})
    surf_T_flux = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_surfTflx.nc"),   chunks={'XC': 100, 'YC': 100})
    surf_FW_flux = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_daily_oceFWflx.nc"), chunks={'XC': 100, 'YC': 100})
    Qnet = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_daily_oceQnet.nc"),          chunks={'XC': 100, 'YC': 100})
    Qsw = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_daily_oceQsw.nc"),            chunks={'XC': 100, 'YC': 100})

    return xr.merge([mld, tau_x, tau_y, surf_S_flux, surf_T_flux, surf_FW_flux, Qnet, Qsw])

def get_time_series(sose_ds, var, lat, lon):
    logging.info(f"Getting time series of {var} at (lat, lon) = ({lat}, {lon})...")
    with ProgressBar():
        time_series = sose_ds[var].sel(XC=lon, YC=lat, method="nearest").values
    return time_series

