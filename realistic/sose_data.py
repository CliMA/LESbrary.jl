import os
import logging

import xgcm
import dask
import numpy as np
import xarray as xr

from datetime import datetime
from dask.diagnostics import ProgressBar

logging.basicConfig(
     format = "%(asctime)s %(levelname)-8s %(message)s",
      level = logging.INFO,
    datefmt = "%Y-%m-%d %H:%M:%S"
)

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
    logging.info("Opening SOSE 3D datasets (this can take a few minutes)...")

    u = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Uvel.nc"),  chunks={'XG': 10, 'YC': 10, 'time': 10}, decode_cf=False)
    v = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Vvel.nc"),  chunks={'XC': 10, 'YG': 10, 'time': 10}, decode_cf=False)
    T = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Theta.nc"), chunks={'XC': 10, 'YC': 10, 'time': 10}, decode_cf=False)
    S = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Salt.nc"),  chunks={'XC': 10, 'YC': 10, 'time': 10}, decode_cf=False)
    N = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Strat.nc"), chunks={'XC': 10, 'YC': 10, 'time': 10}, decode_cf=False)

    return xr.merge([u, v, T, S, N])

def open_sose_advective_flux_datasets(dir):
    logging.info("Opening SOSE advective flux datasets (this can take a few minutes)...")

    uT = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_5day_ADVx_TH.nc"),  chunks={'XG': 10, 'YC': 10, 'time': 10})
    vT = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_5day_ADVy_TH.nc"),  chunks={'XC': 10, 'YG': 10, 'time': 10})
    uS = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_5day_ADVx_SLT.nc"), chunks={'XG': 10, 'YC': 10, 'time': 10})
    vS = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_5day_ADVy_SLT.nc"), chunks={'XC': 10, 'YG': 10, 'time': 10})

    return xr.merge([uT, vT, uS, vS])

def get_times(ds):
    ts = ds.time.values
    # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    ts = [datetime.utcfromtimestamp((dt - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")) for dt in ts]
    return ts

def get_scalar_time_series(ds, var, lat, lon, day_offset, days):
    logging.info(f"Getting time series of {var} at (lat={lat}°N, lon={lon}°E)...")
    time_slice = slice(day_offset, day_offset + days + 1)
    with ProgressBar():
        if var in ["UVEL", "oceTAUX"]:
            time_series = ds[var].isel(time=time_slice).sel(XG=lon, YC=lat, method="nearest").values
        elif var in ["VVEL", "oceTAUY"]:
            time_series = ds[var].isel(time=time_slice).sel(XC=lon, YG=lat, method="nearest").values
        else:
            time_series = ds[var].isel(time=time_slice).sel(XC=lon, YC=lat, method="nearest").values
    return time_series

def get_profile_time_series(ds, var, lat, lon, day_offset, days):
    logging.info(f"Getting time series of {var} at ({lat}°N, {lon}°E) for {days} days...")
    time_slice = slice(day_offset, day_offset + days + 1)
    with ProgressBar():
        if var in ["UVEL", "oceTAUX"]:
            time_series = ds[var].isel(time=time_slice).sel(XG=lon, YC=lat, method="nearest").values
        elif var in ["VVEL", "oceTAUY"]:
            time_series = ds[var].isel(time=time_slice).sel(XC=lon, YG=lat, method="nearest").values
        else:
            time_series = ds[var].isel(time=time_slice).sel(XC=lon, YC=lat, method="nearest").values
    return time_series

def compute_geostrophic_velocities(ds, lat, lon, day_offset, days, g, ρ0, f):
    logging.info(f"Computing geostrophic velocities at ({lat}°N, {lon}°E) for {days} days...")

    # Reverse z index so we calculate cumulative integrals bottom up
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds = ds.reindex(Z=ds.Z[::-1], Zl=ds.Zl[::-1])

    # Only pull out the data we need as time has chunk size 10.
    time_slice = slice(day_offset, day_offset + days + 1)

    U =  ds.UVEL.isel(time=time_slice)
    V =  ds.VVEL.isel(time=time_slice)
    dρdz = ds.DRHODR.isel(time=time_slice)

    # Set up grid metric
    # See: https://xgcm.readthedocs.io/en/latest/grid_metrics.html#Using-metrics-with-xgcm
    ds["drW"] = ds.hFacW * ds.drF  # vertical cell size at u point
    ds["drS"] = ds.hFacS * ds.drF  # vertical cell size at v point
    ds["drC"] = ds.hFacC * ds.drF  # vertical cell size at tracer point

    metrics = {
        ('X',):     ['dxC', 'dxG'], # X distances
        ('Y',):     ['dyC', 'dyG'], # Y distances
        ('Z',):     ['drW', 'drS', 'drC'], # Z distances
        ('X', 'Y'): ['rA', 'rA', 'rAs', 'rAw'] # Areas
    }

    # xgcm grid for calculating derivatives and interpolating
    # Not sure why it's periodic in Y but copied it from the xgcm SOSE example:
    # https://pangeo.io/use_cases/physical-oceanography/SOSE.html#create-xgcm-grid
    grid = xgcm.Grid(ds, metrics=metrics, periodic=('X', 'Y'))

    # Vertical integral from z'=-H to z'=z (cumulative integrals)
    # `extend` sets values outside the array to the nearest array value.
    # (i.e. a limited form of Neumann boundary condition.)
    dρdz = grid.interp(dρdz, 'Z', boundary="extend")
    B = -g/ρ0 * grid.cumint(dρdz, 'Z', boundary="extend")

    dBdx = grid.interp(grid.derivative(B, 'X'), 'Z', boundary="extend")
    dBdy = grid.interp(grid.derivative(B, 'Y'), 'Z', boundary="extend")

    Σdz_dBdx = grid.cumint(dBdx, 'Z', boundary="extend")
    Σdz_dBdy = grid.cumint(dBdy, 'Z', boundary="extend")

    # Velocities at depth
    z_bottom = ds.Z.values[0]
    U_d = U.sel(XG=lon, YC=lat, Z=z_bottom, method="nearest")
    V_d = V.sel(XC=lon, YG=lat, Z=z_bottom, method="nearest")

    print(z_bottom)
    print(U_d)
    print(V_d)

    with ProgressBar():
        U_geo = (U_d - 1/f * Σdz_dBdy).sel(XC=lon, YG=lat, method="nearest").values
        V_geo = (V_d + 1/f * Σdz_dBdx).sel(XG=lon, YC=lat, method="nearest").values

    return U_geo, V_geo
