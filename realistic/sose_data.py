import os
import logging
logging.getLogger().setLevel(logging.INFO)

import xgcm
import dask
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
    logging.info("Opening SOSE 3D datasets (this can take some time)...")

    u = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Uvel.nc"),  chunks={'XG': 10, 'YC': 10, 'time': 10}, decode_cf=False)
    v = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Vvel.nc"),  chunks={'XC': 10, 'YG': 10, 'time': 10}, decode_cf=False)
    T = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Theta.nc"), chunks={'XC': 10, 'YC': 10, 'time': 10}, decode_cf=False)
    S = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Salt.nc"),  chunks={'XC': 10, 'YC': 10, 'time': 10}, decode_cf=False)
    N = xr.open_dataset(os.path.join(dir, "bsose_i122_2013to2017_1day_Strat.nc"), chunks={'XC': 10, 'YC': 10, 'time': 10}, decode_cf=False)

    return xr.merge([u, v, T, S, N])

def open_sose_advective_flux_datasets(dir):
    logging.info("Opening SOSE advective flux datasets (this can take some time)...")

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

def compute_geostrophic_velocities(ds, lat, lon, day_offset, days, zF, α, β, g, f):
    logging.info(f"Computing geostrophic velocities at ({lat}°N, {lon}°E) for {days} days...")

    # Reverse z index so we calculate cumulative integrals bottom up
    ds = ds.reindex(Z=ds.Z[::-1], Zl=ds.Zl[::-1])

    # Only pull out the data we need as time has chunk size 1.
    time_slice = slice(day_offset, day_offset + days + 1)

    U =  ds.UVEL.isel(time=time_slice)
    V =  ds.VVEL.isel(time=time_slice)
    Θ = ds.THETA.isel(time=time_slice)
    S =  ds.SALT.isel(time=time_slice)

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

    # Vertical integrals from z'=-Lz to z'=z (cumulative integrals)
    with dask.config.set({"array.slicing.split_large_chunks": False}):
        Σdz_dΘdx = grid.cumint(grid.derivative(Θ, 'X'), 'Z', boundary="extend")
        Σdz_dΘdy = grid.cumint(grid.derivative(Θ, 'Y'), 'Z', boundary="extend")
        Σdz_dSdx = grid.cumint(grid.derivative(S, 'X'), 'Z', boundary="extend")
        Σdz_dSdy = grid.cumint(grid.derivative(S, 'Y'), 'Z', boundary="extend")

    # Assuming linear equation of state
    Σdz_dBdx = g * (α * Σdz_dΘdx - β * Σdz_dSdx)
    Σdz_dBdy = g * (α * Σdz_dΘdy - β * Σdz_dSdy)

    # Velocities at depth
    z_bottom = ds.Z.values[0]
    U_d = U.sel(XG=lon, YC=lat, Z=z_bottom, method="nearest")
    V_d = V.sel(XC=lon, YG=lat, Z=z_bottom, method="nearest")

    with ProgressBar():
        U_geo = (U_d - 1/f * Σdz_dBdy).sel(XC=lon, YG=lat, method="nearest").values
        V_geo = (V_d + 1/f * Σdz_dBdx).sel(XG=lon, YC=lat, method="nearest").values

    return U_geo, V_geo

def plot_site_analysis(ds, lat, lon, day_offset, days):
    logging.info(f"Plotting site analysis at ({lat}°N, {lon}°E) for {days} days...")

    time = ds.time.values
    time_slice = slice(day_offset, day_offset + days + 1)
    simulation_time = ds.time.isel(time=time_slice).values

    τx  = ds.oceTAUX.sel(XG=lon, YC=lat, method="nearest").values
    τy  = ds.oceTAUY.sel(XC=lon, YG=lat, method="nearest").values
    Qθ  = ds.oceQnet.sel(XC=lon, YC=lat, method="nearest").values
    Qs  = ds.oceFWflx.sel(XC=lon, YC=lat, method="nearest").values
    mld = ds.BLGMLD.sel(XC=lon, YC=lat, method="nearest").values

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 12), dpi=200)

    fig.suptitle(f"LESbrary.jl SOSE site analysis at ({lat}°N, {lon}°E)")

    ax_τ = axes[0]
    ax_τ.plot(time, τx, label=r"$\tau_x$")
    ax_τ.plot(time, τy, label=r"$\tau_y$")
    ax_τ.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_τ.legend(frameon=False)
    ax_τ.set_ylabel(r"Wind stress (N/m$^2$)")
    ax_τ.set_xlim([time[0], time[-1]])
    ax_τ.set_xticklabels([])

    ax_Qθ = axes[1]
    ax_Qθ.plot(time, Qθ)
    ax_Qθ.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_Qθ.set_ylabel(r"Surface heat flux (W/m$^2$)," + "\n>0 increases T")
    ax_Qθ.set_xlim([time[0], time[-1]])
    ax_Qθ.set_xticklabels([])

    ax_Qs = axes[2]
    ax_Qs.plot(time, Qs)
    ax_Qs.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_Qs.set_ylabel("Net surface\n" + r"freshwater flux (kg/m$^2$/s)" + "\n(+=down)," + "\n>0 decreases salinity")
    ax_Qs.set_xlim([time[0], time[-1]])
    ax_Qs.set_xticklabels([])

    ax_mld = axes[3]
    ax_mld.plot(time, mld)
    ax_mld.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_mld.set_xlabel("Time")
    ax_mld.set_ylabel("Mixed layer depth (m)")
    ax_mld.set_xlim([time[0], time[-1]])
    ax_mld.invert_yaxis()

    start_date_str = numpy_datetime_to_date_str(simulation_time[0])
    end_date_str = numpy_datetime_to_date_str(simulation_time[-1])

    plt.savefig(f"lesbrary_site_analysis_latitude{lat}_longitude{lon}_{start_date_str}_to{end_date_str}.png", dpi=300)
    plt.close(fig)

def plot_lateral_flux_site_analysis(ds, lat, lon, day_offset, days):
    logging.info(f"Plotting lateral flux site analysis at ({lat}°N, {lon}°E) for {days} days...")

    time = ds.time.values
    time_slice = slice(day_offset, day_offset + days + 1)
    simulation_time = ds.time.isel(time=time_slice).values

    # Compute column-integrated fluxes and flux differences

    uT = ds.ADVx_TH
    vT = ds.ADVy_TH
    uS = ds.ADVx_SLT
    vS = ds.ADVy_SLT

    with ProgressBar():
        Σdz_uT = uT.integrate(coord="Z").sel(XG=lon, YC=lat, method="nearest").values
        Σdz_vT = vT.integrate(coord="Z").sel(XC=lon, YG=lat, method="nearest").values
        Σdz_uS = uS.integrate(coord="Z").sel(XG=lon, YC=lat, method="nearest").values
        Σdz_vS = vS.integrate(coord="Z").sel(XC=lon, YG=lat, method="nearest").values

        ΔΣdz_uT = uT.integrate(coord="Z").diff("XG").sel(XG=lon, YC=lat, method="nearest").values
        ΔΣdz_vT = vT.integrate(coord="Z").diff("YG").sel(XC=lon, YG=lat, method="nearest").values
        ΔΣdz_uS = uS.integrate(coord="Z").diff("XG").sel(XG=lon, YC=lat, method="nearest").values
        ΔΣdz_vS = vS.integrate(coord="Z").diff("YG").sel(XC=lon, YG=lat, method="nearest").values

    # Plot column-integrated fluxes time series

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    fig.suptitle(f"LESbrary.jl SOSE site analysis: lateral fluxes at ({lat}°N, {lon}°E)")

    ax_T = axes[0]
    ax_T.plot(time, Σdz_uT, label=r"$\int uT \; dz$")
    ax_T.plot(time, Σdz_vT, label=r"$\int vT \; dz$")
    ax_T.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_T.legend(frameon=False)
    ax_T.set_ylabel(r"$\degree C \cdot m^4/s$")
    ax_T.set_xlim([time[0], time[-1]])
    ax_T.set_xticklabels([])

    ax_S = axes[1]
    ax_S.plot(time, Σdz_uS, label=r"$\int uS \; dz$")
    ax_S.plot(time, Σdz_vS, label=r"$\int vS \; dz$")
    ax_S.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_S.legend(frameon=False)
    ax_S.set_ylabel(r"$\mathrm{psu} \cdot m^4/s$")
    ax_S.set_xlim([time[0], time[-1]])

    start_date_str = numpy_datetime_to_date_str(simulation_time[0])
    end_date_str = numpy_datetime_to_date_str(simulation_time[-1])
    filename = f"lesbrary_site_analysis_lateral_fluxes_latitude{lat}_longitude{lon}_{start_date_str}_to_{end_date_str}.png"
    logging.info(f"Saving {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    # Plot flux differences time series

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    fig.suptitle(f"LESbrary.jl SOSE site analysis: lateral flux differences at ({lat}°N, {lon}°E)")

    ax_T = axes[0]
    ax_T.plot(time, ΔΣdz_uT, label=r"$\Delta \int uT \; dz$")
    ax_T.plot(time, ΔΣdz_vT, label=r"$\Delta \int vT \; dz$")
    ax_T.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_T.legend(frameon=False)
    ax_T.set_ylabel(r"\degree C \cdot m^4/s")
    ax_T.set_xlim([time[0], time[-1]])
    ax_T.set_xticklabels([])

    ax_S = axes[1]
    ax_S.plot(time, ΔΣdz_uS, label=r"$\Delta \int uS \; dz$")
    ax_S.plot(time, ΔΣdz_vS, label=r"$\Delta \int vS \; dz$")
    ax_S.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_S.legend(frameon=False)
    ax_S.set_ylabel(r"\mathrm{psu} \cdot m^4/s")
    ax_S.set_xlim([time[0], time[-1]])

    start_date_str = numpy_datetime_to_date_str(simulation_time[0])
    end_date_str = numpy_datetime_to_date_str(simulation_time[-1])
    filename = f"lesbrary_site_analysis_lateral_flux_differences_latitude{lat}_longitude{lon}_{start_date_str}_to_{end_date_str}.png"
    logging.info(f"Saving {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

# https://stackoverflow.com/a/45494027
def numpy_datetime_to_date_str(dt):
    return str(dt)[:10]
