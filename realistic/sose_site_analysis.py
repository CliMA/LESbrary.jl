import os
import logging

import xgcm
import dask
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from datetime import datetime
from dask.diagnostics import ProgressBar

# https://stackoverflow.com/a/45494027
def numpy_datetime_to_date_str(dt):
    return str(dt)[:10]

def plot_surface_forcing_site_analysis(ds, lat, lon, date_offset, n_dates):
    logging.info(f"Plotting site analysis at ({lat}°N, {lon}°E) for {n_dates} dates...")

    time = ds.time.values
    time_slice = slice(date_offset, date_offset + n_dates + 1)
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

    filename = f"lesbrary_site_analysis_surface_forcing_latitude{lat}_longitude{lon}_{start_date_str}_to{end_date_str}.png"
    logging.info(f"Saving {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def plot_lateral_flux_site_analysis(ds_fluxes, ds_2d, lat, lon, depth, date_offset, n_dates, ρ0, cp):
    logging.info(f"Plotting lateral flux site analysis at ({lat}°N, {lon}°E) for {n_dates} dates...")

    time = ds_fluxes.time.values
    time_slice = slice(date_offset, date_offset + n_dates + 1)
    simulation_time = ds_fluxes.time.isel(time=time_slice).values

    # Compute column-integrated fluxes and flux differences

    uT = ds_fluxes.ADVx_TH.sel(Z=slice(0, -depth))
    vT = ds_fluxes.ADVy_TH.sel(Z=slice(0, -depth))
    uS = ds_fluxes.ADVx_SLT.sel(Z=slice(0, -depth))
    vS = ds_fluxes.ADVy_SLT.sel(Z=slice(0, -depth))

    with ProgressBar():
        # We sum instead of integrating since the fluxes are already multipled by an area.
        ΣuT = uT.sum("Z").sel(XG=lon, YC=lat, method="nearest").values
        ΣvT = vT.sum("Z").sel(XC=lon, YG=lat, method="nearest").values
        ΣuS = uS.sum("Z").sel(XG=lon, YC=lat, method="nearest").values
        ΣvS = vS.sum("Z").sel(XC=lon, YG=lat, method="nearest").values

        ΔΣuT = uT.sum("Z").diff("XG").sel(XG=lon, YC=lat, method="nearest").values
        ΔΣvT = vT.sum("Z").diff("YG").sel(XC=lon, YG=lat, method="nearest").values
        ΔΣuS = uS.sum("Z").diff("XG").sel(XG=lon, YC=lat, method="nearest").values
        ΔΣvS = vS.sum("Z").diff("YG").sel(XC=lon, YG=lat, method="nearest").values

    # Plot column-integrated fluxes time series

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    fig.suptitle(f"LESbrary.jl SOSE site analysis: lateral fluxes at ({lat}°N, {lon}°E) down until {depth} m")

    ax_T = axes[0]
    ax_T.plot(time, ΣuT, label=r"$\int uT \; dz$")
    ax_T.plot(time, ΣvT, label=r"$\int vT \; dz$")
    ax_T.plot(time, ΔΣuT + ΔΣvT, label=r"$\Delta \int uT \; dz + \Delta \int vT \; dz$")
    ax_T.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_T.legend(frameon=False)
    ax_T.set_ylabel(r"$\degree C \cdot m^3/s$")
    ax_T.set_xlim([time[0], time[-1]])
    ax_T.set_xticklabels([])

    ax_S = axes[1]
    ax_S.plot(time, ΣuS, label=r"$\int uS \; dz$")
    ax_S.plot(time, ΣvS, label=r"$\int vS \; dz$")
    ax_S.plot(time, ΔΣuS + ΔΣvS, label=r"$\Delta \int uS \; dz + \Delta \int vS \; dz$")
    ax_S.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_S.legend(frameon=False)
    ax_S.set_ylabel(r"$\mathrm{psu} \cdot m^3/s$")
    ax_S.set_xlim([time[0], time[-1]])

    start_date_str = numpy_datetime_to_date_str(simulation_time[0])
    end_date_str = numpy_datetime_to_date_str(simulation_time[-1])
    filename = f"lesbrary_site_analysis_lateral_fluxes_latitude{lat}_longitude{lon}_{start_date_str}_to_{end_date_str}.png"
    logging.info(f"Saving {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    # Plot lateral vs. surface fluxes

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    fig.suptitle(f"LESbrary.jl SOSE site analysis: lateral vs. surface fluxes at ({lat}°N, {lon}°E) down until {depth} m")

    time_2d = ds_2d.time.values

    surface_Qθ = ds_2d.oceQnet.sel(XC=lon, YC=lat, method="nearest").values
    surface_Qs = ds_2d.oceFWflx.sel(XC=lon, YC=lat, method="nearest").values

    dx = ds_fluxes.dxC.sel(XG=lon, YC=lat, method="nearest").values[()]
    dy = ds_fluxes.dyC.sel(XC=lon, YG=lat, method="nearest").values[()]

    # ΔΣuT + ΔΣvT is the net amount of heat advected into the column and we want to
    # convert it into an "equivalent surface heat flux" to compare with the surface heat flux.
    lateral_Qθ = ρ0 * cp * (ΔΣuT + ΔΣvT) / (dx * dy)

    ax_T = axes[0]
    ax_T.plot(time_2d, surface_Qθ, label="surface heat flux")
    ax_T.plot(time, lateral_Qθ, label="lateral heat flux")
    ax_T.axvspan(simulation_time[0], simulation_time[-1], color='gold', alpha=0.5)
    ax_T.legend(frameon=False)
    ax_T.set_ylabel(r"Heat flux ($W/m^2$)")
    ax_T.set_xlim([time[0], time[-1]])
    ax_T.set_xticklabels([])

    filename = f"lesbrary_site_analysis_lateral_vs_surface_fluxes_latitude{lat}_longitude{lon}_{start_date_str}_to_{end_date_str}.png"
    logging.info(f"Saving {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
