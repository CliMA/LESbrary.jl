import xarray as xr
import matplotlib.pyplot as plt

from dask.diagnostics import ProgressBar

# Pacific patch
Px = slice(248, 283)
Py = slice(-64, -52)

# Atlantic patch
Ax = slice(0, 12)
Ay = slice(-67, -55)

mld = xr.open_dataset("bsose_i122_2013to2017_1day_MLD.nc", chunks={'XC': 100, 'YC': 100})
tau_x = xr.open_dataset("bsose_i122_2013to2017_1day_oceTAUX.nc", chunks={'XG': 100, 'YC': 100})
tau_y = xr.open_dataset("bsose_i122_2013to2017_1day_oceTAUY.nc", chunks={'XC': 100, 'YG': 100})

surf_S_flux = xr.open_dataset("bsose_i122_2013to2017_1day_surfSflx.nc", chunks={'XC': 100, 'YC': 100})
surf_T_flux = xr.open_dataset("bsose_i122_2013to2017_1day_surfTflx.nc", chunks={'XC': 100, 'YC': 100})
surf_FW_flux = xr.open_dataset("bsose_i122_2013to2017_daily_oceFWflx.nc", chunks={'XC': 100, 'YC': 100})
Qnet = xr.open_dataset("bsose_i122_2013to2017_daily_oceQnet.nc", chunks={'XC': 100, 'YC': 100})
Qsw = xr.open_dataset("bsose_i122_2013to2017_daily_oceQsw.nc", chunks={'XC': 100, 'YC': 100})


print("Plotting MLD...")
with ProgressBar():
    mld.BLGMLD.sel(XC=Px, YC=Py).mean(dim=('XC', 'YC')).plot(label="Pacific patch")
    mld.BLGMLD.sel(XC=Ax, YC=Ay).mean(dim=('XC', 'YC')).plot(label="Atlantic patch")

plt.ylabel("Mixed layer depth (m)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("MLD_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

ts = slice("2013-01-01", "2013-12-31")
print("Plotting wind stress...")
with ProgressBar():
    tau_x.oceTAUX.sel(XG=Px, YC=Py, time=ts).mean(dim=('XG', 'YC')).plot(label=r"$\tau_x$ (Pacific patch)", linewidth=1)
    tau_x.oceTAUX.sel(XG=Ax, YC=Ay, time=ts).mean(dim=('XG', 'YC')).plot(label=r"$\tau_x$ (Atlantic patch)", linewidth=1)
    tau_y.oceTAUY.sel(XC=Px, YG=Py, time=ts).mean(dim=('XC', 'YG')).plot(label=r"$\tau_y$ (Pacific patch)", linewidth=1)
    tau_y.oceTAUY.sel(XC=Ax, YG=Ay, time=ts).mean(dim=('XC', 'YG')).plot(label=r"$\tau_y$ (Atlantic patch)", linewidth=1)

plt.ylabel(r"Wind stress (N/m$^2$)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Wind_stress_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

print("Plotting surface salinity fluxes...")
with ProgressBar():
    surf_S_flux.SFLUX.sel(XC=Px, YC=Py).mean(dim=('XC', 'YC')).plot(label="Pacific patch")
    surf_S_flux.SFLUX.sel(XC=Ax, YC=Ay).mean(dim=('XC', 'YC')).plot(label="Atlantic patch")

plt.ylabel(r"Surface salinity flux (g/m$^2$/s), >0 increases salt")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Surface_salinity_flux_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

print("Plotting surface heat fluxes...")
with ProgressBar():
    surf_T_flux.TFLUX.sel(XC=Px, YC=Py).mean(dim=('XC', 'YC')).plot(label="Pacific patch")
    surf_T_flux.TFLUX.sel(XC=Ax, YC=Ay).mean(dim=('XC', 'YC')).plot(label="Atlantic patch")

plt.ylabel(r"Surface heat flux (W/m$^2$), >0 increases T")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Surface_heat_flux_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

print("Plotting surface freshwater fluxes...")
with ProgressBar():
    surf_FW_flux.oceFWflx.sel(XC=Px, YC=Py).mean(dim=('XC', 'YC')).plot(label="Pacific patch")
    surf_FW_flux.oceFWflx.sel(XC=Ax, YC=Ay).mean(dim=('XC', 'YC')).plot(label="Atlantic patch")

plt.ylabel(r"Net surface freshwater flux (kg/m$^2$/s) (+=down), >0 decreases salinity")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Surface_freshwater_flux_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

print("Plotting Qnet...")
with ProgressBar():
    Qnet.oceQnet.sel(XC=Px, YC=Py).mean(dim=('XC', 'YC')).plot(label="Pacific patch", linewidth=1)
    Qnet.oceQnet.sel(XC=Ax, YC=Ay).mean(dim=('XC', 'YC')).plot(label="Atlantic patch", linewidth=1)

plt.ylabel(r"$Q_{net}$ (W/m$^2$), (+=down), >0 increases theta")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Surface_Qnet_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

print("Plotting Qsw...")
with ProgressBar():
    Qsw.oceQsw.sel(XC=Px, YC=Py).mean(dim=('XC', 'YC')).plot(label="Pacific patch)", linewidth=1)
    Qsw.oceQsw.sel(XC=Ax, YC=Ay).mean(dim=('XC', 'YC')).plot(label="Atlantic patch", linewidth=1)

plt.ylabel(r"$Q_{sw}$ (W/m$^2$), (+=down), >0 increases theta")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Surface_Qsw_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

