import xarray as xr
import matplotlib.pyplot as plt

from dask.diagnostics import ProgressBar

# Pacific patch
Px = slice(248, 283)
Py = slice(-64, -52)

# Atlantic patch
Ax = slice(0, 12)
Ay = slice(-67, -55)

ADVx_TH = xr.open_dataset("bsose_i122_2013to2017_monthly_ADVx_TH.nc", chunks={'XG': 100, 'YC': 100})
ADVy_TH = xr.open_dataset("bsose_i122_2013to2017_monthly_ADVy_TH.nc", chunks={'XC': 100, 'YG': 100})
ADVr_TH = xr.open_dataset("bsose_i122_2013to2017_monthly_ADVr_TH.nc", chunks={'XC': 100, 'YC': 100})
ADVx_SLT = xr.open_dataset("bsose_i122_2013to2017_monthly_ADVx_SLT.nc", chunks={'XG': 100, 'YC': 100})
ADVy_SLT = xr.open_dataset("bsose_i122_2013to2017_monthly_ADVy_SLT.nc", chunks={'XC': 100, 'YG': 100})
ADVr_SLT = xr.open_dataset("bsose_i122_2013to2017_monthly_ADVr_SLT.nc", chunks={'XC': 100, 'YC': 100})

DFxE_TH = xr.open_dataset("bsose_i122_2013to2017_monthly_DFxE_TH.nc", chunks={'XG': 100, 'YC': 100})
DFyE_TH = xr.open_dataset("bsose_i122_2013to2017_monthly_DFyE_TH.nc", chunks={'XC': 100, 'YG': 100})
DFrI_TH = xr.open_dataset("bsose_i122_2013to2017_monthly_DFrI_TH.nc", chunks={'XC': 100, 'YC': 100})
DFxE_SLT = xr.open_dataset("bsose_i122_2013to2017_monthly_DFxE_SLT.nc", chunks={'XG': 100, 'YC': 100})
DFyE_SLT = xr.open_dataset("bsose_i122_2013to2017_monthly_DFyE_SLT.nc", chunks={'XC': 100, 'YG': 100})
DFrI_SLT = xr.open_dataset("bsose_i122_2013to2017_monthly_DFrI_SLT.nc", chunks={'XC': 100, 'YC': 100})

print("Plotting advective heat fluxes...")
with ProgressBar():
    ADVx_TH.ADVx_TH.sel(XG=Px, YC=Py).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="ADVx (Pacific patch)")
    ADVx_TH.ADVx_TH.sel(XG=Ax, YC=Ay).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="ADVx (Atlantic patch)")
    ADVy_TH.ADVy_TH.sel(XC=Px, YG=Py).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="ADVy (Pacific patch)")
    ADVy_TH.ADVy_TH.sel(XC=Ax, YG=Ay).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="ADVy (Atlantic patch)")
    ADVr_TH.ADVr_TH.sel(XC=Px, YC=Py).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="ADVz (Pacific patch)")
    ADVr_TH.ADVr_TH.sel(XC=Ax, YC=Ay).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="ADVz (Atlantic patch)")

plt.ylabel(r"Advective heat fluxes (K m$^3$/s)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Advective_heat_fluxes_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")


print("Plotting diffusive heat fluxes...")
with ProgressBar():
    DFxE_TH.DFxE_TH.sel(XG=Px, YC=Py).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="DFx (Pacific patch)")
    DFxE_TH.DFxE_TH.sel(XG=Ax, YC=Ay).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="DFx (Atlantic patch)")
    DFyE_TH.DFyE_TH.sel(XC=Px, YG=Py).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="DFy (Pacific patch)")
    DFyE_TH.DFyE_TH.sel(XC=Ax, YG=Ay).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="DFy (Atlantic patch)")
    DFrI_TH.DFrI_TH.sel(XC=Px, YC=Py).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="DFz (Pacific patch)")
    DFrI_TH.DFrI_TH.sel(XC=Ax, YC=Ay).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="DFz (Atlantic patch)")

plt.ylabel(r"Diffusive heat fluxes (K m$^3$/s)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Diffusive_heat_fluxes_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")


print("Plotting advective salt fluxes...")
with ProgressBar():
    ADVx_SLT.ADVx_SLT.sel(XG=Px, YC=Py).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="ADVx (Pacific patch)")
    ADVx_SLT.ADVx_SLT.sel(XG=Ax, YC=Ay).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="ADVx (Atlantic patch)")
    ADVy_SLT.ADVy_SLT.sel(XC=Px, YG=Py).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="ADVy (Pacific patch)")
    ADVy_SLT.ADVy_SLT.sel(XC=Ax, YG=Ay).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="ADVy (Atlantic patch)")
    ADVr_SLT.ADVr_SLT.sel(XC=Px, YC=Py).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="ADVz (Pacific patch)")
    ADVr_SLT.ADVr_SLT.sel(XC=Ax, YC=Ay).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="ADVz (Atlantic patch)")

plt.ylabel(r"Advective salt fluxes (psu m$^3$/s)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Advective_salt_fluxes_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")


print("Plotting diffusive salt fluxes...")
with ProgressBar():
    DFxE_SLT.DFxE_SLT.sel(XG=Px, YC=Py).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="DFx (Pacific patch)")
    DFxE_SLT.DFxE_SLT.sel(XG=Ax, YC=Ay).sum(dim='Z').mean(dim=('XG', 'YC')).plot(label="DFx (Atlantic patch)")
    DFyE_SLT.DFyE_SLT.sel(XC=Px, YG=Py).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="DFy (Pacific patch)")
    DFyE_SLT.DFyE_SLT.sel(XC=Ax, YG=Ay).sum(dim='Z').mean(dim=('XC', 'YG')).plot(label="DFy (Atlantic patch)")
    DFrI_SLT.DFrI_SLT.sel(XC=Px, YC=Py).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="DFz (Pacific patch)")
    DFrI_SLT.DFrI_SLT.sel(XC=Ax, YC=Ay).sum(dim='Zl').mean(dim=('XC', 'YC')).plot(label="DFz (Atlantic patch)")

plt.ylabel(r"Diffusive salt fluxes (psu m$^3$/s)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("Diffusive_salt_fluxes_time_series_PA_patches.png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

