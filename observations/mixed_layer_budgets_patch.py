import dask
import dask.array as dsa
import matplotlib
import numpy as np
import xarray as xr
import xgcm

from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.dpi'] = 200

# Physical constants
c_p = 3.994e3
runit2mass = 1.035e3  # units conversion factor (surface forcing), from vertical r-coordinate unit to mass per unit area [kg/m2]

ds = xr.open_mfdataset("/home/alir/cnhlab004/bsose_i122/monthly/pacific_patch/*.nc", parallel=True)

# A trick for optimization: split the dataset into coordinates and data variables,
# and then drop the coordinates from the data variables. This makes it easier to
# align the data variables in arithmetic operations.
coords = ds.coords.to_dataset().reset_coords()
dsr = ds.reset_coords(drop=True)

grid = xgcm.Grid(ds, periodic=('X', 'Y'))

def tracer_flux_budget(suffix):
    """
    Calculate the convergence of fluxes of tracer `suffix` where `suffix` is `TH` or `SLT`.
    Return a new xarray.Dataset.
    """

    conv_horiz_adv_flux  = -(grid.diff(dsr['ADVx_' + suffix], 'X') + grid.diff(dsr['ADVy_' + suffix], 'Y')).rename('conv_horiz_adv_flux_' + suffix)
    conv_horiz_diff_flux = -(grid.diff(dsr['DFxE_' + suffix], 'X') + grid.diff(dsr['DFyE_' + suffix], 'Y')).rename('conv_horiz_diff_flux_' + suffix)

    # sign convention is opposite for vertical fluxes
    conv_vert_adv_flux  = grid.diff(dsr['ADVr_' + suffix], 'Z', boundary='fill').rename('conv_vert_adv_flux_' + suffix)
    conv_vert_diff_flux = grid.diff(dsr['DFrI_' + suffix], 'Z', boundary='fill').rename('conv_vert_diff_flux_' + suffix)
    # conv_vert_diff_flux = (grid.diff(dsr['DFrE_' + suffix], 'Z', boundary='fill') +
    #                        grid.diff(dsr['DFrI_' + suffix], 'Z', boundary='fill') +
    #                        grid.diff(dsr['KPPg_' + suffix], 'Z', boundary='fill')).rename('conv_vert_diff_flux_' + suffix)

    all_fluxes = [conv_horiz_adv_flux, conv_horiz_diff_flux, conv_vert_adv_flux, conv_vert_diff_flux]
    conv_all_fluxes = sum(all_fluxes).rename('conv_total_flux_' + suffix)

    return xr.merge(all_fluxes + [conv_all_fluxes])

budget_th  = tracer_flux_budget("TH")
budget_slt = tracer_flux_budget("SLT")

# treat the shortwave flux separately from the rest of the surface flux
surf_flux_th = (dsr.TFLUX - dsr.oceQsw) * coords.rA / (c_p * runit2mass)
surf_flux_th_sw = dsr.oceQsw * coords.rA / (c_p * runit2mass)

surf_flux_slt = dsr.SFLUX * coords.rA  / runit2mass

lin_fs_correction_th = -(dsr.WTHMASS.isel(Zl=0, drop=True) * coords.rA)
lin_fs_correction_slt = -(dsr.WSLTMASS.isel(Zl=0, drop=True) * coords.rA)

# in order to align the surface fluxes with the rest of the 3D budget terms,
# we need to give them a z coordinate. We can do that with this function
def surface_to_3d(da):
    da.coords['Z'] = dsr.Z[0]
    return da.expand_dims(dim='Z', axis=1)

def swfrac(coords, fact=1, jwtype=2):
    """
    Clone of MITgcm routine for computing sw flux penetration.
    z: depth of output levels
    """

    rfac = [0.58 , 0.62, 0.67, 0.77, 0.78]
    a1 = [0.35 , 0.6  , 1.0  , 1.5  , 1.4]
    a2 = [23.0 , 20.0 , 17.0 , 14.0 , 7.9 ]

    facz = fact * coords.Zl.sel(Zl=slice(0, -200))
    j = jwtype - 1
    swdk = rfac[j] * np.exp(facz / a1[j]) + (1-rfac[j]) * np.exp(facz / a2[j])

    return swdk.rename("swdk")

_, swdown = xr.align(dsr.Zl, surf_flux_th_sw * swfrac(coords), join='left', )
swdown = swdown.fillna(0)

# now we can add the to the budget datasets and they will align correctly
# into the top cell (lazily filling with NaN's elsewhere)
budget_slt['surface_flux_conv_SLT'] = surface_to_3d(surf_flux_slt)
budget_slt['lin_fs_correction_SLT'] = surface_to_3d(lin_fs_correction_slt)

budget_th['surface_flux_conv_TH'] = surface_to_3d(surf_flux_th)
budget_th['lin_fs_correction_TH'] = surface_to_3d(lin_fs_correction_th)
budget_th['sw_flux_conv_TH'] = -grid.diff(swdown, 'Z', boundary='fill').fillna(0.)

budget_th['total_tendency_TH'] = (budget_th.conv_total_flux_TH +
                                  budget_th.surface_flux_conv_TH.fillna(0.) +
                                  budget_th.lin_fs_correction_TH.fillna(0.) +
                                  budget_th.sw_flux_conv_TH)

budget_slt['total_tendency_SLT'] = (budget_slt.conv_total_flux_SLT +
                                    budget_slt.surface_flux_conv_SLT.fillna(0.) +
                                    budget_slt.lin_fs_correction_SLT.fillna(0.))

volume = (coords.drF * coords.rA * coords.hFacC)
day2seconds = (24*60*60)**-1

budget_th['total_tendency_TH_truth'] = dsr.TOTTTEND * volume * day2seconds
budget_slt['total_tendency_SLT_truth'] = dsr.TOTSTEND * volume * day2seconds

time_slice = dict(time=slice(0, 10))
valid_range = dict(YC=slice(-90,-30))

def check_vertical(budget, suffix):
    ds_chk = (budget[[f'total_tendency_{suffix}', f'total_tendency_{suffix}_truth']]
              .sel(**valid_range).sum(dim=['Z', 'XC']).mean(dim='time'))
    return ds_chk

def check_horizontal(budget, suffix):
    ds_chk = (budget[[f'total_tendency_{suffix}', f'total_tendency_{suffix}_truth']]
              .sel(**valid_range).sum(dim=['YC', 'XC']).mean(dim='time'))
    return ds_chk

print("Validating vertical temperature budget...")
with ProgressBar():
    th_vert = check_vertical(budget_th, 'TH').load()

th_vert.total_tendency_TH.plot(linewidth=2, label=r"$d\theta/dt$")
th_vert.total_tendency_TH_truth.plot(linestyle='--', linewidth=2, label=r"$d\theta/dt$ (true)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("temperature_budget_vertical.png", dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

print("Validating horizontal temperature budget...")
with ProgressBar():
    th_horiz = check_horizontal(budget_th, 'TH').load()

th_horiz.total_tendency_TH.plot(linewidth=2, y='Z', label=r"$d\theta/dt$")
th_horiz.total_tendency_TH_truth.plot(linestyle='--', linewidth=2, y='Z', label=r"$d\theta/dt$ (true)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("temperature_budget_horizontal.png", dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")


print("Validating vertical salinity budget...")
with ProgressBar():
    slt_vert = check_vertical(budget_slt, 'SLT').load()

slt_vert.total_tendency_SLT.plot(linewidth=2, label=r"$dS/dt$")
slt_vert.total_tendency_SLT_truth.plot(linestyle='--', linewidth=2, label=r"$dS/dt$ (true)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("salt_budget_vertical.png", dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")


print("Validating horizontal salinity budget...")
with ProgressBar():
    slt_horiz = check_horizontal(budget_slt, 'SLT').load()

slt_horiz.total_tendency_SLT.plot(linewidth=2, y='Z', label=r"$dS/dt$")
slt_horiz.total_tendency_SLT_truth.plot(linestyle='--', linewidth=2, y='Z', label=r"$dS/dt$ (true)")
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("salt_budget_horizontal.png", dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")


budget_th_sum  = budget_th.sum(dim=('XC', 'YC', 'Z')).load()
budget_slt_sum = budget_slt.sum(dim=('XC', 'YC', 'Z')).load()

plt.figure(figsize=(18, 8))
for v in budget_th_sum.data_vars:
    with ProgressBar():
        print(f"Plotting {v}...")
        budget_th_sum[v].rolling(time=30).mean().plot(label=v)

# plt.ylim([-4e7, 4e7])
plt.grid()
plt.title('Patch heat budget')
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("heat_budget_time_series.png", dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

plt.figure(figsize=(18, 8))
for v in budget_slt_sum.data_vars:
    with ProgressBar():
        print(f"Plotting {v}...")
        budget_slt_sum[v].rolling(time=30).mean().plot(label=v)

# plt.ylim([-4e7, 4e7])
plt.grid()
plt.title('Patch salinity budget')
lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("salinity_budget_time_series.png", dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close("all")

