import dask
import dask.array as dsa
import matplotlib
import numpy as np
import xarray as xr
import xgcm

from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.dpi'] = 200

XC_YC_files = [
    "bsose_i122_2013to2017_monthly_ADVr_TH.nc",
    "bsose_i122_2013to2017_monthly_ADVr_SLT.nc",
    "bsose_i122_2013to2017_monthly_DFrI_TH.nc",
    "bsose_i122_2013to2017_monthly_DFrI_SLT.nc",
    "bsose_i122_2013to2017_monthly_TOTTTEND.nc",
    "bsose_i122_2013to2017_monthly_TOTSTEND.nc",
    "bsose_i122_2013to2017_monthly_WTHMASS.nc",
    "bsose_i122_2013to2017_monthly_WSLTMASS.nc",
    "bsose_i122_2013to2017_5day_oceQsw.nc",
    "bsose_i122_2013to2017_5day_surfTflx.nc",
    "bsose_i122_2013to2017_5day_surfSflx.nc"
]

XC_YG_files = [
    "bsose_i122_2013to2017_monthly_ADVy_TH.nc",
    "bsose_i122_2013to2017_monthly_ADVy_SLT.nc",
    "bsose_i122_2013to2017_monthly_DFyE_TH.nc",
    "bsose_i122_2013to2017_monthly_DFyE_SLT.nc"
]

XG_YC_files = [
    "bsose_i122_2013to2017_monthly_ADVx_TH.nc",
    "bsose_i122_2013to2017_monthly_ADVx_SLT.nc",
    "bsose_i122_2013to2017_monthly_DFxE_TH.nc",
    "bsose_i122_2013to2017_monthly_DFxE_SLT.nc"
]

all_files = XC_YC_files + XC_YG_files + XG_YC_files

x_chunks = 100
y_chunks = 100
x_slice = slice(0, 3)
y_slice = slice(-43, -40)

for fpath in all_files:
    if fpath in XC_YC_files:
        ds = xr.open_dataset(fpath, chunks={'XC': x_chunks, 'YC': y_chunks})
        patch_ds = ds.sel(XC=x_slice, YC=y_slice)
    elif fpath in XC_YG_files:
        ds = xr.open_dataset(fpath, chunks={'XC': x_chunks, 'YG': y_chunks})
        patch_ds = ds.sel(XC=x_slice, YG=y_slice)
    elif fpath in XG_YC_files:
        ds = xr.open_dataset(fpath, chunks={'XG': x_chunks, 'YC': y_chunks})
        patch_ds = ds.sel(XG=x_slice, YC=y_slice)

    new_fpath = fpath.replace(".nc", "_patch.nc")
    print(f"Saving {new_fpath}...")
    with ProgressBar():
        patch_ds.load().to_netcdf(new_fpath, mode="w")

