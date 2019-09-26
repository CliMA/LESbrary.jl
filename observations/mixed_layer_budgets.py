import xarray as xr

import cmocean
import matplotlib.pyplot as plt

files_3d = [
    "bsose_i122_2013to2017_1day_Uvel.nc",
    "bsose_i122_2013to2017_1day_Vvel.nc",
    "bsose_i122_2013to2017_1day_Wvel.nc",
    "bsose_i122_2013to2017_1day_Theta.nc",
    "bsose_i122_2013to2017_1day_Salt.nc",
    "bsose_i122_2013to2017_1day_Strat.nc"
]

files_2d = [
    "bsose_i122_2013to2017_1day_MLD.nc",
    "bsose_i122_2013to2017_1day_oceTAUX.nc",
    "bsose_i122_2013to2017_1day_oceTAUY.nc",
    "bsose_i122_2013to2017_1day_surfSflx.nc",
    "bsose_i122_2013to2017_1day_surfTflx.nc",
    "bsose_i122_2013to2017_daily_oceFWflx.nc",
    "bsose_i122_2013to2017_daily_oceQnet.nc",
    "bsose_i122_2013to2017_daily_oceQsw.nc"
]

fields_3d = xr.open_mfdataset(files_3d, parallel=True, chunks={'time': 10})
fields_2d = xr.open_mfdataset(files_2d, parallel=True, chunks={'time': 10})

