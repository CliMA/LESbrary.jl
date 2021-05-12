from subprocess import run

# Links from http://sose.ucsd.edu/BSOSE6_iter122_solution.html
# Also see: http://sose.ucsd.edu/SO6/ITER122/budgets/available_diagnostics.log

urls = [
    ## Grid

    "http://sose.ucsd.edu/SO6/SETUP/grid.nc",

    ## 3D physical diagnostics

    # Potential temperature [degC]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_Theta.nc",
    # Salinity [g/kg]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_Salt.nc",
    # Zonal component of velocity [m/s]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_Uvel.nc",
    # Meridional component of velocity [m/s]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_Vvel.nc",
    # Vertical component of velocity [m/s]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_Wvel.nc",
    # Stratification dσ/dz [kg/m^4]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_Strat.nc",

    ## 3D advective fluxes

    # Zonal advective flux of potential temperature [degC m^3/s]
    "http://sose.ucsd.edu/SO6/ITER122/budgets/bsose_i122_2013to2017_5day_ADVx_TH.nc",
    # Meridional advective flux of potential temperature [degC m^3/s]
    "http://sose.ucsd.edu/SO6/ITER122/budgets/bsose_i122_2013to2017_5day_ADVy_TH.nc",
    # Zonal advective flux of salinity [psu m^3/s]
    "http://sose.ucsd.edu/SO6/ITER122/budgets/bsose_i122_2013to2017_5day_ADVx_SLT.nc",
    # Meridional advective flux of salinity [psu m^3/s]
    "http://sose.ucsd.edu/SO6/ITER122/budgets/bsose_i122_2013to2017_5day_ADVy_SLT.nc",

    ## 2D diagnostics

    # Zonal surface wind stress, >0 increases uVel [N/m^2]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_oceTAUX.nc",
    # Meridional surface wind stress, >0 increases vVel [N/m^2]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_oceTAUY.nc",
    # Total heat flux, >0 increases theta [W/m^2]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_surfTflx.nc",
    # Total salt flux, >0 increases salt [g/m^2/s]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_surfSflx.nc",
    # Diagnosed mixed layer depth [m]
    "http://sose.ucsd.edu/SO6/ITER122/bsose_i122_2013to2017_1day_MLD.nc",
    # Net surface heat flux into the ocean (+=down), >0 increases theta [W/m^2]
    "http://sose.ucsd.edu/SO6/ITER122/daily/bsose_i122_2013to2017_daily_oceQnet.nc",
    # Net short-wave radiation (+=down), >0 increases theta [W/m^2]
    "http://sose.ucsd.edu/SO6/ITER122/daily/bsose_i122_2013to2017_daily_oceQsw.nc",
    # Net surface fresh-water flux into the ocean (+=down), >0 decreases salinity [kg/m^2/s]
    "http://sose.ucsd.edu/SO6/ITER122/daily/bsose_i122_2013to2017_daily_oceFWflx.nc"
]

for url in urls:
    run(["wget", "-N", url])
