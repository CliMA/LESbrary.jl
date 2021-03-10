# Observational data

The Southern Ocean State Estimate (SOSE) is currently being used for pilot simulations although we might also use ECCO4v3 and ERA5 data in the future.

SOSE data needs to be downloaded first (the files needed are over 3 TiB!). The download script is `download_sose_i122_data.py`.

Once the data is downloaded, you should be able to just run the `pilot_simulation.jl` script. There's a `SOSE_DIR` variable in there that should point to the directory containing the SOSE data. Any Python dependencies will be added by Julia using Conda.jl
