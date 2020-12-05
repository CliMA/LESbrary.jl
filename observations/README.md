# Observational data

The Southern Ocean State Estimate (SOSE) is currently being used for pilot simulations although we might also use ECCO4v3 and ERA5 data.

SOSE data and also lots of plots being produced is on `engaging.mit.edu` at `/home/alir/cnhlab004/bsose_i122`.

## Python conda environment

Some aspects of the LESbrary.jl relies on some Python functionality. If you need this python functionality, you'll have to set up the conda environment:

1. Download and install Miniconda (or Anaconda) if needed: https://docs.conda.io/en/latest/miniconda.html
2. Instantiate and activate the LESbrary.jl conda environment:

   ```bash
   conda env create --name lesbrary --file=environment.yml
   ```

   then `conda activate lesbrary`
3. In Julia set the `PYTHON` environment variable to point to the Python executable provided by Conda. For example: `julia --project` then

   ```julia
   ENV["PYTHON"]="/home/alir/miniconda3/envs/lesbrary/bin/python3.8"
   ```

   at the REPL.

4. Build PyCall: `] build PyCall`
