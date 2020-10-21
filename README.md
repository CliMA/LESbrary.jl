# LESbrary.jl

This package is a framework for building a library of large eddy simulations (LES) of ocean surface boundary layer turbulence — the _LESbrary_ — with [Oceananigans.jl](https://github.com/climate-machine/Oceananigans.jl).
The LESbrary will archive turbulence data for both idealized and realistic oceanic scenarios.

## Adding LESbrary.jl to a julia environment

To add `LESbrary.jl` to your julia environment, type

```julia
julia> using Pkg

julia> Pkg.add("https://github.com/CliMA/LESbrary.jl.git")
```

To add the `LESbrary` package to your julia environment. This will allow you to write `using LESbrary` in your scripts (and thereby run the scripts in `examples/`.

## Python conda environment

Some aspects of the LESbrary.jl relies on some Python functionality. If you need this python functionality, you'll have to set up the conda environment:

1. Download and install Miniconda (or Anaconda) if needed: https://docs.conda.io/en/latest/miniconda.html
2. Instantiate and activate the LESbrary.jl conda environment: `conda env create --name lesbrary --file=environment.yml` then `conda activate lesbrary`
3. In Julia set the `PYTHON` environment variable to point to the Python executable provided by Conda. For example: `julia --project` then `ENV["PYTHON"]="/home/alir/miniconda3/envs/lesbrary/bin/python3.8"` at the REPL.
4. Build PyCall: `] build PyCall`

# Observational data

The Southern Ocean State Estimate (SOSE) is currently being used for pilot simulations although we're planning to also use ECCO v4 and ERA5 data.

SOSE data and also lots of plots being produced is on engaging at `/home/alir/cnhlab004/bsose_i122`.
