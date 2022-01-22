#!/bin/sh

CUDA_VISIBLE_DEVICES=0 julia --project three_layer_constant_fluxes.jl \
    --name free_convection \
    --hours 48 \
    --size 128 128 128 \
    --extent 512 512 256 \
    --thermocline linear \
    --momentum-flux 0.0 \
    --buoyancy-flux 1.2e-7 \
    --animation

# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/three_layer_constant_fluxes.jl --hours 48 --size 256 256 128 --extent 512 512 256 --thermocline linear --momentum-flux -1e-3 --buoyancy-flux 0.0 --name strong_wind --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/three_layer_constant_fluxes.jl --hours 48 --size 256 256 128 --extent 512 512 256 --thermocline linear --momentum-flux -7e-4 --buoyancy-flux 6e-8 --name strong_wind_weak_cooling --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/three_layer_constant_fluxes.jl --hours 48 --size 256 256 128 --extent 512 512 256 --thermocline linear --momentum-flux -3.3e-4 --buoyancy-flux 1.1e-7 --name weak_wind_strong_cooling --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/three_layer_constant_fluxes.jl --hours 48 --size 256 256 128 --extent 512 512 256 --thermocline linear --momentum-flux -1e-3 --buoyancy-flux -4e-8 --name strong_wind_weak_heating --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/three_layer_constant_fluxes.jl --hours 48 --size 256 256 128 --extent 512 512 256 --thermocline linear --momentum-flux -2e-4 --buoyancy-flux 0.0 --coriolis 0.0 --name strong_wind_no_rotation --animation
