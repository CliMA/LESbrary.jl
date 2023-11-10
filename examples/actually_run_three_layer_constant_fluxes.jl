# Example script for generating and analyzing LESbrary data

using Oceananigans
using Oceananigans.Units

using CUDA

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: six_hour_suite_parameters
using LESbrary.IdealizedExperiments: eighteen_hour_suite_parameters
using LESbrary.IdealizedExperiments: twelve_hour_suite_parameters
using LESbrary.IdealizedExperiments: twenty_four_hour_suite_parameters
using LESbrary.IdealizedExperiments: thirty_six_hour_suite_parameters
using LESbrary.IdealizedExperiments: forty_eight_hour_suite_parameters
using LESbrary.IdealizedExperiments: seventy_two_hour_suite_parameters

# LESbrary parameters
# ===================
#
# Canonical LESbrary parameters are organized by the _intended duration_ of the simulation
# --- two days for the "two day suite", four days for the "four day suite", et cetera.
# The strength of the forcing is tuned roughly so that the boundary layer deepens to roughly
# 128 meters, half the depth of a 256 meter domain.
#
# Each suite has five cases:
#
# * :free_convection
# * :weak_wind_strong_cooling
# * :strong_wind_weak_cooling
# * :strong_wind
# * :strong_wind_no_rotation
#
# In addition to selecting the architecture, size, and case to run, we can also tweak
# certain parameters below.

configuration = (;
    architecture = GPU(),
    size = (448, 448, 512),
    #size = (384, 384, 384),
    #size = (64, 64, 64),
    snapshot_time_interval = 10minutes,
    passive_tracers = false,
    jld2_output = true,
    checkpoint = true,
    time_averaged_statistics = true,
    data_directory = "/nobackup/users/glwagner"
)

cases = [
         :free_convection,
         :weak_wind_strong_cooling, 
         :med_wind_med_cooling,     
         :strong_wind_weak_cooling, 
         :strong_wind,
         :strong_wind_no_rotation,
        ]

#for case in cases

#case = :free_convection
#case = :strong_wind_no_rotation
#case = :weak_wind_strong_cooling
#case = :med_wind_med_cooling
#case = :strong_wind_weak_cooling
case = :strong_wind

    parameters = six_hour_suite_parameters[case]
    #parameters = twelve_hour_suite_parameters[case]
    #parameters = eighteen_hour_suite_parameters[case]
    #parameters = seventy_two_hour_suite_parameters[case]

    println("Running $case")
    for (k, v) in parameters
        println("    - ", k, ": ", v)
    end

    simulation = three_layer_constant_fluxes_simulation(; configuration..., parameters...)

    if !isnothing(simulation.model.stokes_drift)
        stokes_drift = simulation.model.stokes_drift
        grid = simulation.model.grid
        @show uˢ₀ = CUDA.@allowscalar stokes_drift.uˢ[1, 1, grid.Nz]

        a★ = stokes_drift.air_friction_velocity
        ρʷ = stokes_drift.water_density
        ρᵃ = stokes_drift.air_density
        u★ = a★ * sqrt(ρᵃ / ρʷ)
        @show La = sqrt(u★ / uˢ₀)
    end

    run!(simulation)
#end

#=
#case = :free_convection
#case = :weak_wind_strong_cooling
#case = :med_wind_med_cooling
#case = :strong_wind_weak_cooling
#case = :strong_wind
case = :strong_wind_no_rotation

parameters = thirty_six_hour_suite_parameters[case]
@info "Running case $case with $parameters..."

start_time = time_ns()
simulation = three_layer_constant_fluxes_simulation(; configuration..., parameters...)
run!(simulation)
elapsed = 1e-9 * (time_ns() - start_time)
=#
