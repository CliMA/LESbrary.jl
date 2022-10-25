# Example script for generating and analyzing LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: half_day_suite_parameters
using LESbrary.IdealizedExperiments: one_day_suite_parameters
using LESbrary.IdealizedExperiments: two_day_suite_parameters
using LESbrary.IdealizedExperiments: three_day_suite_parameters
using LESbrary.IdealizedExperiments: four_day_suite_parameters

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
    size = (128, 128, 128),
    snapshot_time_interval = 10minutes,
    passive_tracers = false,
    time_averaged_statistics = true,
    data_directory = "/home/greg/Projects/LESbrary.jl/data"
)

cases = (:free_convection,
         :weak_wind_strong_cooling, 
         :med_wind_med_cooling,     
         :strong_wind_weak_cooling, 
         :strong_wind,
         :strong_wind_no_rotation)

for case in cases
    parameters = two_day_suite_parameters[case]
    @show "Running with $parameters..."
    simulation = three_layer_constant_fluxes_simulation(; configuration..., parameters...)
    run!(simulation)
end

