# Example script for generating LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: two_day_suite_parameters
using LESbrary.IdealizedExperiments: four_day_suite_parameters
using LESbrary.IdealizedExperiments: six_day_suite_parameters

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
# certain parameters. Below we change the "snapshot_time_interval" (the interval over
# which slices of the simulation is saved) from the default 2 minutes to 1 minute
# (to make pretty movies), and we turn passive tracers off.

architecture = GPU()
size = (256, 256, 256)
case = :strong_wind
snapshot_time_interval = 1minute
passive_tracers = false

simulation = three_layer_constant_fluxes_simulation(; architecture,
                                                      size,
                                                      passive_tracers,
                                                      snapshot_time_interval,
                                                      two_day_suite_parameters[case]...)

run!(simulation)

