# Example script for generating LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: forty_eight_hour_suite_parameters
using LESbrary.IdealizedExperiments: twenty_four_hour_suite_parameters
using LESbrary.IdealizedExperiments: twelve_hour_suite_parameters
using LESbrary.IdealizedExperiments: six_hour_suite_parameters
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
# * :med_wind_med_cooling
# * :strong_wind_weak_cooling
# * :strong_wind
# * :strong_wind_no_rotation
#
# In addition to selecting the architecture, size, and case to run, we can also tweak
# certain parameters. Below we change the "snapshot_time_interval" (the interval over
# which slices of the simulation is saved) from the default 2 minutes to 1 minute
# (to make pretty movies).

architecture = GPU()
#size = (64, 64, 64)
size = (128, 128, 128)
#size = (256, 256, 256)
#size = (256, 256, 384)
# case = :strong_wind
snapshot_time_interval = 10minute
data_directory = "/nobackup/users/glwagner/LESbrary/"

cases = [
    :strong_wind,
    :free_convection,
    :weak_wind_strong_cooling,
    #:med_wind_med_cooling,
    #:strong_wind_weak_cooling,
    #:strong_wind_no_rotation
]

#=
suites = [twelve_hour_suite_parameters,
          twenty_four_hour_suite_parameters,
          forty_eight_hour_suite_parameters]
=#

#suite = six_hour_suite_parameters
suite = twelve_hour_suite_parameters
#suite = twenty_four_hour_suite_parameters
#suite = forty_eight_hour_suite_parameters
#suite = seventy_two_hour_suite_parameters

@inline κˢ(x, y, z, t) = ifelse(z > -5, 1e-2, zero(z))

for case in cases
    suite_parameters = suite[case]

    # suite_parameters[:stokes_drift] = false
    # name = suite_parameters[:name]
    # suite_parameters[:name] = name * "_no_stokes"

    # suite_parameters[:background_diffusivity] = κˢ
    # name = suite_parameters[:name]
    # suite_parameters[:name] = name * "_surface_diffusivity"
    
    suite_parameters[:name] = name * "_with_tracer"
    simulation = three_layer_constant_fluxes_simulation(; architecture,
                                                          size,
                                                          checkpoint = false,
                                                          pickup = false,
                                                          passive_tracers = true,
                                                          data_directory,
                                                          snapshot_time_interval,
                                                          suite_parameters...)
    run!(simulation)
end

