# Example script for generating LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: two_day_suite_parameters
using LESbrary.IdealizedExperiments: four_day_suite_parameters
using LESbrary.IdealizedExperiments: six_day_suite_parameters

# 
# Run 
# case = :free_convection
#
architecture = GPU()
size = (256, 256, 256)
case = :weak_wind_strong_cooling
snapshot_time_interval = 1minutes

parameters = two_day_suite_parameters[case]
simulation = three_layer_constant_fluxes_simulation(; architecture, size, snapshot_time_interval, parameters...)
run!(simulation)
    
