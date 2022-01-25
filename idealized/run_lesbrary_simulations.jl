# Example script for generating LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: run_three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: two_day_suite_parameters
using LESbrary.IdealizedExperiments: four_day_suite_parameters
using LESbrary.IdealizedExperiments: six_day_suite_parameters

architecture = GPU()

all_parameters = tuple(values(two_day_suite_parameters)...,
                       values(four_day_suite_parameters)...,
                       values(six_day_suite_parameters)...)

for size in ((64, 64, 64), (128, 128, 128), (256, 256, 256))
    for parameters in all_parameters
        run_three_layer_constant_fluxes_simulation(; architecture, size, parameters...)
    end
end
    
