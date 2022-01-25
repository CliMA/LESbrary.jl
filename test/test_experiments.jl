include("runtests_preamble.jl")

using Oceananigans.Units
using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation

@testset "Three layer constant flux simulation" begin
    simulation = three_layer_constant_fluxes_simulation(size = (1, 1, 32),
                                                        stop_time = 2.0,
                                                        snapshot_time_interval = 1.0,
                                                        averages_time_interval = 1.0,
                                                        averages_time_window = 1.0)
    run!(simulation)

    @test simulation.stop_time == 2.0
end

