include("runtests_preamble.jl")

using Oceananigans.Units
using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: ConstantFluxStokesDrift, ConstantFluxPeakWavenumber, peak_wavenumber

@testset "ConstantFluxStokesDrift" begin
    grid = RectilinearGrid(size=256, z=(-64, 0), topology=(Flat, Flat, Bounded))
    cfsd = ConstantFluxStokesDrift(grid, -1e-4, 2π/300)
    @test cfsd isa ConstantFluxStokesDrift
    
    # Sort of a regression test
    @test cfsd.uˢ[1, 1, grid.Nz] ≈ 0.13965495
    @test cfsd.∂z_uˢ[1, 1, grid.Nz+1] ≈ 0.94022337
    @test cfsd.∂z_uˢ[1, 1, grid.Nz] ≈ 0.119721326
    
    # Test ConstantFluxPeakWavenumber (experimental)
    cfpw = ConstantFluxPeakWavenumber()
    Cᵡ = cfpw.Cᵡ
    Cᵅ = cfpw.Cᵅ

    @test Cᵡ ≈ 3.755
    @test Cᵅ ≈ -0.287
    @test 1 + Cᵅ ≈ 0.713
    
    # 2 / (1 + Cᵅ) ≈ 2.805, 3.755^2.805 ≈ 41
    @test Cᵡ^(2 / (1 + Cᵅ)) ≈ 40.90808
    @test 2 * (1 + 2Cᵅ) / (1 + Cᵅ) ≈ 1.1949509
    @test 2Cᵅ / (1 + Cᵅ) ≈ -0.8050491
end


@testset "Three layer constant flux simulation" begin
    simulation = three_layer_constant_fluxes_simulation(size = (2, 2, 16),
                                                        stokes_drift = false,
                                                        stop_time = 2.0,
                                                        snapshot_time_interval = 1.0,
                                                        averages_time_interval = 1.0,
                                                        averages_time_window = 1.0)
    run!(simulation)

    @test simulation.stop_time == 2.0

    simulation = three_layer_constant_fluxes_simulation(size = (4, 4, 16),
                                                        stokes_drift_peak_wavenumber = 2π / 300,
                                                        momentum_flux = -1e-4,
                                                        stop_time = 2.0,
                                                        snapshot_time_interval = 1.0,
                                                        averages_time_interval = 1.0,
                                                        averages_time_window = 1.0)
    run!(simulation)

    @test simulation.stop_time == 2.0
end
