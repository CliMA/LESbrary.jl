include("runtests_preamble.jl")

@testset "LESbrary" begin
    include("test_fit_cubic.jl")
    include("test_interpolated_profiles.jl")
    include("test_diagnose_buoyancy_flux.jl")
    include("test_turbulence_statistics.jl")
    include("test_examples.jl")
    include("test_experiments.jl")
end
