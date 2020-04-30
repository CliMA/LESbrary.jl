using Test

@test begin
    include(joinpath(@__DIR__, "simulation", "run_free_convection.jl"))
    @test 1 == 1
end
