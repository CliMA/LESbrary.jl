using Test

@test begin
    include(joinpath(@__DIR__, "idealized", "run_free_convection.jl"))
    @test 1 == 1
end
