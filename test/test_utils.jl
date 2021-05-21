using LESbrary.Utils

using Pkg
Pkg.develop(path=joinpath(@__DIR__, "..", "realistic"))

@testset "fit_cubic" begin
    c = fit_cubic((0, 0), (1, 1), 0, 0)

    # We should get back 3x^2 - 2x^3
    @test c[1] ≈ 0 atol=1e-12
    @test c[2] ≈ 0 atol=1e-12
    @test c[3] ≈ 3
    @test c[4] ≈ -2
end
