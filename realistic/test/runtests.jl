using Test
using Statistics
using RealisticLESbrary

@testset "InterpolatedProfile" begin
    zs = 0:5:50
    Δz = zs.step
    N = length(zs)
    data = randn(N)

    prof = InterpolatedProfile(data, zs, Δz)

    @test prof(0) == prof.data[1]
    @test prof(5) == prof.data[2]

    @test prof(12.5) ≈ (prof.data[3] + prof.data[4]) / 2
end

@testset "InterpolatedProfileTimeSeries" begin
    zs = 0:5:50
    ts = 0:60:600
    Δz = zs.step
    Δt = ts.step
    Nz = length(zs)
    Nt = length(ts)
    data = randn(Nz, Nt)

    prof = InterpolatedProfileTimeSeries(data, zs, ts, Δz, Δt)

    @test prof(0, 0) == prof.data[1, 1]
    @test prof(5, 0) == prof.data[2, 1]
    @test prof(0, 60) == prof.data[1, 2]
    @test prof(5, 60) == prof.data[2, 2]

    @test prof(27.5, 330) ≈ mean(prof.data[6:7, 6:7])
end
