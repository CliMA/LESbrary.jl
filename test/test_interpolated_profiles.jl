using Test
using Statistics
using LESbrary.Utils
import LESbrary.Utils

@testset "InterpolatedProfile" begin

    # interpolation

    zs = 0:5:50
    Δz = zs.step
    N = length(zs)
    data = randn(N)

    prof = InterpolatedProfile(data, zs, Δz, zs[end])

    @test prof(0) == prof.data[1]
    @test prof(5) == prof.data[2]

    @test prof(12.5) ≈ (prof.data[3] + prof.data[4]) / 2

    # derivatives

    zs = 0:5:50
    Δz = zs.step
    N = length(zs)

    f(z) = z^2
    data = f.(zs)

    prof = InterpolatedProfile(data, zs, Δz, zs[end])

    @test Utils.∂z(prof, 2) ≈ (f(5) - f(0)) / Δz
    @test Utils.∂z(prof, 7.4) ≈ (f(10) - f(5)) / Δz
    @test Utils.∂z(prof, 11.8) ≈ (f(15) - f(10)) / Δz

    @test Utils.∂z(prof, 2) == Utils.∂z(prof, 4)
    @test Utils.∂z(prof, 7.4) == Utils.∂z(prof, 9.99)
    @test Utils.∂z(prof, 11.8) == Utils.∂z(prof, 10.01)
end

@testset "InterpolatedProfileTimeSeries" begin

    # interpolation

    zs = 0:5:50
    ts = 0:60:600
    Δz = zs.step
    Δt = ts.step
    Nz = length(zs)
    Nt = length(ts)
    data = randn(Nz, Nt)

    prof = InterpolatedProfileTimeSeries(data, zs, ts, Δz, Δt, zs[end], ts[end])

    @test prof(0, 0) == prof.data[1, 1]
    @test prof(5, 0) == prof.data[2, 1]
    @test prof(0, 60) == prof.data[1, 2]
    @test prof(5, 60) == prof.data[2, 2]

    @test prof(27.5, 330) ≈ mean(prof.data[6:7, 6:7])

    # derivatives

    zs = 0:5:50
    ts = 0:60:600
    Δz = zs.step
    Δt = ts.step
    Nz = length(zs)
    Nt = length(ts)

    f(z, t) = 3z^2 + 0.1t^2
    data = f.(zs, ts')

    prof = InterpolatedProfileTimeSeries(data, zs, ts, Δz, Δt, zs[end], ts[end])

    @test Utils.∂z(prof, 3, 0) ≈ (f(5, 0) - f(0, 0)) / Δz
    @test Utils.∂z(prof, 7.5, 10) ≈ (f(10, 10) - f(5, 10)) / Δz
    @test Utils.∂z(prof, 10.01, 420.69) ≈ (f(15, 420.69) - f(10, 420.69)) / Δz

    @test ∂t(prof, 0, 45) ≈ (f(0, 60) - f(0, 0)) / Δt
    @test ∂t(prof, 6, 100) ≈ (f(6, 120) - f(6, 60)) / Δt
    @test ∂t(prof, 22.2, 310) ≈ (f(22.2, 360) - f(22.2, 300)) / Δt
end
