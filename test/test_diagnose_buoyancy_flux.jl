using Statistics
using LESbrary.Utils

# This test replicates the setup from Souza et al. (2020) and compares
# the answer with Eq. (4). We expect agreement to machine precision.
#
# Souza et al. (2020). Uncertainty quantification of ocean parameterizations: Application to
# the K‐Profile‐Parameterization for penetrative convection. Journal of Advances in Modeling
# Earth Systems, 12, e2020MS002108. DOI: https://doi.org/10.1029/2020MS002108

@testset "diagnose_buoyancy_flux" begin
    α  = 2e-4
    g  = 9.81
    N² = 1e-5
    b(z) = 20*α*g + N²*z

    zs = range(-100, 0, length=64)
    bs = [b(z) for z in zs]

    days = 86400
    Δτ = 8days

    # We want the mixed layer to reach half the domain height.
    depth = mean(zs)

    Qb = diagnose_buoyancy_flux(bs, zs, Δτ, depth)

    h = abs(depth)
    Qb_empirical = h^2  * N² / (2Δτ)

    @test Qb ≈ Qb_empirical
end
