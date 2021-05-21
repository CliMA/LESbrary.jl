using QuadGK

"""
    diagnose_buoyancy_flux(b, z, time_period, depth)

Returns the buoyancy flux (m²/s³) required to extract all the buoyancy from a 1D buoyancy profile `b(z)`
down to some z-coordinate `depth` (< 0) over a `time_period` (s) by integrating the buoyancy profile and
converting the integrated buoyancy content to an equivalent buoyancy flux.
"""
function diagnose_buoyancy_flux(b, z, time_period, depth)

    Δτ = time_period
    Δz = z[2] - z[1]
    @assert first(z) < depth < last(z)

    ℑb = InterpolatedProfile(b, z, Δz, last(z))
    b₀ = first(b)

    integrand(z) = ℑb(z) - b₀
    B, err = quadgk(integrand, depth, last(z))
    @show err

    return B / Δτ
end
