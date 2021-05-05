import Base: getindex

#####
##### Linear interpolated for regularly spaced profiles
##### e.g. T(z) and profile time series, e.g. T(z, t).
#####

@inline fractional_index(x, xs, Δx) = @inbounds (x - xs[1]) / Δx

# Linear Lagrange polynomials
@inline ϕ₀(ξ) = 1 - ξ
@inline ϕ₁(ξ) = ξ

struct InterpolatedProfile{D, Z, Δ}
    data :: D
       z :: Z
      Δz :: Δ
end

@inline function _interpolate(profile::InterpolatedProfile, zᵢ)
    k = 1 + fractional_index(zᵢ, profile.z, profile.Δz)
    ξ, k = mod(k, 1), Base.unsafe_trunc(Int, k)
    return @inbounds ϕ₀(ξ) * profile.data[k] + ϕ₁(ξ) * profile.data[k+1]
end

@inline (profile::InterpolatedProfile)(z) = _interpolate(profile, z)

struct InterpolatedProfileTimeSeries{D, Z, T, ΔZ, ΔT}
    data :: D
       z :: Z
       t :: T
      Δz :: ΔZ
      Δt :: ΔT
end

@inline function _interpolate(profile::InterpolatedProfileTimeSeries, zᵢ, tᵢ)
    k = 1 + fractional_index(zᵢ, profile.z, profile.Δz)
    n = 1 + fractional_index(tᵢ, profile.t, profile.Δt)

    ξ, k = mod(k, 1), Base.unsafe_trunc(Int, k)
    η, n = mod(n, 1), Base.unsafe_trunc(Int, n)

    return @inbounds (  ϕ₀(ξ) * ϕ₀(η) * profile.data[k,   n  ]
                      + ϕ₀(ξ) * ϕ₁(η) * profile.data[k,   n+1]
                      + ϕ₁(ξ) * ϕ₀(η) * profile.data[k+1, n  ]
                      + ϕ₁(ξ) * ϕ₁(η) * profile.data[k+1, n+1])
end

@inline (profile::InterpolatedProfileTimeSeries)(z, t) = _interpolate(profile, z, t)
