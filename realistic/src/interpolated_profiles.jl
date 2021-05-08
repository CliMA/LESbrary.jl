import Adapt

@inline fractional_index(x, xs, Δx) = @inbounds (x - xs[1]) / Δx

# Linear Lagrange polynomials
@inline ϕ₀(ξ) = 1 - ξ
@inline ϕ₁(ξ) = ξ

#####
##### Linear interpolation for regularly spaced profiles e.g. T(z).
#####

struct InterpolatedProfile{D, Z, Δ, T}
     data :: D
        z :: Z
       Δz :: Δ
    z_max :: T
end

rebuild(p::InterpolatedProfile, new_array_type) =
    InterpolatedProfile(new_array_type(p.data), p.z, p.Δz, p.z_max)

@inline function _interpolate(profile::InterpolatedProfile, zᵢ)
    k = 1 + fractional_index(zᵢ, profile.z, profile.Δz)
    ξ, k = mod(k, 1), Base.unsafe_trunc(Int, k)

    # Ensure we don't go out of bounds.
    if zᵢ >= profile.z_max
        return @inbounds profile.data[k]
    else
        return @inbounds ϕ₀(ξ) * profile.data[k] + ϕ₁(ξ) * profile.data[k+1]
    end
end

@inline (profile::InterpolatedProfile)(z) = _interpolate(profile, z)

@inline function ∂z(profile::InterpolatedProfile, zᵢ)
    k = 1 + fractional_index(zᵢ, profile.z, profile.Δz)
    k = Base.unsafe_trunc(Int, k)
    zᵢ >= profile.z_max && (k = k - 1) # Ensure we don't go out of bounds.
    return @inbounds (profile.data[k+1] - profile.data[k]) / profile.Δz
end

Adapt.adapt_structure(to, profile::InterpolatedProfile) =
    InterpolatedProfile(Adapt.adapt(to, profile.data),
                        Adapt.adapt(to, profile.z),
                        Adapt.adapt(to, profile.Δz),
                        Adapt.adapt(to, profile.z_max))

#####
##### Linear interpolated for regularly spaced profile time series, e.g. T(z, t).
#####

struct InterpolatedProfileTimeSeries{D, Z, T, ΔZ, ΔT, FT}
     data :: D
        z :: Z
        t :: T
       Δz :: ΔZ
       Δt :: ΔT
    z_max :: FT
    t_max :: FT
end

rebuild(p::InterpolatedProfileTimeSeries, new_array_type) =
InterpolatedProfileTimeSeries(new_array_type(p.data), p.z, p.t, p.Δz, p.Δt, p.z_max, p.t_max)

@inline function _interpolate(profile::InterpolatedProfileTimeSeries, zᵢ, tᵢ)
    k = 1 + fractional_index(zᵢ, profile.z, profile.Δz)
    n = 1 + fractional_index(tᵢ, profile.t, profile.Δt)

    ξ, k = mod(k, 1), Base.unsafe_trunc(Int, k)
    η, n = mod(n, 1), Base.unsafe_trunc(Int, n)

    # Ensure we don't go out of bounds.
    if zᵢ >= profile.z_max && tᵢ >= profile.t_max
        return @inbounds profile.data[k, n]
    elseif zᵢ >= profile.z_max
        return @inbounds ϕ₀(η) * profile.data[k, n] + ϕ₁(η) * profile.data[k, n+1]
    elseif tᵢ >= profile.t_max
        return @inbounds ϕ₀(ξ) * profile.data[k, n] + ϕ₁(ξ) * profile.data[k+1, n]
    else
        return @inbounds (  ϕ₀(ξ) * ϕ₀(η) * profile.data[k,   n  ]
                          + ϕ₀(ξ) * ϕ₁(η) * profile.data[k,   n+1]
                          + ϕ₁(ξ) * ϕ₀(η) * profile.data[k+1, n  ]
                          + ϕ₁(ξ) * ϕ₁(η) * profile.data[k+1, n+1])
    end
end

@inline (profile::InterpolatedProfileTimeSeries)(z, t) = _interpolate(profile, z, t)

@inline function ∂z(profile::InterpolatedProfileTimeSeries, zᵢ, tᵢ)
    k = 1 + fractional_index(zᵢ, profile.z, profile.Δz)
    n = 1 + fractional_index(tᵢ, profile.t, profile.Δt)

    ξ, k = mod(k, 1), Base.unsafe_trunc(Int, k)
    η, n = mod(n, 1), Base.unsafe_trunc(Int, n)

    # Ensure we don't go out of bounds.
    zᵢ >= profile.z_max && (k = k - 1)
    tᵢ >= profile.t_max && (n = n - 1)

    dataᵏ   = @inbounds ϕ₀(η) * profile.data[k,   n] + ϕ₁(η) * profile.data[k,   n+1]
    dataᵏ⁺¹ = @inbounds ϕ₀(η) * profile.data[k+1, n] + ϕ₁(η) * profile.data[k+1, n+1]

    return (dataᵏ⁺¹ - dataᵏ) / profile.Δz
end

@inline function ∂t(profile::InterpolatedProfileTimeSeries, zᵢ, tᵢ)
    k = 1 + fractional_index(zᵢ, profile.z, profile.Δz)
    n = 1 + fractional_index(tᵢ, profile.t, profile.Δt)

    ξ, k = mod(k, 1), Base.unsafe_trunc(Int, k)
    η, n = mod(n, 1), Base.unsafe_trunc(Int, n)

    # Ensure we don't go out of bounds.
    zᵢ >= profile.z_max && (k = k - 1)
    tᵢ >= profile.t_max && (n = n - 1)

    dataⁿ   = @inbounds ϕ₀(ξ) * profile.data[k,   n] + ϕ₁(ξ) * profile.data[k+1,   n]
    dataⁿ⁺¹ = @inbounds ϕ₀(ξ) * profile.data[k, n+1] + ϕ₁(ξ) * profile.data[k+1, n+1]

    return (dataⁿ⁺¹ - dataⁿ) / profile.Δt
end

Adapt.adapt_structure(to, profile::InterpolatedProfileTimeSeries) =
    InterpolatedProfileTimeSeries(Adapt.adapt(to, profile.data),
                                  Adapt.adapt(to, profile.z),
                                  Adapt.adapt(to, profile.t),
                                  Adapt.adapt(to, profile.Δz),
                                  Adapt.adapt(to, profile.Δt),
                                  Adapt.adapt(to, profile.z_max),
                                  Adapt.adapt(to, profile.t_max))
