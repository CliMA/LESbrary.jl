using Oceanaingans.Buoyancy: g_Earth
using Oceananigans.Operators: ℑxzᶠᵃᶜ, ℑxzᶜᵃᶠ, ℑzᵃᵃᶠ
using SpecialFunctions
using Adapt: adapt_structure

import Oceananigans.StokesDrift: AbstractStokesDrift, 
import Oceananigans.StokesDrift: ∂t_uˢ, ∂t_vˢ, ∂t_wˢ,
                                 x_curl_Uˢ_cross_U,
                                 y_curl_Uˢ_cross_U,
                                 z_curl_Uˢ_cross_U

struct ConstantWindStokesDrift{T, G, UZ, VZ, UT, VT}
    Cᵝ :: T # Toba's constant
    Cʳ :: T # Transition wavenumber parameter
    Cⁱ :: T # Cutoff / isotropic wavenumber parameter
    Cᴮ :: T # Saturation constant
    ρʷ :: T # Water reference density
    ρᵃ :: T # (Constant) air reference density
    τʷ :: T # Water-side friction velocity
    Cᵅ :: T # Fetch relation exponent
    Cᵡ :: T # Fetch relation constant
    tᵢ :: T # Initial time
    kⁿ :: T # Transition wavenumber
    kⁱ :: T # Isotropic wavenumber
    gravitational_acceleration :: T
    grid :: G
    uˢ :: U
    ∂z_uˢ :: UZ
    ∂t_uˢ :: UT
end

Adapt.adapt_structure(to, sssd) = ConstantWindStokesDrift((nothing for i = 1:15)...,
                                                             adapt(to, sssd.∂z_uˢ),
                                                             adapt(to, sssd.∂t_uˢ))

#####
##### Utilities
#####

# ρʷ τʷ = ρᵃ τᵃ
air_friction_velocity(τʷ, ρʷ, ρᵃ) = sqrt(ρʷ * τʷ / ρᵃ)
air_friction_velocity(sssd:ConstantWindStokesDrift) = air_friction_velocity(sssd.τʷ, sssd.ρʷ, sssd.ρᵃ)


function ConstantWindStokesDrift(grid; gravitational_acceleration = g_Earth, # m s⁻²
                                    Cᵝ = 0.105,  # Toba's constant
                                    Cʳ = 9.7e-3, # Transition wavenumber parameter, Lenain and Pizzo 2020 eq 4
                                    Cⁱ = 0.072,  # Cutoff / isotropic wavenumber parameter
                                                 # exp(π/2 - θ₀) / γ) from Lenain and Pizzo 2020 Appendix A
                                    Cᴮ = 7e-3,   # Saturation constant
                                    ρʷ = 1024,   # kg m⁻³, water density
                                    ρᵃ = 1.225,  # kg m⁻³, air density
                                    u★ = air_friction_velocity(τʷ, ρʷ, ρᵃ),
                                    kⁿ = Cʳ * g / u★^2, # Transition wavenumber
                                    kⁱ = Cⁱ * g / u★^2) # Isotropic wavenumber / upper wavenumber cutoff

       uˢ = Field{Nothing, Nothing, Face}(grid)
    ∂z_uˢ = Field{Nothing, Nothing, Center}(grid)
    ∂t_uˢ = Field{Nothing, Nothing, Center}(grid)

    return ConstantWindStokesDrift(Cᵝ, Cʳ, Cⁱ, Cᴮ, ρʷ, ρᵃ, τʷ, Cᵅ, Cᵡ, tᵢ, kⁿ, kⁱ,
                                      gravitational_acceleration, grid, uˢ, ∂z_uˢ, ∂t_uˢ)
end


const SSSD = ConstantWindStokesDrift

@inline ∂t_uˢ(i, j, k, grid, sd::SSSD, time) = @inbounds sd.∂t_uˢ[1, 1, k]
@inline ∂t_vˢ(i, j, k, grid, sd::SSSD, time) = zero(eltype(grid))
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, ::SSSD, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sd::SSSD, U, t) = @inbounds + ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * sd.∂z_uˢ[1, 1, k]
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sd::SSSD, U, t) = @inbounds - ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * ℑzᵃᵃᶠ(1, 1, k, grid, sd.∂z_uˢ)
@inline y_curl_Uˢ_cross_U(i, j, k, grid,   ::SSSD, U, t) = zero(eltype(grid))

#####
##### Stokes drift computation...
#####

function peak_wavenumber(sssd::ConstantWindStokesDrift, t)
    u★ = air_friction_velocity(sssd)
    tᵢ = sssd.tᵢ
    g = sssd.gravitational_acceleration
    c = sssd.Cᵡ
    α = sssd.Cᵅ
    d = c^(2 / (1 + α))
    β = 2α / (1 + α) 
    γ = 2 * (1 + 2α) / (1 + α)

    return d * 1/g * (g / u★)^γ * (t + tᵢ)^β
end

function peak_wavenumber_tendency(t, sssd, kᵖ=peak_wavenumber(sssd, t))
    tᵢ = sssd.tᵢ
     α = sssd.Cᵅ
    return = 2α / (1 + α) * kᵖ / (t + tᵢ)
end

# Equilibrium range contribution
ℓᵉ(k, z) = expinti(2k * z)
Lᵉ(kᵖ, kⁿ, z) = ℓᵉ(kⁿ, z) - ℓᵉ(kᵖ, z)

# Saturation range contribution
ℓˢ(k, z) = 2 / √k * (exp(2k * z) + √(2π * k * abs(z)) * erf(√(2k*abs(z))))
Lˢ(kⁿ, kⁱ, z) = ℓˢ(kⁿ, z) - ℓˢ(kⁱ, z)

function compute_stokes_drift!(sssd::ConstantWindStokesDrift, t)
    kᵖ = peak_wavenumber(sssd, t)
    ∂t_kᵖ = peak_wavenumber_tendency(sssd, t, kᵖ)

    kⁿ = sssd.kⁿ
    kⁱ = sssd.kⁱ
     g = sssd.gravitational_acceleration
    
    # Stokes drift according to Lenain and Pizzo (2020)
    uˢ(z) = u★ * Cᵝ * Lᵉ(kᵖ, kⁿ, z) + 2Cᴮ * √g * Lˢ(kⁿ, kⁱ, z)
    set!(sssd.uˢ, uˢ)
    sssd.∂z_uˢ .= ∂z(uˢ)

    ∂t_uˢ(z) = -exp(2kᵖ*z) * ∂t_kᵖ / kᵖ
    set!(sssd.∂t_uˢ, ∂t_uˢ)

    return nothing
end

@inline function ∂t_Uˢ(t, sssd)
    kᵖ = peak_wavenumber(sssd, t)
    ∂t_kᵖ = peak_wavenumber_tendency(sssd, t, kᵖ)
    return - ∂t_kᵖ / 2kᵖ^2
end

compute_stokes_drift!(simulation::Simulation, sssd) = compute_stokes_drift!(sssd, time(simulation)

function add_compute_stokes_drift_callback!(simulation, sssd)
    callback = Callback(compute_stokes_drift!, IterationInterval(1), parameters=sssd)
    simulation.callbacks[:compute_stokes_drift] = callback
    return nothing
end

