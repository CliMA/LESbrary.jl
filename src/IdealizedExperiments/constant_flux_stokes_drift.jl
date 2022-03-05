using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: Face, Center
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Operators: ℑxzᶠᵃᶜ, ℑxzᶜᵃᶠ, ℑzᵃᵃᶠ
using Oceananigans.Simulations: Simulation

using CUDA
using SpecialFunctions
using Adapt

import Oceananigans.StokesDrift: AbstractStokesDrift
import Oceananigans.StokesDrift: ∂t_uˢ, ∂t_vˢ, ∂t_wˢ,
                                 x_curl_Uˢ_cross_U,
                                 y_curl_Uˢ_cross_U,
                                 z_curl_Uˢ_cross_U

struct ConstantFluxStokesDrift{T, K, U, UZ}
                            Cᵝ :: T   # 1, Toba's constant
                            Cʳ :: T   # 2, Transition wavenumber parameter
                            Cⁱ :: T   # 3, Cutoff / isotropic wavenumber parameter
                            Cᴮ :: T   # 4, Saturation constant
                            kⁿ :: T   # 5, Transition wavenumber
                            kⁱ :: T   # 6, Isotropic wavenumber
    gravitational_acceleration :: T   # 7
                 water_density :: T   # 8, Water reference density
                   air_density :: T   # 9, (Constant) air reference density
         air_sea_momentum_flux :: T   # 10, Water-side kinematic momentum flux
         air_friction_velocity :: T   # 11, Water-side kinematic momentum flux
               peak_wavenumber :: K   # Wavenumber at spectral peak
                            uˢ :: U
                         ∂z_uˢ :: UZ

    function ConstantFluxStokesDrift{T}(args...) where T
        Nconst = 11
        constants = convert.(T, args[1:Nconst])
        kᵖ, uˢ, ∂z_uˢ = args[Nconst+1:end]
        K = typeof(kᵖ)
        U = typeof(uˢ)
        UZ = typeof(∂z_uˢ)
        return new{T, K, U, UZ}(constants..., args[Nconst+1:end]...)
    end
end

function Adapt.adapt_structure(to, cfsd::ConstantFluxStokesDrift)
    names = propertynames(cfsd)
    Nnames = length(names)
    return ConstantFluxStokesDrift((nothing for i = 1:Nnames-1)...,
                                    adapt(to, cfsd.∂z_uˢ))
end

#####
##### Utilities
#####

# LP2020: Lenain and Pizzo (JPO, 2020)
function ConstantFluxStokesDrift(grid, water_kinematic_momentum_flux, peak_wavenumber;
                                 gravitational_acceleration = g_Earth, # m s⁻²
                                 water_density = 1024,                 # kg m⁻³
                                 air_density = 1.225,                  # kg m⁻³
                                 Cᵝ = 0.105,  # Toba's constant
                                 Cʳ = 9.7e-3, # Transition wavenumber parameter, LP2020 eq 4
                                 Cⁱ = 0.072,  # Isotropic wavenumber parameter: exp(π/2 - θ₀) / γ) LP2020 App A
                                 Cᴮ = 7e-3)   # Saturation constant

    # Calculate transition and isotropic wavenumber
    air_sea_momentum_flux = water_density * water_kinematic_momentum_flux
    air_friction_velocity = a★ = sqrt(air_sea_momentum_flux / air_density)
    kⁿ = Cʳ * gravitational_acceleration / a★^2 # Transition wavenumber
    kⁱ = Cⁱ * gravitational_acceleration / a★^2 # Isotropic wavenumber cutoff

    uˢ = Field{Nothing, Nothing, Center}(grid)
    ∂z_uˢ = Field{Nothing, Nothing, Face}(grid)
    T = eltype(grid)

    cfsd = ConstantFluxStokesDrift{T}(Cᵝ, Cʳ, Cⁱ, Cᴮ, kⁿ, kⁱ,
                                      gravitational_acceleration,
                                      water_density,
                                      air_density,
                                      air_sea_momentum_flux,
                                      air_friction_velocity,
                                      peak_wavenumber,
                                      uˢ, ∂z_uˢ)

    compute_stokes_drift!(0, cfsd)

    return cfsd
end

const CFSD = ConstantFluxStokesDrift

@inline ∂t_uˢ(i, j, k, grid, sd::CFSD, time) = @inbounds sd.∂t_uˢ[1, 1, k]
@inline ∂t_vˢ(i, j, k, grid, sd::CFSD, time) = zero(eltype(grid))
@inline ∂t_wˢ(i, j, k, grid::AbstractGrid{FT}, ::CFSD, time) where FT = zero(FT)

@inline x_curl_Uˢ_cross_U(i, j, k, grid, sd::CFSD, U, t) = @inbounds +ℑxzᶠᵃᶜ(i, j, k, grid, U.w) * ℑzᵃᵃᶜ(1, 1, k, grid, sd.∂z_uˢ)
@inline z_curl_Uˢ_cross_U(i, j, k, grid, sd::CFSD, U, t) = @inbounds -ℑxzᶜᵃᶠ(i, j, k, grid, U.u) * sd.∂z_uˢ[1, 1, k]
@inline y_curl_Uˢ_cross_U(i, j, k, grid,   ::CFSD, U, t) = zero(eltype(grid))

#####
##### Stokes drift computation...
#####

peak_wavenumber(t, cfsd) = cfsd.peak_wavenumber

# Equilibrium range contribution
ℓᵉ(k, z) = expinti(2k * z)
Lᵉ(kᵖ, kⁿ, z) = ℓᵉ(kⁿ, z) - ℓᵉ(kᵖ, z)

# Saturation range contribution
ℓˢ(k, z) = 2 / √k * (exp(2k * z) + √(2π * k * abs(z)) * erf(√(2k*abs(z))))
Lˢ(kⁿ, kⁱ, z) = ℓˢ(kⁿ, z) - ℓˢ(kⁱ, z)

function compute_stokes_drift!(t, cfsd::ConstantFluxStokesDrift, kᵖ=peak_wavenumber(t, cfsd))
    a★ = cfsd.air_friction_velocity
    Cᵝ = cfsd.Cᵝ
    Cᴮ = cfsd.Cᴮ
    kⁿ = cfsd.kⁿ
    kⁱ = cfsd.kⁱ
     g = cfsd.gravitational_acceleration
    
    # Stokes drift according to Lenain and Pizzo (2020)
    uˢ(z) = a★ * Cᵝ * Lᵉ(kᵖ, kⁿ, z) + 2Cᴮ * √g * Lˢ(kⁿ, kⁱ, z)
    set!(cfsd.uˢ, uˢ)
    cfsd.∂z_uˢ .= ∂z(cfsd.uˢ)
    k_surface = cfsd.∂z_uˢ.grid.Nz + 1
    @inbounds CUDA.@allowscalar cfsd.∂z_uˢ[1, 1, k_surface] = Cᵝ * a★ * (kⁿ - kᵖ) + 4Cᴮ * √(2g) * (kⁱ - kⁿ)

    return nothing
end

#####
##### Experimental time-dependent peak wavenumber
#####

Base.@kwdef struct ConstantFluxPeakWavenumber{T}
    Cᵅ :: T = -0.287 # Lenain and Melville 2017
    Cᵡ :: T = 3.755  # Lenain and Melville 2017
    tᵏ :: T = 3600.0 # 1 hour, arbitrary...
end

# Time-dependent case --- use with caution, because we need to verify that ∂t_uˢ is small!
function peak_wavenumber(t, cfsd::ConstantFluxStokesDrift{<:Any, <:ConstantFluxPeakWavenumber})
    a★ = cfsd.air_friction_velocity
    g = cfsd.gravitational_acceleration
    tᵏ = cfsd.peak_wavenumber.time_displacement
    b = cfsd.peak_wavenumber.Cᵡ
    α = cfsd.peak_wavenumber.Cᵅ
    c = b^(2 / (1 + α))
    β = 2α / (1 + α) 
    γ = 2 * (1 + 2α) / (1 + α)

    return d * 1/g * (g / a★)^γ * (t + tᵏ)^β
end

