module TurbulenceStatistics

using Oceananigans,
      Oceananigans.AbstractOperations,
      Oceananigans.Architectures,
      Oceananigans.Diagnostics,
      Oceananigans.Grids,
      Oceananigans.Fields,
      Oceananigans.Operators

include("first_through_third_order.jl")

@inline ψ′(i, j, k, grid, ψ, Ψ) = @inbounds ψ[i, j, k] - Ψ[i, j, k]
@inline ψ′²(i, j, k, grid, ψ, Ψ) = @inbounds ψ′(i, j, k, grid, ψ, Ψ)^2
@inline ψ²(i, j, k, grid, ψ) = @inbounds ψ[i, j, k]^2
@inline w∂zΨ(i, j, k, grid, w, Ψ) = @inbounds w[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, Ψ)

include("turbulent_kinetic_energy.jl")
include("shear_production.jl")

include("turbulent_kinetic_energy_budget.jl")

end # module
