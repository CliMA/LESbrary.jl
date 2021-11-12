module TurbulenceStatistics

using Oceananigans
using Oceananigans.AbstractOperations
using Oceananigans.Architectures
using Oceananigans.Diagnostics
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Operators
using Oceanostics

ViscousDissipation(model; data=nothing) = Oceanostics.IsotropicViscousDissipationRate(model)

include("first_through_third_order.jl")
include("subfilter_fluxes.jl")
include("turbulent_kinetic_energy_budget.jl")

end # module
