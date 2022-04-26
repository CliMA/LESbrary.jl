module TurbulenceStatistics

using Oceananigans
using Oceananigans.AbstractOperations
using Oceananigans.Architectures
using Oceananigans.Diagnostics
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Operators
using Oceanostics

ViscousDissipation(model; kw...) = Oceanostics.IsotropicViscousDissipationRate(model)

include("first_through_third_order.jl")
include("subfilter_fluxes.jl")
include("turbulent_kinetic_energy_budget.jl")
include("uiuj_budget.jl")
include("uic_budget.jl")

end # module
