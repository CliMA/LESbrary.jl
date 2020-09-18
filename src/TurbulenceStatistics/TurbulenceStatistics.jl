module TurbulenceStatistics

using Oceananigans,
      Oceananigans.AbstractOperations,
      Oceananigans.Architectures,
      Oceananigans.Diagnostics,
      Oceananigans.Grids,
      Oceananigans.Fields,
      Oceananigans.Operators

include("turbulent_kinetic_energy.jl")
include("first_through_third_order.jl")

end # module
