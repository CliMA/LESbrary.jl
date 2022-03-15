using Statistics
using Printf
using Logging
using JLD2
using NCDatasets
using GeoData
using Oceanostics
using Oceananigans
using Oceananigans.Units

using Oceanostics.TKEBudgetTerms: TurbulentKineticEnergy, ZShearProduction

using LESbrary.Utils: SimulationProgressMessenger, fit_cubic, poly
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant
using LESbrary.TurbulenceStatistics: first_order_statistics,
                                     first_through_second_order,
                                     turbulent_kinetic_energy_budget,
                                     subfilter_momentum_fluxes,
                                     subfilter_tracer_fluxes,
                                     ViscousDissipation

try
    using CairoMakie
catch
    using GLMakie
finally
    @warn "Could not load either CairoMakie or GLMakie; animations are not available."
end

Logging.global_logger(OceananigansLogger())

@inline passive_tracer_forcing(x, y, z, t, p) = p.μ⁺ * exp(-(z - p.z₀)^2 / (2 * p.λ^2)) - p.μ⁻

# Code credit: https://discourse.julialang.org/t/collecting-all-output-from-shell-commands/15592
function execute(cmd::Cmd)
    out, err = Pipe(), Pipe()
    process = run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
    close(out.in)
    close(err.in)
    return (stdout = out |> read |> String, stderr = err |> read |> String, code = process.exitcode)
end

function eddying_channel_simulation(;
  )

    return filepath
end
