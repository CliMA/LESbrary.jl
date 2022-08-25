module IdealizedExperiments

using Oceananigans.Units

# Code credit: https://discourse.julialang.org/t/collecting-all-output-from-shell-commands/15592
function execute(cmd::Cmd)
    out, err = Pipe(), Pipe()
    process = run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
    close(out.in)
    close(err.in)
    return (stdout = out |> read |> String, stderr = err |> read |> String, code = process.exitcode)
end

include("constant_flux_stokes_drift.jl")
include("three_layer_constant_fluxes.jl")

half_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 4.8e-7, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -6.0e-4, :buoyancy_flux => 4.0e-7, :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -9.0e-4, :buoyancy_flux => 3.2e-7, :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -1.2e-3, :buoyancy_flux => 2.0e-7, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -1.4e-3, :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -6.0e-4, :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in half_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 12hours
    set[:stokes_drift] = true
end

one_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 2.4e-7, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -3.0e-4, :buoyancy_flux => 2.0e-7, :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -4.5e-4, :buoyancy_flux => 1.6e-7, :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -5.9e-4, :buoyancy_flux => 1.0e-7, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -6.8e-4, :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -3.0e-4, :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in one_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 1days
    set[:stokes_drift] = true
end

two_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 1.2e-7, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -2.0e-4, :buoyancy_flux => 1.0e-7, :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -3.4e-4, :buoyancy_flux => 8.0e-8, :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -3.8e-4, :buoyancy_flux => 5.0e-8, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -4.5e-4, :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -1.6e-4, :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in two_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 2days
    set[:stokes_drift] = true
end

three_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 8.8e-8, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -1.8e-4, :buoyancy_flux => 7.5e-8, :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -3.0e-4, :buoyancy_flux => 6.0e-8, :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -3.4e-4, :buoyancy_flux => 3.8e-8, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -4.0e-4, :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -1.2e-4, :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in three_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 3days
    set[:stokes_drift] = true
end


four_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 5.5e-8, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -1.5e-4, :buoyancy_flux => 5.0e-8, :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -2.5e-4, :buoyancy_flux => 4.0e-8, :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -2.8e-4, :buoyancy_flux => 2.5e-8, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -3.2e-4, :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -1.0e-4, :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in four_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 4days
    set[:stokes_drift] = true
end

six_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 3.7e-8, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -7.0e-6, :buoyancy_flux => 3.3e-8, :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -1.0e-4, :buoyancy_flux => 3.0e-8, :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -1.5e-4, :buoyancy_flux => 1.6e-8, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -2.0e-4, :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -7.0e-6, :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in six_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 6days
    set[:stokes_drift] = false
end

end # module
