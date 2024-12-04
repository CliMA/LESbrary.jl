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
include("temperature_salinity_constant_fluxes.jl")

six_hour_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 9.6e-7,            :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -5.0e-4, :buoyancy_flux => 8.0e-7,            :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -8.0e-4, :buoyancy_flux => 6.4e-7,            :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -1.2e-3, :buoyancy_flux => 4.0e-7,            :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -1.4e-3, :buoyancy_flux => 0.0,               :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -1.1e-3, :buoyancy_flux => 0.0,               :f => 0.0),
    :strong_wind_and_sunny    => Dict{Symbol, Any}(:momentum_flux => -1.5e-3, :penetrating_buoyancy_flux => -6e-7, :f => 0.0),
)

for (name, set) in six_hour_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 6hours
    set[:stokes_drift] = true
    set[:tracer_forcing_timescale] = 15minutes
end

twelve_hour_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 4.8e-7,            :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -4.0e-4, :buoyancy_flux => 4.0e-7,            :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -6.0e-4, :buoyancy_flux => 3.2e-7,            :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -8.0e-4, :buoyancy_flux => 2.0e-7,            :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -9.0e-4, :buoyancy_flux => 0.0,               :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -6.0e-4, :buoyancy_flux => 0.0,               :f => 0.0),
    :strong_wind_and_sunny    => Dict{Symbol, Any}(:momentum_flux => -9.0e-4, :penetrating_buoyancy_flux => -5e-7, :f => 0.0),
)

for (name, set) in twelve_hour_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 12hours
    set[:stokes_drift] = true
    set[:tracer_forcing_timescale] = 30minutes
end

twenty_four_hour_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 2.4e-7,            :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -3.0e-4, :buoyancy_flux => 2.0e-7,            :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -4.5e-4, :buoyancy_flux => 1.6e-7,            :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -5.9e-4, :buoyancy_flux => 1.0e-7,            :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -6.8e-4, :buoyancy_flux => 0.0,               :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -3.0e-4, :buoyancy_flux => 0.0,               :f => 0.0),
    :strong_wind_and_sunny    => Dict{Symbol, Any}(:momentum_flux => -4.5e-4, :penetrating_buoyancy_flux => -3e-7, :f => 0.0),
)

for (name, set) in twenty_four_hour_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 24hours
    set[:stokes_drift] = true
    set[:tracer_forcing_timescale] = 1hours
end

forty_eight_hour_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 1.2e-7,            :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -2.0e-4, :buoyancy_flux => 1.0e-7,            :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -3.4e-4, :buoyancy_flux => 8.0e-8,            :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -3.8e-4, :buoyancy_flux => 5.0e-8,            :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -4.5e-4, :buoyancy_flux => 0.0,               :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -1.6e-4, :buoyancy_flux => 0.0,               :f => 0.0),
    :strong_wind_and_sunny    => Dict{Symbol, Any}(:momentum_flux => -2.0e-4, :penetrating_buoyancy_flux => -1e-7, :f => 0.0),
)

for (name, set) in forty_eight_hour_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 48hours
    set[:stokes_drift] = true
    set[:tracer_forcing_timescale] = 2hours
end

seventy_two_hour_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 8.7e-8,            :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -1.8e-4, :buoyancy_flux => 7.5e-8,            :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -2.9e-4, :buoyancy_flux => 6.0e-8,            :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -3.4e-4, :buoyancy_flux => 3.8e-8,            :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -4.1e-4, :buoyancy_flux => 0.0,               :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -1.1e-4, :buoyancy_flux => 0.0,               :f => 0.0),
    :strong_wind_and_sunny    => Dict{Symbol, Any}(:momentum_flux => -1.3e-4, :penetrating_buoyancy_flux => -5e-8, :f => 0.0),
)

for (name, set) in seventy_two_hour_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 72hours
    set[:stokes_drift] = true
    set[:tracer_forcing_timescale] = 4hours
end

end # module

