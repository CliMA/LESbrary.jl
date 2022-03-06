# Example script for generating and analyzing LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: two_day_suite_parameters
using LESbrary.IdealizedExperiments: four_day_suite_parameters
using LESbrary.IdealizedExperiments: six_day_suite_parameters

# LESbrary parameters
# ===================
#
# Canonical LESbrary parameters are organized by the _intended duration_ of the simulation
# --- two days for the "two day suite", four days for the "four day suite", et cetera.
# The strength of the forcing is tuned roughly so that the boundary layer deepens to roughly
# 128 meters, half the depth of a 256 meter domain.
#
# Each suite has five cases:
#
# * :free_convection
# * :weak_wind_strong_cooling
# * :strong_wind_weak_cooling
# * :strong_wind
# * :strong_wind_no_rotation
#
# In addition to selecting the architecture, size, and case to run, we can also tweak
# certain parameters. Below we change the "snapshot_time_interval" (the interval over
# which slices of the simulation is saved) from the default 2 minutes to 1 minute
# (to make pretty movies), and we turn passive tracers off.

configuration = (;
    architecture = GPU(),
    size = (64, 64, 64),
    snapshot_time_interval = 1minute,
    passive_tracers = false,
    time_averaged_statistics = false,
    stokes_drift = true,
    stokes_drift_peak_wavenumber = 2π / 300 # m⁻¹
)

case = :weak_wind_strong_cooling

parameters = Dict(
    :name => "very_weak_wind_very_strong_cooling",
    :momentum_flux => -1e-4,
    :buoyancy_flux => 1e-7,
    :f => 1e-4,
    :stop_time => 2days,
)

#=
two_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 1.2e-7, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -1e-3,   :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -7e-4,   :buoyancy_flux => 6e-8,   :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -3.3e-4, :buoyancy_flux => 1.1e-7, :f => 1e-4),
    :strong_wind_weak_heating => Dict{Symbol, Any}(:momentum_flux => -1e-3,   :buoyancy_flux => -4e-8,  :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -2e-4,   :buoyancy_flux => 0.0,    :f => 0.0),
)
=#

simulation = three_layer_constant_fluxes_simulation(; configuration..., parameters...)

run!(simulation)

#=
using CairoMakie
using ElectronDisplay

filepath = simulation.output_writers[:statistics].filepath

u = FieldTimeSeries(filepath, "u", architecture=CPU())
v = FieldTimeSeries(filepath, "v", architecture=CPU())
T = FieldTimeSeries(filepath, "T", architecture=CPU())
e = FieldTimeSeries(filepath, "e", architecture=CPU())

z = znodes(u)

fig = Figure(resolution=(1800, 600))
ax_T = Axis(fig[1, 1])
ax_u = Axis(fig[1, 2])
ax_e = Axis(fig[1, 3])

n = length(u.times)

Tn = interior(T[n])[1, 1, :]
un = interior(u[n])[1, 1, :]
vn = interior(v[n])[1, 1, :]
en = interior(e[n])[1, 1, :]

lines!(ax_T, Tn, z)
lines!(ax_u, un, z)
lines!(ax_u, vn, z)
lines!(ax_e, en, z)

display(fig)
=#

