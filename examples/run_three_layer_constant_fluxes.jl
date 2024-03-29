# Example script for generating and analyzing LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: two_day_suite_parameters

using GLMakie
#using CairoMakie
#using ElectronDisplay

# LESbrary parameters
# ===================
#
# Canonical LESbrary parameters are organized by the _intended duration_ of the simulation
# --- two days for the "two day suite", four days for the "four day suite", et cetera.
# The strength of the forcing is tuned roughly so that the boundary layer deepens to roughly
# 128 meters, half the depth of a 256 meter domain.
#
# Each suite has six cases:
#
# * :free_convection
# * :weak_wind_strong_cooling
# * :med_wind_med_cooling
# * :strong_wind_weak_cooling
# * :strong_wind
# * :strong_wind_no_rotation
#
# In addition to selecting the architecture, size, and case to run, we can also tweak
# certain parameters below.

configuration = (;
    architecture = GPU(),
    size = (128, 128, 128),
    snapshot_time_interval = 10minutes,
    passive_tracers = false,
    time_averaged_statistics = false,
)

half_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 4.8e-7, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -5.0e-4, :buoyancy_flux => 4.0e-7, :f => 1e-4),
    :med_wind_med_cooling     => Dict{Symbol, Any}(:momentum_flux => -7.0e-4, :buoyancy_flux => 3.2e-7, :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -8.0e-4, :buoyancy_flux => 2.0e-7, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -1.0e-3, :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -6.0e-4, :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in half_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 12hours
    set[:stokes_drift] = true
end

p = half_day_suite_parameters[:strong_wind_no_rotation]

@show "Running with $p..."
simulation = three_layer_constant_fluxes_simulation(; configuration..., p...)

run!(simulation)

filepath = simulation.output_writers[:statistics].filepath

u = FieldTimeSeries(filepath, "u", architecture=CPU())
v = FieldTimeSeries(filepath, "v", architecture=CPU())
T = FieldTimeSeries(filepath, "T", architecture=CPU())
e = FieldTimeSeries(filepath, "e", architecture=CPU())
Nt = length(e.times)

z = znodes(u)

fig = Figure(resolution=(1800, 600))
ax_T = Axis(fig[2, 1], xlabel="Temperature (ᵒC)", ylabel="z (m)")
ax_u = Axis(fig[2, 2], xlabel="Velocity (cm s⁻¹)", ylabel="z (m)")
ax_e = Axis(fig[2, 3], xlabel="Turbulent kinetic energy (cm² s⁻²)", ylabel="z (m)")

slider = Slider(fig[3, 1:3], range=1:Nt, startvalue=1)
n = slider.value

name = replace(string(p[:name]), "_" => " ")
title = @lift string(name, " at t = ", prettytime(e.times[$n]))
Label(fig[1, 1:3], title)

Tn = @lift interior(T[$n], 1, 1, :)
un = @lift interior(u[$n], 1, 1, :)
vn = @lift interior(v[$n], 1, 1, :)
en = @lift interior(e[$n], 1, 1, :)

@show Tmin = minimum(minimum(T[n]) for n in 1:Nt)
@show Tmax = maximum(minimum(T[n]) for n in 1:Nt)
hlines!(ax_T, -128, xmin=Tmin, xmax=Tmax, color=:gray23)
lines!(ax_T, Tn, z)

umin = minimum(minimum(u[n]) for n in 1:Nt)
umax = maximum(maximum(u[n]) for n in 1:Nt)
vmin = minimum(minimum(v[n]) for n in 1:Nt)
vmax = maximum(maximum(v[n]) for n in 1:Nt)

@show umin = min(umin, vmin)
@show umax = max(umax, vmax)

hlines!(ax_u, -128, xmin=umin, xmax=umax, color=:gray23)
lines!(ax_u, un, z)
lines!(ax_u, vn, z)

@show emax = maximum(maximum(e[n]) for n in 1:Nt)
hlines!(ax_e, -128, xmin=-0.1emax, xmax=emax, color=:gray23)
scatter!(ax_e, en, z)
xlims!(ax_e, -0.1emax, emax)

display(fig)
