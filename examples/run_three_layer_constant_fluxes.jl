# Example script for generating and analyzing LESbrary data

using Oceananigans
using Oceananigans.Units

using LESbrary.IdealizedExperiments: three_layer_constant_fluxes_simulation
using LESbrary.IdealizedExperiments: two_day_suite_parameters

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
# certain parameters below.

configuration = (;
    architecture = GPU(),
    size = (128, 128, 128),
    snapshot_time_interval = 2minute,
    passive_tracers = true,
    time_averaged_statistics = false,
)

for parameters in values(two_day_suite_parameters)
    @show "Running with $parameters..."
    simulation = three_layer_constant_fluxes_simulation(; configuration..., parameters...)
    run!(simulation)
end

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
ax_T = Axis(fig[1, 1], xlabel="Temperature (ᵒC)", ylabel="z (m)", title=parameters[:name])
ax_u = Axis(fig[1, 2], xlabel="Velocity (cm s⁻¹)", ylabel="z (m)", title=parameters[:name])
ax_e = Axis(fig[1, 3], xlabel="Turbulent kinetic energy (cm² s⁻²)", ylabel="z (m)", title=parameters[:name])

n = length(u.times)

Tn = interior(T[n])[1, 1, :]
un = interior(u[n])[1, 1, :]
vn = interior(v[n])[1, 1, :]
en = interior(e[n])[1, 1, :]

@show Tmin = minimum(Tn)
@show Tmax = maximum(Tn)
hlines!(ax_T, -128, xmin=Tmin, xmax=Tmax, color=:gray23)
lines!(ax_T, Tn, z)

@show umin = min(minimum(1e2 * un), minimum(1e2 * vn))
@show umax = max(maximum(1e2 * un), maximum(1e2 * vn))
hlines!(ax_u, -128, xmin=umin, xmax=umax, color=:gray23)
lines!(ax_u, 1e2 * un, z)
lines!(ax_u, 1e2 * vn, z)

emin = minimum(1e4 * en)
emax = maximum(1e4 * en)
hlines!(ax_e, -128, xmin=emin, xmax=emax, color=:gray23)
lines!(ax_e, 1e4 * en, z)

display(fig)
=#
