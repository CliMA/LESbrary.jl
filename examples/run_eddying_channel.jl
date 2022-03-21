using GLMakie
using Oceananigans
using Oceananigans.Units
using LESbrary.IdealizedExperiments: eddying_channel_simulation

simulation = eddying_channel_simulation(architecture=GPU(), stop_time=2years)
 
run!(simulation)

u, v, w = simulation.model.velocities
ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)

b = simulation.model.tracers.b
Nz = simulation.model.grid.Nz
bn = Array(interior(b, :, :, Nz))
ζn = Array(interior(ζ, :, :, Nz))

fig = Figure(resolution=(1200, 800))
ax_b = Axis(fig[1, 1])
ax_ζ = Axis(fig[1, 2])
heatmap!(ax_b, bn) 
heatmap!(ax_ζ, ζn) 

display(fig)
