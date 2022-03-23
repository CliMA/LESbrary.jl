using GLMakie
using CUDA
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using LESbrary.IdealizedExperiments: eddying_channel_simulation

#####
##### Setup and run the simulation
#####

#boundary_layer_closure = CATKEVerticalDiffusivity()
boundary_layer_closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1.0)

simulation = eddying_channel_simulation(; boundary_layer_closure,
                                        architecture = GPU(),
                                        peak_momentum_flux = 1.5e-4,
                                        size = (160, 80, 40),
                                        stop_time = 1year,
                                        vertical_grid_refinement = 8,
                                        initial_Δt = 20minutes,
                                        max_Δt = 20minutes)
 
wall_time = time_ns()

run!(simulation)

elapsed = 1e-9 * (time_ns() - wall_time)

@info "Simulation took " * prettytime(elapsed) * " to run."

#####
##### Visualization
#####

u, v, w = simulation.model.velocities
b = simulation.model.tracers.b
c = simulation.model.tracers.c

ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)

B = Field(Average(b, dims=1))
C = Field(Average(c, dims=1))
U = Field(Average(u, dims=1))

compute!(B)
compute!(C)
compute!(U)

Nz = simulation.model.grid.Nz
Bn = Array(interior(B, 1, :, :))
Cn = Array(interior(C, 1, :, :))
Un = Array(interior(U, 1, :, :))
bn = Array(interior(b, :, :, Nz))
ζn = Array(interior(ζ, :, :, Nz))

xb, yb, zb = nodes(b)
xu, yu, zu = nodes(u)
xζ, yζ, zζ = nodes(ζ)

zb = CUDA.@allowscalar Array(zb)
zu = CUDA.@allowscalar Array(zu)

fig = Figure(resolution=(1200, 800))

ax_b = Axis(fig[1, 1], aspect=2, xlabel="x", ylabel="y", title="Buoyancy")
ax_ζ = Axis(fig[1, 2], aspect=2, xlabel="x", ylabel="y", title="Vertical vorticity")

ζmax = maximum(abs, ζn)
ζlim = ζmax / 2
colorrange = (-ζlim, ζlim)
heatmap!(ax_b, xb, yb, bn) 
heatmap!(ax_ζ, xζ, yζ, ζn; colorrange, colormap=:redblue) 

ax_c = Axis(fig[2, 1], aspect=2, xlabel="y", ylabel="z", title="Tracer")
ax_u = Axis(fig[2, 2], aspect=2, xlabel="y", ylabel="z", title="Zonal velocity")

heatmap!(ax_c, yb, zb, Cn)
contour!(ax_c, yb, zb, Bn, levels=15, linewidth=2, color=(:black, 0.6)) 

umax = maximum(abs, Un)
ulim = umax
colorrange = (-ulim, ulim)
heatmap!(ax_u, yu, zu, Un; colormap=:redblue, colorrange)
contour!(ax_u, yb, zb, Bn, levels=15, linewidth=2, color=(:black, 0.6)) 

display(fig)
