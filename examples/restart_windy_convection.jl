using Oceananigans, Oceananigans.OutputWriters, Oceananigans.Grids, Oceananigans.Fields,
      Oceananigans.Diagnostics, Oceananigans.AbstractOperations,
      Oceananigans.TimeSteppers

using PyPlot

using LESbrary

prefix = "windy_convection_Qu1.0e-04_Qb1.0e-08_Nsq1.0e-05_N64"

data = joinpath(@__DIR__, "..", "data", prefix)

iteration = 17932
filename = "windy_convection_Qu1.0e-04_Qb1.0e-08_Nsq1.0e-05_N64_checkpoint_iteration$iteration.jld2"
checkpoint = joinpath(data, filename)

model = restore_from_checkpoint(checkpoint)

statistics = LESbrary.Statistics.first_through_third_order(model)

simulation = Simulation(model, Δt=1e-2, stop_iteration=model.clock.iteration+3)

simulation.output_writers[:statistics] = JLD2OutputWriter(model, statistics,
                                                               force = true,
                                                           frequency = 1,
                                                                 dir = dirname(@__FILE__),
                                                              prefix = "restart_windy_convection")

print(model.clock.iteration)

run!(simulation)

#=
time_step!(model, 1e-6)

total_pressure = CellField(CPU(), model.grid)
@. total_pressure.data = model.pressures.pHY′.data + model.pressures.pNHS.data

pslice = interior(total_pressure)[:, 1, :]
wslice = interior(model.velocities.w)[:, 1, :]

pmax = maximum(abs, pslice)
plim = pmax / 10
levels = vcat([-pmax], range(-plim, stop=plim, length=10), [pmax])

close("all")
fig, axs = subplots(ncols=2)

sca(axs[1])
contourf(rotr90(pslice), vmin=-plim, vmax=plim, levels=levels, cmap="RdBu_r")
colorbar()

sca(axs[2])
contourf(rotr90(wslice))
colorbar()

# Create scratch space for calculations
scratch = CellField(model.architecture, model.grid)

# Extract short field names
u, v, w = model.velocities
θ, c = model.tracers
νₑ = model.diffusivities.νₑ

# Define horizontal averages
U = HorizontalAverage(u)
V = HorizontalAverage(v)
T = HorizontalAverage(θ)
C = HorizontalAverage(c)

▶T = HorizontalAverage(@at((Cell, Cell, Face), 1 * θ), scratch)
▶U = HorizontalAverage(@at((Cell, Cell, Face), 1 * u), scratch)

ph = model.pressures.pHY′
pn = model.pressures.pNHS

p = model.pressures.pHY′ + model.pressures.pNHS
P = HorizontalAverage(p, scratch)
Ph = HorizontalAverage(ph)
Pn = HorizontalAverage(pn)

# Vertical momentum balance:
#
# ∂t w + u ⋅ ∇ w + ∂z p - b = ...
#
# ∂t w + ∇ ⋅ (u w) + ∂z p - b = ∇ ⋅ τ₃
#
# The horizontal average is
#
# ∂z (w w) + ∂z P - B = ∂z τ₃₃
#
# We use a decomposition by which
#
# ∂z P_h - B = 0
#
# and
#
# ∂z (w w) + ∂z P_n = ∂z τ₃₃
#
∂z_w² = HorizontalAverage(@at((Cell, Cell, Face), ∂z(w * w)), scratch)

∂z_p = HorizontalAverage(∂z(p), scratch)

∂z_ph = HorizontalAverage(∂z(ph), scratch)
∂z_pn = HorizontalAverage(∂z(pn), scratch)

α = model.buoyancy.equation_of_state.α
g = model.buoyancy.gravitational_acceleration
B = HorizontalAverage(@at((Cell, Cell, Face), 0.001962 * θ), scratch)

τ₃₃ = @at (Cell, Cell, Cell) νₑ * (-∂z(w) - ∂z(w))

∂z_T₃₃ = HorizontalAverage(∂z(τ₃₃), scratch)

 Tz = HorizontalAverage(∂z(θ), scratch)
 w² = HorizontalAverage(w * w,     scratch)
 wθ = HorizontalAverage(w * θ,     scratch)
wwθ = HorizontalAverage(w * w * θ, scratch)
∂z_wwθ = HorizontalAverage(∂z(w * w * θ), scratch)

 wu = HorizontalAverage(w * u,     scratch)
wwu = HorizontalAverage(w * w * u, scratch)
wuu = HorizontalAverage(w * u * u, scratch)

w²U = w²(model) .* ▶U(model)
wwu′ = wwu(model) .- w²U

w²T = w²(model) .* ▶T(model)
wwθ′ = wwθ(model) .- w²T

wwTz = w²(model) .* Tz(model)

∂z_wwθ′ = @. (wwθ′[2:end] - wwθ′[1:end-1]) / model.grid.Δz

zC = znodes(Cell, model.grid)
zF = znodes(Face, model.grid)

fig, axs = subplots(nrows=3, ncols=5)

sca(axs[1, 1])
plot(U(model)[2:end-1], zC)
plot(V(model)[2:end-1], zC)

sca(axs[1, 2])
plot(wu(model)[2:end], zF)

sca(axs[1, 3])
plot(wwu(model)[2:end], zF)

sca(axs[2, 1])
plot(T(model)[2:end-1], zC)

sca(axs[2, 2])
plot(wθ(model)[2:end], zF)

sca(axs[2, 3])
#plot(∂z_wwθ(model)[2:end-1], zC, label=L"\overline{\partial_z(w w \theta)}")
plot(∂z_wwθ′[1:end-1], zC, label=L"\overline{\partial_z(w' w' \theta')}")
plot(wwTz[2:end], zF, label=L"\overline{w' w' \partial_z T}")
legend()

sca(axs[2, 4])
plot(wwθ(model)[2:end], zF, label=L"\overline{w' w' \theta}")
plot(w²T[2:end], zF, label=L"\overline{w'^2} T")
plot(wwθ′[2:end], zF, label=L"\overline{w' w' \theta'}")
legend()

sca(axs[2, 5])
plot(w′w′θ′[2:end], zF, label=L"\overline{w w \theta'}")

sca(axs[3, 1])
plot(Ph(model)[2:end-1], zC, label=L"\bar{p}_{hy}")
plot(Pn(model)[2:end-1], zC, label=L"\bar{p}_{nh}")

plot(P(model)[2:end-1], zC, label=L"\bar{p}")
legend()

sca(axs[3, 2])

plot( .- ∂z_p(model)[3:end], zF[2:end],  label=L"- \partial_z \bar{p}")
plot( .- ∂z_ph(model)[3:end], zF[2:end], label=L"- \partial_z \bar{p}_{hy}")
plot( .- ∂z_pn(model)[3:end], zF[2:end], label=L"- \partial_z \bar{p}_{nh}")

plot( ∂z_w²(model)[3:end], zF[2:end], label=L"\partial_z \overline{w'^2}")
plot(     B(model)[3:end], zF[2:end], label=L"B")
plot( .- ∂z_T₃₃(model)[3:end], zF[2:end], label=L"-\partial_z \tau_{zz}")

legend()
=#
