# # Free convection

# This script runs a simulation of convection driven by cooling at the
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary
using Printf
using Statistics
using Oceananigans
using Oceananigans.Units
using LESbrary.Utils: SimulationProgressMessenger

# Domain

Nx = Ny = Nz = 128
grid = RectilinearGrid(GPU(), size=(Nx, Ny, Nz), x=(0, 512), y=(0, 512), z=(-256, 0))

# Buoyancy and boundary conditions

Qᵇ = 1e-7
N² = 1e-5

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            buoyancy = BuoyancyTracer(),
                            advection = WENO(),
                            tracers = :b,
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (; b=b_bcs))

# # Initial condition

Ξ(z) = rand() * exp(z / 8)
bᵢ(x, y, z) = N² * z + 1e-6 * Ξ(z) * N² * grid.Lz
set!(model, b=bᵢ)

# # Prepare the simulation

# Adaptive time-stepping

simulation = Simulation(model, Δt=2.0, stop_time=8hour)

wizard = TimeStepWizard(cfl=1.5, max_change=1.1, max_Δt=30.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))
simulation.callbacks[:progress] = Callback(SimulationProgressMessenger(wizard), IterationInterval(100))

# Prepare Output

filename = @sprintf("free_convection_Qb%.1e_Nsq%.1e_Nh%d_Nz%d", Qᵇ, N², grid.Nx, grid.Nz)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                                                      schedule = TimeInterval(4hour),
                                                      filename = filename * "_fields",
                                                      overwrite_existing = true)

simulation.output_writers[:slices] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = AveragedTimeInterval(1hour, window=15minute),
                                                      filename = filename * "_slices",
                                                      indices = (:, floor(Int, grid.Ny/2), :),
                                                      overwrite_existing = true)

# Horizontally-averaged turbulence statistics
turbulence_statistics = LESbrary.TurbulenceStatistics.first_through_second_order(model)
tke_budget_statistics = LESbrary.TurbulenceStatistics.turbulent_kinetic_energy_budget(model)

simulation.output_writers[:statistics] =
    JLD2OutputWriter(model, merge(turbulence_statistics, tke_budget_statistics),
                     schedule = AveragedTimeInterval(1hour, window=15minute),
                     filename = filename * "_statistics",
                     overwrite_existing = true)

# # Run

run!(simulation)

