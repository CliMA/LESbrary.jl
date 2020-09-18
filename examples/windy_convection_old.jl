# # Free convection

# This script runs a simulation of convection driven by cooling at the
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary, Random, Printf, Statistics

# # Set up the model
#
# ## Grid

using Oceananigans.Grids

grid = RegularCartesianGrid(
                            size = (64, 64, 64),
                               x = (0, 128),
                               y = (0, 128),
                               z = (-64, 0)
                            )
# ## Boundary conditions

using Oceananigans.BoundaryConditions

N² = 1e-5

b_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, 1e-8),
                                       bottom = BoundaryCondition(Gradient, N²))

c_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Value, 0),
                                       bottom = BoundaryCondition(Value, 1))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, -1e-4))

# ## Tracer forcing

using Oceananigans.Utils: hour, minute
using Oceananigans.Forcing: Relaxation

c_forcing = Relaxation(; rate=1/hour, target=(x, y, z, t) -> 1)

# ## Stokes drift

struct SteadyStokesShear{T} <: Function
    a :: T
    k :: T
    g :: T

    function SteadyStokesShear(a, k, g=9.81; T=Float64)
        return new{T}(a, k, g)
    end
end

@inline (uˢ::SteadyStokesShear)(z, t) = 2 * (uˢ.a * uˢ.k)^2 * sqrt(uˢ.g * uˢ.k) * exp(2 * uˢ.k * z)

# ## Model instantiation

using Oceananigans
using Oceananigans.SurfaceWaves: UniformStokesDrift
using CUDA: has_cuda

model = IncompressibleModel(architecture = has_cuda() ? GPU() : CPU(),
                                    grid = grid,
                                 tracers = (:b, :c),
                                buoyancy = BuoyancyTracer(),
                                coriolis = FPlane(f=1e-4),
                           surface_waves = UniformStokesDrift(∂z_uˢ=SteadyStokesShear(0.8, 2π/60)),
                                 closure = AnisotropicMinimumDissipation(),
                     boundary_conditions = (u=u_bcs, b=b_bcs, c=c_bcs),
                                 forcing = ModelForcing(c=c_forcing))

# ## Initial condition

set!(model,
     c = 1,
     b = (x, y, z) -> N² * z + 1e-6 * N² * grid.Lz * exp(z / 8) * randn())

# # Prepare the simulation

using Oceananigans.Utils: hour, minute
using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.2, Δt=1e-1, max_change=1.1, max_Δt=10.0)

simulation = Simulation(model, Δt=wizard, stop_time=12hour, progress_frequency=100,
                        progress=SimulationProgressMessenger(model, wizard))

# ## Checkpointer

using Oceananigans.Utils: GiB
using Oceananigans.OutputWriters: Checkpointer, JLD2OutputWriter

prefix = @sprintf("windy_convection_Qu%.1e_Qb%.1e_Nsq%.1e_N%d",
                  abs(model.velocities.u.boundary_conditions.z.top.condition),
                  model.tracers.b.boundary_conditions.z.top.condition,
                  model.tracers.b.boundary_conditions.z.bottom.condition,
                  grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

statistics = LESbrary.Statistics.first_through_third_order(model)

simulation.output_writers[:checkpointer] = Checkpointer(model, force = true,
                                                            interval = 6hour, # every quarter period
                                                                 dir = data_directory,
                                                              prefix = prefix * "_checkpoint")

simulation.output_writers[:statistics] = JLD2OutputWriter(model, statistics,
                                                              force = true,
                                                           interval = 15minute,
                                                                dir = data_directory,
                                                             prefix = prefix * "_statistics")

# # Run
run!(simulation)

exit() # Release GPU memory
