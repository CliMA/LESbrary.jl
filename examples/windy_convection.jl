# # Free convection

# This script runs a simulation of convection driven by cooling at the
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary, Random, Printf, Statistics

# # Set up the model

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(64, 64, 64), x=(0, 128), y=(0, 128), z=(-64, 0))

using Oceananigans.Buoyancy, Oceananigans.BoundaryConditions

Qᵇ = 1e-8
N² = 1e-6

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)
α, g = buoyancy.gravitational_acceleration, buoyancy.equation_of_state.α

  Qᶿ = Qᵇ / (α * g)
dθdz = N² / (α * g)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz))

c_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Value, 0),
                                       bottom = BoundaryCondition(Value, 1))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, -1e-4))

using Oceananigans
using CUDAapi: has_cuda

model = IncompressibleModel(architecture = has_cuda() ? GPU() : CPU(),
                                    grid = grid,
                                 tracers = (:T, :c),
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(),
                     boundary_conditions = (T=θ_bcs, u=u_bcs))

set!(model,
     c = 1,
     T = (x, y, z) -> dθdz * z + 1e-6 * dθdz * grid.Lz * exp(z / 8) * randn())

# # Prepare the simulation

using Oceananigans.Utils: hour, minute
using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.2, Δt=1e-1, max_change=1.1, max_Δt=10.0)

simulation = Simulation(model, Δt=wizard, stop_time=12hour, progress_frequency=100,
                        progress=SimulationProgressMessenger(model, wizard))

# Prepare Output

using Oceananigans.Utils: GiB
using Oceananigans.OutputWriters: FieldOutputs, JLD2OutputWriter
using LESbrary.Statistics: horizontal_averages

prefix = @sprintf("windy_convection_Qu%.1e_Qb%.1e_Nsq%.1e_N%d",
                  abs(u_bcs.z.top.condition), Qᵇ, N², grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# Three-dimensional field output
fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, FieldOutputs(merge(model.velocities, model.tracers));
                            force = true,
                         interval = 2hour, # every quarter period
                     max_filesize = 1GiB,
                              dir = data_directory,
                           prefix = prefix * "_fields")

# Horizontal averages
simulation.output_writers[:averages] =
    JLD2OutputWriter(model, LESbrary.Statistics.horizontal_averages(model);
                        force = true,
                     interval = 15minute,
                          dir = data_directory,
                       prefix = prefix * "_averages")

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

exit() # Release GPU memory
