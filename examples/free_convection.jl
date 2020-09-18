# # Free convection

# This script runs a simulation of convection driven by cooling at the 
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary, Printf, Statistics

# Domain

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(64, 64, 64), x=(0, 128), y=(0, 128), z=(-64, 0))

# Buoyancy and boundary conditions

using Oceananigans.Buoyancy, Oceananigans.BoundaryConditions

Qᵇ = 1e-7
N² = 1e-5

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

  Qᶿ = Qᵇ / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)
dθdz = N² / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz))

# LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# Instantiate Oceananigans.IncompressibleModel

using Oceananigans

model = IncompressibleModel(architecture = CPU(),
                             timestepper = :RungeKutta3,
                                    grid = grid,
                                 tracers = :T,
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                     boundary_conditions = (T=θ_bcs,))

# # Initial condition

Ξ(z) = rand() * exp(z / 8)

θᵢ(x, y, z) = dθdz * z + 1e-6 * Ξ(z) * dθdz * grid.Lz

set!(model, T=θᵢ)

# # Prepare the simulation

using Oceananigans.Utils: hour, minute
using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.5, Δt=2.0, max_change=1.1, max_Δt=30.0)

simulation = Simulation(model, Δt=wizard, stop_time=8hour, iteration_interval=100, 
                        progress=SimulationProgressMessenger(model, wizard))

# Prepare Output

using Oceananigans.Utils: GiB
using Oceananigans.OutputWriters: JLD2OutputWriter

prefix = @sprintf("free_convection_Qb%.1e_Nsq%.1e_Nh%d_Nz%d", Qᵇ, N², grid.Nx, grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers); 
                                                      time_interval = 4hour, # every quarter period
                                                             prefix = prefix * "_fields",
                                                                dir = data_directory,
                                                       max_filesize = 2GiB,
                                                              force = true)
    
# Horizontally-averaged turbulence statistics
turbulence_statistics = LESbrary.TurbulenceStatistics.first_through_second_order(model)
tke_budget_statistics = LESbrary.TurbulenceStatistics.turbulent_kinetic_energy_budget(model)

simulation.output_writers[:statistics] =
    JLD2OutputWriter(model, merge(turbulence_statistics, tke_budget_statistics),
                     time_averaging_window = 5minute,
                             time_interval = 1hour,
                                    prefix = prefix * "_statistics",
                                       dir = data_directory,
                                     force = true)

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

# # Load and plot data

using JLD2, Plots

file = jldopen(simulation.output_writers[:statistics].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

z = znodes(Cell, grid)

linewidth = 3

E = file["timeseries/turbulent_kinetic_energy/$(iterations[end])"][1, 1, :]

tke_plot = plot(E, z,
                size = (1000, 1000),
                linewidth = linewidth,
                xlabel = "Turbulent kinetic energy / TKE (m² s⁻²)",
                ylabel = "z (m)",
                label = nothing)

# Terms in the TKE budget
      buoyancy_flux = file["timeseries/buoyancy_flux/$(iterations[end])"][1, 1, :]
   shear_production = file["timeseries/shear_production/$(iterations[end])"][1, 1, :]
        dissipation = file["timeseries/dissipation/$(iterations[end])"][1, 1, :]
 pressure_transport = file["timeseries/pressure_transport/$(iterations[end])"][1, 1, :]
advective_transport = file["timeseries/advective_transport/$(iterations[end])"][1, 1, :]

transport = pressure_transport .+ advective_transport

tke_budget_plot = plot([buoyancy_flux dissipation transport], z,
                             size = (1000, 1000),
                        linewidth = linewidth,
                           xlabel = "TKE budget terms",
                           ylabel = "z (m)",
                            label = ["buoyancy flux" "shear production" "dissipation" "transport"])

plot(tke_plot, tke_budget_plot, layout=(1, 2), label="Turbulence statistics")

# exit() # release GPU memory
