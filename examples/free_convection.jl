# # Free convection

# This script runs a simulation of convection driven by cooling at the
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary, Printf, Statistics

# Domain

using Oceananigans.Grids

grid = RegularRectilinearGrid(size=(32, 32, 32), x=(0, 128), y=(0, 128), z=(-64, 0))

# Buoyancy and boundary conditions

using Oceananigans.Buoyancy, Oceananigans.BoundaryConditions

Qᵇ = 1e-7
N² = 1e-5

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

## Compute temperature flux and gradient from buoyancy flux and gradient
Qᵀ = Qᵇ / (α * g)
dTdz = N² / (α * g)

T_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵀ),
                                       bottom = BoundaryCondition(Gradient, dTdz))

# LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# Instantiate Oceananigans.IncompressibleModel

using Oceananigans
using Oceananigans.Advection: WENO5

model = IncompressibleModel(architecture = CPU(),
                             timestepper = :RungeKutta3,
                               advection = WENO5(),
                                    grid = grid,
                                 tracers = :T,
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                     boundary_conditions = (T=T_bcs,))

# # Initial condition

Ξ(z) = rand() * exp(z / 8)

Tᵢ(x, y, z) = dTdz * z + 1e-6 * Ξ(z) * dTdz * grid.Lz

set!(model, T=Tᵢ)

# # Prepare the simulation

using Oceananigans.Utils: hour, minute
using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=1.5, Δt=2.0, max_change=1.1, max_Δt=30.0)

simulation = Simulation(model, Δt=wizard, stop_time=8hour, iteration_interval=100,
                        progress=SimulationProgressMessenger(wizard))

# Prepare Output

using Oceananigans.Utils: GiB
using Oceananigans.OutputWriters

prefix = @sprintf("free_convection_Qb%.1e_Nsq%.1e_Nh%d_Nz%d", Qᵇ, N², grid.Nx, grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                                                           schedule = TimeInterval(4hour),
                                                             prefix = prefix * "_fields",
                                                                dir = data_directory,
                                                       max_filesize = 2GiB,
                                                              force = true)

simulation.output_writers[:slices] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                           schedule = AveragedTimeInterval(1hour, window=15minute),
                                                             prefix = prefix * "_slices",
                                                       field_slicer = FieldSlicer(j=floor(Int, grid.Ny/2)),
                                                                dir = data_directory,
                                                       max_filesize = 2GiB,
                                                              force = true)


# Horizontally-averaged turbulence statistics
turbulence_statistics = LESbrary.TurbulenceStatistics.first_through_second_order(model)
tke_budget_statistics = LESbrary.TurbulenceStatistics.turbulent_kinetic_energy_budget(model)

simulation.output_writers[:statistics] =
    JLD2OutputWriter(model, merge(turbulence_statistics, tke_budget_statistics),
                     schedule = AveragedTimeInterval(1hour, window=15minute),
                       prefix = prefix * "_statistics",
                          dir = data_directory,
                        force = true)

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

# # Load and plot turbulence statistics

using JLD2, Plots

## Some plot parameters
linewidth = 3
ylim = (-64, 0)
plot_size = (1000, 500)
zC = znodes(Center, grid)
zF = znodes(Face, grid)

## Load data
file = jldopen(simulation.output_writers[:statistics].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))
iter = iterations[end] # plot final iteration

## Temperature
T = file["timeseries/T/$iter"][1, 1, :]

## Velocity variances
w² = file["timeseries/ww/$iter"][1, 1, :]
e  = file["timeseries/e/$iter"][1, 1, :]

## Terms in the TKE budget
buoyancy_flux =   file["timeseries/tke_buoyancy_flux/$iter"][1, 1, :]
  dissipation = - file["timeseries/tke_dissipation/$iter"][1, 1, :]

 pressure_flux = - file["timeseries/tke_pressure_flux/$iter"][1, 1, :]
advective_flux = - file["timeseries/tke_advective_flux/$iter"][1, 1, :]

transport = zeros(grid.Nz)
transport = (pressure_flux[2:end] .+ advective_flux[2:end]
             .- pressure_flux[1:end-1] .- advective_flux[1:end-1]) / grid.Δz

## For mixing length calculation
wT = file["timeseries/wT/$iter"][1, 1, 2:end-1]

close(file)

## Post-process the data to determine the mixing length

## Mixing length, computed at cell interfaces and omitting boundaries
Tz = @. (T[2:end] - T[1:end-1]) / grid.Δz
bz = @. α * g * Tz
eᶠ = @. (e[1:end-1] + e[2:end]) / 2

## Mixing length model: wT ∝ - ℓᵀ √e ∂z T ⟹  ℓᵀ = wT / (√e ∂z T)
ℓ_measured = @. - wT / (√(eᶠ) * Tz)
ℓ_estimated = @. min(-zF[2:end-1], sqrt(eᶠ / max(0, bz)))

# Plot data

temperature = plot(T, zC, size = plot_size,
                     linewidth = linewidth,
                        xlabel = "Temperature (ᵒC)",
                        ylabel = "z (m)",
                          ylim = ylim,
                         label = nothing)

variances = plot(e, zC, size = plot_size,
                     linewidth = linewidth,
                        xlabel = "Velocity variances (m² s⁻²)",
                        ylabel = "z (m)",
                          ylim = ylim,
                         label = "(u² + v² + w²) / 2")

plot!(variances, 1/2 .* w², zF, linewidth = linewidth,
                                    label = "w² / 2")

budget = plot([buoyancy_flux dissipation transport], zC, size = plot_size,
              linewidth = linewidth,
                 xlabel = "TKE budget terms",
                 ylabel = "z (m)",
                   ylim = ylim,
                  label = ["buoyancy flux" "dissipation" "kinetic energy transport"])

mixing_length = plot([ℓ_measured ℓ_estimated], zF[2:end-1], size = plot_size,
                                                       linewidth = linewidth,
                                                          xlabel = "Mixing length (m)",
                                                          ylabel = "z (m)",
                                                            xlim = (-5, 20),
                                                            ylim = ylim,
                                                           label = ["measured" "estimated"])

plot(temperature, variances, budget, mixing_length, layout=(1, 4))
