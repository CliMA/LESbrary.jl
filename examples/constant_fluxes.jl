# # Free convection

# This script runs a simulation of convection driven by cooling at the 
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary, Printf, Statistics

# Domain

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(64, 64, 64), x=(0, 512), y=(0, 512), z=(-256, 0))

# Buoyancy and boundary conditions

using Oceananigans.Buoyancy, Oceananigans.BoundaryConditions

#Qᵘ = - 1e-4 # U₁₀ = 6 m/s, roughly speaking
Qᵘ = - 1e-3 # U₁₀ = 19 m/s, roughly speaking

Qᵇ = + 1e-7 # cooling at 208 W / m²
#Qᵇ = - 1e-7 # heating at 208 W / m²

z_transition = -48
z_deep = -80

N²_shallow    = 1e-7
N²_transition = 1e-5
N²_deep       = 1e-6

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

Qᶿ = Qᵇ / (α * g)
dθdz_shallow    = N²_shallow    / (α * g)
dθdz_transition = N²_transition / (α * g)
dθdz_deep       = N²_deep       / (α * g)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz_deep))

u_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

c_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Value, 1.0),
                                       bottom = BoundaryCondition(Value, 0.0))

# Tracer forcing

struct SymmetricExponentialTarget{C}
         center :: C
    decay_scale :: C
end

SymmetricExponentialTarget(FT=Float64; decay_scale, center=0) =
    SymmetricExponentialTarget{FT}(center, decay_scale)

@inline (e::SymmetricExponentialTarget)(x, y, z, t) = exp(-abs(z - e.center) / e.decay_scale)

c_forcing = Relaxation(; rate=1/hour, target=SymmetricExponentialTarget(decay_scale=24))

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
                                 tracers = (:T, :c),
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                     boundary_conditions = (T=θ_bcs, u=u_bcs),
                                 forcing = ModelForcing(c=c_forcing))

# # Initial condition

Ξ(z) = rand() * exp(z / 8)

θ_surface = 20

function initial_temperature(x, y, z)
    if z > z_transition
        return θ_surface + dθdz_shallow * z + 1e-6 * Ξ(z) * dθdz_shallow * grid.Lz
    elseif z > z_deep
        θ_transition = θ_surface + z_transition * dθdz_shallow
        return θ_transition + dθdz_transition * z
    else
        θ_deep = θ_surface + z_transition * dθdz_shallow + (z_deep - z_transition) * dθdz_transition
        return θ_deep + dθdz_deep * z
    end
end

set!(model, T = initial_temperature, c = (x, y, z) -> c_forcing.target(x, y, z, 0))

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

prefix = @sprintf("constant_fluxes_Qu%.1e_Qb%.1e_Nh%d_Nz%d", abs(Qᵘ), Qᵇ, grid.Nx, grid.Nz)

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
#turbulence_statistics = LESbrary.TurbulenceStatistics.first_through_third_order(model)
tke_budget_statistics = LESbrary.TurbulenceStatistics.turbulent_kinetic_energy_budget(model)

simulation.output_writers[:statistics] = JLD2OutputWriter(model, tke_budget_statistics,
                                                          time_averaging_window = 5minute,
                                                                  time_interval = 1hour,
                                                                         prefix = prefix * "_statistics",
                                                                            dir = data_directory,
                                                                          force = true)

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

exit() # Release GPU memory
