# # Turbulent mixing of a three layer boundary layer driven by constant surface fluxes

# This script runs a simulation of a turbulent oceanic boundary layer with an initial
# three-layer temperature stratification. Turbulent mixing is driven by constant fluxes
# of momentum and heat at the surface.

using ArgParse

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--Nh"
            help = "The number of grid points in x, y."
            default = 32
            arg_type = Int

        "--Nz"
            help = "The number of grid points in z."
            default = 32
            arg_type = Int

        "--buoyancy-flux",
            help = """The surface buoyancy flux in units of m² s⁻³.
                      A positive buoyancy flux implies cooling.

                      Note:
                          buoyancy-flux = + 1e-7 corresponds to cooling at 208 W / m²
                          buoyancy-flux = - 1e-7 corresponds to heating at 208 W / m²
                   """
            default = 1e-7
            arg_type = Float64

        "--momentum-flux",
            help = """The surface x-momentum flux divided by density in units of m² s⁻².
                      A negative flux drives currents in the positive x-direction.

                      Note:
                        momentum-flux = - 1e-4 corresponds to U₁₀ = +6 m/s, roughly speaking
                        momentum-flux = - 1e-3 corresponds to U₁₀ = +19 m/s, roughly speaking
                      """

            default = -1e-4
            arg_type = Float64

        "--surface-temperature",
            help = """The temperature at the surface in ᵒC."""
            default = 20
            arg_type = Float64

        "--surface-layer-depth",
            help = """The depth of the surface layer in units of m."""
            default = 48
            arg_type = Float64

        "--thermocline-width",
            help = """The width of the thermocline in units of m."""
            default = 32
            arg_type = Float64

        "--surface-layer-buoyancy-gradient",
            help = """The buoyancy gradient in the surface layer in units s⁻²."""
            default = 1e-7
            arg_type = Float64

        "--thermocline-buoyancy-gradient",
            help = """The buoyancy gradient in the thermocline in units s⁻²."""
            default = 1e-5
            arg_type = Float64

        "--deep-buoyancy-gradient",
            help = """The buoyancy gradient below the thermocline in units s⁻²."""
            default = 1e-6
            arg_type = Float64

        "--device", "-d"
            help = "The CUDA device index on which to run the simulation."
            default = 0
            arg_type = Int
    end

    return parse_args(settings)
end

args = parse_command_line_arguments()

using LESbrary, Printf, Statistics

# Domain

using Oceananigans.Grids

Nh, Nz = args["Nh"], args["Nz"]

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, 512), y=(0, 512), z=(-256, 0))

# Buoyancy and boundary conditions

using Oceananigans.Buoyancy, Oceananigans.BoundaryConditions

Qᵘ = args["momentum_flux"]
Qᵇ = args["buoyancy_flux"]

thermocline_width = args["thermocline_width"]
surface_layer_depth = args["surface_layer_depth"]
thermocline_base = surface_layer_depth + thermocline_width

N²_surface_layer = args["surface_layer_buoyancy_gradient"]
N²_thermocline = args["thermocline_buoyancy_gradient"]
N²_deep = args["deep_buoyancy_gradient"]

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

Qᶿ = Qᵇ / (α * g)
dθdz_surface_layer = N²_surface_layer / (α * g)
dθdz_thermocline = N²_thermocline / (α * g)
dθdz_deep = N²_deep / (α * g)

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

θ_surface = args["surface_temperature"]

function initial_temperature(x, y, z)
    if z > -surface_layer_depth
        return θ_surface + dθdz_surface_layer * z + 1e-6 * Ξ(z) * dθdz_surface_layer * grid.Lz

    elseif z > -surface_layer_depth - thermocline_width

        θ_transition = θ_surface + surface_layer_depth * dθdz_surface_layer
        return θ_transition + dθdz_thermocline * z

    else

        θ_deep = (θ_surface 
                    + surface_layer_depth * dθdz_surface_layer 
                    + thermocline_width * dθdz_thermocline)

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
