# # Free convection

# This script runs a simulation of convection driven by cooling at the 
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary

using
    Oceananigans,
    Oceananigans.Diagnostics,
    Oceananigans.OutputWriters,
    Oceananigans.Coriolis,
    Oceananigans.BoundaryConditions,
    Oceananigans.Forcing,
    Oceananigans.Utils,
    Oceananigans.TurbulenceClosures,
    Oceananigans.Buoyancy

using Random, Printf, Statistics, ArgParse

using Oceananigans: @hascuda

# # Argument parsing

"Returns a dictionary of parsed command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--Nh"
            help = "The number of grid points in x, y."
            default = 16
            arg_type = Int

        "--Nz"
            help = "The number of grid points in z."
            default = 16
            arg_type = Int

        "--buoyancy_flux", "-Q"
            help = """The surface buoyancy flux that drives convection in units of m² s⁻³. 
                      A positive buoyancy flux implies cooling."""
            default = 1e-9
            arg_type = Float64

        "--buoyancy_gradient"
            help = """The buoyancy gradient, or the square of the Brunt-Vaisala frequency N²,
                      at the start of the simulation in units s⁻²."""
            default = 1e-6
            arg_type = Float64

        "--coriolis"
            help = "The Coriolis parameter."
            default = 1e-4
            arg_type = Float64

        "--hours"
            help = "The stop time for the simulation in hours."
            default = 0.1
            arg_type = Float64

        "--device", "-d"
            help = "The CUDA device index on which to run the simulation."
            default = 0
            arg_type = Int
    end

    return parse_args(settings)
end

args = parse_command_line_arguments()

## The first thing we do is to select the GPU to run on as specified on the command line.
@hascuda LESbrary.Utils.select_device!(args["device"])

# # Numerical and physical parameters

## These parameters are set on the command line.
       Nh = args["Nh"]                # Number of grid points in x, y
       Nz = args["Nz"]                # Number of grid points in z
       Qᵇ = args["buoyancy_flux"]     # [m² s⁻³] Buoyancy flux at surface
       N² = args["buoyancy_gradient"] # [s⁻²] Initial buoyancy gradient
        f = args["coriolis"]          # [s⁻¹] Coriolis parameter
stop_time = args["hours"] * hour

## These parameters are 'fixed'.
Lh = 256                       # [m] Grid spacing in x, y (meters)
Lz = 128                       # [m] Grid spacing in z (meters)
θ₀ = 20.0                      # [ᵒC] Surface temperature

## Create the grid
grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, Lh), y=(0, Lh), z=(-Lz, 0))

#####
##### Buoyancy, equation of state, temperature flux, and initial temperature gradient
#####

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

   Qᶿ = Qᵇ / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)
dθdz₀ = N² / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)
 dθdz = dθdz₀

# # Near-wall LES diffusivity modification + temperature flux specification

using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant,
                                            SurfaceFluxDiffusivityBoundaryConditions,
                                            save_closure_parameters!

# We use a wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# This is necessary to obtain smooth buoyancy profiles near the boundary.
Δz = Lz/Nz
Cᴬᴹᴰ = SurfaceEnhancedModelConstant(Δz, C₀=1/12, enhancement=7, decay_scale=4Δz)

# In addition to using a surface-enhanced model constant, we provide a model for
# the eddy diffusivity at the surface.
κₑ_bcs = SurfaceFluxDiffusivityBoundaryConditions(grid, Qᵇ; Cʷ=1.0)

κ₀ = κₑ_bcs.z.top.condition # surface diffusivity
dθdz_surface = - Qᶿ / κ₀    # set temperature gradient = - flux / diffusivity

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Gradient, dθdz_surface),
                                       bottom = BoundaryCondition(Gradient, dθdz₀))

# # Sponge layer specification

using LESbrary.SpongeLayers: Fu, Fv, Fw, Fθ

τ = 60  # [s] Sponge layer damping time-scale
δ = 4   # [m] Sponge layer width

u_forcing = ParameterizedForcing(Fu, (δ=δ, τ=τ))
v_forcing = ParameterizedForcing(Fv, (δ=δ, τ=τ))
w_forcing = ParameterizedForcing(Fw, (δ=δ, τ=τ))
θ_forcing = ParameterizedForcing(Fθ, (δ=δ, τ=τ, dθdz=N²))

# # Model instantiation, initial condition, and model run

prefix = @sprintf("free_convection_Qb%.1e_Nsq%.1e_stop%.1f_Nh%d_Nz%d", Qᵇ, N², 
                  stop_time / hour, Nh, Nz)

using CUDAapi: has_cuda

model = IncompressibleModel(       architecture = has_cuda() ? GPU() : CPU(),
                                           grid = grid,
                                        tracers = (:T,),
                                       buoyancy = buoyancy,
                                       coriolis = FPlane(f=f),
                                        closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                            boundary_conditions = (θ=θ_bcs, κₑ=(θ=κₑ_bcs,)),
                                        forcing = ModelForcing(u=u_forcing, v=v_forcing, w=w_forcing, θ=θ_forcing)
                           )

# # Initial condition

# ## Noise
ε₀ = 1e-6               # Non-dimensional noise amplitude
Lϵ = 2                  # Decay scale
Δθ = dθdz₀ * Lz         # Temperature perturbation scale
w★ = (Qᵇ * Lz)^(1/3)    # Vertical velocity scale

Ξ(ε₀, L, z) = ε₀ * randn() * z / Lz * exp(z / L) # rapidly decaying noise

θᵢ(x, y, z) = Ξ(ε₀ * Δθ, Lϵ, z) + dθdz₀ * z
uᵢ(x, y, z) = Ξ(ε₀ * w★, Lϵ, z)

Oceananigans.set!(model, T=θᵢ, u=uᵢ, v=uᵢ, w=uᵢ)

"Save a few things that we might want when we analyze the data."
function init(file, model; kwargs...)
    file["sponge_layer/δ"] = δ
    file["sponge_layer/τ"] = τ
    file["initial_conditions/N²"] = N²
    file["boundary_conditions/Qᵇ"] = Qᵇ
    file["boundary_conditions/Qᶿ"] = Qᶿ
    save_closure_parameters!(file, model.closure)
    return nothing
end

# # Prepare the simulation

using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(       cfl = 0.1,
                                Δt = 1e-1,
                        max_change = 1.1,
                            max_Δt = 10.0)

messenger = SimulationProgressMessenger(model, wizard)

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, progress_frequency=100, progress=messenger)

# # Output

using LESbrary.Statistics: horizontal_averages

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# Three-dimensional field output
fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,),
                         LESbrary.Utils.prefix_tuple_names(:κₑ, model.diffusivities.κₑ))

field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); force=true, init=init,
                                    interval = 4hour, # every quarter period
                                max_filesize = 2GiB,
                                         dir = data_directory,
                                      prefix = prefix * "_fields")

simulation.output_writers[:fields] = field_writer


# Horizontal averages
averages_writer = JLD2OutputWriter(model, LESbrary.Statistics.horizontal_averages(model); 
                                      force = true, 
                                       init = init,
                                   interval = 10minute,
                                        dir = data_directory,
                                     prefix = prefix*"_averages")

simulation.output_writers[:averages] = averages_writer

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

exit() # Release GPU memory
