#= run_free_convection.jl

This script runs a simulation of free convection. It is meant to be
used to 'spin up' a turbulent boundary layer, for use as an initial 
condition for simulations of boundary layer turbulence forced by the
growth of surface waves.
=#
using 
    LESbrary.SpongeLayers,
    LESbrary.Statistics,
    LESbrary.NearSurfaceTurbulenceModels,
    LESbrary.Utils

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

using CUDAapi: has_cuda

using Oceananigans: @hascuda

"Returns a dictionary of command line arguments."
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

@hascuda select_device!(args["device"])

# # Set numerical and physical parameters

# These parameters are set on the command line.
Nh = args["Nh"]                # Number of grid points in x, y
Nz = args["Nz"]                # Number of grid points in z
Qᵇ = args["buoyancy_flux"]     # [m² s⁻³] Buoyancy flux at surface
N² = args["buoyancy_gradient"] # [s⁻²] Initial buoyancy gradient

Lh = 128                       # [m] Grid spacing in x, y (meters)
Lz = 64                        # [m] Grid spacing in z (meters)
θ₀ = 20.0                      # [ᵒC] Surface temperature
 f = args["coriolis"]          # [s⁻¹] Coriolis parameter

# Create the grid 
grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, Lh), y=(0, Lh), z=(-Lz, 0))

# Calculate stop time as time when boundary layer depth is h = Lz/2.
# Uses a conservative estimate based on 
#
#   h ∼ √(2 * Qᵇ * stop_time / N²)

stop_time = args["hours"] * hour

# # Near-wall LES diffusivity modification + temperature flux specification

# Wall-aware AMD model constant
Δz = Lz/Nz
Cᴬᴹᴰ = SurfaceEnhancedModelConstant(Δz, C₀=1/12, enhancement=7, decay_scale=4Δz)

κₑ_bcs = SurfaceFluxDiffusivityBoundaryConditions(grid, Qᵇ; Cʷ=1.0)

κ₀ = κₑ_bcs.z.top.condition # surface diffusivity
dbdz_surface = - Qᵇ / κ₀    # set temperature gradient = - flux / diffusivity

b_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Gradient, dbdz_surface),
                                       bottom = BoundaryCondition(Gradient, N²))

# # Sponge layer

τ = 60  # [s] Sponge layer damping time-scale
δ = 4   # [m] Sponge layer width

u_forcing = ParameterizedForcing(Fu, (δ=δ, τ=τ))
v_forcing = ParameterizedForcing(Fv, (δ=δ, τ=τ))
w_forcing = ParameterizedForcing(Fw, (δ=δ, τ=τ))
b_forcing = ParameterizedForcing(Fb, (δ=δ, τ=τ, dbdz=N²))

# # Model instantiation, initial condition, and model run

prefix = @sprintf("free_convection_Qb%.1e_Nsq%.1e_stop%.1f_Nh%d_Nz%d", Qᵇ, N², 
                  stop_time / hour, Nh, Nz)

model = IncompressibleModel(       architecture = has_cuda() ? GPU() : CPU(),
                                           grid = grid,
                                        tracers = :b,
                                       buoyancy = BuoyancyTracer(),
                                       coriolis = FPlane(f=f),
                                        closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                            boundary_conditions = (b=b_bcs, κₑ=(b=κₑ_bcs,)),
                                        forcing = ModelForcing(u=u_forcing, v=v_forcing, w=w_forcing, b=b_forcing)
                           )

# Initial condition
ε₀ = 1e-6
Δb = N² * Lz
w★ = (Qᵇ * Lz)^(1/3)
Lϵ = 2 # noise decay scale, meters

Ξ(ε₀, L, z) = ε₀ * randn() * z / Lz * exp(z / L) # rapidly decaying noise

bᵢ(x, y, z) = Ξ(ε₀ * Δb, Lϵ, z) + N² * z
uᵢ(x, y, z) = Ξ(ε₀ * w★, Lϵ, z)

Oceananigans.set!(model, b=bᵢ, u=uᵢ, v=uᵢ, w=uᵢ)

"Save a few things that we might want when we analyze the data."
function init(file, model; kwargs...)
    file["sponge_layer/δ"] = δ
    file["sponge_layer/τ"] = τ
    file["initial_conditions/N²"] = N²
    file["boundary_conditions/Qᵇ"] = Qᵇ
    save_closure_parameters!(file, model.closure)
    return nothing
end

# # Prepare the simulation

# Adaptive time-stepping
wizard = TimeStepWizard(       cfl = 0.1,
                                Δt = 1e-1,
                        max_change = 1.1,
                            max_Δt = 10.0)

messenger = SimulationProgressMessenger(model, wizard)

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, progress_frequency=100, progress=messenger)

# # Output

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# Three-dimensional field output
fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,),
                         prefix_tuple_names(:κₑ, model.diffusivities.κₑ))

field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); force=true, init=init,
                                    interval = 4hour, # every quarter period
                                max_filesize = 2GiB,
                                         dir = data_directory,
                                      prefix = prefix * "_fields")

simulation.output_writers[:fields] = field_writer


# Horizontal averages
averages_writer = JLD2OutputWriter(model, horizontal_averages(model); force=true, init=init,
                                   interval = 10minute,
                                        dir = data_directory,
                                     prefix = prefix*"_averages")

simulation.output_writers[:averages] = averages_writer

# # Run

print_banner(simulation)

run!(simulation)

exit() # Release GPU memory
