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
    Oceananigans.Grids,
    Oceananigans.TurbulenceClosures,
    Oceananigans.Buoyancy

using Oceananigans.BoundaryConditions: UVelocityBoundaryFunction, VVelocityBoundaryFunction

using Random, Printf, Statistics, ArgParse

using Oceananigans: @hascuda

# # Argument parsing

#=
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


# # Numerical and physical parameters

## These parameters are set on the command line.
       Nh = args["Nh"]                # Number of grid points in x, y
       Nz = args["Nz"]                # Number of grid points in z
       Qᵇ = args["buoyancy_flux"]     # [m² s⁻³] Buoyancy flux at surface
       N² = args["buoyancy_gradient"] # [s⁻²] Initial buoyancy gradient
        f = args["coriolis"]          # [s⁻¹] Coriolis parameter
stop_time = args["hours"] * hour
=#

## The first thing we do is to select the GPU to run on as specified on the command line.
@hascuda LESbrary.Utils.select_device!(0)

## These parameters are set on the command line.
Nh = 32         # Number of grid points in x, y
Nz = 64         # Number of grid points in z

              thermocline_N² = 1e-5       # [s⁻²]
           thermocline_width = 40         # [m]
                     deep_N² = 1e-6       # [s⁻²]
          coriolis_parameter =      f = 1e-4         # [s⁻¹]
              maximum_stress = max_Qᵘ = 1e-3        # [m² s⁻²]
              storm_duration =      T = 1day             # [s]
   initial_mixed_layer_depth =      h = 40 # [m]     
  transition_layer_thickness =     Δh = 6  # [m]     
                   stop_time = 8 * 2π/f

## These parameters are 'fixed'.
Lh = 512                       # [m] Grid spacing in x, y (meters)
Lz = 256                       # [m] Grid spacing in z (meters)
θ₀ = 20.0                      # [ᵒC] Surface temperature

## Create the grid
grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, Lh), y=(0, Lh), z=(-Lz, 0))

#####
##### Buoyancy, equation of state, temperature flux, and initial temperature gradient
#####

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

       deep_dθdz = deep_N² / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)
thermocline_dθdz = thermocline_N² / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)

# # Near-wall LES diffusivity modification + temperature flux specification

stress_amplitude(t, p) = p.max_Qᵘ * p.e * t * exp(-t^2 / (2 * p.T)^2)
rotating_stress_x(x, y, t, p) = stress_amplitude(t, p) * cos(p.f * t)
rotating_stress_y(x, y, t, p) = stress_amplitude(t, p) * sin(p.f * t)

stress_x_wrapper = UVelocityBoundaryFunction(:z, rotating_stress_x, (max_Qᵘ=max_Qᵘ, f=f, T=T, e=exp(1)))
stress_y_wrapper = VVelocityBoundaryFunction(:z, rotating_stress_y, (max_Qᵘ=max_Qᵘ, f=f, T=T, e=exp(1)))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, stress_x_wrapper))
v_bcs = VVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, stress_y_wrapper))
θ_bcs = TracerBoundaryConditions(grid, bottom = BoundaryCondition(Gradient, deep_dθdz))

# # Sponge layer specification

using LESbrary.SpongeLayers: Fu, Fv, Fw, Fθ

τ = 120 # [s] Sponge layer damping time-scale
δ = 16  # [m] Sponge layer width

u_forcing = ParameterizedForcing(Fu, (δ=δ, τ=τ))
v_forcing = ParameterizedForcing(Fv, (δ=δ, τ=τ))
w_forcing = ParameterizedForcing(Fw, (δ=δ, τ=τ))
θ_forcing = ParameterizedForcing(Fθ, (δ=δ, τ=τ, dθdz=deep_dθdz, θ₀=θ₀ + thermocline_dθdz * h))

# # Model instantiation, initial condition, and model run

prefix = @sprintf("inertial_deepening_Qu%.1e_Nsq%.1e_Nh%d_Nz%d", max_Qᵘ, thermocline_N², Nh, Nz)

using CUDAapi: has_cuda

model = IncompressibleModel(       architecture = has_cuda() ? GPU() : CPU(),
                                           grid = grid,
                                        tracers = (:T,),
                                       buoyancy = buoyancy,
                                       coriolis = FPlane(f=f),
                                        closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (u=u_bcs, v=v_bcs, T=θ_bcs,),
                                        #forcing = ModelForcing(u=u_forcing, v=v_forcing, w=w_forcing, T=θ_forcing)
                           )

# # Initial condition

# ## Noise
ε₀ = 1e-6            # Non-dimensional noise amplitude
Lϵ = 2               # Decay scale
Δθ = thermocline_dθdz * Lz      # Temperature perturbation scale
u★ = sqrt(abs(max_Qᵘ))   # Vertical velocity scale

Ξ(ε₀, L, z) = ε₀ * randn() * z / Lz * exp(z / L) # rapidly decaying noise

step(z, z₀, δ) = 1/2 * (1 + tanh( (z₀ - z) / δ))
θᵢ(x, y, z) = θ₀ + Ξ(ε₀ * Δθ, Lϵ, z) + thermocline_dθdz * (z + h) * step(z, -(h + 3Δh), Δh)
uᵢ(x, y, z) = Ξ(ε₀ * u★, Lϵ, z)

Oceananigans.set!(model, T=θᵢ, u=uᵢ, v=uᵢ, w=uᵢ)

"Save a few things that we might want when we analyze the data."
function init(file, model; kwargs...)
    file["sponge_layer/δ"] = δ
    file["sponge_layer/τ"] = τ
    file["initial_conditions/deep_N²"] = deep_N²
    file["initial_conditions/thermocline_N²"] = thermocline_N²
    file["boundary_conditions/max_Qᵘ"] = max_Qᵘ
    return nothing
end

# # Prepare the simulation

using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(       cfl = 0.1,
                                Δt = 1.0,
                        max_change = 1.1,
                            max_Δt = 20.0)

messenger = SimulationProgressMessenger(model, wizard)

#simulation = Simulation(model, Δt=wizard, stop_time=stop_time, progress_frequency=100, progress=messenger)
simulation = Simulation(model, Δt=wizard, stop_iteration=1, progress_frequency=100, progress=messenger)

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
                                    interval = π/(2f), # every quarter period
                                max_filesize = 2GiB,
                                         dir = data_directory,
                                      prefix = prefix * "_fields")

simulation.output_writers[:fields] = field_writer


# Horizontal averages
averages_writer = JLD2OutputWriter(model, LESbrary.Statistics.horizontal_averages(model); 
                                      force = true, 
                                       init = init,
                                   interval = 15minute,
                                        dir = data_directory,
                                     prefix = prefix * "_averages")

simulation.output_writers[:averages] = averages_writer

# # Run

LESbrary.Utils.print_banner(simulation)

using PyPlot

close("all")
fig, axs = subplots(ncols=2, figsize=(10, 6))

function makeplot!(axs, model)

    u = model.velocities.u
    w = model.velocities.w

    xF, yC, zC = nodes(u)
    xC, yC, zF = nodes(w)

    Nx, Ny, Nz = size(u)

    xu = repeat(reshape(xF, Nx, 1), 1, Nz)
    zu = repeat(reshape(zC, 1, Nz), Nx, 1)

    xw = repeat(reshape(xC, Nx, 1), 1, Ny)
    yw = repeat(reshape(yC, 1, Ny), Nx, 1)

    T = dropdims(mean(model.tracers.T.data, dims=(1, 2)), dims=(1, 2))
    Tz = (T[2:end] - T[1:end-1]) / u.grid.Δz

    sca(axs[1]); cla()
    #plot(T[2:end], zC[:])
    plot(Tz, zF[1:end-1][:])
    #pcolormesh(xw, yw, interior(model.velocities.w)[:, :, end-1])

    sca(axs[2]); cla()
    pcolormesh(xu, zu, interior(model.velocities.u)[:, 8, :])

    title(@sprintf("\$ t = %.8f f / 2\\pi \$", model.clock.time * f / 2π))

    tight_layout()

    return nothing
end

while model.clock.time < stop_time
    simulation.stop_iteration += 100

    run!(simulation)

    makeplot!(axs, model)
end

exit() # Release GPU memory
