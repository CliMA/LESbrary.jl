using Logging
using Printf
using ArgParse
using CUDA

using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators
using SeawaterPolynomials.TEOS10

using Dates: Date, DateTime, Second, Millisecond, now, format
using Oceananigans: Center # Not sure why I need this.
using Oceananigans.Architectures: array_type
using Oceananigans.BuoyancyModels: BuoyancyField
using RealisticLESbrary: ∂z, ∂t

Logging.global_logger(OceananigansLogger())

# We want to time how long simulation setup takes.
t0 = time_ns()

# Remove this fix once https://github.com/CliMA/Oceananigans.jl/pull/1642
# makes it into a tagged release.

import Adapt
using Oceananigans.Forcings: DiscreteForcing

Adapt.adapt_structure(to, forcing::DiscreteForcing) =
    DiscreteForcing(Adapt.adapt(to, forcing.func),
                    Adapt.adapt(to, forcing.parameters))

include("load_sose_data.jl")
include("interpolate_sose_data.jl")
include("make_plots_and_movies.jl")
include("mixed_layer_depth.jl")

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--architecture"
            help = "Architecture to run on: CPU (default) or GPU"
            default = "CPU"
            arg_type = String

        "--size"
            help = "The number of grid points in x, y, and z."
            nargs = 3
            arg_type = Int

        "--extent"
            help = "The length of the x, y, and z dimensions."
            nargs = 3
            arg_type = Float64

        "--latitude"
            help = "Site latitude in degrees North (°N). Note that the SOSE dataset only goes up to -30°N."
            arg_type = Float64

        "--longitude"
            help = "Site longitude in degrees East (°E) between 0°E and 360°E."
            arg_type = Float64

        "--start"
            help = "Start date (format YYYY-mm-dd) between 2013-01-01 and 2018-01-01."
            arg_type = String

        "--end"
            help = "End date (format YYYY-mm-dd) between 2013-01-01 and 2018-01-01. Must be after the start date."
            arg_type = String

        "--sose-dir"
            help = "Directory containing the SOSE datasets."
            arg_type = String

        "--output-dir"
            help = "Directory to write output to. Defaults to the current working directory."
            default = pwd()
            arg_type = String

        "--no-surface-forcing"
            help = "Disable surface forcing of momentum, temperature, and salinity."
            action = :store_true

        "--no-background-state"
            help = "Disable the geostrophic background state and relaxation of temperature and salinity."
            action = :store_true

        "--no-animate"
            help = "Do not produce plots and animations after the simulation has finished."
            action = :store_true
    end

    return parse_args(settings)
end


@info "Parsing command line arguments..."

args = parse_command_line_arguments()

Nx, Ny, Nz = args["size"]
Lx, Ly, Lz = args["extent"]
lat = args["latitude"]
lon = args["longitude"]

start_date = Date(args["start"])
end_date = Date(args["end"])
validate_sose_dates(start_date, end_date)
day_offset, n_days = dates_to_offset(start_date, end_date)

arch_str2type = Dict("CPU" => CPU(), "GPU" => GPU())
arch = arch_str2type[args["architecture"]]

sose_dir = args["sose-dir"]
output_dir = abspath(args["output-dir"])
mkpath(output_dir)

apply_surface_forcing = !args["no-surface-forcing"]
apply_background_state = !args["no-background-state"]
make_animation = !args["no-animate"]

@info "Mapping grid..."

topology = (Periodic, Periodic, Bounded)
grid = RegularRectilinearGrid(topology=topology, size=(Nx, Ny, Nz), x=(0.0, Lx), y=(0.0, Ly), z=(-Lz, 0.0), halo=(3, 3, 3))


@info "Spinning up a tangent plane..."

coriolis = FPlane(latitude=lat)


@info "Surfacing a buoyancy model..."

buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())


@info "Summoning SOSE data and diagnosing geostrophic background state..."

# Don't pass an array_type to keep this data on the CPU until we interpolate it.
sose_datetimes, sose_grid, sose_surface_forcings, sose_profiles =
    load_sose_data(sose_dir, lat, lon, day_offset, n_days, grid, buoyancy, coriolis)

dates = convert.(Date, sose_datetimes)
start_date = dates[day_offset]
stop_date = dates[day_offset + n_days]
@info "Simulation start date: $start_date, stop date: $stop_date"


@info "Interpolating SOSE data..."

times = day * (0:n_days)
interpolated_surface_forcings = interpolate_surface_forcings(sose_surface_forcings, times, array_type=array_type(arch))
interpolated_profiles = interpolate_profiles(sose_profiles, sose_grid, grid, times, array_type=array_type(arch))

# We need a CPU version for set! and output writing (to avoid CUDA scalar operations).
interpolated_surface_forcings_cpu = interpolate_surface_forcings(sose_surface_forcings, times)
interpolated_profiles_cpu = interpolate_profiles(sose_profiles, sose_grid, grid, times)


@info "Plotting initial state for inspection..."

plot_initial_args = (sose_profiles, sose_grid, interpolated_profiles, grid, lat, lon, start_date)
CUDA.@allowscalar plot_initial_state(plot_initial_args..., z_bottom=-Lz, filepath=joinpath(output_dir, "initial_state.png"))
CUDA.@allowscalar plot_initial_state(plot_initial_args..., z_bottom=max(-10Lz, minimum(sose_grid.zF)), filepath=joinpath(output_dir, "initial_state_deep.png"))


@info "Forcing mean-flow interactions and relaxing tracers..."

## Set up forcing forcings to
##   1. include mean flow interactions in the momentum equation, and
##   2. weakly relax tracer fields to the base state.

# Fu′ = - w′∂z(U) - U∂x(u′) - V∂y(u′)
# FIXME? Do we need to use the flux form operator ∇·(Uũ′) instead of ũ′·∇U ?
@inline Fu′(i, j, k, grid, t, u′, v′, w′, p) =
    @inbounds - w′[i, j, k] * ∂z(p.ℑU, grid.zC[k], t) - p.ℑU(grid.zC[k], t) * ∂xᶜᵃᵃ(i, j, k, grid, u′) - p.ℑV(grid.zC[k], t) * ∂yᵃᶜᵃ(i, j, k, grid, u′)

# Fv′ = - w′∂z(V) - U∂x(v′) - V∂y(v′)
@inline Fv′(i, j, k, grid, t, u′, v′, w′, p) =
    @inbounds - w′[i, j, k] * ∂z(p.ℑV, grid.zC[k], t) - p.ℑU(grid.zC[k], t) * ∂xᶜᵃᵃ(i, j, k, grid, v′) - p.ℑV(grid.zC[k], t) * ∂yᵃᶜᵃ(i, j, k, grid, v′)

# Fw′ = - U∂x(w′) - V∂y(w′)
@inline Fw′(i, j, k, grid, t, u′, v′, w′, p) =
    @inbounds - p.ℑU(grid.zC[k], t) * ∂xᶜᵃᵃ(i, j, k, grid, u′) - p.ℑV(grid.zC[k], t) * ∂yᵃᶜᵃ(i, j, k, grid, v′)

# Fθ′ = - ∂t(Θ) - U∂x(θ′) - V∂y(θ′) - w′∂z(Θ)
# FIXME? Do we need to include the ∂t(Θ) terms?
@inline Fθ′(i, j, k, grid, t, u′, v′, w′, T′, p) =
    @inbounds - p.ℑU(grid.zC[k], t) * ∂xᶠᵃᵃ(i, j, k, grid, T′) - p.ℑV(grid.zC[k], t) * ∂yᵃᶜᵃ(i, j, k, grid, T′) - w′[i, j, k] * ∂z(p.ℑΘ, grid.zC[k], t)

# Fs′ = - ∂t(S) - U∂x(s′) - V∂y(s′) - w′∂z(S)
@inline Fs′(i, j, k, grid, t, u′, v′, w′, S′, p) =
    @inbounds - p.ℑU(grid.zC[k], t) * ∂xᶠᵃᵃ(i, j, k, grid, S′) - p.ℑV(grid.zC[k], t) * ∂yᵃᶜᵃ(i, j, k, grid, S′) - w′[i, j, k] * ∂z(p.ℑS, grid.zC[k], t)

# Timescale for relaxation to large-scale solution.
week = 7days
μ = (T=1/week, S=1/week)

# FIXME: Should be μ(C - c̅) so I need to add horizontal averages to parameters.
@inline Fθ_μ(i, j, k, grid, t, T′, p) = @inbounds p.μ.T * (p.ℑΘ(grid.zC[k], t) - T′[i, j, k])
@inline FS_μ(i, j, k, grid, t, S′, p) = @inbounds p.μ.S * (p.ℑS(grid.zC[k], t) - S′[i, j, k])

@inline u_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fu′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, params)
@inline v_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fv′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, params)
@inline w_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fw′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, params)
@inline T_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fθ′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, fields.T, params) + Fθ_μ(i, j, k, grid, clock.time, fields.T, params)
@inline S_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fs′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, fields.S, params) + FS_μ(i, j, k, grid, clock.time, fields.S, params)

ℑτx = interpolated_surface_forcings.τx
ℑτy = interpolated_surface_forcings.τy
ℑQθ = interpolated_surface_forcings.Qθ
ℑQs = interpolated_surface_forcings.Qs
 ℑU = interpolated_profiles.Ugeo
 ℑV = interpolated_profiles.Vgeo
 ℑΘ = interpolated_profiles.Θ
 ℑS = interpolated_profiles.S

# Only give each forcing function the parameters it needs to avoid an
# ERROR: CUDA error: a PTX JIT compilation failed
# due to kernel arguments taking up too much parameter space on the GPU.

forcings = (
    u = Forcing(u_forcing_wrapper, discrete_form=true, parameters=(; ℑU, ℑV)),
    v = Forcing(v_forcing_wrapper, discrete_form=true, parameters=(; ℑU, ℑV)),
    w = Forcing(w_forcing_wrapper, discrete_form=true, parameters=(; ℑU, ℑV)),
    T = Forcing(T_forcing_wrapper, discrete_form=true, parameters=(; ℑU, ℑV, ℑΘ, μ)),
    S = Forcing(S_forcing_wrapper, discrete_form=true, parameters=(; ℑU, ℑV, ℑS, μ))
)


@info "Enforcing boundary conditions..."

## Set up boundary conditions to
##   1. impose wind stresses at the ocean surface, and
##   2. impose heat and salt fluxes at the ocean surface.

# Physical constants.
const ρ₀ = 1027.0  # Density of seawater [kg/m³]
const cₚ = 4000.0  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

@inline wind_stress_x(x, y, t, p) =   p.ℑτx(t) / ρ₀
@inline wind_stress_y(x, y, t, p) =   p.ℑτy(t) / ρ₀
@inline     heat_flux(x, y, t, p) = - p.ℑQθ(t) / ρ₀ / cₚ
@inline     salt_flux(x, y, t, p) =   p.ℑQs(t) / ρ₀

u′_bcs = UVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(wind_stress_x, parameters=(; ℑτx)))
v′_bcs = VVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(wind_stress_y, parameters=(; ℑτy)))
θ′_bcs =    TracerBoundaryConditions(grid, top=FluxBoundaryCondition(heat_flux, parameters=(; ℑQθ)))
s′_bcs =    TracerBoundaryConditions(grid, top=FluxBoundaryCondition(salt_flux, parameters=(; ℑQs)))

boundary_conditions = (u=u′_bcs, v=v′_bcs, T=θ′_bcs, S=s′_bcs)


@info "Framing the model..."

model = IncompressibleModel(
           architecture = arch,
                   grid = grid,
            timestepper = :RungeKutta3,
              advection = arch isa CPU ? UpwindBiasedThirdOrder() : WENO5(),
                tracers = (:T, :S),
               buoyancy = buoyancy,
               coriolis = coriolis,
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = apply_surface_forcing ? boundary_conditions : NamedTuple(),
                forcing = apply_background_state ? forcings : NamedTuple()
)


@info "Initializing conditions..."

ε(μ) = μ * randn() # noise

U₀(x, y, z) = 0
V₀(x, y, z) = 0
W₀(x, y, z) = ε(1e-10)
Θ₀(x, y, z) = interpolated_profiles_cpu.Θ(z, 0)
S₀(x, y, z) = interpolated_profiles_cpu.S(z, 0)

Oceananigans.set!(model, u=U₀, v=V₀, w=W₀, T=Θ₀, S=S₀)


@info "Summoning the time step wizard..."

wizard = TimeStepWizard(
           cfl = 0.5,
            Δt = 1second,
    max_change = 1.1,
        min_Δt = 0.001second,
        max_Δt = 1minute
)

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)


@info "Saddling up progress messenger..."

mutable struct ProgressTicker
    interval_start_time :: Float64
end

function (ticker::ProgressTicker)(simulation)
    model = simulation.model
    p = simulation.parameters

    # Compute simulation progress
    i, t = model.clock.iteration, model.clock.time
    simulation_date_time = DateTime(p.start_date) + Millisecond(round(Int, 1000t))

    wall_time = (time_ns() - ticker.interval_start_time) * 1e-9
    progress = model.clock.time / simulation.stop_time
    ETA = (1 - progress) / progress * simulation.run_time
    ETA_datetime = now() + Second(round(Int, ETA))

    # Find maximum velocities.
    umax = maximum(abs, model.velocities.u)
    vmax = maximum(abs, model.velocities.v)
    wmax = maximum(abs, model.velocities.w)

    # Find tracer extrema
    Tmin = minimum(model.tracers.T)
    Tmax = maximum(model.tracers.T)
    Smin = minimum(model.tracers.S)
    Smax = maximum(model.tracers.S)

    # Find maximum ν and κ.
     νmax = maximum(model.diffusivities.νₑ)
    κTmax = maximum(model.diffusivities.κₑ.T)
    κSmax = maximum(model.diffusivities.κₑ.S)

    mld = p.mixed_layer_depth(model)

    @info @sprintf("[%06.2f%%] iteration: %d, simulation time: %s, mixed layer depth: %.2f m, CFL: %.2e, νCFL: %.2e, next Δt: %s",
                   100 * progress, i, simulation_date_time, mld, cfl(model), dcfl(model), prettytime(simulation.Δt.Δt))

    @info @sprintf("          ├── u⃗_max: (%.2e, %.2e, %.2e) m/s, T: (min=%.2f, max=%.2f) °C, S: (min=%.2f, max=%.2f) psu, νκ_max: (ν=%.2e, κT=%.2e, κS=%.2e)",
                   umax, vmax, wmax, Tmin, Tmax, Smin, Smax, νmax, κTmax, κSmax)

    @info @sprintf("          └── ETA: %s (%s), Δ(wall time): %s / iteration",
                   format(ETA_datetime, "yyyy-mm-dd HH:MM:SS"), prettytime(ETA), prettytime(wall_time / simulation.iteration_interval))

    ticker.interval_start_time = time_ns()

    return nothing
end


@info "Conjuring the simulation..."

simulation = Simulation(model,
                    Δt = wizard,
             stop_time = n_days * days,
    iteration_interval = 10,
              progress = ProgressTicker(time_ns()),
            parameters = (; start_date, end_date, mixed_layer_depth)
)


@info "Garnishing output writers..."

filepath_prefix = joinpath(output_dir, "lesbrary_latitude$(lat)_longitude$(lon)_$(start_date)_to_$(stop_date)")

global_attributes = Dict(
    "latitude" => lat,
    "longitude" => lon,
    "start_date" => string(start_date),
    "end_date" => string(end_date)
)

u, v, w, T, S = fields(model)
b = BuoyancyField(model)
fields_to_output = (; u, v, w, T, S, b)

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields_to_output, global_attributes = global_attributes,
                       schedule = TimeInterval(6hours),
                       filepath = filepath_prefix * "_fields.nc",
                           mode = "c")

simulation.output_writers[:surface] =
    NetCDFOutputWriter(model, fields_to_output, global_attributes = global_attributes,
                            schedule = TimeInterval(5minutes),
                            filepath = filepath_prefix * "_surface.nc",
                                mode = "c",
                        field_slicer = FieldSlicer(k=grid.Nz))

simulation.output_writers[:slice] =
    NetCDFOutputWriter(model, fields_to_output, global_attributes = global_attributes,
                            schedule = TimeInterval(5minutes),
                            filepath = filepath_prefix * "_slice.nc",
                                mode = "c",
                        field_slicer = FieldSlicer(i=1))


@info "Squeezing out statistics..."

profiles = (
    U = AveragedField(u, dims=(1, 2)),
    V = AveragedField(v, dims=(1, 2)),
    T = AveragedField(T, dims=(1, 2)),
    S = AveragedField(S, dims=(1, 2)),
    B = AveragedField(b, dims=(1, 2))
)

simulation.output_writers[:profiles] =
    NetCDFOutputWriter(model, profiles, global_attributes = global_attributes,
                       schedule = TimeInterval(5minutes),
                       filepath = filepath_prefix * "_profiles.nc",
                           mode = "c")


@info "Inscribing background state..."

large_scale_outputs = Dict(
    "τx" => model -> interpolated_surface_forcings_cpu.τx.(model.clock.time),
    "τy" => model -> interpolated_surface_forcings_cpu.τy.(model.clock.time),
    "QT" => model -> interpolated_surface_forcings_cpu.Qθ.(model.clock.time),
    "QS" => model -> interpolated_surface_forcings_cpu.Qs.(model.clock.time),
     "u" => model -> interpolated_profiles_cpu.U.(znodes(Center, model.grid)[:], model.clock.time),
     "v" => model -> interpolated_profiles_cpu.V.(znodes(Center, model.grid)[:], model.clock.time),
     "T" => model -> interpolated_profiles_cpu.Θ.(znodes(Center, model.grid)[:], model.clock.time),
     "S" => model -> interpolated_profiles_cpu.S.(znodes(Center, model.grid)[:], model.clock.time),
  "Ugeo" => model -> interpolated_profiles_cpu.Ugeo.(znodes(Center, model.grid)[:], model.clock.time),
  "Vgeo" => model -> interpolated_profiles_cpu.Vgeo.(znodes(Center, model.grid)[:], model.clock.time),
  "mld_SOSE" => model -> interpolated_surface_forcings_cpu.mld.(model.clock.time),
  "mld_LES"  => mixed_layer_depth
)

large_scale_dims = Dict(
      "τx" => (),
      "τy" => (),
      "QT" => (),
      "QS" => (),
       "u" => ("zC",),
       "v" => ("zC",),
       "T" => ("zC",),
       "S" => ("zC",),
    "∂ρ∂z" => ("zF",),
    "Ugeo" => ("zC",),
    "Vgeo" => ("zC",),
    "mld_SOSE" => (),
    "mld_LES"  => ()
)

simulation.output_writers[:large_scale] =
    NetCDFOutputWriter(model, large_scale_outputs, global_attributes = global_attributes,
                         schedule = TimeInterval(5minutes),
                         filepath = filepath_prefix * "_large_scale.nc",
                       dimensions = large_scale_dims,
                             mode = "c")


wave = raw"""
           _.====.._
         ,:._       ~-_
             `\        ~-_
               |          `.
             ,/             ~-_
..__-..__..-''                 ~~--..__...----... LESbrary.jl ..__..--
"""

fish = raw"""
o                      o                    o
  o                     o                   o
 o                     o                    o
o   .''''.            o   .''''.            o   .''''.
 o /O)    './|         o /O)    './|         o /O)    './|
   > ) \| .'\|           > ) \| .'\|           > ) \| .'\|
    `....`                `....`                `....`
      ` `                   ` `                   ` `

             o                      o                   o
            o                      o                    o
            o   .''''.            o   .''''.             o  .''''.
             o /O)    './|         o /O)    './|         o /O)    './|
               > ) \| .'\|           > ) \| .'\|           > ) \| .'\|
                `....`                `....`                `....`
                  ` `                   ` `                   ` `
"""

print(wave)
println()

@printf("                    arch : %s                       \n", typeof(arch))
@printf("                       N : %d, %d, %d               \n", grid.Nx, grid.Ny, grid.Nz)
@printf("                       L : %.3g, %.3g, %.3g (meters)\n", grid.Lx, grid.Ly, grid.Lz)
@printf("                       Δ : %.3g, %.3g, %.3g (meters)\n", grid.Δx, grid.Δy, grid.Δz)
@printf("                    φ, λ : %.2f°N, %.2f°E           \n", lat, lon)
@printf("                   start : %s                       \n", start_date)
@printf("                     end : %s                       \n", end_date)

println()

@printf("              output dir : %s\n", output_dir)
@printf("   apply surface forcing : %s\n", apply_surface_forcing)
@printf("  apply background state : %s\n", apply_background_state)
@printf("         make animations : %s\n", make_animation)

println()
print(fish)


t1 = time_ns()
t_setup = prettytime((t1 - t0) * 1e-9)
@info "Simulation setup took $t_setup."


@info "Teaching the simulation to run!..."

run!(simulation)


@info "Plotting stuff and making movies..."

if make_animation
    plot_surface_forcings(filepath_prefix)
    animate_fields(lat, lon, start_date, filepath_prefix)
    animate_profiles(lat, lon, start_date, filepath_prefix)
end
