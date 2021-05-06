using Logging
using Printf

using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators
using SeawaterPolynomials.TEOS10

using Oceananigans.BuoyancyModels: BuoyancyField
using Dates: Date, DateTime, Second, Millisecond, now, format
using RealisticLESbrary: ∂z, ∂t

Logging.global_logger(OceananigansLogger())

# include("load_sose_data.jl")
include("interpolate_sose_data.jl")
include("make_plots_and_movies.jl")

## Pick architecture and float type

arch = CPU()
FT = Float64

## Picking site

@info "Finding an interesting spot in the Southern Ocean..."

lat, lon = -35, 210

## Pick simulation time

@info "Setting the clock..."

sose_start_date = Date(2013, 1, 1)
sose_end_date = Date(2018, 1, 1)

start_date = Date(2013, 6, 1)
stop_date = Date(2013, 6, 10)

@assert start_date >= sose_start_date
@assert stop_date <= sose_end_date

day_offset = (start_date - sose_start_date).value + 1
n_days = (stop_date - start_date).value

## Set up grid

@info "Mapping grid..."

Nx = Ny = 32
Nz = 2Nx
Lz = 500.0
Lx = Ly = Lz/2

topology = (Periodic, Periodic, Bounded)
grid = RegularRectilinearGrid(topology=topology, size=(Nx, Ny, Nz), x=(0.0, Lx), y=(0.0, Ly), z=(-Lz, 0.0))

## Pick a Coriolis approximation

@info "Spinning up a tangent plane..."

coriolis = FPlane(latitude=lat)

## Pick an equation of state

@info "Surfacing a buoyancy model..."

buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState(FT))

## Load large-scale (base state) solution from SOSE

@info "Summoning SOSE data and diagnosing geostrophic background state..."

SOSE_DIR = "/storage3/bsose_i122/"

# sose_datetimes, sose_grid, sose_surface_forcings, sose_profiles =
#     load_sose_data(SOSE_DIR, lat, lon, day_offset, n_days, grid, buoyancy, coriolis)

dates = convert.(Date, sose_datetimes)
start_date = dates[day_offset]
stop_date = dates[day_offset + n_days]
@info "Simulation start date: $start_date, stop date: $stop_date"

## Create linear interpolations for base state solution

@info "Interpolating SOSE data..."

times = day * (0:n_days)

interpolated_surface_forcings = interpolate_surface_forcings(sose_surface_forcings, times)
interpolated_profiles = interpolate_profiles(sose_profiles, sose_grid, grid, times)

@info "Plotting initial state for inspection..."

plot_initial_args = (sose_profiles, sose_grid, interpolated_profiles, grid, lat, lon, start_date)
plot_initial_state(plot_initial_args..., z_bottom=-Lz, filepath="initial_state.png")
plot_initial_state(plot_initial_args..., z_bottom=-10Lz, filepath="initial_state_deep.png")

## Set up forcing forcings to
##   1. include mean flow interactions in the momentum equation, and
##   2. weakly relax tracer fields to the base state.

@info "Forcing mean-flow interactions and relaxing tracers..."

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

parameters = (
    ℑτx = interpolated_surface_forcings.τx,
    ℑτy = interpolated_surface_forcings.τy,
    ℑQθ = interpolated_surface_forcings.Qθ,
    ℑQs = interpolated_surface_forcings.Qs,
    ℑU  = interpolated_profiles.Ugeo,
    ℑV  = interpolated_profiles.Vgeo,
    ℑΘ  = interpolated_profiles.Θ,
    ℑS  = interpolated_profiles.S,
    μ   = μ
)

forcings = (
    u = Forcing(u_forcing_wrapper, discrete_form=true, parameters=parameters),
    v = Forcing(v_forcing_wrapper, discrete_form=true, parameters=parameters),
    w = Forcing(w_forcing_wrapper, discrete_form=true, parameters=parameters),
    T = Forcing(T_forcing_wrapper, discrete_form=true, parameters=parameters),
    S = Forcing(S_forcing_wrapper, discrete_form=true, parameters=parameters)
)

## Set up boundary conditions to
##   1. impose wind stresses at the ocean surface, and
##   2. impose heat and salt fluxes at the ocean surface.

@info "Enforcing boundary conditions..."

# Physical constants.
const ρ₀ = 1027.0  # Density of seawater [kg/m³]
const cₚ = 4000.0  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

@inline wind_stress_x(x, y, t, p) =   p.ℑτx(t) / ρ₀
@inline wind_stress_y(x, y, t, p) =   p.ℑτy(t) / ρ₀
@inline     heat_flux(x, y, t, p) = - p.ℑQθ(t) / ρ₀ / cₚ
@inline     salt_flux(x, y, t, p) =   p.ℑQs(t) / ρ₀

u′_bcs = UVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(wind_stress_x; parameters))
v′_bcs = VVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(wind_stress_y; parameters))
θ′_bcs =    TracerBoundaryConditions(grid, top=FluxBoundaryCondition(heat_flux; parameters))
s′_bcs =    TracerBoundaryConditions(grid, top=FluxBoundaryCondition(salt_flux; parameters))

boundary_conditions = (u=u′_bcs, v=v′_bcs, T=θ′_bcs, S=s′_bcs)

## Model setup

@info "Framing the model..."

model = IncompressibleModel(
           architecture = arch,
             float_type = FT,
                   grid = grid,
            timestepper = :RungeKutta3,
              advection = UpwindBiasedThirdOrder(),
                tracers = (:T, :S),
               buoyancy = buoyancy,
               coriolis = coriolis,
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = boundary_conditions,
                forcing = forcings
)

## Initial conditions

@info "Initializing conditions..."

ε(μ) = μ * randn() # noise

U₀(x, y, z) = 0
V₀(x, y, z) = 0
W₀(x, y, z) = ε(1e-10)
Θ₀(x, y, z) = interpolated_profiles.Θ(z, 0)
S₀(x, y, z) = interpolated_profiles.S(z, 0)

Oceananigans.set!(model, u=U₀, v=V₀, w=W₀, T=Θ₀, S=S₀)

## Simulation setup

@info "Conjuring the simulation..."

wizard = TimeStepWizard(cfl=0.5, Δt=1second, max_change=1.1, min_Δt=0.001second, max_Δt=1minute)

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)


mutable struct ProgressTicker
    interval_start_time :: Float64
end

function (p::ProgressTicker)(simulation)
    model = simulation.model

    # Compute simulation progress
    i, t = model.clock.iteration, model.clock.time
    date_time = DateTime(start_time) + Millisecond(round(Int, 1000t))

    wall_time = (time_ns() - p.interval_start_time) * 1e-9
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

    @info @sprintf("[%06.2f%%] iteration: %d, simulation time: %s, CFL: %.2e, νCFL: %.2e, next Δt: %s",
                   100 * progress, i, date_time, cfl(model), dcfl(model), prettytime(simulation.Δt.Δt))

    @info @sprintf("          ├── u⃗_max: (%.2e, %.2e, %.2e) m/s, T: (min=%.2f, max=%.2f) °C, S: (min=%.2f, max=%.2f) psu, νκ_max: (ν=%.2e, κT=%.2e, κS=%.2e)",
                   umax, vmax, wmax, Tmin, Tmax, Smin, Smax, νmax, κTmax, κSmax)

    @info @sprintf("          └── ETA: %s (%s), Δ(wall time): %s / iteration",
                   format(ETA_datetime, "yyyy-mm-dd HH:MM:SS"), prettytime(ETA), prettytime(wall_time / simulation.iteration_interval))

    p.interval_start_time = time_ns()

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=n_days * days, iteration_interval=10, progress=ProgressTicker(time_ns()))

## Diagnosing mixed layer depth

function mixed_layer_depth(model)
    T = model.tracers.T
    ∂T̄∂z = AveragedField(∂z(T), dims=(1, 2))
    compute!(∂T̄∂z)

    _, k_boundary_layer = findmax(abs.(interior(∂T̄∂z)))

    if k_boundary_layer isa CartesianIndex
        k_boundary_layer = k_boundary_layer.I[3]
    end

    mixed_layer_depth = - model.grid.zF[k_boundary_layer]

    @info "Mixed layer depth is $mixed_layer_depth meters"

    return mixed_layer_depth
end

## Output 3D fields

@info "Garnishing output writers..."

filename_prefix = "lesbrary_latitude$(lat)_longitude$(lon)_$(start_date)_to_$(stop_date)"

global_attributes = Dict(
    "latitude" => lat,
    "longitude" => lon
)

u, v, w, T, S = fields(model)
b = BuoyancyField(model)
fields_to_output = (; u, v, w, T, S, b)

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields_to_output, global_attributes = global_attributes,
                       schedule = TimeInterval(6hours),
                       filepath = filename_prefix * "_fields.nc",
                           mode = "c")

simulation.output_writers[:surface] =
    NetCDFOutputWriter(model, fields_to_output, global_attributes = global_attributes,
                            schedule = TimeInterval(5minutes),
                            filepath = filename_prefix * "_surface.nc",
                                mode = "c",
                        field_slicer = FieldSlicer(k=grid.Nz))

simulation.output_writers[:slice] =
    NetCDFOutputWriter(model, fields_to_output, global_attributes = global_attributes,
                            schedule = TimeInterval(5minutes),
                            filepath = filename_prefix * "_slice.nc",
                                mode = "c",
                        field_slicer = FieldSlicer(i=1))

## Output statistics (horizontal averages)

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
                       filepath = filename_prefix * "_profiles.nc",
                           mode = "c")

## Output background state

@info "Inscribing background state..."

# large_scale_outputs = Dict(
#     "τx" => model -> ℑτx.(model.clock.time),
#     "τy" => model -> ℑτy.(model.clock.time),
#     "QT" => model -> ℑQθ.(model.clock.time),
#     "QS" => model -> ℑQs.(model.clock.time),
#      "u" => model ->  ℑU.(model.clock.time, znodes(Center, model.grid)[:]),
#      "v" => model ->  ℑV.(model.clock.time, znodes(Center, model.grid)[:]),
#      "T" => model ->  ℑΘ.(model.clock.time, znodes(Center, model.grid)[:]),
#      "S" => model ->  ℑS.(model.clock.time, znodes(Center, model.grid)[:]),
#   "∂ρ∂z" => model ->  ℑN.(model.clock.time, znodes(Face, model.grid)[:]),
#   "Ugeo" => model -> ℑUgeo.(model.clock.time, znodes(Center, model.grid)[:]),
#   "Vgeo" => model -> ℑVgeo.(model.clock.time, znodes(Center, model.grid)[:]),
#   "mld_SOSE" => model -> ℑmld.(model.clock.time),
#   "mld_LES"  => mixed_layer_depth
# )

# large_scale_dims = Dict(
#       "τx" => (),
#       "τy" => (),
#       "QT" => (),
#       "QS" => (),
#        "u" => ("zC",),
#        "v" => ("zC",),
#        "T" => ("zC",),
#        "S" => ("zC",),
#     "∂ρ∂z" => ("zF",),
#     "Ugeo" => ("zC",),
#     "Vgeo" => ("zC",),
#     "mld_SOSE" => (),
#     "mld_LES"  => ()
# )

# simulation.output_writers[:large_scale] =
#     NetCDFOutputWriter(model, large_scale_outputs, global_attributes = global_attributes,
#                          schedule = TimeInterval(5minutes),
#                          filepath = filename_prefix * "_large_scale.nc",
#                        dimensions = large_scale_dims,
#                              mode = "c")

## Banner!

wave = raw"""
           _.====.._
         ,:._       ~-_
             `\        ~-_
               |          `.
             ,/             ~-_
    -..__..-''                 ~~--..__...----... LESbrary.jl ...
"""

fish = raw"""
                 o                     o
                 o                    o
                o                     o
               o   .''''.             o   .''''.
                o /O)    './|          o /O)    './|
                  > ) \| .'\|            > ) \| .'\|
                   `....`                 `....`
                     ` `                    ` `

       o                      o                    o
      o                      o                     o
      o   .''''.            o   .''''.              o  .''''.
       o /O)    './|         o /O)    './|          o /O)    './|
         > ) \| .'\|           > ) \| .'\|            > ) \| .'\|
          `....`                `....`                 `....`
            ` `                   ` `                    ` `
"""

@printf("""%s
           N : %d, %d, %d
           L : %.3g, %.3g, %.3g [m]
           Δ : %.3g, %.3g, %.3g [m]
        φ, λ : %.2f, %.2f [latitude, longitude]
           f : %.3e [s⁻¹]
       start : %s
         end : %s
        %s""",
        wave,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Lx, grid.Ly, grid.Lz,
        grid.Δx, grid.Δy, grid.Δz,
        lat, lon, model.coriolis.f,
        start_date, stop_date,
        fish)

@info "Teaching the simulation to run!..."

run!(simulation)
