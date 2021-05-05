using Logging
using Printf

using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators
using SeawaterPolynomials.TEOS10

using Oceananigans.BuoyancyModels: BuoyancyField

using Dates: Date, DateTime, Millisecond # Avoid conflicts with Oceananigans.Units

using RealisticLESbrary

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
Lx = Ly = 250.0
Lz = 2Lx

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

#=

ℑτx  = interpolate((ts,), τx,  Gridded(Linear()))
ℑτy  = interpolate((ts,), τy,  Gridded(Linear()))
ℑQθ  = interpolate((ts,), Qθ,  Gridded(Linear()))
ℑQs  = interpolate((ts,), Qs,  Gridded(Linear()))
ℑmld = interpolate((ts,), mld, Gridded(Linear()))

# Coordinates needs to be in increasing order for Interpolations.jl
reverse!(zC_SOSE)
reverse!(zF_SOSE)

U = reverse(U, dims=2)
V = reverse(V, dims=2)
Θ = reverse(Θ, dims=2)
S = reverse(S, dims=2)
N = reverse(N, dims=2)
Ugeo = reverse(Ugeo, dims=2)
Vgeo = reverse(Vgeo, dims=2)

ℑU = interpolate((ts, zC_SOSE), U, Gridded(Linear()))
ℑV = interpolate((ts, zC_SOSE), V, Gridded(Linear()))
ℑΘ = interpolate((ts, zC_SOSE), Θ, Gridded(Linear()))
ℑS = interpolate((ts, zC_SOSE), S, Gridded(Linear()))
ℑN = interpolate((ts, zF_SOSE), S, Gridded(Linear()))
ℑUgeo = interpolate((ts, zC_SOSE), Ugeo, Gridded(Linear()))
ℑVgeo = interpolate((ts, zC_SOSE), Vgeo, Gridded(Linear()))

## Set up forcing forcings to
##   1. include mean flow interactions in the momentum equation, and
##   2. weakly relax tracer fields to the base state.

@info "Forcing mean-flow interactions and relaxing tracers..."

# Fu′ = - w′∂z(U) - U∂x(u′) - V∂y(u′)
# FIXME? Do we need to use the flux form operator ∇·(Uũ′) instead of ũ′·∇U ?
@inline Fu′(i, j, k, grid, t, u′, v′, w′, p) =
    @inbounds - w′[i, j, k] * ∇(p.ℑU, t, grid.zC[k])[2] - p.ℑU(t, grid.zC[k]) * ∂xᶜᵃᵃ(i, j, k, grid, u′) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, u′)

# Fv′ = - w′∂z(V) - U∂x(v′) - V∂y(v′)
@inline Fv′(i, j, k, grid, t, u′, v′, w′, p) =
    @inbounds - w′[i, j, k] * ∇(p.ℑV, t, grid.zC[k])[2] - p.ℑU(t, grid.zC[k]) * ∂xᶜᵃᵃ(i, j, k, grid, v′) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, v′)

# Fw′ = - U∂x(w′) - V∂y(w′)
@inline Fw′(i, j, k, grid, t, u′, v′, w′, p) =
    @inbounds - p.ℑU(t, grid.zC[k]) * ∂xᶜᵃᵃ(i, j, k, grid, u′) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, v′)

# Fθ′ = - ∂t(Θ) - U∂x(θ′) - V∂y(θ′) - w′∂z(Θ)
# FIXME? Do we need to include the ∂t(Θ) terms?
@inline Fθ′(i, j, k, grid, t, u′, v′, w′, T′, p) =
    @inbounds - p.ℑU(t, grid.zC[k]) * ∂xᶠᵃᵃ(i, j, k, grid, T′) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, T′) - w′[i, j, k] * ∇(p.ℑΘ, t, grid.zC[k])[2]

# Fs′ = - ∂t(S) - U∂x(s′) - V∂y(s′) - w′∂z(S)
@inline Fs′(i, j, k, grid, t, u′, v′, w′, S′, p) =
    @inbounds - p.ℑU(t, grid.zC[k]) * ∂xᶠᵃᵃ(i, j, k, grid, S′) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, S′) - w′[i, j, k] * ∇(p.ℑS, t, grid.zC[k])[2]

# Timescale for relaxation to large-scale solution.
week = 7day
μ = (T=1/week, S=1/week)

# FIXME: Should be μ(C - c̅) so I need to add horizontal averages to parameters.
@inline Fθ_μ(i, j, k, grid, t, T′, p) = @inbounds p.μ.T * (p.ℑΘ(t, grid.zC[k]) - T′[i, j, k])
@inline FS_μ(i, j, k, grid, t, S′, p) = @inbounds p.μ.S * (p.ℑS(t, grid.zC[k]) - S′[i, j, k])

@inline u_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fu′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, params)
@inline v_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fv′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, params)
@inline w_forcing_wrapper(i, j, k, grid, clock, fields, params) = Fw′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, params)

@inline T_forcing_wrapper(i, j, k, grid, clock, fields, params) =
    Fθ′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, fields.T, params) + Fθ_μ(i, j, k, grid, clock.time, fields.T, params)

@inline S_forcing_wrapper(i, j, k, grid, clock, fields, params) =
    Fs′(i, j, k, grid, clock.time, fields.u, fields.v, fields.w, fields.S, params) + FS_μ(i, j, k, grid, clock.time, fields.S, params)

parameters = (ℑτx=ℑτx, ℑτy=ℑτy, ℑQθ=ℑQθ, ℑQs=ℑQs, ℑU=ℑUgeo, ℑV=ℑVgeo, ℑΘ=ℑΘ, ℑS=ℑS, μ=μ)

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

@inline wind_stress_x(x, y, t, p) = p.ℑτx(t) / ρ₀
@inline wind_stress_y(x, y, t, p) = p.ℑτy(t) / ρ₀
@inline     heat_flux(x, y, t, p) = - p.ℑQθ(t) / ρ₀ / cₚ
@inline     salt_flux(x, y, t, p) =   p.ℑQs(t) / ρ₀  # Minus sign because a freshwater flux would decrease salinity.

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
Θ₀(x, y, z) = ℑΘ(0, z)
S₀(x, y, z) = ℑS(0, z)

Oceananigans.set!(model, u=U₀, v=V₀, w=W₀, T=Θ₀, S=S₀)

## Simulation setup

@info "Conjuring the simulation..."

wizard = TimeStepWizard(cfl=0.5, Δt=1second, max_change=1.1, min_Δt=0.001second, max_Δt=1minute)

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)

function print_progress(simulation)
    model = simulation.model

    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / simulation.stop_time)

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

    # Print progress statement.
    i, t = model.clock.iteration, model.clock.time
    date_time = DateTime(start_time) + Millisecond(round(Int, 1000t))

    @info @sprintf("[%06.2f%%] iteration: %d, simulation time: %s, CFL: %.2e, νCFL: %.2e, next Δt: %s",
                   progress, i, date_time, cfl(model), dcfl(model), prettytime(simulation.Δt.Δt))

    @info @sprintf("          └── u⃗_max: (%.2e, %.2e, %.2e) m/s, T: (min=%.2f, max=%.2f) °C, S: (min=%.2f, max=%.2f) psu, νκ_max: (ν=%.2e, κT=%.2e, κS=%.2e)",
                   umax, vmax, wmax, Tmin, Tmax, Smin, Smax, νmax, κTmax, κSmax)

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=n_days * days, iteration_interval=10, progress=print_progress)

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

large_scale_outputs = Dict(
    "τx" => model -> ℑτx.(model.clock.time),
    "τy" => model -> ℑτy.(model.clock.time),
    "QT" => model -> ℑQθ.(model.clock.time),
    "QS" => model -> ℑQs.(model.clock.time),
     "u" => model ->  ℑU.(model.clock.time, znodes(Center, model.grid)[:]),
     "v" => model ->  ℑV.(model.clock.time, znodes(Center, model.grid)[:]),
     "T" => model ->  ℑΘ.(model.clock.time, znodes(Center, model.grid)[:]),
     "S" => model ->  ℑS.(model.clock.time, znodes(Center, model.grid)[:]),
  "∂ρ∂z" => model ->  ℑN.(model.clock.time, znodes(Face, model.grid)[:]),
  "Ugeo" => model -> ℑUgeo.(model.clock.time, znodes(Center, model.grid)[:]),
  "Vgeo" => model -> ℑVgeo.(model.clock.time, znodes(Center, model.grid)[:]),
  "mld_SOSE" => model -> ℑmld.(model.clock.time),
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
                         filepath = filename_prefix * "_large_scale.nc",
                       dimensions = large_scale_dims,
                             mode = "c")

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

## Plot surface and slice

using CairoMakie
using GeoData
using NCDatasets

function squeeze(A)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims=singleton_dims)
end

ds_xy = NCDstack(filename_prefix * "_surface.nc")
ds_yz = NCDstack(filename_prefix * "_slice.nc")

times = ds_xy[:time]
Nt = length(times)

xc = ds_xy[:xC].data
xf = ds_xy[:xF].data
yc = ds_xy[:yC].data
yf = ds_xy[:yF].data
zc = ds_yz[:zC].data
zf = ds_yz[:zF].data

fig = Figure(resolution=(1920, 1080))

u_max = max(maximum(abs, ds_xy[:u]), maximum(abs, ds_yz[:u]))
v_max = max(maximum(abs, ds_xy[:v]), maximum(abs, ds_yz[:v]))
w_max = max(maximum(abs, ds_xy[:w]), maximum(abs, ds_yz[:w]))
U_max = max(u_max, v_max, w_max)
U_lims = 0.5 .* (-U_max, +U_max)

frame = Node(1)

plot_title = @lift @sprintf("Realistic SOSE LESbrary.jl (%.2f°N, %.2f°E): time = %s", lat, lon, start_time + Millisecond(round(Int, 1000 * times[$frame])))

u_xy = @lift ds_xy[:u][Ti=$frame].data |> squeeze
v_xy = @lift ds_xy[:v][Ti=$frame].data |> squeeze
w_xy = @lift ds_xy[:w][Ti=$frame].data |> squeeze
T_xy = @lift ds_xy[:T][Ti=$frame].data |> squeeze
S_xy = @lift ds_xy[:S][Ti=$frame].data |> squeeze

u_yz = @lift ds_yz[:u][Ti=$frame].data |> squeeze
v_yz = @lift ds_yz[:v][Ti=$frame].data |> squeeze
w_yz = @lift ds_yz[:w][Ti=$frame].data |> squeeze
T_yz = @lift ds_yz[:T][Ti=$frame].data |> squeeze
S_yz = @lift ds_yz[:S][Ti=$frame].data |> squeeze

ax_u_xy = fig[1, 1] = Axis(fig, title="u-velocity")
hm_u_xy = heatmap!(ax_u_xy, xf, yc, u_xy, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_u_xy)

ax_v_xy = fig[1, 2] = Axis(fig, title="v-velocity")
hm_v_xy = heatmap!(ax_v_xy, xc, yf, v_xy, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_v_xy)

ax_w_xy = fig[1, 3] = Axis(fig, title="w-velocity")
hm_w_xy = heatmap!(ax_w_xy, xc, yc, w_xy, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_w_xy)

ax_T_xy = fig[1, 4] = Axis(fig, title="conservative temperature")
hm_T_xy = heatmap!(ax_T_xy, xc, yc, T_xy, colormap=:thermal, colorrange=extrema(ds_xy[:T]))
hidedecorations!(ax_T_xy)

ax_S_xy = fig[1, 5] = Axis(fig, title="absolute salinity")
hm_S_xy = heatmap!(ax_S_xy, xc, yc, S_xy, colormap=:haline, colorrange=extrema(ds_xy[:S]))
hidedecorations!(ax_S_xy)

ax_u_yz = fig[2, 1] = Axis(fig)
hm_u_yz = heatmap!(ax_u_yz, xf, zc, u_yz, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_u_yz)

ax_v_yz = fig[2, 2] = Axis(fig)
hm_v_yz = heatmap!(ax_v_yz, xc, zc, v_yz, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_v_yz)

ax_w_yz = fig[2, 3] = Axis(fig)
hm_w_yz = heatmap!(ax_w_yz, xc, zf, w_yz, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_w_yz)

ax_T_yz = fig[2, 4] = Axis(fig)
hm_T_yz = heatmap!(ax_T_yz, xc, zc, T_yz, colormap=:thermal, colorrange=extrema(ds_yz[:T]))
hidedecorations!(ax_T_yz)

ax_S_yz = fig[2, 5] = Axis(fig)
hm_S_yz = heatmap!(ax_S_yz, xc, zc, S_yz, colormap=:haline, colorrange=extrema(ds_yz[:S]))
hidedecorations!(ax_S_yz)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

filepath = filename_prefix * "_surface_slice_movie.mp4"
record(fig, filepath, 1:10:Nt, framerate=30) do n
    @info "Animating surface and slice movie frame $n/$Nt..."
    frame[] = n
end

@info "Movie saved: $filepath"

## Plot background, SOSE, and LES profiles

ds_p = NCDstack(filename_prefix * "_profiles.nc")
ds_b = NCDstack(filename_prefix * "_large_scale.nc")

times = ds_p[:time]
Nt = length(times)

zc = ds_p[:zC].data
zf = ds_p[:zF].data

fig = Figure(resolution=(2500, 1080))

frame = Node(1)

plot_title = @lift @sprintf("Realistic SOSE LESbrary.jl (%.2f°N, %.2f°E): time = %s", lat, lon, start_time + Millisecond(round(Int, 1000 * times[$frame])))

U_LES = @lift ds_p[:U][Ti=$frame].data
V_LES = @lift ds_p[:V][Ti=$frame].data
T_LES = @lift ds_p[:T][Ti=$frame].data
S_LES = @lift ds_p[:S][Ti=$frame].data
B_LES = @lift ds_p[:B][Ti=$frame].data

U_SOSE = @lift ds_b[:u][Ti=$frame].data
V_SOSE = @lift ds_b[:v][Ti=$frame].data
T_SOSE = @lift ds_b[:T][Ti=$frame].data
S_SOSE = @lift ds_b[:S][Ti=$frame].data
∂ρ∂z_SOSE = @lift ds_b[:∂ρ∂z][Ti=$frame].data

U_geo = @lift ds_b[:Ugeo][Ti=$frame].data
V_geo = @lift ds_b[:Vgeo][Ti=$frame].data

# time_so_far = @lift ds_b[:time][1:$frame].data
# τx_SOSE = @lift ds_b[:τx][Ti=1:$frame].data
# τy_SOSE = @lift ds_b[:τy][Ti=1:$frame].data
# QΘ_SOSE = @lift ds_b[:QT][Ti=1:$frame].data
# QS_SOSE = @lift ds_b[:QS][Ti=1:$frame].data

colors = ["dodgerblue2", "crimson", "forestgreen"]

ax_U = fig[1, 1] = Axis(fig, xlabel="m/s", ylabel="z (m)")
line_U_SOSE = lines!(ax_U, U_SOSE, zc, label="U (SOSE)", linewidth=3, color=colors[1], linestyle=:dash)
line_U_geo  = lines!(ax_U, U_geo, zc, label="U (geo)", linewidth=3, color=colors[1], linestyle=:dot)
line_U_LES  = lines!(ax_U, U_LES, zc, label="U (LES)", linewidth=3, color=colors[1])
axislegend(ax_U, position=:rb, framevisible=false)
xlims!(ax_U, extrema([extrema(ds_p[:U])..., extrema(ds_b[:u])..., extrema(ds_b[:Ugeo])...]))
ylims!(ax_U, extrema(zf))

ax_V = fig[1, 2] = Axis(fig, xlabel="m/s", ylabel="z (m)")
line_V_SOSE = lines!(ax_V, V_SOSE, zc, label="V (SOSE)", linewidth=3, color=colors[2], linestyle=:dash)
line_V_geo  = lines!(ax_V, V_geo, zc, label="V (geo)", linewidth=3, color=colors[2], linestyle=:dot)
line_V_LES  = lines!(ax_V, V_LES, zc, label="V (LES)", linewidth=3, color=colors[2])
axislegend(ax_V, position=:rb, framevisible=false)
xlims!(ax_V, extrema([extrema(ds_p[:U])..., extrema(ds_b[:v])..., extrema(ds_b[:Vgeo])...]))
ylims!(ax_V, extrema(zf))

ax_T = fig[1, 3] = Axis(fig, xlabel="°C", ylabel="z (m)")
line_T_SOSE = lines!(ax_T, T_SOSE, zc, label="Θ (SOSE)", linewidth=3, color=colors[3], linestyle=:dash)
line_T_LES  = lines!(ax_T, T_LES, zc, label="Θ (LES)", linewidth=3, color=colors[3])
axislegend(ax_T, position=:rb, framevisible=false)
xlims!(ax_T, extrema([extrema(ds_p[:T])..., extrema(ds_b[:T])...]))
ylims!(ax_T, extrema(zf))

ax_S = fig[1, 4] = Axis(fig, xlabel="psu", ylabel="z (m)")
line_S_SOSE = lines!(ax_S, S_SOSE, zc, label="S (SOSE)", linewidth=3, color=colors[3], linestyle=:dash)
line_S_LES  = lines!(ax_S, S_LES, zc, label="S (LES)", linewidth=3, color=colors[3])
axislegend(ax_S, position=:rb, framevisible=false)
xlims!(ax_S, extrema([extrema(ds_p[:S])..., extrema(ds_b[:S])...]))
ylims!(ax_S, extrema(zf))

ax_B = fig[1, 5] = Axis(fig, xlabel="m/s²", ylabel="z (m)")
line_B_LES  = lines!(ax_B, B_LES, zc, label="B (LES)", linewidth=3, color=colors[3])
axislegend(ax_B, position=:rb, framevisible=false)
xlims!(ax_B, extrema(ds_p[:B]))
ylims!(ax_B, extrema(zf))

ax_N = fig[1, 6] = Axis(fig, xlabel="kg/m⁴", ylabel="z (m)")
line_N_SOSE  = lines!(ax_N, ∂ρ∂z_SOSE, zf, label="∂ρ∂z (SOSE)", linewidth=3, color=colors[3], linestyle=:dash)
axislegend(ax_N, position=:rb, framevisible=false)
xlims!(ax_N, extrema(ds_b[:∂ρ∂z]))
ylims!(ax_N, extrema(zf))

# ax_τ = fig[2, :] = Axis(fig, ylabel="N/m²")
# line_τx = lines!(ax_τ, time_so_far, τx_SOSE, label="τx", linewidth=3, color=colors[1])
# line_τy = lines!(ax_τ, time_so_far, τy_SOSE, label="τy", linewidth=3, color=colors[2])
# axislegend(ax_τ, position=:rb, framevisible=false)
# xlims!(ax_τ, extrema(ds_b[:time]))
# ylims!(ax_τ, extrema([extrema(ds_b[:τx])..., extrema(ds_b[:τy])...]))
# ax_τ.height = Relative(0.15)

# ax_QΘ = fig[3, :] = Axis(fig, ylabel="QΘ (W/m²)")
# line_QΘ = lines!(ax_QΘ, time_so_far, QΘ_SOSE, linewidth=3, color=colors[1])
# xlims!(ax_QΘ, extrema(ds_b[:time]))
# ylims!(ax_QΘ, extrema(ds_b[:QT]))
# ax_QΘ.height = Relative(0.15)

# ax_QS = fig[4, :] = Axis(fig, ylabel="QS (kg/m²/s))")
# line_QS = lines!(ax_QS, time_so_far, QS_SOSE, linewidth=3, color=colors[1])
# xlims!(ax_QS, extrema(ds_b[:time]))
# ylims!(ax_QS, extrema(ds_b[:QS]))
# ax_QS.height = Relative(0.15)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

filepath = filename_prefix * "_first_order_statistics.mp4"
record(fig, filepath, 1:2:Nt, framerate=30) do n
    @info "Animating first-order statistics movie frame $n/$Nt..."
    frame[] = n
end

@info "Movie saved: $filepath"

## Plot surface forcings

ds_b = NCDstack(filename_prefix * "_large_scale.nc")

times = ds_p[:time] / day
Nt = length(times)

# Makie doesn't support DateTime plotting yet :(
# date_times = [start_time + Millisecond(round(Int, 1000t)) for t in times]

τx_SOSE = ds_b[:τx].data
τy_SOSE = ds_b[:τy].data
τ_SOSE = @. √(τx_SOSE^2 + τy_SOSE^2)
QΘ_SOSE = ds_b[:QT].data
QS_SOSE = ds_b[:QS].data
mld_SOSE = ds_b[:mld_SOSE].data
mld_LES = ds_b[:mld_LES].data

fig = Figure(resolution=(1920, 1080))
plot_title = @sprintf("Realistic SOSE LESbrary.jl (%.2f°N, %.2f°E) surface forcings", lat, lon)

ax_τ = fig[1, 1] = Axis(fig, ylabel="N/m²")
line_τx = lines!(ax_τ, times, τx_SOSE, label="τx", linewidth=3, color=colors[1])
line_τy = lines!(ax_τ, times, τy_SOSE, label="τy", linewidth=3, color=colors[2])
line_τ  = lines!(ax_τ, times, τ_SOSE, label="√(τx² + τy²)", linewidth=3, color=colors[3])
axislegend(ax_τ, position=:rb, framevisible=false)
xlims!(ax_τ, extrema(times))
ylims!(ax_τ, extrema([extrema(τx_SOSE)..., extrema(τy_SOSE)..., extrema(τ_SOSE)...]))
hidexdecorations!(ax_τ, grid=false)

ax_QΘ = fig[2, 1] = Axis(fig, ylabel="QΘ (W/m²)")
line_QΘ = lines!(ax_QΘ, times, QΘ_SOSE, linewidth=3, color=colors[3])
xlims!(ax_QΘ, extrema(times))
ylims!(ax_QΘ, extrema(QΘ_SOSE))
hidexdecorations!(ax_τ, grid=false)

ax_QS = fig[3, 1] = Axis(fig, xlabel="time (days)", ylabel="QS (kg/m²/s)")
line_QS = lines!(ax_QS, times, QS_SOSE, linewidth=3, color=colors[3])
xlims!(ax_QS, extrema(times))
ylims!(ax_QS, extrema(QS_SOSE))

ax_mld = fig[4, 1] = Axis(fig, xlabel="time (days)", ylabel="Mixed layer depth (m)")
line_mld_SOSE = lines!(ax_mld, times, mld_SOSE, label="SOSE", linewidth=3, color=colors[3], linestyle=:dash)
line_mld_LES = lines!(ax_mld, times, mld_LES, label="LES", linewidth=3, color=colors[3])
axislegend(ax_mld, position=:rb, framevisible=false)
xlims!(ax_mld, extrema(times))
ylims!(ax_mld, extrema([extrema(mld_SOSE)..., extrema(mld_LES)...]))

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

filepath = filename_prefix * "_surface_forcings.png"
save(filepath, fig)

@info "Figure saved: $filepath"
