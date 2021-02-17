using Printf
using Dates
using PyCall
using Conda

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Fields
using Oceananigans.Advection
using Oceananigans.Buoyancy
using Oceananigans.BoundaryConditions
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.Utils
using SeawaterPolynomials.TEOS10

using Oceananigans.Utils: second, minute, minutes, hour, hours, day, days
using Interpolations: interpolate, gradient, Gridded, Linear
const ∇ = gradient

# Install needed Python packages
# Conda.add("xarray")
# Conda.add_channel("conda-forge")
# Conda.add("xgcm", channel="conda-forge")
# Conda.add("netcdf4")

# Needed to import local Python modules like sose_data
# See: https://github.com/JuliaPy/PyCall.jl/issues/48
py"""
import sys
sys.path.insert(0, ".")
"""

## Set up grid

@info "Mapping grid..."

lat, lon = -50, 275
day_offset, n_days = 250, 10

arch = CPU()
FT = Float64

Nx = Ny = 32
Nz = 2Nx
Lx = Ly = 250.0
Lz = 2Lx

topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(0.0, Lx), y=(0.0, Ly), z=(-Lz, 0.0))

## Pick a Coriolis approximation

@info "Spinning up a tangent plane..."

coriolis = FPlane(latitude=lat)

## Pick an equation of state

@info "Surfacing a buoyancy model..."

buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState(FT))

## Load large-scale (base state) solution from SOSE

@info "Summoning SOSE data..."

sose = pyimport("sose_data")

SOSE_DIR = "/storage3/bsose_i122/"

# Don't have to wait minutes to load 3D data if we already did so.
if (!@isdefined ds2) && (!@isdefined ds3)
    ds2 = sose.open_sose_2d_datasets(SOSE_DIR)
    ds3 = sose.open_sose_3d_datasets(SOSE_DIR)
end

date_times = sose.get_times(ds2)

τx = sose.get_scalar_time_series(ds2, "oceTAUX",  lat, lon, day_offset, n_days) |> Array{FT}
τy = sose.get_scalar_time_series(ds2, "oceTAUY",  lat, lon, day_offset, n_days) |> Array{FT}
Qθ = sose.get_scalar_time_series(ds2, "oceQnet",  lat, lon, day_offset, n_days) |> Array{FT}
Qs = sose.get_scalar_time_series(ds2, "oceFWflx", lat, lon, day_offset, n_days) |> Array{FT}

U = sose.get_profile_time_series(ds3, "UVEL",  lat, lon, day_offset, n_days) |> Array{FT}
V = sose.get_profile_time_series(ds3, "VVEL",  lat, lon, day_offset, n_days) |> Array{FT}
Θ = sose.get_profile_time_series(ds3, "THETA", lat, lon, day_offset, n_days) |> Array{FT}
S = sose.get_profile_time_series(ds3, "SALT",  lat, lon, day_offset, n_days) |> Array{FT}

# Nominal values for α, β to compute geostrophic velocities
# FIXME: Use TEOS-10 (Θ, Sᴬ, Z) dependent values
α = 1.67e-4
β = 7.80e-4

@info "Diagnosing geostrophic background state..."

# Don't have to wait minutes to compute Ugeo and Vgeo if we already did so.
if (!@isdefined Ugeo) && (!@isdefined Vgeo)
    Ugeo, Vgeo = sose.compute_geostrophic_velocities(ds3, lat, lon, day_offset, n_days, grid.zF, α, β,
                                                     buoyancy.gravitational_acceleration, coriolis.f)

    ds2.close()
    ds3.close()
end

## Create linear interpolations for base state solution

@info "Interpolating SOSE data..."

ts = day * (0:n_days-1) |> collect
zC = ds3.Z.values

ℑτx = interpolate((ts,), τx, Gridded(Linear()))
ℑτy = interpolate((ts,), τy, Gridded(Linear()))
ℑQθ = interpolate((ts,), Qθ, Gridded(Linear()))
ℑQs = interpolate((ts,), Qs, Gridded(Linear()))

# z coordinate needs to be in increasing order.
reverse!(zC)
U = reverse(U, dims=2)
V = reverse(V, dims=2)
Θ = reverse(Θ, dims=2)
S = reverse(S, dims=2)
Ugeo = reverse(U, dims=2)
Vgeo = reverse(V, dims=2)

ℑU = interpolate((ts, zC), U, Gridded(Linear()))
ℑV = interpolate((ts, zC), V, Gridded(Linear()))
ℑΘ = interpolate((ts, zC), Θ, Gridded(Linear()))
ℑS = interpolate((ts, zC), S, Gridded(Linear()))
ℑUgeo = interpolate((ts, zC), Ugeo, Gridded(Linear()))
ℑVgeo = interpolate((ts, zC), Vgeo, Gridded(Linear()))

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
    umax = interiorparent(model.velocities.u) |> Array |> maximum
    vmax = interiorparent(model.velocities.v) |> Array |> maximum
    wmax = interiorparent(model.velocities.w) |> Array |> maximum

    # Find tracer extrema
    Tmin, Tmax = interiorparent(model.tracers.T) |> Array |> extrema
    Smin, Smax = interiorparent(model.tracers.S) |> Array |> extrema

    # Find maximum ν and κ.
    νmax = interiorparent(model.diffusivities.νₑ) |> Array |> maximum
    κTmax = interiorparent(model.diffusivities.κₑ.T) |> Array |> maximum
    κSmax = interiorparent(model.diffusivities.κₑ.S) |> Array |> maximum

    # Print progress statement.
    i, t = model.clock.iteration, model.clock.time

    @info @sprintf("[%06.2f%%] iteration: %d, time: %s, CFL: %.2e, νCFL: %.2e, next Δt: %s",
                   progress, i, prettytime(t), cfl(model), dcfl(model), simulation.Δt.Δt)

    @info @sprintf("          └── u⃗_max: (%.2e, %.2e, %.2e) m/s, T: (min=%.2e, max=%.2e) °C, S: (min=%.2e, max=%.2e) psu, νκ_max: (ν=%.2e, κT=%.2e, κS=%.2e)",
                   umax, vmax, wmax, Tmin, Tmax, Smin, Smax, νmax, κTmax, κSmax)

    return nothing
end

simulation = Simulation(model, Δt=wizard, stop_time=7days, iteration_interval=10, progress=print_progress)

## Output writing

@info "Garnishing output writers..."

u, v, w, T, S = fields(model)

profiles = (
    U = AveragedField(u, dims=(1, 2)),
    V = AveragedField(v, dims=(1, 2)),
    T = AveragedField(T, dims=(1, 2)),
    S = AveragedField(S, dims=(1, 2))
)

filename_prefix = "lesbrary_latitude$(lat)_longitude$(lon)_days$(n_days)"

global_attributes = Dict(
    "latitude" => lat,
    "longitude" => lon
)

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields(model), global_attributes = global_attributes,
                       schedule = TimeInterval(6hours),
                       filepath = filename_prefix * "_fields.nc",
                           mode = "c")

simulation.output_writers[:surface] =
    NetCDFOutputWriter(model, fields(model), global_attributes = global_attributes,
                            schedule = TimeInterval(5minutes),
                            filepath = filename_prefix * "_surface.nc",
                                mode = "c",
                        field_slicer = FieldSlicer(k=grid.Nz))

simulation.output_writers[:slice] =
    NetCDFOutputWriter(model, fields(model), global_attributes = global_attributes,
                            schedule = TimeInterval(5minutes),
                            filepath = filename_prefix * "_slice.nc",
                                mode = "c",
                        field_slicer = FieldSlicer(i=1))

simulation.output_writers[:profiles] =
    NetCDFOutputWriter(model, profiles, global_attributes = global_attributes,
                       schedule = TimeInterval(5minutes),
                       filepath = filename_prefix * "_profiles.nc",
                           mode = "c")

large_scale_outputs = Dict(
    "τx" => model -> ℑτx.(model.clock.time),
    "τy" => model -> ℑτy.(model.clock.time),
    "QT" => model -> ℑQθ.(model.clock.time),
    "QS" => model -> ℑQs.(model.clock.time),
     "u" => model ->  ℑU.(model.clock.time, znodes(Center, model.grid)[:]),
     "v" => model ->  ℑV.(model.clock.time, znodes(Center, model.grid)[:]),
     "T" => model ->  ℑΘ.(model.clock.time, znodes(Center, model.grid)[:]),
     "S" => model ->  ℑS.(model.clock.time, znodes(Center, model.grid)[:]),
  "Ugeo" => model -> ℑUgeo.(model.clock.time, znodes(Center, model.grid)[:]),
  "Vgeo" => model -> ℑVgeo.(model.clock.time, znodes(Center, model.grid)[:])
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
    "Ugeo" => ("zC",),
    "Vgeo" => ("zC",)
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
        days : %d
        %s""",
        wave,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Lx, grid.Ly, grid.Lz,
        grid.Δx, grid.Δy, grid.Δz,
        lat, lon, model.coriolis.f, days,
        fish)

run!(simulation)
