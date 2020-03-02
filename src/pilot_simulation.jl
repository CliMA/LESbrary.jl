import Dates
using Printf
using PyCall

using Interpolations: interpolate, gradient, Gridded, Linear
const ∇ = gradient

using Oceananigans
using Oceananigans.Operators
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.Utils

# Needed to import local modules
# See: https://github.com/JuliaPy/PyCall.jl/issues/48
py"""
import sys
sys.path.insert(0, ".")
"""

#####
##### Load large-scale (base state) solution from SOSE
#####

sose = pyimport("sose_data")

ds2 = sose.open_sose_2d_datasets("/home/alir/cnhlab004/bsose_i122/")
ds3 = sose.open_sose_3d_datasets("/home/alir/cnhlab004/bsose_i122/")

date_times = sose.get_times(ds2)

lat, lon, days = -60, 275, 10

arch = CPU()
FT = Float64

τx = sose.get_scalar_time_series(ds2, "oceTAUX",  lat, lon, days) |> Array{FT}
τy = sose.get_scalar_time_series(ds2, "oceTAUY",  lat, lon, days) |> Array{FT}
Qθ = sose.get_scalar_time_series(ds2, "oceQnet",  lat, lon, days) |> Array{FT}
Qs = sose.get_scalar_time_series(ds2, "oceFWflx", lat, lon, days) |> Array{FT}

U = sose.get_profile_time_series(ds3, "UVEL",  lat, lon, days) |> Array{FT}
V = sose.get_profile_time_series(ds3, "VVEL",  lat, lon, days) |> Array{FT}
Θ = sose.get_profile_time_series(ds3, "THETA", lat, lon, days) |> Array{FT}
S = sose.get_profile_time_series(ds3, "SALT",  lat, lon, days) |> Array{FT}

ds2.close()
ds3.close()

#####
##### Create linear interpolations for base state solution
#####

ts = day * (0:days-1) |> collect
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

ℑU = interpolate((ts, zC), U, Gridded(Linear()))
ℑV = interpolate((ts, zC), V, Gridded(Linear()))
ℑΘ = interpolate((ts, zC), Θ, Gridded(Linear()))
ℑS = interpolate((ts, zC), S, Gridded(Linear()))

#####
##### Set up the grid
#####

Nx = Ny = 32
Nz = 2Nx
Lx = Ly = 1000.0
Lz = 2Lx
topology = (Periodic, Periodic, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), x=(0.0, Lx), y=(0.0, Ly), z=(-Lz, 0.0))

#####
##### Set up forcing forcings to
#####   1. include mean flow interactions in the momentum equation, and
#####   2. weakly relax tracer fields to the base state.
#####

# Fu′ = - w′∂z(U) - U∂x(u′) - V∂y(u′)
# FIXME? Do we need to use the flux form operator ∇·(Uũ′) instead of ũ′·∇U ?
@inline Fu′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - ũ′.w[i, j, k] * ∇(p.ℑU, t, grid.zC[k])[2] - p.ℑU(t, grid.zC[k]) * ∂xᶜᵃᵃ(i, j, k, grid, ũ′.u) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.u)

# Fv′ = - w′∂z(V) - U∂x(v′) - V∂y(v′)
@inline Fv′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - ũ′.w[i, j, k] * ∇(p.ℑV, t, grid.zC[k])[2] - p.ℑU(t, grid.zC[k]) * ∂xᶜᵃᵃ(i, j, k, grid, ũ′.v) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v)

# Fw′ = - U∂x(w′) - V∂y(w′)
@inline Fw′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - p.ℑU(t, grid.zC[k]) * ∂xᶜᵃᵃ(i, j, k, grid, ũ′.u) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v)

# Fθ′ = - ∂t(Θ) - U∂x(θ′) - V∂y(θ′) - w′∂z(Θ)
# FIXME? Do we need to include the ∂t(Θ) terms?
@inline Fθ′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - p.ℑU(t, grid.zC[k]) * ∂xᶠᵃᵃ(i, j, k, grid, c′.T) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v) - ũ′.w[i, j, k] * ∇(p.ℑΘ, t, grid.zC[k])[2]

# Fs′ = - ∂t(S) - U∂x(s′) - V∂y(s′) - w′∂z(S)
@inline Fs′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - p.ℑU(t, grid.zC[k]) * ∂xᶠᵃᵃ(i, j, k, grid, c′.S) - p.ℑV(t, grid.zC[k]) * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v) - ũ′.w[i, j, k] * ∇(p.ℑS, t, grid.zC[k])[2]

# Timescale for relaxation to large-scale solution.
week = 7day
μ = (T=1/week, S=1/week)

# FIXME: Should be μ(C - c̅) so I need to add horizontal averages to parameters.
@inline Fθ_μ(i, j, k, grid, t, ũ′, c′, p) = @inbounds p.μ.T * (p.ℑΘ(t, grid.zC[k]) - c′.T[i, j, k])
@inline FS_μ(i, j, k, grid, t, ũ′, c′, p) = @inbounds p.μ.S * (p.ℑS(t, grid.zC[k]) - c′.S[i, j, k])

forcings = ModelForcing(u=Fu′, v=Fv′, w=Fw′, T=Fθ′, S=Fs′)

#####
##### Set up boundary conditions to
#####   1. impose wind stresses at the ocean surface, and
#####   2. impose heat and salt fluxes at the ocean surface.
#####

# Physical constants.
const ρ₀ = 1027.0  # Density of seawater [kg/m³]
const cₚ = 4000.0  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

@inline wind_stress_x(i, j, grid, t, I, ũ′, c′, p) = p.ℑτx(t) / ρ₀
@inline wind_stress_y(i, j, grid, t, I, ũ′, c′, p) = p.ℑτy(t) / ρ₀
@inline     heat_flux(i, j, grid, t, I, ũ′, c′, p) = - p.ℑQθ(t) / ρ₀ / cₚ
@inline     salt_flux(i, j, grid, t, I, ũ′, c′, p) =   p.ℑQs(t) / ρ₀  # Minus sign because a freshwater flux would decrease salinity.

u′_bcs = UVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(wind_stress_x))
v′_bcs = UVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(wind_stress_y))
θ′_bcs =    TracerBoundaryConditions(grid, top=FluxBoundaryCondition(heat_flux))
s′_bcs =    TracerBoundaryConditions(grid, top=FluxBoundaryCondition(salt_flux))

#####
##### Model setup
#####

model = IncompressibleModel(
    architecture = arch,
    float_type = FT,
    grid = grid,
    tracers = (:T, :S),
    coriolis = FPlane(latitude=lat),
    boundary_conditions = (u=u′_bcs, v=v′_bcs, T=θ′_bcs, S=s′_bcs),
    closure = AnisotropicMinimumDissipation(),
    forcing = forcings,
    parameters = (ℑτx=ℑτx, ℑτy=ℑτy, ℑQθ=ℑQθ, ℑQs=ℑQs, ℑU=ℑU, ℑV=ℑV, ℑΘ=ℑΘ, ℑS=ℑS, μ=μ)
)

#####
##### Initial conditions
#####

ε(μ) = μ * randn() # noise

U₀(x, y, z) = ℑU(0, z)
V₀(x, y, z) = ℑV(0, z)
W₀(x, y, z) = ε(1e-10)
Θ₀(x, y, z) = ℑΘ(0, z)
S₀(x, y, z) = ℑS(0, z)

Oceananigans.set!(model, u=U₀, v=V₀, w=W₀, T=Θ₀, S=S₀)

#####
##### Setting up diagnostics
#####

Up = HorizontalAverage(model.velocities.u,     return_type=Array)
Vp = HorizontalAverage(model.velocities.v,     return_type=Array)
Wp = HorizontalAverage(model.velocities.w,     return_type=Array)
Tp = HorizontalAverage(model.tracers.T,        return_type=Array)
Sp = HorizontalAverage(model.tracers.S,        return_type=Array)

νp  = HorizontalAverage(model.diffusivities.νₑ,   return_type=Array)
κTp = HorizontalAverage(model.diffusivities.κₑ.T, return_type=Array)
κSp = HorizontalAverage(model.diffusivities.κₑ.S, return_type=Array)

u, v, w = model.velocities
T, S = model.tracers

uu = HorizontalAverage(u*u, model, return_type=Array)
vv = HorizontalAverage(v*v, model, return_type=Array)
ww = HorizontalAverage(w*w, model, return_type=Array)
uv = HorizontalAverage(u*v, model, return_type=Array)
uw = HorizontalAverage(u*w, model, return_type=Array)
vw = HorizontalAverage(v*w, model, return_type=Array)
wT = HorizontalAverage(w*T, model, return_type=Array)
wS = HorizontalAverage(w*S, model, return_type=Array)

#####
##### Setting up output writers
#####

filename_prefix = "lesbrary_lat$(lat)_lon$(lon)_days$(days)"

global_attributes = Dict(
    "creator" => "CliMA Ocean LESbrary project",
    "creation time" => string(Dates.now()),
    "lat" => lat, "lon" => lon
)

output_attributes = Dict(
    "τx" => Dict("longname" => "Wind stress in the x-direction", "units" => "N/m"),
    "τy" => Dict("longname" => "Wind stress in the y-direction", "units" => "N/m"),
    "QT" => Dict("longname" => "Net surface heat flux into the ocean (+=down), >0 increases theta", "units" => "W/m²"),
    "QS" => Dict("longname" => "net surface freshwater flux into the ocean (+=down), >0 decreases salinity", "units" => "kg/m²/s"),
    "ν"  => Dict("longname" => "Eddy viscosity", "units" => "m²/s"),
    "κT" => Dict("longname" => "Eddy diffusivity of conservative temperature", "units" => "m²/s"),
    "κS" => Dict("longname" => "Eddy diffusivity of absolute salinity", "units" => "m²/s"),
    "uu" => Dict("longname" => "Velocity covariance between u and u", "units" => "m²/s²"),
    "vv" => Dict("longname" => "Velocity covariance between v and v", "units" => "m²/s²"),
    "ww" => Dict("longname" => "Velocity covariance between w and w", "units" => "m²/s²"),
    "uv" => Dict("longname" => "Velocity covariance between u and v", "units" => "m²/s²"),
    "uw" => Dict("longname" => "Velocity covariance between u and w", "units" => "m²/s²"),
    "vw" => Dict("longname" => "Velocity covariance between v and w", "units" => "m²/s²"),
    "wT" => Dict("longname" => "Vertical turbulent heat flux", "units" => "K*m/s"),
    "wS" => Dict("longname" => "Vertical turbulent salinity flux", "units" => "g/kg*m/s")
)

#####
##### Fields and slices output writers
#####

fields = Dict(
    "u"  => model.velocities.u,
    "v"  => model.velocities.v,
    "w"  => model.velocities.w,
    "T"  => model.tracers.T,
    "S"  => model.tracers.S,
    "ν"  => model.diffusivities.νₑ,
    "κT" => model.diffusivities.κₑ.T,
    "κS" => model.diffusivities.κₑ.S
)

field_output_writer =
    NetCDFOutputWriter(model, fields, filename=filename_prefix * "_fields.nc", interval=6hour,
                      global_attributes=global_attributes, output_attributes=output_attributes)

surface_output_writer =
    NetCDFOutputWriter(model, fields, filename=filename_prefix * "_surface.nc", interval=10minute,
                      global_attributes=global_attributes, output_attributes=output_attributes,
                      zC=Nz, zF=Nz)

slice_output_writer =
    NetCDFOutputWriter(model, fields, filename=filename_prefix * "_slice.nc", interval=6hour,
                      global_attributes=global_attributes, output_attributes=output_attributes,
                      xC=1, xF=1)

#####
##### Horizontal averages output writer
#####

profiles = Dict(
    "u"  => model ->  Up(model)[1+model.grid.Hz:end-model.grid.Hz],
    "v"  => model ->  Vp(model)[1+model.grid.Hz:end-model.grid.Hz],
    "w"  => model ->  Wp(model)[1+model.grid.Hz:end-model.grid.Hz],
    "T"  => model ->  Tp(model)[1+model.grid.Hz:end-model.grid.Hz],
    "S"  => model ->  Sp(model)[1+model.grid.Hz:end-model.grid.Hz],
    "ν"  => model ->  νp(model)[1+model.grid.Hz:end-model.grid.Hz],
    "κT" => model -> κTp(model)[1+model.grid.Hz:end-model.grid.Hz],
    "κS" => model -> κSp(model)[1+model.grid.Hz:end-model.grid.Hz],
    "uu" => model ->  uu(model)[1+model.grid.Hz:end-model.grid.Hz],
    "vv" => model ->  vv(model)[1+model.grid.Hz:end-model.grid.Hz],
    "ww" => model ->  ww(model)[1+model.grid.Hz:end-model.grid.Hz],
    "uv" => model ->  uv(model)[1+model.grid.Hz:end-model.grid.Hz],
    "uw" => model ->  uw(model)[1+model.grid.Hz:end-model.grid.Hz],
    "vw" => model ->  vw(model)[1+model.grid.Hz:end-model.grid.Hz],
    "wT" => model ->  wT(model)[1+model.grid.Hz:end-model.grid.Hz],
    "wS" => model ->  wS(model)[1+model.grid.Hz:end-model.grid.Hz]
)

profile_dims = Dict(k => ("zC",) for k in keys(profiles))
profile_dims["w"] = ("zF",)

profile_output_writer =
    NetCDFOutputWriter(model, profiles, filename=filename_prefix * "_profiles.nc", interval=10minute,
                      global_attributes=global_attributes, output_attributes=output_attributes,
                      dimensions=profile_dims)

#####
##### Large scale solution output writer
#####

large_scale_outputs = Dict(
    "τx" => model -> ℑτx.(model.clock.time),
    "τy" => model -> ℑτy.(model.clock.time),
    "QT" => model -> ℑQθ.(model.clock.time),
    "QS" => model -> ℑQs.(model.clock.time),
     "u" => model ->  ℑU.(model.clock.time, model.grid.zC),
     "v" => model ->  ℑV.(model.clock.time, model.grid.zC),
     "T" => model ->  ℑΘ.(model.clock.time, model.grid.zC),
     "S" => model ->  ℑS.(model.clock.time, model.grid.zC)
)

large_scale_dims = Dict(
    "τx" => (), "τy" => (), "QT" => (), "QS" => (),
    "u" => ("zC",), "v" => ("zC",), "T" => ("zC",), "S" => ("zC",)
)


large_scale_output_writer =
    NetCDFOutputWriter(model, large_scale_outputs, filename=filename_prefix * "_large_scale.nc", interval=10minute,
                      global_attributes=global_attributes, output_attributes=output_attributes,
                      dimensions=large_scale_dims)

#####
##### Banner!
#####

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

#####
##### Set up and run simulation
#####

wizard = TimeStepWizard(cfl=0.1, Δt=1.0, max_change=1.2, max_Δt=5.0)

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)

function print_progress(simulation)
    model = simulation.model
    
    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / simulation.stop_time)

    # Find maximum velocities.
    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    # Find maximum ν and κ.
    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)

    # Print progress statement.
    i, t = model.clock.iteration, model.clock.time
    @printf("[%06.2f%%] i: %d, t: %.3f days, umax: (%.2e, %.2e, %.2e) m/s, CFL: %.2e, νκmax: (%.2e, %.2e), νκCFL: %.2e, next Δt: %.2e s\n",
            progress, i, t / day, umax, vmax, wmax, cfl(model), νmax, κmax, dcfl(model), simulation.Δt.Δt)
end

simulation = Simulation(model, Δt=wizard, stop_time=1day, progress_frequency=20, progress=print_progress)

simulation.output_writers[:fields] = field_output_writer
simulation.output_writers[:surface] = surface_output_writer
simulation.output_writers[:slice] = field_output_writer
simulation.output_writers[:profiles] = profile_output_writer
simulation.output_writers[:large_scale] = large_scale_output_writer

run!(simulation)

