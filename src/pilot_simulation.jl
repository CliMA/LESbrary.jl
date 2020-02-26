import Dates
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

lat, lon, days = 190, -55, 10

arch = CPU()
FT = Float64

τx = sose.get_scalar_time_series(ds2, "oceTAUX", lat, lon, days) |> Array{FT}
τy = sose.get_scalar_time_series(ds2, "oceTAUY", lat, lon, days) |> Array{FT}
Qθ = sose.get_scalar_time_series(ds2, "oceQnet", lat, lon, days) |> Array{FT}
Qs = sose.get_scalar_time_series(ds2, "SFLUX",   lat, lon, days) |> Array{FT}

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

Nx = Ny = Nz = 32
Lx = Ly = 500.0
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
@inline     heat_flux(i, j, grid, t, I, ũ′, c′, p) = p.ℑQθ(t) / ρ₀ / cₚ
@inline     salt_flux(i, j, grid, t, I, ũ′, c′, p) = p.ℑQs(t)  # FIXME?

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
    forcing = forcings,
    parameters = (ℑτx=ℑτx, ℑτy=ℑτy, ℑQθ=ℑQθ, ℑQs=ℑQs, ℑU=ℑU, ℑV=ℑV, ℑΘ=ℑΘ, ℑS=ℑS, μ=μ)
)

#####
##### Setting up diagnostics
#####

nan_checker = NaNChecker(model, frequency=1000, fields=Dict(:w => model.velocities.w))

Δtₚ = 10minute  # Time interval for computing and saving profiles.

Up = HorizontalAverage(model.velocities.u,     return_type=Array)
Vp = HorizontalAverage(model.velocities.v,     return_type=Array)
Wp = HorizontalAverage(model.velocities.w,     return_type=Array)
Tp = HorizontalAverage(model.tracers.T,        return_type=Array)
Sp = HorizontalAverage(model.tracers.S,        return_type=Array)
νp = HorizontalAverage(model.diffusivities.νₑ, return_type=Array)

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
    "creation time" => Dates.now(),
    "lat" => lat, "lon" => lon
)

output_attributes = Dict(
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

function horizontal_average_interior(model, H)
    Nz, Hz = model.grid.Nz, model.grid.Hz
    _horizontal_average_interior(model) = H(model)[Hz:end-Hz]
    return _horizontal_average_interior
end

profiles = Dict(
    "u" => Up, "v" => Vp, "w" => Wp,
    "T" => Tp, "S" => Sp,
    "ν" => νp, "κT" => κTp, "κS" => κSp,
    "uu" => uu, "vv" => vv, "ww" => ww,
    "uv" => uv, "uw" => uw, "vw" => vw,
    "wT" => wT, "wS" => wS
)

profile_dims = Dict(k => "zC" for k in keys(profiles))
profile_dims["ww"] = "zF"

profile_output_writer =
    NetCDFOutputWriter(model, profiles, filename=filename_prefix * "_profiles.nc", interval=6hour,
                      global_attributes=global_attributes, output_attributes=output_attributes,
                      dimensions=profile_dims)

large_scale_outputs = Dict(
    "τx" => model -> ℑτx.(model.clock.time),
    "τy" => model -> ℑτy.(model.clock.time),
    "QT" => model -> ℑQθ.(model.clock.time),
    "QS" => model -> ℑQs.(model.clock.time),
    "u" => model -> ℑU.(model.clock.time, model.grid.zC),
    "v" => model -> ℑV.(model.clock.time, model.grid.zC),
    "T" => model -> ℑΘ.(model.clock.time, model.grid.zC),
    "S" => model -> ℑS.(model.clock.time, model.grid.zC)
)

simulation = Simulation(model, Δt=..., stop_time=days*day)

simulation.output_writers[:fields] = field_output_writer
