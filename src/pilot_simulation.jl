using PyCall

# Needed to import local modules
# See: https://github.com/JuliaPy/PyCall.jl/issues/48
py"""
import sys
sys.path.insert(0, ".")
"""

sose = pyimport("sose_data")

ds2 = sose.open_sose_2d_datasets("/home/alir/cnhlab004/bsose_i122/")
ds3 = sose.open_sose_3d_datasets("/home/alir/cnhlab004/bsose_i122/")

date_times = sose.get_times(ds2)

lat, lon, days = 190, -55, 10

τx   = sose.get_scalar_time_series(ds2, "oceTAUX", lat, lon)
τy   = sose.get_scalar_time_series(ds2, "oceTAUY", lat, lon)
Qnet = sose.get_scalar_time_series(ds2, "oceQnet", lat, lon)

U = sose.get_profile_time_series(ds3, "UVEL",  lat, lon, days)
V = sose.get_profile_time_series(ds3, "VVEL",  lat, lon, days)
Θ = sose.get_profile_time_series(ds3, "THETA", lat, lon, days)
S = sose.get_profile_time_series(ds3, "SALT",  lat, lon, days)

ds2.close()
ds3.close()

ts = day * (0:days-1)
zC = ds3.Z.values

ℑτx = interpolate(ts, τx,   Gridded(Linear()))
ℑτy = interpolate(ts, τy,   Gridded(Linear()))
ℑQ  = interpolate(ts, Qnet, Gridded(Linear()))

ℑU = interpolate((ts, -z), U, Gridded(Linear()))
ℑV = interpolate((ts, -z), V, Gridded(Linear()))
ℑΘ = interpolate((ts, -z), Θ, Gridded(Linear()))
ℑS = interpolate((ts, -z), S, Gridded(Linear()))

Nx = Ny = Nz = 32
Lx = Ly = Lz = 100
topology = (Periodic, Bounded, Bounded)
grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz) x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# Fu′ = - w′∂z(U) - U∂x(u′) - V∂y(u′)
# FIXME? Do we need to use the flux form operator ∇·(Uũ′) instead of ũ′·∇U ?
@inline Fu′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - ũ′.w[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, p.U) - p.U[k] * ∂xᶜᵃᵃ(i, j, k, grid, ũ′.u) - p.V[k] * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.u)

# Fv′ = - w′∂z(V) - U∂x(v′) - V∂y(v′)
@inline Fv′(i, j, k, grid, t, ũ′, c′, p) =
@inbounds - ũ′.w[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, p.V) - p.U[k] * ∂xᶜᵃᵃ(i, j, k, grid, ũ′.v) - p.V[k] * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v)

# Fw′ = - U∂x(w′) - V∂y(w′)
@inline Fw′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - p.U[k] * ∂xᶜᵃᵃ(i, j, k, grid, ũ′.u) - p.V[k] * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v)

# Fθ′ = - ∂t(Θ) - U∂x(θ′) - V∂y(θ′) - w′∂z(Θ)
# FIXME? Do we need to include the ∂t(Θ) terms?
@inline Fθ′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - p.U[k] * ∂xᶠᵃᵃ(i, j, k, grid, c′.T) - p.V[k] * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v) - ũ′.w[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, p.Θ)

# Fs′ = - ∂t(S) - U∂x(s′) - V∂y(s′) - w′∂z(S)
@inline Fs′(i, j, k, grid, t, ũ′, c′, p) =
    @inbounds - p.U[k] * ∂xᶠᵃᵃ(i, j, k, grid, c′.S) - p.V[k] * ∂yᵃᶜᵃ(i, j, k, grid, ũ′.v) - ũ′.w[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, p.S)

forcings = ModelForcing(u=Fu′, v=Fv′, w=Fw′, T=Fθ′, S=Fs′)

# Physical constants.
const ρ₀ = 1027.0  # Density of seawater [kg/m³]
const cₚ = 4000.0  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

u′_bcs = UVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(τx / ρ₀))
v′_bcs = UVelocityBoundaryConditions(grid, top=FluxBoundaryCondition(τy / ρ₀))
θ′_bcs = TracerBoundaryConditions(grid, top=FluxBoundaryCondition(Qnet / ρ / cₚ))
s′_bcs = TracerBoundaryConditions(grid, top=FluxBoundaryConditions(surf_S_flux))  # FIXME???

arch = CPU()
FT = Float64

model = IncompressibleModel(
    architecture = arch,
    float_type = FT,
    grid = grid,
    tracrs = (:T, :S),
    coriolis = FPlane(latitude=lat),
    boundary_conditions = (u=u′_bcs, v=v′_bcs, T=θ′_bcs, S=s′_bcs),
    forcing = forcings
)

