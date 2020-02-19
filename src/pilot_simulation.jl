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

u_bcs = UVelocityBoundaryConditions

model = IncompressibleModel(
    architecture = CPU(),
    float_type = Float64,
    grid = grid,
    coriolis = FPlane(latitude=lat)
)

