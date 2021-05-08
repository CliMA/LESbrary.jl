using PyCall
using Conda

# Install needed Python packages
Conda.add("xarray")
Conda.add_channel("conda-forge")
Conda.add("xgcm", channel="conda-forge")
Conda.add("netcdf4")

# Needed to import local Python modules like sose_data
# See: https://github.com/JuliaPy/PyCall.jl/issues/48
py"""
import sys
sys.path.insert(0, ".")
"""

sose = pyimport("sose_data")

function load_sose_data(sose_dir, lat, lon, day_offset, n_days, grid, buoyancy, coriolis; array_type=Array{Float64})

    @info "Opening SOSE datasets..."

    ds2 = sose.open_sose_2d_datasets(sose_dir)
    ds3 = sose.open_sose_3d_datasets(sose_dir)

    times = sose.get_times(ds2)

    zC = ds3.Z.values
    zF = ds3.Zl.values

    @info "Extracting surface forcings and profiles..."

    τx  = sose.get_scalar_time_series(ds2, "oceTAUX",  lat, lon, day_offset, n_days) |> array_type
    τy  = sose.get_scalar_time_series(ds2, "oceTAUY",  lat, lon, day_offset, n_days) |> array_type
    Qθ  = sose.get_scalar_time_series(ds2, "oceQnet",  lat, lon, day_offset, n_days) |> array_type
    Qs  = sose.get_scalar_time_series(ds2, "oceFWflx", lat, lon, day_offset, n_days) |> array_type
    mld = sose.get_scalar_time_series(ds2, "BLGMLD",   lat, lon, day_offset, n_days) |> array_type

    U = sose.get_profile_time_series(ds3, "UVEL",   lat, lon, day_offset, n_days) |> array_type
    V = sose.get_profile_time_series(ds3, "VVEL",   lat, lon, day_offset, n_days) |> array_type
    Θ = sose.get_profile_time_series(ds3, "THETA",  lat, lon, day_offset, n_days) |> array_type
    S = sose.get_profile_time_series(ds3, "SALT",   lat, lon, day_offset, n_days) |> array_type
    N = sose.get_profile_time_series(ds3, "DRHODR", lat, lon, day_offset, n_days) |> array_type

    # Nominal values for α, β to compute geostrophic velocities
    # FIXME: Linear equation of state does not apply!
    # FIXME: Diagnose buoyancy from stratification!
    α = 1.67e-4
    β = 7.80e-4

    @info "Diagnosing geostrophic velocities..."

    Ugeo, Vgeo = sose.compute_geostrophic_velocities(ds3, lat, lon, day_offset, n_days, grid.zF, α, β,
                                                     buoyancy.gravitational_acceleration, coriolis.f)

    Ugeo = array_type(Ugeo)
    Vgeo = array_type(Vgeo)

    ds2.close()
    ds3.close()

    sose_vertical_grid = (; zC, zF)
    surface_forcings = (; τx, τy, Qθ, Qs, mld)
    profiles = (; U, V, Θ, S, N, Ugeo, Vgeo)

    return times, sose_vertical_grid, surface_forcings, profiles
end
