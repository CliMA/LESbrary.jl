using Interpolations
using RealisticLESbrary

function interpolate_surface_forcings(sose_surface_forcings, times)
    Δt = times[2] - times[1]
    return (
         τx = InterpolatedProfile(sose_surface_forcings.τx,  times, Δt),
         τy = InterpolatedProfile(sose_surface_forcings.τy,  times, Δt),
         Qθ = InterpolatedProfile(sose_surface_forcings.Qθ,  times, Δt),
         Qs = InterpolatedProfile(sose_surface_forcings.Qs,  times, Δt),
        mld = InterpolatedProfile(sose_surface_forcings.mld, times, Δt)
    )
end

function interpolate_profile(profile, sose_grid, grid, times)
    Δt = times[2] - times[1]

    ## First we interpolate from the SOSE stretched grid onto the Oceananigans.jl regular grid.

    # Coordinates needs to be in increasing order for Interpolations.jl.
    sose_zC = reverse(sose_grid.zC)
    profile = reverse(profile, dims=2)

    ℑprofile = interpolate((times, sose_zC), profile, Gridded(Linear()))

    interior_zC = grid.zC[1:grid.Nz]
    interpolated_data = ℑprofile.(times', interior_zC)

    ## Then we construct and return an InterpolatedProfileTimeSeries for fast linear interpolation in kernels.

    return InterpolatedProfileTimeSeries(interpolated_data, interior_zC, times, grid.Δz, Δt)

    return ℑprofile
end

function interpolate_profiles(sose_profiles, sose_grid, grid, times)
    return (
        U    = interpolate_profile(sose_profiles.U,    sose_grid, grid, times),
        V    = interpolate_profile(sose_profiles.V,    sose_grid, grid, times),
        Θ    = interpolate_profile(sose_profiles.Θ,    sose_grid, grid, times),
        S    = interpolate_profile(sose_profiles.S,    sose_grid, grid, times),
        Ugeo = interpolate_profile(sose_profiles.Ugeo, sose_grid, grid, times),
        Vgeo = interpolate_profile(sose_profiles.Vgeo, sose_grid, grid, times)
    )
end
