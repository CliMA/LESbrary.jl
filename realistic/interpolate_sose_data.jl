using RealisticLESbrary

using Interpolations: Gridded, Linear, interpolate

function interpolate_surface_forcings(sose_surface_forcings, times; array_type=Array{Float64})
    Δt = times[2] - times[1]
    t_max = last(times)
    return (
         τx = InterpolatedProfile(array_type(sose_surface_forcings.τx),  times, Δt, t_max),
         τy = InterpolatedProfile(array_type(sose_surface_forcings.τy),  times, Δt, t_max),
         Qθ = InterpolatedProfile(array_type(sose_surface_forcings.Qθ),  times, Δt, t_max),
         Qs = InterpolatedProfile(array_type(sose_surface_forcings.Qs),  times, Δt, t_max),
        mld = InterpolatedProfile(array_type(sose_surface_forcings.mld), times, Δt, t_max)
    )
end

function interpolate_profile(profile, sose_grid, grid, times, array_type)
    Δt = times[2] - times[1]
    t_max = last(times)
    z_max = grid.zC[grid.Nz]

    ## First we interpolate from the SOSE stretched grid onto the Oceananigans.jl regular grid.

    # Coordinates needs to be in increasing order for Interpolations.jl.
    sose_zC = reverse(sose_grid.zC)
    profile = reverse(profile, dims=2)

    ℑprofile = interpolate((times, sose_zC), profile, Gridded(Linear()))

    interior_zC = grid.zC[1:grid.Nz]
    interpolated_data = ℑprofile.(times', interior_zC)

    ## Then we construct and return an InterpolatedProfileTimeSeries for fast linear interpolation in kernels.

    return InterpolatedProfileTimeSeries(array_type(interpolated_data), interior_zC, times, grid.Δz, Δt, z_max, t_max)

    return ℑprofile
end

function interpolate_profiles(sose_profiles, sose_grid, grid, times; array_type=Array{Float64})
    return (
        U    = interpolate_profile(sose_profiles.U,    sose_grid, grid, times, array_type),
        V    = interpolate_profile(sose_profiles.V,    sose_grid, grid, times, array_type),
        Θ    = interpolate_profile(sose_profiles.Θ,    sose_grid, grid, times, array_type),
        S    = interpolate_profile(sose_profiles.S,    sose_grid, grid, times, array_type),
        Ugeo = interpolate_profile(sose_profiles.Ugeo, sose_grid, grid, times, array_type),
        Vgeo = interpolate_profile(sose_profiles.Vgeo, sose_grid, grid, times, array_type)
    )
end
