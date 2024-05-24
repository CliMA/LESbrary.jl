using Oceananigans
using JLD2
using GLMakie

set_theme!(Theme(fontsize=28))

dir = "/Users/gregorywagner/Projects/LESbrary.jl/data"

fig = Figure(size=(1500, 1200))

#ax_b1 = Axis(fig[1, 4], xlabel="Temperature (ᵒC)", ylabel="z (m)")
#ax_b2 = Axis(fig[2, 4], xlabel="Temperature (ᵒC)", ylabel="z (m)")
#ax_u = Axis(fig[c, 3], xlabel="Velocities (m s⁻¹)", ylabel="z (m)", yaxisposition=:right)

ax_per_row = 3

case1 = (
    suite = "6_hour_suite",
    res = "1m",
    prefix = "strong_wind_no_rotation",
    row = 1,
    label = "(a)",
)

case2 = (
    suite = "6_hour_suite",
    res = "1m",
    prefix = "strong_wind_weak_cooling",
    row = 1,
    label = "(b)",
)

case3 = (
    suite = "6_hour_suite",
    res = "1m",
    prefix = "free_convection",
    row = 1,
    label = "(c)",
)

case4 = (
    suite = "12_hour_suite",
    res = "1m",
    prefix = "strong_wind_no_rotation",
    row = 2,
    label = "(d)",
)

case5 = (
    suite = "12_hour_suite",
    res = "1m",
    prefix = "strong_wind_weak_cooling",
    row = 2,
    label = "(e)",
)

case6 = (
    suite = "12_hour_suite",
    res = "1m",
    prefix = "free_convection",
    row = 2,
    label = "(f)",
)

case7 = (
    suite = "72_hour_suite",
    res = "1m",
    prefix = "strong_wind_no_rotation",
    row = 3,
    label = "(g)",
)

case8 = (
    suite = "72_hour_suite",
    res = "1m",
    prefix = "strong_wind_weak_cooling",
    row = 3,
    label = "(h)",
)

case9 = (
    suite = "72_hour_suite",
    res = "1m",
    prefix = "free_convection",
    row = 3,
    label = "(i)",
)

cases = [case1, case2, case3,
         case4, case5, case6,
         case7, case8, case9]

for (c, case) in enumerate(cases)

    suite = case.suite
    res = case.res
    prefix = case.prefix
    row = case.row
    label = case.label

    xy_filepath = joinpath(dir, suite, res, prefix * "_xy_slice.jld2")
    yz_filepath = joinpath(dir, suite, res, prefix * "_yz_slice.jld2")
    xz_filepath = joinpath(dir, suite, res, prefix * "_xz_slice.jld2")

    w_xy_t = FieldTimeSeries(xy_filepath, "w")
    w_xz_t = FieldTimeSeries(xz_filepath, "w")
    w_yz_t = FieldTimeSeries(yz_filepath, "w")

    statistics_filepath = joinpath(dir, suite, res, prefix * "_instantaneous_statistics.jld2")

    times = w_xy_t.times
    Nt = length(times)

    grid = w_xy_t.grid

    Nx, Ny, Nz = size(grid)
    x, y, z = nodes(w_xy_t)
    Lx = grid.Lx
    Ly = grid.Ly
    Lz = grid.Lz

    Nz += 1

    x_xz = repeat(x, 1, Nz)
    z_xz = repeat(reshape(z, 1, Nz), Nx, 1)
    y_xz = 0.995 * Ly * ones(Nx, Nz)

    y_yz = repeat(y, 1, Nz)
    z_yz = repeat(reshape(z, 1, Nz), grid.Ny, 1)
    x_yz = 0.995 * Lx * ones(Ny, Nz)

    # Slight displacements to "stitch" the cube together
    x_xy = x
    y_xy = y
    z_xy = -0.001 * Lz * ones(Nx, Ny)

    azimuth = 6.7
    elevation = 0.50
    perspectiveness = 0.1
    xlabel = "x (m)"
    ylabel = "y (m)"
    zlabel = "z (m)"
    aspect = :data
    xlabeloffset = 60
    zlabeloffset = 80

    j = c - ax_per_row * (row - 1)
    ax_w = fig[row, j] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness,
                               xlabeloffset, zlabeloffset)

    if c != 2
        hidedecorations!(ax_w)
    end

    n = length(w_xy_t) #140
    w_xy = interior(w_xy_t[n], :, :, 1)
    w_xz = interior(w_xz_t[n], :, 1, :)
    w_yz = interior(w_yz_t[n], 1, :, :)

    wmax = maximum(abs, w_xy)
    wlim = wmax / 2
    @show wmax
    colorrange_w = (-wlim, wlim)
    colormap_w = :balance

    pl = surface!(ax_w, x_xz, y_xz, z_xz; color=w_xz, colormap=colormap_w, colorrange=colorrange_w)
         surface!(ax_w, x_yz, y_yz, z_yz; color=w_yz, colormap=colormap_w, colorrange=colorrange_w)
         surface!(ax_w, x_xy, y_xy, z_xy; color=w_xy, colormap=colormap_w, colorrange=colorrange_w)

    xtext = 440
    ytext = 0
    ztext = 120
    text!(ax_w, xtext, ytext, ztext, text=label)

    xlims!(ax_w, 0, 512)
    ylims!(ax_w, 0, 512)
    zlims!(ax_w, -256, 0)
end

display(fig)
save("les_w_visualization.png", fig)

