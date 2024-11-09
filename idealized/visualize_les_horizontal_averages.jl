using Oceananigans
using JLD2
using GLMakie
using Statistics

fonts = (; regular=texfont())
set_theme!(Theme(fontsize=32, linewidth=6; fonts))

dir = "/Users/gregorywagner/Projects/LESbrary.jl/data"

fig = Figure(size=(1680, 670))

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

case = case5

suite = case.suite
res = case.res
prefix = case.prefix
row = case.row
label = case.label

xy_filepath = joinpath(dir, suite, res, prefix * "_xy_slice.jld2")
yz_filepath = joinpath(dir, suite, res, prefix * "_yz_slice.jld2")
xz_filepath = joinpath(dir, suite, res, prefix * "_xz_slice.jld2")

# T_xy_t = FieldTimeSeries(xy_filepath, "w")
# T_xz_t = FieldTimeSeries(xz_filepath, "w")
# T_yz_t = FieldTimeSeries(yz_filepath, "w")

T_xy_t = FieldTimeSeries(xy_filepath, "T")
T_xz_t = FieldTimeSeries(xz_filepath, "T")
T_yz_t = FieldTimeSeries(yz_filepath, "T")

statistics_filepath = joinpath(dir, suite, res, prefix * "_instantaneous_statistics.jld2")

Tt = FieldTimeSeries(statistics_filepath, "T")
Bt = FieldTimeSeries(statistics_filepath, "b")
Ut = FieldTimeSeries(statistics_filepath, "u")
Vt = FieldTimeSeries(statistics_filepath, "v")
W²t = FieldTimeSeries(statistics_filepath, "ww")
Et = FieldTimeSeries(statistics_filepath, "e")

Tn = interior(Tt[end], 1, 1, :)
Bn = interior(Bt[end], 1, 1, :)
Un = interior(Ut[end], 1, 1, :)
Vn = interior(Vt[end], 1, 1, :)
En = interior(Et[end], 1, 1, :)
W²n = interior(W²t[end], 1, 1, :)

times = T_xy_t.times
Nt = length(times)
grid = T_xy_t.grid

Nx, Ny, Nz = size(grid)
x, y, z = nodes(T_xy_t)
Lx = grid.Lx
Ly = grid.Ly
Lz = grid.Lz

# Nz += 1

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
xlabel = L"x \, \mathrm{(m)}"
ylabel = L"y \, \mathrm{(m)}"
zlabel = L"z \, \mathrm{(m)}"
aspect = :data
xlabeloffset = 90
ylabeloffset = 70
zlabeloffset = 100

ax_b = fig[2, 1] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness,
                         xlabeloffset, ylabeloffset, zlabeloffset)

n = length(T_xy_t) #140
T_xy = interior(T_xy_t[n], :, :, 1)
T_xz = interior(T_xz_t[n], :, 1, :)
T_yz = interior(T_yz_t[n], 1, :, :)

T_xy .-= Tn[end-1]
T_xz .-= reshape(Tn, 1, Nz)
T_yz .-= reshape(Tn, 1, Nz)

ϵ = mean(Bn ./ Tn) # <<eyes>>
b_xy = ϵ .* T_xy
b_xz = ϵ .* T_xz
b_yz = ϵ .* T_yz

bmax = maximum(abs, b_xz)
blim = bmax / 6
colorrange_b = (-blim, blim)
colormap_b = :balance

pl = surface!(ax_b, x_xz, y_xz, z_xz; color=b_xz, colormap=colormap_b, colorrange=colorrange_b)
     surface!(ax_b, x_yz, y_yz, z_yz; color=b_yz, colormap=colormap_b, colorrange=colorrange_b)
     surface!(ax_b, x_xy, y_xy, z_xy; color=b_xy, colormap=colormap_b, colorrange=colorrange_b)

xlims!(ax_b, 0, 512)
ylims!(ax_b, 0, 512)
zlims!(ax_b, -256, 0)

ticks = ([-2e-5, -1e-5, 0, 1e-5, 2e-5], ["-2", "-1", "0", "1", "2"])
Colorbar(fig[1, 1], pl; ticks, label=L"\mathrm{Buoyancy \, perturbation} \, b\prime \, (10^{-5} \times \mathrm{m \, s^{-2}})", vertical=false, width=Relative(0.7))

# Horizontal averages
xticks = ([0, 1e-4, 2e-4, 3e-4, 4e-4], ["0", "1", "2", "3", "4"])
ax_B = Axis(fig[1:2, 2]; xticks, xlabel="Buoyancy \n (10⁻⁴ × m s⁻²)", ylabel=L"z \, \mathrm{(m)}")

z = znodes(Bt)
lines!(ax_B, Bn .- Bn[1], z)

xticks = ([-0.1, 0, 0.1], ["-0.1", "0", "0.1"])
ax_u = Axis(fig[1:2, 3]; xticks, xlabel="Velocities \n (m s⁻¹)", ylabel=L"z \, \mathrm{(m)}")

lines!(ax_u, Un, z, color=:black, label=L"u")
lines!(ax_u, Vn, z, color=:forestgreen, label=L"v")
xlims!(ax_u, -0.19, 0.19)
axislegend(ax_u, position=:lb)

xticks = ([0, 2e-3, 4e-3], ["0", "2", "4"])
ax_e = Axis(fig[1:2, 4]; xticks, xlabel="Kinetic energies \n (10⁻³ × m² s⁻²)", ylabel=L"z \, \mathrm{(m)}", yaxisposition=:right)

zw = znodes(W²t)

lines!(ax_e, En, z,   label=L"\mathscr{E}")
lines!(ax_e, W²n, zw, label=L"w'^2")
axislegend(ax_e, position=:rb)

#colsize!(fig.layout, 1, Relative(0.6))
colsize!(fig.layout, 2, Relative(0.17))
colsize!(fig.layout, 3, Relative(0.17))
colsize!(fig.layout, 4, Relative(0.17))

hidespines!(ax_B, :t, :r)
hidespines!(ax_u, :t, :l, :r)
hidespines!(ax_e, :t, :l)
hideydecorations!(ax_u, grid=false)

xtext = 440
ytext = 0
ztext = 120
text!(ax_b, xtext, ytext, ztext, text="(a)")

text!(ax_B, 0.95, 0.03, text="(b)", align=(:right, :bottom), space=:relative)
text!(ax_u, 0.95, 0.03, text="(c)", align=(:right, :bottom), space=:relative)
text!(ax_e, 0.95, 0.25, text="(d)", align=(:right, :bottom), space=:relative)

display(fig)

save("les_horizontal_averages.png", fig)

