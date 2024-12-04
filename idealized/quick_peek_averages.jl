using Oceananigans
using GLMakie
#using CairoMakie

set_theme!(Theme(fontsize=24, linewidth=3))
basedir = "/Users/gregorywagner/Projects/LESbrary.jl/data"
suite = "48_hour_suite"
resolution = "2m"
suffix = "_instantaneous_statistics.jld2"
datadir = joinpath(basedir, suite, resolution)

cases = [
    "free_convection_with_tracer",
    "strong_wind_and_sunny_with_tracer",
    "strong_wind_no_rotation_with_tracer",
    "strong_wind_with_tracer",
    "strong_wind_weak_cooling_with_tracer",
    "weak_wind_strong_cooling_with_tracer",
    "med_wind_med_cooling_with_tracer",
]

fig = Figure(size=(1400, 500))

axb = Axis(fig[1, 1])
axc = Axis(fig[1, 2])
axu = Axis(fig[1, 3])

#xlims!(axb, -1e-3, 0)
xlims!(axc, -2, 6)

filepath1 = joinpath(datadir, first(cases) * suffix)
b1 = FieldTimeSeries(filepath1, "b")
Nt = length(b1)
z = znodes(b1)
t = b1.times

slider = Slider(fig[2, 1:3], startvalue=Nt, range=1:Nt)
n = slider.value

title = @lift string("t = ", prettytime(t[$n]))
Label(fig[0, 1:3], title)

colors = [:black, :blue, :red, :seagreen, :orange, :purple, :brown]

for (c, case) in enumerate(cases)

    filepath = joinpath(datadir, case * suffix)
    bt = FieldTimeSeries(filepath, "b")
    ct = FieldTimeSeries(filepath, "c")
    ut = FieldTimeSeries(filepath, "u")
    vt = FieldTimeSeries(filepath, "v")

    times = bt.times
    t = times[end]
    @show prettytime(t)

    bn = @lift interior(bt[$n], 1, 1, :)
    cn = @lift interior(ct[$n], 1, 1, :)
    un = @lift interior(ut[$n], 1, 1, :)
    vn = @lift interior(vt[$n], 1, 1, :)

    lines!(axb, bn, z; color=colors[c], linestyle=:solid)
    lines!(axc, cn, z; color=colors[c], linestyle=:solid)
    lines!(axu, un, z; color=colors[c], linestyle=:solid, label="u")
    lines!(axu, vn, z; color=colors[c], linestyle=:dash,  label="v")
end
       
display(fig)

