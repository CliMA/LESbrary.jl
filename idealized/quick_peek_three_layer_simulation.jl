using Oceananigans
using GLMakie
#using CairoMakie

basedir = "/Users/gregorywagner/Projects/LESbrary.jl/data"
suite = "12_hour_suite"
resolution = "1m"
filename = "strong_wind_no_stokes_xz_slice.jld2"

set_theme!(Theme(fontsize=24, linewidth=3))
fig = Figure(size=(1200, 1200))
ax = Axis(fig[1, 1])
datadir = joinpath(basedir, suite, resolution)
filepath = joinpath(datadir, filename)
wt = FieldTimeSeries(filepath, "w")
wn = interior(wt[end], :, 1, :)
times = wt.times
t = times[end]
@show prettytime(t)

wlim = 3 * maximum(wn) / 4
heatmap!(ax, wn, colormap=:balance, colorrange=(-wlim, wlim))
   
display(fig)


