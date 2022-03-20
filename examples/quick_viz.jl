using JLD2, Statistics, Random
using GLMakie

jl_file = jldopen("/storage4/andre/lesbrary_mesoscale/eddying_channel_zonal_time_average.jld2")

t_keys = keys(jl_file["timeseries"]["b"])[2:end]

fig = Figure()
ax = Axis(fig[1, 1])
b_field = jl_file["timeseries"]["b"][t_keys[end]][1, :, :]
field = jl_file["timeseries"]["vb"][t_keys[end]][1, :, :] #u, v, w, b, 
clims = quantile.(Ref(field[:]), (0.05, 0.95))
println("clims are ", clims)

heatmap1 = heatmap!(ax, 0 .. 1e6, -3000 .. 0, field, interpolate=true, colormap=:balance, colorrange=clims)
contour!(ax, 0 .. 1e6, -3000 .. 0, b_field, levels=20, linewidth=4, color=:black, alpha=0.5)

##
jlfts = jldopen("/storage4/andre/lesbrary_mesoscale/eddying_channel_top_slice.jld2") #jld2 file top slice

t_keys = keys(jlfts["timeseries"]["b"])
top_bfield = jlfts["timeseries"]["b"][t_keys[end]][:, :, 1]

##
fig2 = Figure()
ax2 = Axis(fig2[1, 1])
heatmap!(ax2, top_bfield, interpolate=true, colormap=:thermometer)

##
using NCDatasets
ds = Dataset("eddying_channel_check_output_zonal_time_averaged_statistics.nc")

keys(ds)
b_field = ds["b"][:, :, end]
field = ds["vb"][:, :, end]

fig = Figure()
ax = Axis(fig[1, 1])

clims = quantile.(Ref(field[:]), (0.05, 0.95))
println("clims are ", clims)
heatmap1 = heatmap!(ax, 0 .. 1e6, -3000 .. 0, field, interpolate=true, colormap=:balance, colorrange=clims)
contour!(ax, 0 .. 1e6, -3000 .. 0, b_field, levels=20, linewidth=4, color=:black, alpha=0.5)