using JLD2, GLMakie, Statistics, Random

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