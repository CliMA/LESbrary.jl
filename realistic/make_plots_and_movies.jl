using CairoMakie

function plot_initial_state(sose_profiles, sose_grid, interpolated_profiles, grid, lat, lon, start_date; z_bottom=-grid.Lz, filepath)

    zc_sose = sose_grid.zC
    zf_sose = sose_grid.zF

    U₀ = sose_profiles.U[1, :]
    V₀ = sose_profiles.V[1, :]
    Θ₀ = sose_profiles.Θ[1, :]
    S₀ = sose_profiles.S[1, :]

    Ugeo₀ = sose_profiles.Ugeo[1, :]
    Vgeo₀ = sose_profiles.Vgeo[1, :]

    zc = grid.zC[1:grid.Nz]
    zf = grid.zF[1:grid.Nz+1]

    ℑU₀ = interpolated_profiles.U.(zc, 0)
    ℑV₀ = interpolated_profiles.V.(zc, 0)
    ℑΘ₀ = interpolated_profiles.Θ.(zc, 0)
    ℑS₀ = interpolated_profiles.S.(zc, 0)

    ℑUgeo₀ = interpolated_profiles.Ugeo.(zc, 0)
    ℑVgeo₀ = interpolated_profiles.Vgeo.(zc, 0)

    colors = ["dodgerblue2", "crimson", "forestgreen"]

    fig = Figure(resolution=(2500, 1080))

    ax_U = fig[1, 1] = Axis(fig, xlabel="m/s", ylabel="z (m)")
    lines!(ax_U, U₀, zc_sose, label="U (SOSE)", linewidth=3, color=colors[1])
    lines!(ax_U, ℑU₀, zc, label="U (SOSE, interpolated)", linewidth=3, color=colors[1], linestyle=:dash)
    lines!(ax_U, Ugeo₀, zc_sose, label="U (geo)", linewidth=3, color=colors[2])
    lines!(ax_U, ℑUgeo₀, zc, label="U (geo, interpolated)", linewidth=3, color=colors[2], linestyle=:dash)
    axislegend(ax_U, position=:rb, framevisible=false)
    ylims!(ax_U, (z_bottom, 0))

    ax_V = fig[1, 2] = Axis(fig, xlabel="m/s", ylabel="z (m)")
    lines!(ax_V, V₀, zc_sose, label="V (SOSE)", linewidth=3, color=colors[1])
    lines!(ax_V, ℑV₀, zc, label="V (SOSE, interpolated)", linewidth=3, color=colors[1], linestyle=:dash)
    lines!(ax_V, Vgeo₀, zc_sose, label="V (geo)", linewidth=3, color=colors[2])
    lines!(ax_V, ℑVgeo₀, zc, label="V (geo, interpolated)", linewidth=3, color=colors[2], linestyle=:dash)
    axislegend(ax_V, position=:rb, framevisible=false)
    ylims!(ax_V, (z_bottom, 0))

    ax_Θ = fig[1, 3] = Axis(fig, xlabel="°C", ylabel="z (m)")
    lines!(ax_Θ, Θ₀, zc_sose, label="Θ (SOSE)", linewidth=3, color=colors[3])
    lines!(ax_Θ, ℑΘ₀, zc, label="Θ (SOSE, interpolated)", linewidth=3, color=colors[3], linestyle=:dash)
    axislegend(ax_Θ, position=:rb, framevisible=false)
    ylims!(ax_Θ, (z_bottom, 0))
    xlims!(ax_Θ, extrema(z_bottom < -grid.Lz ? filter(!iszero, Θ₀) : ℑΘ₀))

    ax_S = fig[1, 4] = Axis(fig, xlabel="psu", ylabel="z (m)")
    lines!(ax_S, S₀, zc_sose, label="S (SOSE)", linewidth=3, color=colors[3])
    lines!(ax_S, ℑS₀, zc, label="S (SOSE, interpolated)", linewidth=3, color=colors[3], linestyle=:dash)
    axislegend(ax_S, position=:rb, framevisible=false)
    ylims!(ax_S, (z_bottom, 0))
    xlims!(ax_S, extrema(z_bottom < -grid.Lz ? filter(!iszero, S₀) : ℑS₀))

    plot_title = @sprintf("SOSE LESbrary.jl (%.2f°N, %.2f°E) initial state at %s", lat, lon, start_date)
    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

    @info "Saving $filepath..."
    save(filepath, fig)

    return nothing
end

#=

## Plot surface and slice

using CairoMakie
using GeoData
using NCDatasets

function squeeze(A)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims=singleton_dims)
end

ds_xy = NCDstack(filename_prefix * "_surface.nc")
ds_yz = NCDstack(filename_prefix * "_slice.nc")

times = ds_xy[:time]
Nt = length(times)

xc = ds_xy[:xC].data
xf = ds_xy[:xF].data
yc = ds_xy[:yC].data
yf = ds_xy[:yF].data
zc = ds_yz[:zC].data
zf = ds_yz[:zF].data

fig = Figure(resolution=(1920, 1080))

u_max = max(maximum(abs, ds_xy[:u]), maximum(abs, ds_yz[:u]))
v_max = max(maximum(abs, ds_xy[:v]), maximum(abs, ds_yz[:v]))
w_max = max(maximum(abs, ds_xy[:w]), maximum(abs, ds_yz[:w]))
U_max = max(u_max, v_max, w_max)
U_lims = 0.5 .* (-U_max, +U_max)

frame = Node(1)

plot_title = @lift @sprintf("Realistic SOSE LESbrary.jl (%.2f°N, %.2f°E): time = %s", lat, lon, start_time + Millisecond(round(Int, 1000 * times[$frame])))

u_xy = @lift ds_xy[:u][Ti=$frame].data |> squeeze
v_xy = @lift ds_xy[:v][Ti=$frame].data |> squeeze
w_xy = @lift ds_xy[:w][Ti=$frame].data |> squeeze
T_xy = @lift ds_xy[:T][Ti=$frame].data |> squeeze
S_xy = @lift ds_xy[:S][Ti=$frame].data |> squeeze

u_yz = @lift ds_yz[:u][Ti=$frame].data |> squeeze
v_yz = @lift ds_yz[:v][Ti=$frame].data |> squeeze
w_yz = @lift ds_yz[:w][Ti=$frame].data |> squeeze
T_yz = @lift ds_yz[:T][Ti=$frame].data |> squeeze
S_yz = @lift ds_yz[:S][Ti=$frame].data |> squeeze

ax_u_xy = fig[1, 1] = Axis(fig, title="u-velocity")
hm_u_xy = heatmap!(ax_u_xy, xf, yc, u_xy, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_u_xy)

ax_v_xy = fig[1, 2] = Axis(fig, title="v-velocity")
hm_v_xy = heatmap!(ax_v_xy, xc, yf, v_xy, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_v_xy)

ax_w_xy = fig[1, 3] = Axis(fig, title="w-velocity")
hm_w_xy = heatmap!(ax_w_xy, xc, yc, w_xy, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_w_xy)

ax_T_xy = fig[1, 4] = Axis(fig, title="conservative temperature")
hm_T_xy = heatmap!(ax_T_xy, xc, yc, T_xy, colormap=:thermal, colorrange=extrema(ds_xy[:T]))
hidedecorations!(ax_T_xy)

ax_S_xy = fig[1, 5] = Axis(fig, title="absolute salinity")
hm_S_xy = heatmap!(ax_S_xy, xc, yc, S_xy, colormap=:haline, colorrange=extrema(ds_xy[:S]))
hidedecorations!(ax_S_xy)

ax_u_yz = fig[2, 1] = Axis(fig)
hm_u_yz = heatmap!(ax_u_yz, xf, zc, u_yz, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_u_yz)

ax_v_yz = fig[2, 2] = Axis(fig)
hm_v_yz = heatmap!(ax_v_yz, xc, zc, v_yz, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_v_yz)

ax_w_yz = fig[2, 3] = Axis(fig)
hm_w_yz = heatmap!(ax_w_yz, xc, zf, w_yz, colormap=:balance, colorrange=U_lims)
hidedecorations!(ax_w_yz)

ax_T_yz = fig[2, 4] = Axis(fig)
hm_T_yz = heatmap!(ax_T_yz, xc, zc, T_yz, colormap=:thermal, colorrange=extrema(ds_yz[:T]))
hidedecorations!(ax_T_yz)

ax_S_yz = fig[2, 5] = Axis(fig)
hm_S_yz = heatmap!(ax_S_yz, xc, zc, S_yz, colormap=:haline, colorrange=extrema(ds_yz[:S]))
hidedecorations!(ax_S_yz)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

filepath = filename_prefix * "_surface_slice_movie.mp4"
record(fig, filepath, 1:10:Nt, framerate=30) do n
    @info "Animating surface and slice movie frame $n/$Nt..."
    frame[] = n
end

@info "Movie saved: $filepath"

## Plot background, SOSE, and LES profiles

ds_p = NCDstack(filename_prefix * "_profiles.nc")
ds_b = NCDstack(filename_prefix * "_large_scale.nc")

times = ds_p[:time]
Nt = length(times)

zc = ds_p[:zC].data
zf = ds_p[:zF].data

fig = Figure(resolution=(2500, 1080))

frame = Node(1)

plot_title = @lift @sprintf("Realistic SOSE LESbrary.jl (%.2f°N, %.2f°E): time = %s", lat, lon, start_time + Millisecond(round(Int, 1000 * times[$frame])))

U_LES = @lift ds_p[:U][Ti=$frame].data
V_LES = @lift ds_p[:V][Ti=$frame].data
T_LES = @lift ds_p[:T][Ti=$frame].data
S_LES = @lift ds_p[:S][Ti=$frame].data
B_LES = @lift ds_p[:B][Ti=$frame].data

U_SOSE = @lift ds_b[:u][Ti=$frame].data
V_SOSE = @lift ds_b[:v][Ti=$frame].data
T_SOSE = @lift ds_b[:T][Ti=$frame].data
S_SOSE = @lift ds_b[:S][Ti=$frame].data
∂ρ∂z_SOSE = @lift ds_b[:∂ρ∂z][Ti=$frame].data

U_geo = @lift ds_b[:Ugeo][Ti=$frame].data
V_geo = @lift ds_b[:Vgeo][Ti=$frame].data

# time_so_far = @lift ds_b[:time][1:$frame].data
# τx_SOSE = @lift ds_b[:τx][Ti=1:$frame].data
# τy_SOSE = @lift ds_b[:τy][Ti=1:$frame].data
# QΘ_SOSE = @lift ds_b[:QT][Ti=1:$frame].data
# QS_SOSE = @lift ds_b[:QS][Ti=1:$frame].data

colors = ["dodgerblue2", "crimson", "forestgreen"]

ax_U = fig[1, 1] = Axis(fig, xlabel="m/s", ylabel="z (m)")
line_U_SOSE = lines!(ax_U, U_SOSE, zc, label="U (SOSE)", linewidth=3, color=colors[1], linestyle=:dash)
line_U_geo  = lines!(ax_U, U_geo, zc, label="U (geo)", linewidth=3, color=colors[1], linestyle=:dot)
line_U_LES  = lines!(ax_U, U_LES, zc, label="U (LES)", linewidth=3, color=colors[1])
axislegend(ax_U, position=:rb, framevisible=false)
xlims!(ax_U, extrema([extrema(ds_p[:U])..., extrema(ds_b[:u])..., extrema(ds_b[:Ugeo])...]))
ylims!(ax_U, extrema(zf))

ax_V = fig[1, 2] = Axis(fig, xlabel="m/s", ylabel="z (m)")
line_V_SOSE = lines!(ax_V, V_SOSE, zc, label="V (SOSE)", linewidth=3, color=colors[2], linestyle=:dash)
line_V_geo  = lines!(ax_V, V_geo, zc, label="V (geo)", linewidth=3, color=colors[2], linestyle=:dot)
line_V_LES  = lines!(ax_V, V_LES, zc, label="V (LES)", linewidth=3, color=colors[2])
axislegend(ax_V, position=:rb, framevisible=false)
xlims!(ax_V, extrema([extrema(ds_p[:U])..., extrema(ds_b[:v])..., extrema(ds_b[:Vgeo])...]))
ylims!(ax_V, extrema(zf))

ax_T = fig[1, 3] = Axis(fig, xlabel="°C", ylabel="z (m)")
line_T_SOSE = lines!(ax_T, T_SOSE, zc, label="Θ (SOSE)", linewidth=3, color=colors[3], linestyle=:dash)
line_T_LES  = lines!(ax_T, T_LES, zc, label="Θ (LES)", linewidth=3, color=colors[3])
axislegend(ax_T, position=:rb, framevisible=false)
xlims!(ax_T, extrema([extrema(ds_p[:T])..., extrema(ds_b[:T])...]))
ylims!(ax_T, extrema(zf))

ax_S = fig[1, 4] = Axis(fig, xlabel="psu", ylabel="z (m)")
line_S_SOSE = lines!(ax_S, S_SOSE, zc, label="S (SOSE)", linewidth=3, color=colors[3], linestyle=:dash)
line_S_LES  = lines!(ax_S, S_LES, zc, label="S (LES)", linewidth=3, color=colors[3])
axislegend(ax_S, position=:rb, framevisible=false)
xlims!(ax_S, extrema([extrema(ds_p[:S])..., extrema(ds_b[:S])...]))
ylims!(ax_S, extrema(zf))

ax_B = fig[1, 5] = Axis(fig, xlabel="m/s²", ylabel="z (m)")
line_B_LES  = lines!(ax_B, B_LES, zc, label="B (LES)", linewidth=3, color=colors[3])
axislegend(ax_B, position=:rb, framevisible=false)
xlims!(ax_B, extrema(ds_p[:B]))
ylims!(ax_B, extrema(zf))

ax_N = fig[1, 6] = Axis(fig, xlabel="kg/m⁴", ylabel="z (m)")
line_N_SOSE  = lines!(ax_N, ∂ρ∂z_SOSE, zf, label="∂ρ∂z (SOSE)", linewidth=3, color=colors[3], linestyle=:dash)
axislegend(ax_N, position=:rb, framevisible=false)
xlims!(ax_N, extrema(ds_b[:∂ρ∂z]))
ylims!(ax_N, extrema(zf))

# ax_τ = fig[2, :] = Axis(fig, ylabel="N/m²")
# line_τx = lines!(ax_τ, time_so_far, τx_SOSE, label="τx", linewidth=3, color=colors[1])
# line_τy = lines!(ax_τ, time_so_far, τy_SOSE, label="τy", linewidth=3, color=colors[2])
# axislegend(ax_τ, position=:rb, framevisible=false)
# xlims!(ax_τ, extrema(ds_b[:time]))
# ylims!(ax_τ, extrema([extrema(ds_b[:τx])..., extrema(ds_b[:τy])...]))
# ax_τ.height = Relative(0.15)

# ax_QΘ = fig[3, :] = Axis(fig, ylabel="QΘ (W/m²)")
# line_QΘ = lines!(ax_QΘ, time_so_far, QΘ_SOSE, linewidth=3, color=colors[1])
# xlims!(ax_QΘ, extrema(ds_b[:time]))
# ylims!(ax_QΘ, extrema(ds_b[:QT]))
# ax_QΘ.height = Relative(0.15)

# ax_QS = fig[4, :] = Axis(fig, ylabel="QS (kg/m²/s))")
# line_QS = lines!(ax_QS, time_so_far, QS_SOSE, linewidth=3, color=colors[1])
# xlims!(ax_QS, extrema(ds_b[:time]))
# ylims!(ax_QS, extrema(ds_b[:QS]))
# ax_QS.height = Relative(0.15)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

filepath = filename_prefix * "_first_order_statistics.mp4"
record(fig, filepath, 1:2:Nt, framerate=30) do n
    @info "Animating first-order statistics movie frame $n/$Nt..."
    frame[] = n
end

@info "Movie saved: $filepath"

## Plot surface forcings

ds_b = NCDstack(filename_prefix * "_large_scale.nc")

times = ds_p[:time] / day
Nt = length(times)

# Makie doesn't support DateTime plotting yet :(
# date_times = [start_time + Millisecond(round(Int, 1000t)) for t in times]

τx_SOSE = ds_b[:τx].data
τy_SOSE = ds_b[:τy].data
τ_SOSE = @. √(τx_SOSE^2 + τy_SOSE^2)
QΘ_SOSE = ds_b[:QT].data
QS_SOSE = ds_b[:QS].data
mld_SOSE = ds_b[:mld_SOSE].data
mld_LES = ds_b[:mld_LES].data

fig = Figure(resolution=(1920, 1080))
plot_title = @sprintf("Realistic SOSE LESbrary.jl (%.2f°N, %.2f°E) surface forcings", lat, lon)

ax_τ = fig[1, 1] = Axis(fig, ylabel="N/m²")
line_τx = lines!(ax_τ, times, τx_SOSE, label="τx", linewidth=3, color=colors[1])
line_τy = lines!(ax_τ, times, τy_SOSE, label="τy", linewidth=3, color=colors[2])
line_τ  = lines!(ax_τ, times, τ_SOSE, label="√(τx² + τy²)", linewidth=3, color=colors[3])
axislegend(ax_τ, position=:rb, framevisible=false)
xlims!(ax_τ, extrema(times))
ylims!(ax_τ, extrema([extrema(τx_SOSE)..., extrema(τy_SOSE)..., extrema(τ_SOSE)...]))
hidexdecorations!(ax_τ, grid=false)

ax_QΘ = fig[2, 1] = Axis(fig, ylabel="QΘ (W/m²)")
line_QΘ = lines!(ax_QΘ, times, QΘ_SOSE, linewidth=3, color=colors[3])
xlims!(ax_QΘ, extrema(times))
ylims!(ax_QΘ, extrema(QΘ_SOSE))
hidexdecorations!(ax_τ, grid=false)

ax_QS = fig[3, 1] = Axis(fig, xlabel="time (days)", ylabel="QS (kg/m²/s)")
line_QS = lines!(ax_QS, times, QS_SOSE, linewidth=3, color=colors[3])
xlims!(ax_QS, extrema(times))
ylims!(ax_QS, extrema(QS_SOSE))

ax_mld = fig[4, 1] = Axis(fig, xlabel="time (days)", ylabel="Mixed layer depth (m)")
line_mld_SOSE = lines!(ax_mld, times, mld_SOSE, label="SOSE", linewidth=3, color=colors[3], linestyle=:dash)
line_mld_LES = lines!(ax_mld, times, mld_LES, label="LES", linewidth=3, color=colors[3])
axislegend(ax_mld, position=:rb, framevisible=false)
xlims!(ax_mld, extrema(times))
ylims!(ax_mld, extrema([extrema(mld_SOSE)..., extrema(mld_LES)...]))

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

filepath = filename_prefix * "_surface_forcings.png"
save(filepath, fig)

@info "Figure saved: $filepath"


=#
