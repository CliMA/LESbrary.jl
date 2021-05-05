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
