using Oceananigans
using CairoMakie
using MathTeXEngine

basedir = "/Users/gregorywagner/Projects/LESbrary.jl/data"
suite = "6_hour_suite"
resolutions = ["1m", "2m", "4m"]

labels = Dict(
    #"0.75m" => "0.75 meter",
    "1m" => "1 meter",
    "2m" => "2 meter",
    "4m" => "4 meter",
)

cases = [
    "free_convection",
    #"weak_wind_strong_cooling",
    #"strong_wind_weak_cooling",
    "med_wind_med_cooling",
    "strong_wind",
    #"strong_wind_no_rotation",
    #"strong_wind_weak_cooling",
    "strong_wind_and_sunny",
]

titles = Dict(
    "free_convection" => "Free \n convection",
    "strong_wind" => "Strong wind",
    "strong_wind_no_rotation" => "Strong wind \n no rotation",
    "weak_wind_strong_cooling" => "Weak wind \n strong cooling",
    "med_wind_med_cooling" => "Mid wind \n mid cooling",
    "strong_wind_weak_cooling" => "Strong wind \n weak cooling",
    "strong_wind_and_sunny" => "Strong wind \n and sunny",
)

# case = "free_convection"

fonts = (; regular=texfont())
set_theme!(Theme(fontsize=24, linewidth=3; fonts))
fig = Figure(size=(1200, 500))
axc = []

for (c, case) in enumerate(cases)
    colors = Makie.wong_colors()
    colors = reverse(colors)

    if c == length(cases)
        yaxisposition = :right
    else
        yaxisposition = :left
    end

    x = 1:9
    xticks = (x .* 1e-4, [string(i) for i in x])

    # if c == 1 || c == 2
    #     xticks = ([1, 2, 3, 4] .* 1e-4, ["1", "2", "3", "4"])
    # else
    #     xticks = ([1, 3, 5, 7] .* 1e-4, ["1", "3", "5", "7", "9"])
    # end
    ax = Axis(fig[2, c], xlabel="Buoyancy \n (10⁻⁴ × m s⁻²)", ylabel=L"z \, \mathrm{(m)}"; yaxisposition, xticks)
    title = titles[case]
    Label(fig[1, c], title, tellwidth=false)
    push!(axc, ax)

    if c != 1 && c != length(cases)
        hideydecorations!(ax, grid=false)
    end

    if c == 1
        hidespines!(ax, :t, :r)
    elseif c == 4
        hidespines!(ax, :t, :l)
    else
        hidespines!(ax, :t, :l, :r)
    end
    
    #res = "1m"
    for res in resolutions

    #for waves in ("", "_no_stokes")
        #filename = string(case, waves, "_instantaneous_statistics.jld2")
        
        #try
            filename = string(case, "_with_tracer_instantaneous_statistics.jld2")
            datadir = joinpath(basedir, suite, res)
            filepath = joinpath(datadir, filename)
            bt = FieldTimeSeries(filepath, "b")
            bn = interior(bt[end], 1, 1, :) .- bt[1][1, 1, 1]
            @show length(bt)
            z = znodes(bt)

            # if waves == ""
            #     label = "with \nStokes drift"
            # else
            #     label = "without"
            # end

            label = labels[res]
            if res == "1m"
            #if waves == ""
                linewidth = 5
                color = (:black, 0.8)
                linestyle = :solid
            else
                linewidth = 3
                linestyle = :dash
                color = pop!(colors)
            end
            lines!(ax, bn, z; label, linewidth, linestyle, color)
        #catch
        #end
    end

    ylims!(ax, -200, 5)
end

Nc = length(cases)
#Legend(fig[2, Nc+1], first(axc))
axislegend(axc[1], position=(0, 0.9))

# For Stokes drift comparison
#xlims!(axc[1], 1.7e-4, 3.7e-4)
#xlims!(axc[2], 1.7e-4, 4.8e-4)
#xlims!(axc[3], 1.7e-4, 6e-4)
#xlims!(axc[4], 1.7e-4, 7.1e-4)

# For 12 hour suite resolution comparison
if suite == "6_hour_suite"
    xlims!(axc[1], 1e-4, 3.3e-4)
    xlims!(axc[2], 1e-4, 3.9e-4)
    xlims!(axc[3], 1e-4, 5.5e-4)
    xlims!(axc[4], 1e-4, 7.0e-4)
elseif suite == "24_hour_suite"
    xlims!(axc[1], 1.7e-4, 3.3e-4)
    xlims!(axc[2], 1.7e-4, 3.9e-4)
    xlims!(axc[3], 1.7e-4, 5.5e-4)
    xlims!(axc[4], 1.7e-4, 9.3e-4)
else
    xlims!(axc[1], 1.7e-4, 3.3e-4)
    xlims!(axc[2], 1.7e-4, 3.9e-4)
    xlims!(axc[3], 1.7e-4, 5.5e-4)
    xlims!(axc[4], 1.7e-4, 8.0e-4)
end

for ax in axc
    ylims!(ax, -160, 0)
end
    
display(fig)
save("les_resolution_dependence_$suite.pdf", fig)

