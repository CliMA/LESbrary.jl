using NCDatasets, Statistics

files = []
descriptor = []

nb_qd_h = Dataset("/storage4/andre/lesbrary_mesoscale/eddying_channel_without_beta_half_tau_zonal_time_averaged_statistics.nc")
push!(files, nb_qd_h)
push!(descriptor, "half the windstress and no beta")

nb_qd = Dataset("/storage4/andre/lesbrary_mesoscale/eddying_channel_without_beta_quadratic_drag_zonal_time_averaged_statistics.nc")
push!(files, nb_qd)
push!(descriptor, "default windstress and no beta")

nb_qd_dub = Dataset("/storage4/andre/lesbrary_mesoscale/eddying_channel_without_beta_double_tau_zonal_time_averaged_statistics.nc")
push!(files, nb_qd_dub)
push!(descriptor, "double windstress and no beta")

b_qd = Dataset("/storage4/andre/lesbrary_mesoscale/eddying_channel_with_beta_quadratic_drag_zonal_time_averaged_statistics.nc")
push!(files, b_qd)
push!(descriptor, "default windstress and with beta")

b_qd_h = Dataset("/storage4/andre/lesbrary_mesoscale/eddying_channel_with_beta_half_tau_zonal_time_averaged_statistics.nc")
push!(files, b_qd_h)
push!(descriptor, "half the windstress and with beta")

b_qd_dub = Dataset("/storage4/andre/lesbrary_mesoscale/eddying_channel_with_beta_double_tau_zonal_time_averaged_statistics.nc")
push!(files, b_qd_dub)
push!(descriptor, "double the windstress and with beta")

b_qd_nt = Dataset("/storage4/andre/lesbrary_mesoscale/eddying_channel_with_beta_no_tau_zonal_time_averaged_statistics.nc")
push!(files, b_qd_nt)
push!(descriptor, "no windstress and with beta")

for (i, cfile) in enumerate(files)
    vb = cfile["vb"][:, :, end]
    v = (cfile["v"][1:end-1, :, end] + cfile["v"][2:end, :, end]) * 0.5
    b = cfile["b"][:, :, end]

    vpbp = vb .- (v .* b)
    println("---")
    println("for the case with ", descriptor[i])
    println("meriodional flux extrema ", extrema(vpbp))
    qt = quantile.(Ref(vpbp[:]), (0.05, 0.95))
    println("the meriodional flux quantile  (0.05, 0.95) is ", qt)

    qt = quantile.(Ref(vpbp[:]), (0.1, 0.9))
    println("the meriodional flux quantile  (0.1, 0.9) is ", qt)

    println("---")
end

##
#=
i = 3
function plotthing(i)
    println("looking at ", descriptor[i])
    cfile = files[i]
    vb = cfile["vb"][:, 1:end-4, end]
    ∫vb = sum(vb, dims=2)[:, 1]
    y = cfile["yC"][:]
    fig, _, _ = lines(y, ∫vb)
    display(fig)
end


function plotthinghm(i)
    println("looking at ", descriptor[i])
    cfile = files[i]
    vb = cfile["vb"][:, :, end]
    y = cfile["yC"][:]
    z = cfile["zC"][:]
    fig, _, _ = heatmap(y, z, vb, interpolate = true, colormap = :balance, colorrange = (-3e-5, 3e-5))
    display(fig)
end
=#

