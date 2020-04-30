module FileManagement

export
    get_iters,
    get_time,
    get_final_time,
    get_grid,
    get_fields,
    get_statistics,
    get_wind_stress,
    get_surface_wave_parameters,
    get_parameter,
    get_multiple_parameters,
    set_from_file!,
    calculate_horizontal_average_timeseries,
    collect_horizontal_averages

using JLD2

function get_iters(filename)
    file = jldopen(filename)
    iters = parse.(Int, keys(file["timeseries/t"]))
    close(file)
    return iters
end

function get_time(filename, i)
    file = jldopen(filename)
    t = file["timeseries/t/$i"]
    close(file)
    return t
end

get_final_time(filename) = get_time(filename, get_iters(filename)[end])

function get_grid(filename)
    file = jldopen(filename)

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]
    Nz = file["grid/Nz"]

    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    Lz = file["grid/Lz"]

    close(file)

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))

    return grid
end

function get_fields(filename, i)
    file = jldopen(filename)

    t = file["timeseries/t/$i"]
    u = file["timeseries/u/$i"]
    v = file["timeseries/v/$i"]
    w = file["timeseries/w/$i"]
    b = file["timeseries/b/$i"]

    close(file)

    return t, u, v, w, b
end

function get_statistics(filename, i)
    t, u, v, w, b = get_fields(filename, i)
    grid = get_grid(filename)
    b′, U, V, B, Bz, E, w², e = calculate_statistics(grid, u, v, w, b)
    return t, b′, U, V, B, Bz, E, w², e
end

function get_averages(filename, i)

    U = mean(u, dims=(1, 2))
    V = mean(v, dims=(1, 2))
    B = mean(b, dims=(1, 2))
    W² = mean(w.^2, dims=(1, 2))

    return t, U, V, W², B
end

function get_wind_stress(filename)
    file = jldopen(filename)
    τ = abs(file["boundary_conditions/Qᵘ₀"])
    close(file)
    return τ
end

function get_surface_wave_parameters(filename)
    file = jldopen(filename)

    aˢʷ = file["surface_waves/aˢʷ"]
    kˢʷ = file["surface_waves/kˢʷ"]

    close(file)

    return aˢʷ, kˢʷ
end

function get_parameter(filename, group, parameter_name)
    parameter = nothing

    jldopen(filename) do file

        if parameter_name ∈ keys(file["$group"])
            parameter = file["$group/$parameter_name"]
        end

    end

    return parameter
end

get_multiple_parameters(filename, group, names...) =
    Tuple(get_parameter(filename, group, name) for name in names)

function set_from_file!(model, filename; i=length(get_iters(filename)))

    file = jldopen(filename)

    # Load initial condition from file
    iter = parse(Int, keys(file["timeseries/t"])[i])
    
    u₀ = file["timeseries/u/$iter"]
    v₀ = file["timeseries/v/$iter"]
    w₀ = file["timeseries/w/$iter"]
    b₀ = file["timeseries/b/$iter"]
    
    close(file)

    array_type = typeof(model.velocities.u.data.parent)

    # Set initial condition
    model.velocities.u.data.parent .= array_type(u₀)
    model.velocities.v.data.parent .= array_type(v₀)
    model.velocities.w.data.parent .= array_type(w₀)
    model.tracers.b.data.parent .= array_type(b₀)

    return nothing
end

function calculate_horizontal_average_timeseries(directory; part=0)

    part > 9 && error("Part must be less than 10.")

    filenames = cd(() -> glob("*fields*"), directory)
    sortby(filename) = parse(Int, filename[end-5:end-5])
    sort!(filenames, by=sortby)
    part = part == 0 ? length(filenames) : part

    t, U, V, B, Bz, w², E = [[] for i=1:7]

    filepath = joinpath(directory, filenames[part])

    grid = get_grid(filepath)

    for filename in filenames
        filepath = joinpath(directory, filename)
        iters = get_iters(filepath)

        for iter in iters
            tᵢ, u, v, w, b = get_fields(filepath, iter)

            push!(t, tᵢ)

            b′, Uᵢ, Vᵢ, Bᵢ, Bzᵢ, Eᵢ, w²ᵢ, e = calculate_statistics(grid, u, v, w, b)

            push!(U, Uᵢ)
            push!(V, Vᵢ)
            push!(B, Bᵢ)
            push!(Bz, Bzᵢ)
            push!(w², w²ᵢ)
            push!(E, Eᵢ)
        end
    end

    ii = sortperm(t)

     t = t[ii]
     U = U[ii]
     V = V[ii]
     B = B[ii]
    Bz = Bz[ii]
    w² = w²[ii]
     E = E[ii]

    S = [sqrt.(Ui.^2 .+ Vi.^2) for (Ui, Vi) in zip(U, V)]

    return t, U, V, S, B, Bz, w², E
end

function collect_horizontal_averages(filename)
    iters = get_iters(filename)
    grid = get_grid(filename)

    cell_quantity_names = (:U, :V, :b, :U², :V², :E, :νₑ, :κₑ_b)
    cell_quantities = NamedTuple{cell_quantity_names}(Tuple(zeros(grid.Nz, length(iters)) 
                                                            for i = 1:length(cell_quantity_names)))

    face_quantity_names = (:W², :W³, :wu, :wv, :wb, :τ₁₃, :τ₂₃, :τ₃₃, :q₃_b)
    face_quantities = NamedTuple{face_quantity_names}(Tuple(zeros(grid.Nz+1, length(iters))
                                                            for i = 1:length(face_quantity_names)))

    t = zeros(length(iters))

    file = jldopen(filename)

    for (j, iter) in enumerate(iters)
        for i = 1:length(cell_quantities)
            c = cell_quantities[i]
            name = propertynames(cell_quantities)[i]
            c[:, j] .= file["timeseries/$name/$iter"][:][2:end-1]
        end

        for i = 1:length(face_quantities)
            f = face_quantities[i]
            name = propertynames(face_quantities)[i]
            f[:, j] .= file["timeseries/$name/$iter"][:][2:end]
        end

        t[j] = file["timeseries/t/$iter"]
    end

    close(file)

    return merge((t=t,), cell_quantities, face_quantities)
end

end # module
