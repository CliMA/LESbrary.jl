using Oceananigans

# TODO: Use `BuoyancyField`!
function mixed_layer_depth(model)
    T = model.tracers.T
    ∂T̄∂z = AveragedField(Oceananigans.∂z(T), dims=(1, 2))
    compute!(∂T̄∂z)

    _, k_boundary_layer = findmax(abs.(interior(∂T̄∂z)))

    if k_boundary_layer isa CartesianIndex
        k_boundary_layer = k_boundary_layer.I[3]
    end

    mixed_layer_depth = - model.grid.zF[k_boundary_layer]

    return mixed_layer_depth
end
