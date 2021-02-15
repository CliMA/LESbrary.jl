struct ShearProduction{A, G, U, V, W, Ua, Va} <: AbstractField{Center, Center, Center, A, G}
    data :: A
    grid :: G
       u :: U
       v :: V
       w :: W
       U :: Ua
       V :: Va
end

"""
    ShearProduction(model)

Returns an `AbstractField` representing the shear production term that arises
in the turbulent kinetic energy budget for `model`.

Calling `compute!(shear_production::ShearProduction)` computes the current shear production
term and stores in `tke.data`.
"""
function ShearProduction(model; data = nothing, 
                                   U = AveragedField(model.velocities.u, dims=(1, 2)),
                                   V = AveragedField(model.velocities.v, dims=(1, 2)))

    if isnothing(data)
        data = new_data(model.architecture, model.grid, (Center, Center, Center))
    end

    u, v, w = model.velocities

    return ShearProduction(data, model.grid, u, v, w, U, V)
end

function compute!(sp::ShearProduction)

    compute!(sp.U)
    compute!(sp.V)

    arch = architecture(sp.data)

    workgroup, worksize = work_layout(sp.grid, :xyz, location=(Center, Center, Center))

    compute_kernel! = compute_shear_production!(device(arch), workgroup, worksize)

    event = compute_kernel!(sp.data, sp.grid, sp.u, sp.v, sp.w, sp.U, sp.V; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

@inline w∂zΨ(i, j, k, grid, w, Ψ) = @inbounds w[i, j, k] * ∂zᵃᵃᶠ(i, j, k, grid, Ψ)

@kernel function compute_shear_production!(shear_production, grid, u, v, w, U, V)
    i, j, k = @index(Global, NTuple)

    @inbounds shear_production[i, j, k] = - (
        ℑxᶜᵃᵃ(i, j, k, grid, ψ′, u, U) * ℑzᵃᵃᶜ(i, j, k, grid, w∂zΨ, w, U) +
        ℑyᵃᶜᵃ(i, j, k, grid, ψ′, v, V) * ℑzᵃᵃᶜ(i, j, k, grid, w∂zΨ, w, V) )
end

Adapt.adapt_structure(to, sp::ShearProduction) = Adapt.adapt(to, sp.data)
