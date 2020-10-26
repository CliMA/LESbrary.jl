struct AdvectiveFluxDivergence{A, G, W, E} <: AbstractField{Cell, Cell, Cell, A, G}
    data :: A
    grid :: G
       w :: W
       e :: E
end

"""
    AdvectiveFluxDivergence(model)

Returns an `AbstractField` representing the shear production term that arises
in the turbulent kinetic energy budget for `model`.

Calling `compute!(shear_production::ShearProduction)` computes the current shear production
term and stores in `tke.data`.
"""
function AdvectiveFluxDivergence(model; data = nothing, 
                                 e = TurbulentKineticEnergy(model)

    if isnothing(data)
        data = new_data(model.architecture, model.grid, (Cell, Cell, Cell))
    end

    u, v, w = model.velocities

    return ShearProduction(data, model.grid, u, v, w, U, V)
end

function compute!(sp::ShearProduction)

    compute!(sp.U)
    compute!(sp.V)

    arch = architecture(sp.data)

    workgroup, worksize = work_layout(sp.grid, :xyz, location=(Cell, Cell, Cell))

    compute_kernel! = compute_shear_production!(device(arch), workgroup, worksize)

    event = compute_kernel!(sp.data, sp.grid, sp.u, sp.v, sp.w, sp.U, sp.V; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

@kernel function compute_shear_production!(shear_production, grid, u, v, w, U, V)
    i, j, k = @index(Global, NTuple)

    @inbounds shear_production[i, j, k] = (
        ℑxᶜᵃᵃ(i, j, k, grid, ψ′, u, U) * ℑzᵃᵃᶜ(i, j, k, grid, w∂zΨ, w, U) +
        ℑyᵃᶜᵃ(i, j, k, grid, ψ′, v, U) * ℑzᵃᵃᶜ(i, j, k, grid, w∂zΨ, w, V) )
end

Adapt.adapt_structure(to, sp::ShearProduction) = Adapt.adapt(to, sp.data)
