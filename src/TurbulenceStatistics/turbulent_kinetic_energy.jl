using KernelAbstractions
using Adapt
using Statistics

using Oceananigans.Fields
using Oceananigans.Utils: work_layout
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ
using Oceananigans.Fields: AbstractField, new_data

import Oceananigans.Fields: compute!

struct TurbulentKineticEnergy{A, G, U, V, W, Ua, Va} <: AbstractField{Cell, Cell, Cell, A, G}
    data :: A
    grid :: G
       u :: U
       v :: V
       w :: W
       U :: Ua # horizontally-averaged u
       V :: Va # horizontally-averaged v
end

"""
    TurbulentKineticEnergy(model)

Returns an `AbstractField` representing the turbulent kinetic energy of `model`.

Calling `compute!(tke::TurbulentKineticEnergy)` computes the turbulent kinetic energy of `model`
and stores it in `tke.data`.
"""
function TurbulentKineticEnergy(model;
                                data = nothing, 
                                   U = AveragedField(model.velocities.u, dims=(1, 2)),
                                   V = AveragedField(model.velocities.v, dims=(1, 2)))

    if isnothing(data)
        data = new_data(model.architecture, model.grid, (Cell, Cell, Cell))
    end

    u, v, w = model.velocities

    return TurbulentKineticEnergy(data, model.grid, u, v, w, U, V)
end

function compute!(tke::TurbulentKineticEnergy)

    compute!(tke.U)
    compute!(tke.V)

    arch = architecture(tke.data)

    workgroup, worksize = work_layout(tke.grid, :xyz, location=(Cell, Cell, Cell))

    compute_kernel! = compute_turbulent_kinetic_energy!(device(arch), workgroup, worksize)

    event = compute_kernel!(tke.data, tke.grid, tke.u, tke.v, tke.w, tke.U, tke.V; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end


@kernel function compute_turbulent_kinetic_energy!(tke, grid, u, v, w, U, V)
    i, j, k = @index(Global, NTuple)

    @inbounds tke[i, j, k] = (
                              ℑxᶜᵃᵃ(i, j, k, grid, ψ′², u, U) +
                              ℑyᵃᶜᵃ(i, j, k, grid, ψ′², v, V) +
                              ℑzᵃᵃᶜ(i, j, k, grid, ψ², w)
                             ) / 2
end

Adapt.adapt_structure(to, tke::TurbulentKineticEnergy) = Adapt.adapt(to, tke.data)
