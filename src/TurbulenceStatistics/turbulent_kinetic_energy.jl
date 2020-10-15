using KernelAbstractions
using Statistics

using Oceananigans.Utils: work_layout
using Oceananigans.Fields
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ
using Oceananigans.Fields: AbstractReducedField, new_data
import Oceananigans.Fields: compute!

function subfilter_viscous_dissipation(model)

    u, v, w = model.velocities

    νₑ = model.diffusivities.νₑ

    Σˣˣ = ∂x(u)
    Σʸʸ = ∂y(v)
    Σᶻᶻ = ∂z(w)
    Σˣʸ = (∂y(u) + ∂x(v)) / 2
    Σˣᶻ = (∂z(u) + ∂x(w)) / 2
    Σʸᶻ = (∂z(v) + ∂y(w)) / 2

    ϵ = νₑ * 2 * ( Σˣˣ^2 + Σʸʸ^2 + Σᶻᶻ^2 + Σˣʸ^2 + Σˣᶻ^2 + Σʸᶻ^2 )

    return ϵ
end

"""
    turbulent_kinetic_energy_budget(model; b = BuoyancyField(model),
                                           w_scratch = ZFaceField(model.architecture, model.grid),
                                           c_scratch = CellField(model.architecture, model.grid))

Returns a `Dict` with `AveragedField`s correpsonding to terms in the turbulent kinetic energy budget.
The turbulent kinetic energy equation is

`` ∂_t E = - ∂_z ⟨w′e′ + w′p′⟩ - ⟨w′u′⟩ ∂_z U - ⟨w′v′⟩ ∂_z V + ⟨w′b′⟩ - ϵ ``,

where uppercase variables denote a horizontal mean, and primed variables denote deviations from
the horizontal mean.

The terms on the right side of the turbulent kinetic energy equation and their correpsonding keys are

1. `:advective_flux_divergence`, ``∂_z ⟨w′e′⟩``
2. `:pressure_flux_divergence`, ``∂_z ⟨w′p′⟩``
3. `:shear_production`, ``⟨w′u′⟩ ∂_z U``
4. `:buoyancy_flux`, ``⟨w′b′⟩``, where ``b`` is buoyancy
5. `:dissipation`, ``ϵ = ⟨2 νₑ Σᵢⱼ²⟩``, where ``νₑ`` is the subfilter eddy viscosity and ``Σᵢⱼ`` is the strain-rate tensor.

In addition, the return statistics `Dict` includes

6. `:advective_flux`, ``⟨w′e′⟩``
7. `:pressure_flux`, ``⟨w′p′⟩``
8. `:turbulent_kinetic_energy`, ``E = 1/2 (u′² + v′² + w′²)``

All variables are located at cell centers and share memory space with `c_scratch.data`, except `:advective_flux` and
`:pressure_flux`, which are located at `(Cell, Cell, Face)` and use `w_scratch`.

Note that these diagnostics do not compile on the GPU currently.
"""
function turbulent_kinetic_energy_budget(model; b = BuoyancyField(model),
                                                u_scratch = XFaceField(model.architecture, model.grid),
                                                v_scratch = YFaceField(model.architecture, model.grid),
                                                w_scratch = ZFaceField(model.architecture, model.grid),
                                                c_scratch = CellField(model.architecture, model.grid))

    statistics = Dict()

    dissipation = subfilter_viscous_dissipation(model)

    u, v, w = model.velocities
    p = pressure(model)

    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))

    turbulent_kinetic_energy  = @at (Cell, Cell, Cell) 0.5 * ( (u - U)^2 + (v - V)^2 + w^2 )
    buoyancy_flux             = @at (Cell, Cell, Cell) w * b
    shear_production          = @at (Cell, Cell, Cell) (u - U) * w * ∂z(U) + (v - V) * w * ∂z(V)

    advective_flux            = @at (Cell, Cell, Face) w * turbulent_kinetic_energy
    advective_flux_divergence = @at (Cell, Cell, Cell) ∂z(w * turbulent_kinetic_energy)

    pressure_flux             = @at (Cell, Cell, Face) w * p
    pressure_flux_divergence  = @at (Cell, Cell, Cell) ∂z(w * p)

    statistics[:turbulent_kinetic_energy]  = AveragedField(turbulent_kinetic_energy, dims=(1, 2), operand_data=c_scratch.data)
    statistics[:buoyancy_flux]             = AveragedField(buoyancy_flux,            dims=(1, 2), operand_data=c_scratch.data)
    statistics[:shear_production]          = AveragedField(shear_production,         dims=(1, 2), operand_data=c_scratch.data)
    statistics[:dissipation]               = AveragedField(dissipation,               dims=(1, 2), operand_data=c_scratch.data)
    statistics[:advective_flux]            = AveragedField(advective_flux,            dims=(1, 2), operand_data=w_scratch.data)
    statistics[:pressure_flux]             = AveragedField(pressure_flux,             dims=(1, 2), operand_data=w_scratch.data)
    statistics[:advective_flux_divergence] = AveragedField(advective_flux_divergence, dims=(1, 2), operand_data=c_scratch.data)
    statistics[:pressure_flux_divergence]  = AveragedField(pressure_flux_divergence,  dims=(1, 2), operand_data=c_scratch.data)

    return statistics
end

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

    event = compute_kernel!(tke.data, tke.grid, tke.u, tke.v, tke.w, tke.U.data, tke.V.data; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

@inline u′²(i, j, k, grid, u, U) = @inbounds (u[i, j, k] - U[k])^2
@inline w²(i, j, k, grid, w) = @inbounds w[i, j, k]^2

@kernel function compute_turbulent_kinetic_energy!(tke, grid, u, v, w, U, V)
    i, j, k = @index(Global, NTuple)

    @inbounds tke[i, j, k] = (
                              ℑxᶜᵃᵃ(i, j, k, grid, u′², u, U) +
                              ℑyᵃᶜᵃ(i, j, k, grid, u′², v, V) +
                              ℑzᵃᵃᶜ(i, j, k, grid, w², w)
                             ) / 2
end
