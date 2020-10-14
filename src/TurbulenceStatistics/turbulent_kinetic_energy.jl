using Oceananigans.Fields
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ
using Oceananigans.Fields: AbstractField, new_data
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

    #u′ = ComputedField(u - U, data=u_scratch.data)
    #v′ = ComputedField(v - V, data=v_scratch.data)
    
    u′² = @at (Cell, Cell, Cell) (u - U)^2
    v′² = @at (Cell, Cell, Cell) (v - V)^2

    turbulent_kinetic_energy  = @at (Cell, Cell, Cell) 0.5 * ( (u - U)^2 + (v - V)^2 + w^2 )
    buoyancy_flux             = @at (Cell, Cell, Cell) w * b
    shear_production          = @at (Cell, Cell, Cell) (u - U) * w * ∂z(U) + (v - V) * w * ∂z(V)

    advective_flux            = @at (Cell, Cell, Face) w * turbulent_kinetic_energy
    advective_flux_divergence = @at (Cell, Cell, Cell) ∂z(w * turbulent_kinetic_energy)

    pressure_flux             = @at (Cell, Cell, Face) w * p
    pressure_flux_divergence  = @at (Cell, Cell, Cell) ∂z(w * p)

    #statistics[:turbulent_kinetic_energy] = AveragedField(turbulent_kinetic_energy, dims=(1, 2), operand_data=c_scratch.data)

    statistics[:turbulent_u_variance] = AveragedField(u′², dims=(1, 2), operand_data=c_scratch.data)
    statistics[:turbulent_v_variance] = AveragedField(v′², dims=(1, 2), operand_data=c_scratch.data)

    #=
    statistics[:buoyancy_flux]            = AveragedField(buoyancy_flux,            dims=(1, 2), operand_data=c_scratch.data)
    statistics[:shear_production]         = AveragedField(shear_production,         dims=(1, 2), operand_data=c_scratch.data)

    
    statistics[:advective_flux]            = AveragedField(advective_flux,            dims=(1, 2), operand_data=w_scratch.data)
    statistics[:pressure_flux]             = AveragedField(pressure_flux,             dims=(1, 2), operand_data=w_scratch.data)
    statistics[:advective_flux_divergence] = AveragedField(advective_flux_divergence, dims=(1, 2), operand_data=c_scratch.data)
    statistics[:pressure_flux_divergence]  = AveragedField(pressure_flux_divergence,  dims=(1, 2), operand_data=c_scratch.data)
    =#

    #statistics[:dissipation]               = AveragedField(dissipation,               dims=(1, 2), operand_data=c_scratch.data)

    return statistics
end

struct TurbulentKineticEnergy{A, G, E, U, V, W, Ua, Va} <: AbstractReducedField{Nothing, Nothing, Cell, A, G, 2}
    data :: A
    grid :: G
    e :: E # CellField...
    u :: U
    v :: V
    w :: W
    U :: Ua
    V :: Va
end

function TurbulentKineticEnergy(model)
    u, v, w = model.velocities

    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))

    e = CellField(model.architecture, model.grid)

    data = new_data(model.architecture, model.grid, (Nothing, Nothing, Cell))

    return TurbulentKineticEnergy(data, model.grid, e, u, v, w, U, V)
end

function compute!(tke::TurbulentKineticEnergy)

    compute!(tke.U)
    compute!(tke.V)

    arch = architecture(tke.data)

    workgroup, worksize = work_layout(comp.grid, :xyz, location=(Cell, Cell, Cell))

    compute_kernel! = compute_turbulent_kinetic_energy!(device(arch), workgroup, worksize)

    event = compute_kernel!(tke.e, tke.grid, tke.u, tke.v, tke.w, tke.U.data, tke.V.data; dependencies=Event(device(arch)))

    wait(device(arch), event)

    zero_halo_regions!(tke.e, dims=(1, 2))

    sum!(tke.data.parent, tke.e.data.parent)

    tke.data.parent ./= (tke.grid.Nx * tke.grid.Ny)

    return nothing
end

@inline u′²(i, j, k, grid, u, U) = @inbounds (u[i, j, k] - U[k])^2

@kernel function _compute_turbulent_kinetic_energy!(tke, grid, u, v, w, U, V)
    i, j, k = @index(Global, NTuple)

    @inbounds tke[i, j, k] = (
                              ℑxᶜᵃᵃ(i, j, k, grid, u′², u, U) +
                              ℑyᵃᶜᵃ(i, j, k, grid, u′², v, V) +
                              ℑzᵃᵃᶜ(i, j, k, grid, w², w)
                             ) / 2

    return nothing
end
