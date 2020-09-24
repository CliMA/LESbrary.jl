using Oceananigans.Fields

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
                                                w_scratch = ZFaceField(model.architecture, model.grid),
                                                c_scratch = CellField(model.architecture, model.grid))

    u, v, w = model.velocities

    νₑ = model.diffusivities.νₑ

    dissipation = subfilter_viscous_dissipation(model)
    p = pressure(model)

    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))

    statistics = Dict()

    turbulent_kinetic_energy = @at (Cell, Cell, Cell) ( (u - U)^2 + (v - V)^2 + w^2 ) / 2
    buoyancy_flux            = @at (Cell, Cell, Cell) w * b
    shear_production         = @at (Cell, Cell, Cell) (u - U) * w * ∂z(U) + (v - V) * w * ∂z(V)

    statistics[:turbulent_kinetic_energy] = AveragedField(turbulent_kinetic_energy, dims=(1, 2), operand_data=c_scratch.data)
    statistics[:buoyancy_flux]            = AveragedField(buoyancy_flux,            dims=(1, 2), operand_data=c_scratch.data)
    statistics[:shear_production]         = AveragedField(shear_production,         dims=(1, 2), operand_data=c_scratch.data)

    advective_flux            = @at (Cell, Cell, Face) w * turbulent_kinetic_energy
    advective_flux_divergence = @at (Cell, Cell, Cell) ∂z(w * turbulent_kinetic_energy)

    pressure_flux            = @at (Cell, Cell, Face) w * p
    pressure_flux_divergence = @at (Cell, Cell, Cell) ∂z(w * p)

    statistics[:advective_flux]            = AveragedField(advective_flux,            dims=(1, 2), operand_data=w_scratch.data)
    statistics[:pressure_flux]             = AveragedField(pressure_flux,             dims=(1, 2), operand_data=w_scratch.data)
    statistics[:advective_flux_divergence] = AveragedField(advective_flux_divergence, dims=(1, 2), operand_data=c_scratch.data)
    statistics[:pressure_flux_divergence]  = AveragedField(pressure_flux_divergence,  dims=(1, 2), operand_data=c_scratch.data)
    statistics[:dissipation]               = AveragedField(dissipation,               dims=(1, 2), operand_data=c_scratch.data)

    return statistics
end
