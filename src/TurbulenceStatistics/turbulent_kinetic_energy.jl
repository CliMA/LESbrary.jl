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
                                                c_scratch = CellField(model.architecture, model.grid))

    u, v, w = model.velocities

    νₑ = model.diffusivities.νₑ

    p = pressure(model)

    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))

    statistics = Dict()

    turbulent_kinetic_energy = @at (Cell, Cell, Cell) ( (u - U)^2 + (v - V)^2 + w^2 ) / 2
    buoyancy_flux            = @at (Cell, Cell, Cell) w * b
    shear_production         = @at (Cell, Cell, Cell) (u - U) * w * ∂z(U) + (v - V) * w * ∂z(V)

    statistics[:turbulent_kinetic_energy] = AveragedField(turbulent_kinetic_energy, dims=(1, 2), computed_data=c_scratch.data)
    statistics[:buoyancy_flux]            = AveragedField(buoyancy_flux,            dims=(1, 2), computed_data=c_scratch.data)
    statistics[:shear_production]         = AveragedField(shear_production,         dims=(1, 2), computed_data=c_scratch.data)

    advective_transport = @at (Cell, Cell, Cell) ∂z(w * turbulent_kinetic_energy)
    pressure_transport  = @at (Cell, Cell, Cell) ∂z(w * p)

    statistics[:advective_transport] = AveragedField(advective_transport, dims=(1, 2), computed_data=c_scratch.data)
    statistics[:pressure_transport] = AveragedField(pressure_transport, dims=(1, 2), computed_data=c_scratch.data)
    statistics[:dissipation] = AveragedField(subfilter_viscous_dissipation(model), dims=(1, 2), computed_data=c_scratch.data)

    return statistics
end
