
function uic_budget(model;
                    terms = (:u, :c),
                    with_flux_divergences = false,
                    b = BuoyancyField(model),
                    p = model.pressures.pHY′ + model.pressures.pNHS)

    u = Field(@at (Center, Center, Center) model.velocities[terms[1]])
    c = Field(@at (Center, Center, Center) model.tracers[terms[2]])
    w = model.velocities.w

    U = Field(Average(u, dims=(1, 2)))
    C = Field(Average(c, dims=(1, 2)))
    B = Field(Average(b, dims=(1, 2)))
    P = Field(Average(p, dims=(1, 2)))
    W = Field(Average(w, dims=(1, 2)))

    ∂x1 = directional_derivative(terms[1])

    advective_flux = @at (Center, Center, Center) model.velocities.w * u * c
    if terms[1] == :w
        buoyancy_flux = @at (Center, Center, Center) c * b
    else
        buoyancy_flux = ZeroField()
    end

    shear_production    = @at (Center, Center, Center) (w - W) * ((u - U) * ∂z(C) + (c - C) * ∂z(U))
    viscous_dissipation = TermWiseViscousDissipationRate(model, U, C; vel_name1 = terms[1], vel_name2 = terms[2])
    pressure_term       = @at (Center, Center, Center) (c - C) * ∂x1(p - P) 
    turbulence_statistics = Dict()

    turbulence_statistics[:uic]                   = Field(Average((u - U) * (c - C),   dims=(1, 2)))
    turbulence_statistics[:uic_shear_production] = Field(Average(shear_production,    dims=(1, 2)))
    turbulence_statistics[:uic_advective_flux]   = Field(Average(advective_flux,      dims=(1, 2)))
    turbulence_statistics[:uic_pressure_term]    = Field(Average(pressure_term,       dims=(1, 2)))
    turbulence_statistics[:uic_dissipation]      = Field(Average(viscous_dissipation, dims=(1, 2)))
    turbulence_statistics[:uic_buoyancy_flux]    = Field(Average(buoyancy_flux,       dims=(1, 2)))

    if with_flux_divergences
        advective_flux_field      = Field(advective_flux)
        advective_flux_divergence = ∂z(advective_flux_field)
        turbulence_statistics[:uic_advective_flux_divergence] = Field(Average(advective_flux_divergence, dims=(1, 2)))
    end

    return turbulence_statistics
end
