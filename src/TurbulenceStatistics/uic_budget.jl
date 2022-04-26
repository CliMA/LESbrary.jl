
# Assuming that if there is a velocity it's in terms[1]
function uic_budget(model;
                    terms = (:u, :c),
                    with_flux_divergences = false,
                    b = BuoyancyField(model),
                    p = model.pressures.pHY′ + model.pressures.pNHS)

    fields = merge(model.velocities, model.tracers)

    u = Field(fields[terms[1]])
    c = Field(fields[terms[2]])
    w = model.velocities.w

    U = Field(Average(u, dims=(1, 2)))
    C = Field(Average(c, dims=(1, 2)))
    B = Field(Average(b, dims=(1, 2)))
    P = Field(Average(p, dims=(1, 2)))
    W = Field(Average(w, dims=(1, 2)))

    u′ = u - U
    c′ = c - C
    w′ = w - W
    b′ = b - B
    p′ = p - P   
    
    ∂x1 = directional_derivative(terms[1])

    advective_flux = @at (Center, Center, Center) w′ * u′ * c′
    if terms[1] == :w
        buoyancy_flux = @at (Center, Center, Center) c′ * b′
    else
        buoyancy_flux = ZeroField()
    end

    shear_production    = @at (Center, Center, Center) w′ * (u′ * ∂z(C) + c′ * ∂z(U))
    viscous_dissipation = TermWiseViscousDissipationRate(model, U, C; name1 = terms[1], name2 = terms[2])
    pressure_term       = @at (Center, Center, Center) c′ * ∂x1(p′) 
    turbulence_statistics = Dict()

    turbulence_statistics[:uic]                  = Field(Average(u′ * c′,             dims=(1, 2)))
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
