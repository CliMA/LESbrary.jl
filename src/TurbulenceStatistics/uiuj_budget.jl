
# To change to 3D derivatives! (we need to update the Oceananigans dependency)
using Oceananigans.Operators: ∂xᶜᵃᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶠ, 
                              ℑxyᶜᶜᵃ, ℑxzᶜᵃᶜ, 
                              ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, 
                              ℑxyᶜᶜᵃ, ℑyzᵃᶜᶜ, 
                              ∂zᵃᵃᶜ, 
                              ℑxzᶜᵃᶜ, ℑyzᵃᶜᶜ, 
                              ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ

function uiuj_budget(model;
                    terms = (:u, :v),
                    with_flux_divergences = false,
                    b = BuoyancyField(model),
                    p = model.pressures.pHY′ + model.pressures.pNHS)

    u1 = Field(model.velocities[terms[1]])
    u2 = Field(model.velocities[terms[2]])

    U1 = Field(Average(u1, dims=(1, 2)))
    U2 = Field(Average(u2, dims=(1, 2)))

    w  = model.velocities.w
    W  = Field(Average(w, dims=(1, 2)))
    B  = Field(Average(b, dims=(1, 2)))
    P  = Field(Average(p, dims=(1, 2)))

    u1′ = u1 - U1
    u2′ = u2 - U2
    
    w′ = w - W
    b′ = b - B
    p′ = p - P   
    
    ∂x1 = directional_derivative(terms[1])
    ∂x2 = directional_derivative(terms[2])

    advective_flux = @at (Center, Center, Center) w′ * u1′ * u2′
    if terms[1] == :w && terms[2] == :w
        buoyancy_flux = @at (Center, Center, Center) (u1′ + u2′) * b′
    elseif terms[1] == :w
        buoyancy_flux = @at (Center, Center, Center) u2′ * b′
    elseif terms[2] == :w
        buoyancy_flux = @at (Center, Center, Center) u1′ * b′ 
    else
        buoyancy_flux = ZeroField()
    end

    shear_production    = @at (Center, Center, Center) w′ * (u1′ * ∂z(U2) + u2′ * ∂z(U1))
    viscous_dissipation = TermWiseViscousDissipationRate(model, U1, U2; name1 = terms[1], name2 = terms[2])
    pressure_term       = @at (Center, Center, Center) u2′ * ∂x1(p′) + u1′ * ∂x2(p′)
    turbulence_statistics = Dict()

    turbulence_statistics[:uiuj]                  = Field(Average(u1′ * u2′, dims=(1, 2)))
    turbulence_statistics[:uiuj_shear_production] = Field(Average(shear_production,      dims=(1, 2)))
    turbulence_statistics[:uiuj_advective_flux]   = Field(Average(advective_flux,        dims=(1, 2)))
    turbulence_statistics[:uiuj_pressure_term]    = Field(Average(pressure_term,         dims=(1, 2)))
    turbulence_statistics[:uiuj_dissipation]      = Field(Average(viscous_dissipation,   dims=(1, 2)))
    turbulence_statistics[:uiuj_buoyancy_flux]    = Field(Average(buoyancy_flux,         dims=(1, 2)))

    if with_flux_divergences
        advective_flux_field      = Field(advective_flux)
        advective_flux_divergence = ∂z(advective_flux_field)
        turbulence_statistics[:uiuj_advective_flux_divergence] = Field(Average(advective_flux_divergence, dims=(1, 2)))
    end

    return turbulence_statistics
end

@inline directional_derivative(term) = term == :u ? ∂x : term == :v ? ∂y : ∂z

# using Oceanostics.TKEBudgetTerms: validate_dissipative_closure
# using Oceanostics: _νᶜᶜᶜ 

############# To remove (I do not seem to be able to use using from Oceanostics)
using Oceananigans.TurbulenceClosures: νᶜᶜᶜ

@inline _νᶜᶜᶜ(args...) = νᶜᶜᶜ(args...)
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple{}, K::Tuple{}, clock) = zero(eltype(grid))
@inline _νᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, K::Tuple, clock) =
     νᶜᶜᶜ(i, j, k, grid, closure_tuple[1],     K[1],     clock) + 
    _νᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], K[2:end], clock)
    
using Oceananigans.TurbulenceClosures: κᶜᶜᶜ

@inline _κᶜᶜᶜ(args...) = κᶜᶜᶜ(args...)
@inline _κᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple{}, K::Tuple{}, id, clock) = zero(eltype(grid))
@inline _κᶜᶜᶜ(i, j, k, grid, closure_tuple::Tuple, K::Tuple, id, clock) =
     κᶜᶜᶜ(i, j, k, grid, closure_tuple[1],     K[1],     id, clock) + 
    _κᶜᶜᶜ(i, j, k, grid, closure_tuple[2:end], K[2:end], id, clock)

#############

# Temporary hack since it is not possible to pass symbols to GPU: TOFIX! 
# (also because like this we do not get the correct diffusivity)

@inline name(term) = term == :u ? 1 : term == :v ? 2 : term == :w ? 3 : 4

function TermWiseViscousDissipationRate(model, mean1, mean2; name1 = :u, name2 = :v) 
    
    # validate_dissipative_closure(model.closure)

    field1 = merge(model.velocities, model.tracers)[name1]
    field2 = merge(model.velocities, model.tracers)[name2]

    parameters = (closure = model.closure,
                  diffusivity_fields = model.diffusivity_fields,
                  clock = model.clock,
                  vel_names = (name(name1), name(name2)))

    return KernelFunctionOperation{Center, Center, Center}(termwise_viscous_dissipation_rate_ccc, model.grid;
                        computed_dependencies=(field1 - mean1, field2 - mean2),
                        parameters)
end

#=
 here it is assumed that τᵢⱼ ≈ ∂ⱼuᵢ
=#

@inline function termwise_viscous_dissipation_rate_ccc(i, j, k, grid, u1, u2, p)

    ∂x, ∂y, ∂z, ℑx, ℑy, ℑz = derivatives(p.name[1])
 
    dx1 = ℑx(i, j, k, grid, ∂x, u1)
    dy1 = ℑy(i, j, k, grid, ∂y, u1)
    dz1 = ℑz(i, j, k, grid, ∂z, u1)

    ∂x, ∂y, ∂z, ℑx, ℑy, ℑz = derivatives(p.name[2])

    dx2 = ℑx(i, j, k, grid, ∂x, u2)
    dy2 = ℑy(i, j, k, grid, ∂y, u2)
    dz2 = ℑz(i, j, k, grid, ∂z, u2)

    K = diffusion_coefficient(i, j, k, grid, p.closure, p.diffusivity_fields, p.clock, p.vel_name)
    return K * (dx1 * dx2 + dy1 * dy2 + dz1 * dz2)
end

function diffusion_coefficient(i, j, k, grid, closure, diffusivity_fields, clock, name)
   return _νᶜᶜᶜ(i, j, k, grid, closure, diffusivity_fields, clock)
end

@inline ℑᵢ(i, j, k, grid, u) = u[i, j, k]
@inline ℑᵢ(i, j, k, grid, f::Function, args...) = f(i, j, k, grid, args...)

function derivatives(name)
    if name == 1
        ∂x = ∂xᶜᵃᵃ
        ∂y = ∂yᵃᶠᵃ
        ∂z = ∂zᵃᵃᶠ
        ℑx = ℑᵢ
        ℑy = ℑxyᶜᶜᵃ
        ℑz = ℑxzᶜᵃᶜ
    elseif name == 2
        ∂x = ∂xᶠᵃᵃ
        ∂y = ∂yᵃᶜᵃ
        ∂z = ∂zᵃᵃᶠ
        ℑx = ℑxyᶜᶜᵃ
        ℑy = ℑᵢ
        ℑz = ℑyzᵃᶜᶜ
    elseif name == 3
        ∂x = ∂xᶠᵃᵃ
        ∂y = ∂yᵃᶠᵃ
        ∂z = ∂zᵃᵃᶜ
        ℑx = ℑxzᶜᵃᶜ
        ℑy = ℑyzᵃᶜᶜ
        ℑz = ℑᵢ
    else
        ∂x = ∂xᶠᵃᵃ
        ∂y = ∂yᵃᶠᵃ
        ∂z = ∂zᵃᵃᶠ
        ℑx = ℑxᶜᵃᵃ
        ℑy = ℑyᵃᶜᵃ
        ℑz = ℑzᵃᵃᶜ
    end
    return ∂x, ∂y, ∂z, ℑx, ℑy, ℑz
end
