using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation

subfilter_diffusivity(::AnisotropicMinimumDissipation, diffusivity_fields, name) =
    getproperty(diffusivity_fields.κₑ, name)

"""
    subfilter_momentum_fluxes(model)

Returns a dictionary of horizontally-averaged subfilter momentum fluxes.
"""
function subfilter_momentum_fluxes(model)

    u, v, w = model.velocities
    νₑ = model.diffusivity_fields.νₑ

    averages = Dict(
                    :νₑ_∂z_u => Average(∂z(u) * νₑ, dims=(1, 2)),
                    :νₑ_∂z_v => Average(∂z(v) * νₑ, dims=(1, 2)),
                    :νₑ_∂z_w => Average(∂z(w) * νₑ, dims=(1, 2))
                   )

    return averages
end

"""
    subfilter_tracer_fluxes(model)

Returns a dictionary of horizontally-averaged subfilter tracer fluxes.
"""
function subfilter_tracer_fluxes(model)

    averages = Dict()

    for tracer in propertynames(model.tracers)
        c = getproperty(model.tracers, tracer)
        κₑ = subfilter_diffusivity(model.closure, model.diffusivity_fields, tracer)

        name = Symbol(:κₑ_∂z_, tracer)

        averages[name] = Average(∂z(c) * κₑ, dims=(1, 2))
    end

    return averages
end
