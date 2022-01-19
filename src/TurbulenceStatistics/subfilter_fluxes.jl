using Oceananigans.Fields: ZFaceField, CenterField
using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation

subfilter_diffusivity(::AnisotropicMinimumDissipation, diffusivity_fields, name) =
    getproperty(diffusivity_fields.κₑ, name)

"""
    subfilter_momentum_fluxes(model,
                              uz_scratch = Field{Face, Center, Face}(model.grid),
                              vz_scratch = Field{Center, Face, Face}(model.grid),
                              c_scratch = CenterField(model.grid))

Returns a dictionary of horizontally-averaged subfilter momentum fluxes.
"""
function subfilter_momentum_fluxes(model;
                                   uz_scratch = Field{Face, Center, Face}(model.grid),
                                   vz_scratch = Field{Center, Face, Face}(model.grid),
                                   c_scratch = CenterField(model.grid))

    u, v, w = model.velocities

    νₑ = model.diffusivity_fields.νₑ

    averages = Dict(
                    :νₑ_∂z_u => Field(Average(∂z(u) * νₑ, dims=(1, 2))),
                    :νₑ_∂z_v => Field(Average(∂z(v) * νₑ, dims=(1, 2))),
                    :νₑ_∂z_w => Field(Average(∂z(w) * νₑ, dims=(1, 2)))
                   )

    return averages
end

"""
    subfilter_tracer_fluxes(model, w_scratch=ZFaceField(model.grid))

Returns a dictionary of horizontally-averaged subfilter tracer fluxes.
"""
function subfilter_tracer_fluxes(model; w_scratch = ZFaceField(model.grid))

    averages = Dict()

    for tracer in propertynames(model.tracers)
        c = getproperty(model.tracers, tracer)
        κₑ = subfilter_diffusivity(model.closure, model.diffusivity_fields, tracer)

        name = Symbol(:κₑ_∂z_, tracer)

        averages[name] = Field(Average(∂z(c) * κₑ, dims=(1, 2)))
    end

    return averages
end
