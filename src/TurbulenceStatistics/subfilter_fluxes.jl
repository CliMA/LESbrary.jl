using Oceananigans.Fields: ZFaceField, CenterField
using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation

subfilter_diffusivity(::AnisotropicMinimumDissipation, diffusivities, name) =
    getproperty(diffusivities.κₑ, name)

"""
    subfilter_momentum_fluxes(model,
                              uz_scratch = Field(Face, Center, Face, model.architecture, model.grid),
                              vz_scratch = Field(Center, Face, Face, model.architecture, model.grid),
                              c_scratch = CenterField(model.architecture, model.grid))

Returns a dictionary of horizontally-averaged subfilter momentum fluxes.
"""
function subfilter_momentum_fluxes(model;
                                   uz_scratch = Field(Face, Center, Face, model.architecture, model.grid),
                                   vz_scratch = Field(Center, Face, Face, model.architecture, model.grid),
                                   c_scratch = CenterField(model.architecture, model.grid))

    u, v, w = model.velocities

    νₑ = model.diffusivity_fields.νₑ

    averages = Dict(
                    :νₑ_∂z_u => AveragedField(∂z(u) * νₑ, dims=(1, 2)),
                    :νₑ_∂z_v => AveragedField(∂z(v) * νₑ, dims=(1, 2)),
                    :νₑ_∂z_w => AveragedField(∂z(w) * νₑ, dims=(1, 2))
                   )

    return averages
end

"""
    subfilter_tracer_fluxes(model, w_scratch=ZFaceField(model.architecture, model.grid))

Returns a dictionary of horizontally-averaged subfilter tracer fluxes.
"""
function subfilter_tracer_fluxes(model; w_scratch = ZFaceField(model.architecture, model.grid))

    averages = Dict()

    for tracer in propertynames(model.tracers)
        c = getproperty(model.tracers, tracer)
        κₑ = subfilter_diffusivity(model.closure, model.diffusivity_fields, tracer)

        name = Symbol(:κₑ_∂z_, tracer)

        averages[name] = AveragedField(∂z(c) * κₑ, dims=(1, 2))
    end

    return averages
end
