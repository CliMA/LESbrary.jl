using Oceananigans.Fields: ZFaceField, CellField
using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation

subfilter_diffusivity(::AnisotropicMinimumDissipation, diffusivities, name) =
    getproperty(diffusivities.κₑ, name)

"""
    subfilter_momentum_fluxes(model,
                              uz_scratch = Field(Face, Cell, Face, model.architecture, model.grid),
                              vz_scratch = Field(Cell, Face, Face, model.architecture, model.grid),
                              c_scratch = CellField(model.architecture, model.grid))

Returns a dictionary of horizontally-averaged subfilter momentum fluxes.
"""
function subfilter_momentum_fluxes(model;
                                   uz_scratch = Field(Face, Cell, Face, model.architecture, model.grid),
                                   vz_scratch = Field(Cell, Face, Face, model.architecture, model.grid),
                                   c_scratch = CellField(model.architecture, model.grid))

    u, v, w = model.velocities

    νₑ = model.diffusivities.νₑ

    averages = Dict(
                    :νₑ_∂z_u => AveragedField(∂z(u) * νₑ, dims=(1, 2), operand_data=uz_scratch.data),
                    :νₑ_∂z_v => AveragedField(∂z(v) * νₑ, dims=(1, 2), operand_data=vz_scratch.data),
                    :νₑ_∂z_w => AveragedField(∂z(w) * νₑ, dims=(1, 2), operand_data=c_scratch.data),
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
        κₑ = subfilter_diffusivity(model.closure, model.diffusivities, tracer)

        name = Symbol(:κₑ_∂z_, tracer) 

        averages[name] = AveragedField(∂z(c) * κₑ, dims=(1, 2), operand_data=w_scratch.data)
    end

    return averages
end
