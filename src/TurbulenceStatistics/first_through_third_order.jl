using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField, CenterField, PressureField
using Oceananigans.Buoyancy: BuoyancyField

has_buoyancy_tracer(model) = :b ∈ keys(model.tracers)

"""
    horizontally_averaged_velocities(model)

Returns a dictionary in which `:u` represents the horizontal average of
`model.velocities.u` and `:v` represents the horizontal average of
`model.velocities.v`.
"""
function horizontally_averaged_velocities(model)

    u, v, w = model.velocities

    averages = Dict(
                    :u => AveragedField(u, dims=(1, 2)),
                    :v => AveragedField(v, dims=(1, 2))
                   )

    return averages
end

"""
    horizontally_averaged_tracers(model)

Returns a dictionary containing horizontal averages of
every tracer in `model.tracers`. The dictionary keys
correspond to the names of the tracer.
"""
function horizontally_averaged_tracers(model)

    averages = Dict()

    for tracer in keys(model.tracers)
        c = getproperty(model.tracers, tracer)
        averages[tracer] = AveragedField(c, dims=(1, 2))
    end

    return averages
end

"""
    velocity_covariances(model; u_scratch = XFaceField(model.architecture, model.grid),
                                v_scratch = YFaceField(model.architecture, model.grid),
                                w_scratch = ZFaceField(model.architecture, model.grid),
                                c_scratch = CenterField(model.architecture, model.grid))

Returns a dictionary containing horizontal averages of the velocity covariances uᵢuⱼ.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.
"""
function velocity_covariances(model; u_scratch = XFaceField(model.architecture, model.grid),
                                     v_scratch = YFaceField(model.architecture, model.grid),
                                     w_scratch = ZFaceField(model.architecture, model.grid),
                                     c_scratch = CenterField(model.architecture, model.grid))

    u, v, w = model.velocities

    covariances = Dict(
                       :uu => AveragedField(u * u, dims=(1, 2), operand_data=u_scratch.data),
                       :vv => AveragedField(v * v, dims=(1, 2), operand_data=v_scratch.data),
                       :ww => AveragedField(w * w, dims=(1, 2), operand_data=w_scratch.data),
                       :uv => AveragedField(u * v, dims=(1, 2), operand_data=u_scratch.data),
                       :wv => AveragedField(w * v, dims=(1, 2), operand_data=w_scratch.data),
                       :wu => AveragedField(w * u, dims=(1, 2), operand_data=w_scratch.data)
                       )

    return covariances
end

"""
    tracer_covariances(model; b = BuoyancyField(model),   
                              u_scratch = XFaceField(model.architecture, model.grid),
                              v_scratch = YFaceField(model.architecture, model.grid),
                              w_scratch = ZFaceField(model.architecture, model.grid),
                              c_scratch = CenterField(model.architecture, model.grid))

Returns a dictionary containing horizontal averages of
tracer variance, tracer-velocity covariances, and tracer-buoyancy covariance.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.

`b` is the `BuoyancyField(model)` for model. If `b` is not specified, new memory is allocated to
store the computation of buoyancy in three-dimensions over `model.grid`.
"""
function tracer_covariances(model; b = BuoyancyField(model),   
                                   u_scratch = XFaceField(model.architecture, model.grid),
                                   v_scratch = YFaceField(model.architecture, model.grid),
                                   w_scratch = ZFaceField(model.architecture, model.grid),
                                   c_scratch = CenterField(model.architecture, model.grid))

    u, v, w = model.velocities

    covariances = Dict()

    for tracer in keys(model.tracers)
        c = getproperty(model.tracers, tracer)

        # Keys
        cc = Symbol(tracer, tracer)
        uc = Symbol(:u, tracer)
        vc = Symbol(:v, tracer)
        wc = Symbol(:w, tracer)

        covariances[cc] = AveragedField(c * c, dims=(1, 2), operand_data=c_scratch.data)
        covariances[uc] = AveragedField(u * c, dims=(1, 2), operand_data=u_scratch.data)
        covariances[vc] = AveragedField(v * c, dims=(1, 2), operand_data=v_scratch.data)
        covariances[wc] = AveragedField(w * c, dims=(1, 2), operand_data=w_scratch.data)

        # Add covariance of tracer with buoyancy
        if tracer != :b && !has_buoyancy_tracer(model)
            bc = Symbol(:b, tracer)
            covariances[bc] = AveragedField(b * c, dims=(1, 2), operand_data=c_scratch.data)
        end
    end

    return covariances
end

"""
    third_order_velocity_statistics(model, u_scratch = XFaceField(model.architecture, model.grid),
                                           v_scratch = YFaceField(model.architecture, model.grid),
                                           w_scratch = ZFaceField(model.architecture, model.grid),
                                           c_scratch = CenterField(model.architecture, model.grid),
                                           p = PressureField(model))

Returns a dictionary of functions that calculate horizontally-averaged third-order statistics
that involve the velocity field. Includes statistics associated with pressure.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.
"""
function third_order_velocity_statistics(model; u_scratch = XFaceField(model.architecture, model.grid),
                                                v_scratch = YFaceField(model.architecture, model.grid),
                                                w_scratch = ZFaceField(model.architecture, model.grid),
                                                c_scratch = CenterField(model.architecture, model.grid),
                                                p = PressureField(model))

    u, v, w = model.velocities

    # Pressure-strain terms
    Σˣʸ = 0.5 * (∂y(u) + ∂x(v)) # FFC
    Σˣᶻ = 0.5 * (∂z(u) + ∂x(w)) # FCF
    Σʸᶻ = 0.5 * (∂z(v) + ∂y(w)) # CFF

    third_order_statistics = Dict(
                                   :uuu => AveragedField(u * u * u, dims=(1, 2), operand_data=u_scratch.data),
                                   :uuv => AveragedField(u * u * v, dims=(1, 2), operand_data=u_scratch.data),
                                   :uuw => AveragedField(u * u * w, dims=(1, 2), operand_data=u_scratch.data),
                                   :uvv => AveragedField(u * v * v, dims=(1, 2), operand_data=u_scratch.data),
                                   :uww => AveragedField(u * w * w, dims=(1, 2), operand_data=u_scratch.data),

                                   :vvv => AveragedField(v * v * v, dims=(1, 2), operand_data=v_scratch.data),
                                   :vvw => AveragedField(v * v * w, dims=(1, 2), operand_data=v_scratch.data),
                                   :vww => AveragedField(v * w * w, dims=(1, 2), operand_data=v_scratch.data),
 
                                   :www => AveragedField(w * w * w, dims=(1, 2), operand_data=w_scratch.data),
                                   :wvu => AveragedField(w * v * u, dims=(1, 2), operand_data=w_scratch.data),
 
                                    :up => AveragedField(u * p,     dims=(1, 2), operand_data=u_scratch.data),
                                    :vp => AveragedField(v * p,     dims=(1, 2), operand_data=v_scratch.data),
                                    :wp => AveragedField(w * p,     dims=(1, 2), operand_data=w_scratch.data),
  
                                  :pux => AveragedField(p * ∂x(u), dims=(1, 2), operand_data=c_scratch.data),
                                  :puy => AveragedField(p * ∂y(u), dims=(1, 2), operand_data=c_scratch.data),
                                  :puz => AveragedField(p * ∂z(u), dims=(1, 2), operand_data=c_scratch.data),

                                  :pvx => AveragedField(p * ∂x(v), dims=(1, 2), operand_data=c_scratch.data),
                                  :pvy => AveragedField(p * ∂y(v), dims=(1, 2), operand_data=c_scratch.data),
                                  :pvz => AveragedField(p * ∂z(v), dims=(1, 2), operand_data=c_scratch.data),

                                  :pwx => AveragedField(p * ∂x(w), dims=(1, 2), operand_data=c_scratch.data),
                                  :pwy => AveragedField(p * ∂y(w), dims=(1, 2), operand_data=c_scratch.data),
                                  :pwz => AveragedField(p * ∂z(w), dims=(1, 2), operand_data=c_scratch.data),
                                 )

    return third_order_statistics
end

"""
    third_order_tracer_statistics(model; u_scratch = XFaceField(model.architecture, model.grid),
                                         v_scratch = YFaceField(model.architecture, model.grid),
                                         w_scratch = ZFaceField(model.architecture, model.grid),
                                         c_scratch = CenterField(model.architecture, model.grid))

Returns a dictionary of functions that calculate horizontally-averaged third-order statistics
that involve tracers.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.
"""
function third_order_tracer_statistics(model; u_scratch = XFaceField(model.architecture, model.grid),
                                              v_scratch = YFaceField(model.architecture, model.grid),
                                              w_scratch = ZFaceField(model.architecture, model.grid),
                                              c_scratch = CenterField(model.architecture, model.grid),
                                              p = PressureField(model))

    u, v, w = model.velocities

    third_order_statistics = Dict()

    for tracer in keys(model.tracers)
        c = getproperty(model.tracers, tracer)

        # Keys
        wcc = Symbol(:w, tracer, tracer)
        cwu = Symbol(tracer, :wu)
        cpx = Symbol(tracer, :px)
        cpy = Symbol(tracer, :py)
        cpz = Symbol(tracer, :pz)

        third_order_statistics[wcc] = AveragedField(w * c * c, dims=(1, 2), operand_data=w_scratch.data)
        third_order_statistics[cwu] = AveragedField(c * w * u, dims=(1, 2), operand_data=c_scratch.data)
        third_order_statistics[cpx] = AveragedField(c * ∂x(p), dims=(1, 2), operand_data=c_scratch.data)
        third_order_statistics[cpy] = AveragedField(c * ∂y(p), dims=(1, 2), operand_data=c_scratch.data)
        third_order_statistics[cpz] = AveragedField(c * ∂z(p), dims=(1, 2), operand_data=c_scratch.data)
    end

    return third_order_statistics
end

function first_order_statistics(model; b = BuoyancyField(model),
                                       u_scratch = XFaceField(model.architecture, model.grid),
                                       v_scratch = YFaceField(model.architecture, model.grid),
                                       w_scratch = ZFaceField(model.architecture, model.grid),
                                       c_scratch = CenterField(model.architecture, model.grid),
                                       p = PressureField(model))

    output = merge(
                   horizontally_averaged_velocities(model),
                   horizontally_averaged_tracers(model),
                   )

    output[:p] = AveragedField(p, dims=(1, 2))

    if !has_buoyancy_tracer(model)
        output[:b] = AveragedField(b, dims=(1, 2))
    end

    return output
end

function second_order_statistics(model; b = BuoyancyField(model),
                                        u_scratch = XFaceField(model.architecture, model.grid),
                                        v_scratch = YFaceField(model.architecture, model.grid),
                                        w_scratch = ZFaceField(model.architecture, model.grid),
                                        c_scratch = CenterField(model.architecture, model.grid))

    output = merge(
                   velocity_covariances(model, u_scratch = u_scratch,
                                               v_scratch = v_scratch,
                                               w_scratch = w_scratch,
                                               c_scratch = c_scratch),

                   tracer_covariances(model, b = b,
                                             u_scratch = u_scratch,
                                             v_scratch = v_scratch,
                                             w_scratch = w_scratch,
                                             c_scratch = c_scratch)
                                                     
                  )

    if !has_buoyancy_tracer(model)
        u, v, w = model.velocities

        output[:ub] = AveragedField(u * b, dims=(1, 2), operand_data=u_scratch.data)
        output[:vb] = AveragedField(v * b, dims=(1, 2), operand_data=v_scratch.data)
        output[:wb] = AveragedField(w * b, dims=(1, 2), operand_data=w_scratch.data)
    end

    return output
end

function third_order_statistics(model; u_scratch = XFaceField(model.architecture, model.grid),
                                       v_scratch = YFaceField(model.architecture, model.grid),
                                       w_scratch = ZFaceField(model.architecture, model.grid),
                                       c_scratch = CenterField(model.architecture, model.grid),
                                       p = PressureField(model))

    output = merge(
                   third_order_velocity_statistics(model, p = p,
                                                   u_scratch = u_scratch,
                                                   v_scratch = v_scratch,
                                                   w_scratch = w_scratch,
                                                   c_scratch = c_scratch),

                   third_order_tracer_statistics(model, p = p,
                                                 u_scratch = u_scratch,
                                                 v_scratch = v_scratch,
                                                 w_scratch = w_scratch,
                                                 c_scratch = c_scratch)
                   )

    return output
end

function first_through_second_order(model; b = BuoyancyField(model), p = PressureField(model),
                                           u_scratch = XFaceField(model.architecture, model.grid),
                                           v_scratch = YFaceField(model.architecture, model.grid),
                                           w_scratch = ZFaceField(model.architecture, model.grid),
                                           c_scratch = CenterField(model.architecture, model.grid))
                                       

    output = merge(
                   first_order_statistics(model, b = b, p = p,
                                          u_scratch = u_scratch,
                                          v_scratch = v_scratch,
                                          w_scratch = w_scratch,
                                          c_scratch = c_scratch),

                   second_order_statistics(model,
                                           u_scratch = u_scratch,
                                           v_scratch = v_scratch,
                                           w_scratch = w_scratch,
                                           c_scratch = c_scratch,
                                                   b = b),

                  )
                  
    return output
end

function first_through_third_order(model; b = BuoyancyField(model), p = PressureField(model),
                                          u_scratch = XFaceField(model.architecture, model.grid),
                                          v_scratch = YFaceField(model.architecture, model.grid),
                                          w_scratch = ZFaceField(model.architecture, model.grid),
                                          c_scratch = CenterField(model.architecture, model.grid))
                                          

    output = merge(
                   first_order_statistics(model, b = b, p = p,
                                          u_scratch = u_scratch,
                                          v_scratch = v_scratch,
                                          w_scratch = w_scratch,
                                          c_scratch = c_scratch),

                   second_order_statistics(model,
                                           u_scratch = u_scratch,
                                           v_scratch = v_scratch,
                                           w_scratch = w_scratch,
                                           c_scratch = c_scratch,
                                                   b = b),

                   third_order_statistics(model, p = p,
                                          u_scratch = u_scratch,
                                          v_scratch = v_scratch,
                                          w_scratch = w_scratch,
                                          c_scratch = c_scratch)

                  )
                  
    return output
end
