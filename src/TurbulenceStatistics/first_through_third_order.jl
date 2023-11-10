using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField, CenterField
using Oceananigans.BuoyancyModels: BuoyancyField

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
                    :u => Average(u, dims=(1, 2)),
                    :v => Average(v, dims=(1, 2))
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
        averages[tracer] = Average(c, dims=(1, 2))
    end

    return averages
end

"""
    velocity_covariances(model; u_scratch = XFaceField(model.grid),
                                v_scratch = YFaceField(model.grid),
                                w_scratch = ZFaceField(model.grid),
                                c_scratch = CenterField(model.grid))

Returns a dictionary containing horizontal averages of the velocity covariances uᵢuⱼ.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.
"""
function velocity_covariances(model; u_scratch = XFaceField(model.grid),
                                     v_scratch = YFaceField(model.grid),
                                     w_scratch = ZFaceField(model.grid),
                                     c_scratch = CenterField(model.grid))

    u, v, w = model.velocities

    covariances = Dict(
                       :uu => Average(u * u, dims=(1, 2)),
                       :vv => Average(v * v, dims=(1, 2)),
                       :ww => Average(w * w, dims=(1, 2)),
                       :uv => Average(u * v, dims=(1, 2)),
                       :wv => Average(w * v, dims=(1, 2)),
                       :wu => Average(w * u, dims=(1, 2))
                       )

    return covariances
end

"""
    tracer_covariances(model; b = BuoyancyField(model),
                              u_scratch = XFaceField(model.grid),
                              v_scratch = YFaceField(model.grid),
                              w_scratch = ZFaceField(model.grid),
                              c_scratch = CenterField(model.grid))

Returns a dictionary containing horizontal averages of
tracer variance, tracer-velocity covariances, and tracer-buoyancy covariance.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.

`b` is the `BuoyancyField(model)` for model. If `b` is not specified, new memory is allocated to
store the computation of buoyancy in three-dimensions over `model.grid`.
"""
function tracer_covariances(model; b = BuoyancyField(model),
                                   u_scratch = XFaceField(model.grid),
                                   v_scratch = YFaceField(model.grid),
                                   w_scratch = ZFaceField(model.grid),
                                   c_scratch = CenterField(model.grid))

    u, v, w = model.velocities

    covariances = Dict()

    for tracer in keys(model.tracers)
        c = getproperty(model.tracers, tracer)

        # Keys
        cc = Symbol(tracer, tracer)
        uc = Symbol(:u, tracer)
        vc = Symbol(:v, tracer)
        wc = Symbol(:w, tracer)

        covariances[cc] = Average(c * c, dims=(1, 2))
        covariances[uc] = Average(u * c, dims=(1, 2))
        covariances[vc] = Average(v * c, dims=(1, 2))
        covariances[wc] = Average(w * c, dims=(1, 2))

        # Add covariance of tracer with buoyancy
        if tracer != :b && !has_buoyancy_tracer(model)
            bc = Symbol(:b, tracer)
            covariances[bc] = Average(b * c, dims=(1, 2))
        end
    end

    return covariances
end

"""
    third_order_velocity_statistics(model, u_scratch = XFaceField(model.grid),
                                           v_scratch = YFaceField(model.grid),
                                           w_scratch = ZFaceField(model.grid),
                                           c_scratch = CenterField(model.grid),
                                           p = model.pressures.pHY′ + model.pressures.pNHS)

Returns a dictionary of functions that calculate horizontally-averaged third-order statistics
that involve the velocity field. Includes statistics associated with pressure.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.
"""
function third_order_velocity_statistics(model; u_scratch = XFaceField(model.grid),
                                                v_scratch = YFaceField(model.grid),
                                                w_scratch = ZFaceField(model.grid),
                                                c_scratch = CenterField(model.grid),
                                                p = model.pressures.pHY′ + model.pressures.pNHS)

    u, v, w = model.velocities

    # Pressure-strain terms
    Σˣʸ = 0.5 * (∂y(u) + ∂x(v)) # FFC
    Σˣᶻ = 0.5 * (∂z(u) + ∂x(w)) # FCF
    Σʸᶻ = 0.5 * (∂z(v) + ∂y(w)) # CFF

    third_order_statistics = Dict(
                                  :uuu => Average(u * u * u, dims=(1, 2)),
                                  :uuv => Average(u * u * v, dims=(1, 2)),
                                  :uuw => Average(u * u * w, dims=(1, 2)),
                                  :uvv => Average(u * v * v, dims=(1, 2)),
                                  :uww => Average(u * w * w, dims=(1, 2)),

                                  :vvv => Average(v * v * v, dims=(1, 2)),
                                  :vvw => Average(v * v * w, dims=(1, 2)),
                                  :vww => Average(v * w * w, dims=(1, 2)),

                                  :www => Average(w * w * w, dims=(1, 2)),
                                  :wvu => Average(w * v * u, dims=(1, 2)),

                                   :up => Average(u * p,     dims=(1, 2)),
                                   :vp => Average(v * p,     dims=(1, 2)),
                                   :wp => Average(w * p,     dims=(1, 2)),

                                  :pux => Average(p * ∂x(u), dims=(1, 2)),
                                  :puy => Average(p * ∂y(u), dims=(1, 2)),
                                  :puz => Average(p * ∂z(u), dims=(1, 2)),

                                  :pvx => Average(p * ∂x(v), dims=(1, 2)),
                                  :pvy => Average(p * ∂y(v), dims=(1, 2)),
                                  :pvz => Average(p * ∂z(v), dims=(1, 2)),

                                  :pwx => Average(p * ∂x(w), dims=(1, 2)),
                                  :pwy => Average(p * ∂y(w), dims=(1, 2)),
                                  :pwz => Average(p * ∂z(w), dims=(1, 2)),
                                 )

    return third_order_statistics
end

"""
    third_order_tracer_statistics(model; u_scratch = XFaceField(model.grid),
                                         v_scratch = YFaceField(model.grid),
                                         w_scratch = ZFaceField(model.grid),
                                         c_scratch = CenterField(model.grid))

Returns a dictionary of functions that calculate horizontally-averaged third-order statistics
that involve tracers.

The scratch fields `u_scratch`, `v_scratch`, `w_scratch` and `c_scratch` specify
scratch space for computations at `u`, `v`, `w`, and tracer ("`c`") locations, respectively.
"""
function third_order_tracer_statistics(model; u_scratch = XFaceField(model.grid),
                                              v_scratch = YFaceField(model.grid),
                                              w_scratch = ZFaceField(model.grid),
                                              c_scratch = CenterField(model.grid),
                                              p = model.pressures.pHY′ + model.pressures.pNHS)

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

        third_order_statistics[wcc] = Average(w * c * c, dims=(1, 2))
        third_order_statistics[cwu] = Average(c * w * u, dims=(1, 2))
        third_order_statistics[cpx] = Average(c * ∂x(p), dims=(1, 2))
        third_order_statistics[cpy] = Average(c * ∂y(p), dims=(1, 2))
        third_order_statistics[cpz] = Average(c * ∂z(p), dims=(1, 2))
    end

    return third_order_statistics
end

function first_order_statistics(model; b = BuoyancyField(model),
                                       u_scratch = XFaceField(model.grid),
                                       v_scratch = YFaceField(model.grid),
                                       w_scratch = ZFaceField(model.grid),
                                       c_scratch = CenterField(model.grid),
                                       p = model.pressures.pHY′ + model.pressures.pNHS)

    output = merge(horizontally_averaged_velocities(model),
                   horizontally_averaged_tracers(model))

    output[:p] = Average(p, dims=(1, 2))

    if !has_buoyancy_tracer(model)
        output[:b] = Average(b, dims=(1, 2))
    end

    return output
end

function second_order_statistics(model; b = BuoyancyField(model),
                                        u_scratch = XFaceField(model.grid),
                                        v_scratch = YFaceField(model.grid),
                                        w_scratch = ZFaceField(model.grid),
                                        c_scratch = CenterField(model.grid))

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

        output[:ub] = Average(u * b, dims=(1, 2))
        output[:vb] = Average(v * b, dims=(1, 2))
        output[:wb] = Average(w * b, dims=(1, 2))
    end

    return output
end

function third_order_statistics(model; u_scratch = XFaceField(model.grid),
                                       v_scratch = YFaceField(model.grid),
                                       w_scratch = ZFaceField(model.grid),
                                       c_scratch = CenterField(model.grid),
                                       p = model.pressures.pHY′ + model.pressures.pNHS)

    output = merge(third_order_velocity_statistics(model, p = p,
                                                   u_scratch = u_scratch,
                                                   v_scratch = v_scratch,
                                                   w_scratch = w_scratch,
                                                   c_scratch = c_scratch),

                   third_order_tracer_statistics(model, p = p,
                                                 u_scratch = u_scratch,
                                                 v_scratch = v_scratch,
                                                 w_scratch = w_scratch,
                                                 c_scratch = c_scratch))

    return output
end

function first_through_second_order(model; b = BuoyancyField(model),
                                           u_scratch = XFaceField(model.grid),
                                           v_scratch = YFaceField(model.grid),
                                           w_scratch = ZFaceField(model.grid),
                                           c_scratch = CenterField(model.grid),
                                           p = model.pressures.pHY′ + model.pressures.pNHS)


    output = merge(first_order_statistics(model, b = b, p = p,
                                          u_scratch = u_scratch,
                                          v_scratch = v_scratch,
                                          w_scratch = w_scratch,
                                          c_scratch = c_scratch),

                   second_order_statistics(model,
                                           u_scratch = u_scratch,
                                           v_scratch = v_scratch,
                                           w_scratch = w_scratch,
                                           c_scratch = c_scratch,
                                                   b = b))

    return output
end

function first_through_third_order(model; b = BuoyancyField(model),
                                          u_scratch = XFaceField(model.grid),
                                          v_scratch = YFaceField(model.grid),
                                          w_scratch = ZFaceField(model.grid),
                                          c_scratch = CenterField(model.grid),
                                          p = model.pressures.pHY′ + model.pressures.pNHS)


    output = merge(first_order_statistics(model, b = b, p = p,
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
                                          c_scratch = c_scratch))
    
    return output
end

