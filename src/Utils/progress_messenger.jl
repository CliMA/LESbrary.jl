using Dates: now, Second, format

mutable struct SimulationProgressMessenger{T} <: Function
    wall_time₀ :: T  # Wall time at simulation start
    wall_time⁻ :: T  # Wall time at previous calback
end

SimulationProgressMessenger() =
    SimulationProgressMessenger(
                      1e-9 * time_ns(),
                      1e-9 * time_ns())

function (pm::SimulationProgressMessenger)(simulation)
    model = simulation.model

    i, t = model.clock.iteration, model.clock.time

    # Avoid dividing by progress = 0 on iteration 0.
    i == 0 && return nothing

    progress = t / simulation.stop_time
    ETA = (1 - progress) / progress * simulation.run_wall_time
    ETA_datetime = now() + Second(round(Int, ETA))

    current_wall_time = 1e-9 * time_ns() - pm.wall_time₀
    time_since_last_callback = 1e-9 * time_ns() - pm.wall_time⁻
    wall_time_per_step = time_since_last_callback / simulation.callbacks[:progress].schedule.interval
    pm.wall_time⁻ = 1e-9 * time_ns()

    u_max = maximum(abs, model.velocities.u)
    v_max = maximum(abs, model.velocities.v)
    w_max = maximum(abs, model.velocities.w)
    ν_max = maximum(abs, model.diffusivity_fields.νₑ)

    @info @sprintf("[%06.2f%%] iteration: % 6d, time: % 10s, Δt: % 10s, wall time: % 8s (% 8s / time step), ETA: %s (%s)",
                    100 * progress, i, prettytime(t),
                    prettytime(simulation.Δt),
                    prettytime(current_wall_time),
                    prettytime(wall_time_per_step),
                    format(ETA_datetime, "yyyy-mm-dd HH:MM:SS"),
                    prettytime(ETA))

    adv_cfl = AdvectiveCFL(simulation.Δt)
    dif_cfl = DiffusiveCFL(simulation.Δt)
    @info @sprintf("          └── u⃗_max: (%.2e, %.2e, %.2e) m/s, CFL: %.2e, ν_max: %.2e m²/s, νCFL: %.2e",
                    u_max, v_max, w_max, adv_cfl(model), ν_max, dif_cfl(model))

    @info ""

    return nothing
end
