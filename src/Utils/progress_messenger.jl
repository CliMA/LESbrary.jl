using Oceananigans.Fields: interior

mutable struct SimulationProgressMessenger{T} <: Function
    wall_time₀ :: T  # Wall time at simulation start
    wall_time⁻ :: T  # Wall time at previous callback
    iteration⁻ :: Int  # Iteration at previous callback
end

SimulationProgressMessenger(Δt) =
    SimulationProgressMessenger(
                      1e-9 * time_ns(),
                      1e-9 * time_ns(),
                      0)

function (pm::SimulationProgressMessenger)(simulation)
    model = simulation.model

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    current_wall_time = 1e-9 * time_ns() - pm.wall_time₀
    time_since_last_callback = 1e-9 * time_ns() - pm.wall_time⁻
    iterations_since_last_callback = i - pm.iteration⁻
    wall_time_per_step = time_since_last_callback / iterations_since_last_callback

    pm.wall_time⁻ = 1e-9 * time_ns()
    pm.iteration⁻ = i

    u_max = maximum(abs, interior(model.velocities.u))
    v_max = maximum(abs, interior(model.velocities.v))
    w_max = maximum(abs, interior(model.velocities.w))
    #ν_max = maximum(abs, model.diffusivity_fields.νₑ)

    @info @sprintf("[%06.2f%%] iteration: % 6d, time: % 10s, Δt: % 10s, wall time: % 8s (% 8s / time step)",
                    progress, i, prettytime(t), prettytime(simulation.Δt),
                    prettytime(current_wall_time), prettytime(wall_time_per_step))

    @info @sprintf("          └── u⃗_max: (%.2e, %.2e, %.2e) m/s", #, ν_max: %.2e m²/s",
                   u_max, v_max, w_max) #, ν_max)

    @info ""

    return nothing
end
