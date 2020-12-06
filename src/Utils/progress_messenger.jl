mutable struct SimulationProgressMessenger{T, U, V, W, N, A, D, Δ} <: Function
    wall_time₀ :: T  # Wall time at simulation start
    wall_time⁻ :: T  # Wall time at previous calback
          umax :: U
          vmax :: V
          wmax :: W
          νmax :: N
       adv_cfl :: A
       dif_cfl :: D
            Δt :: Δ
end

SimulationProgressMessenger(model, Δt) =
    SimulationProgressMessenger(
                      1e-9 * time_ns(),
                      1e-9 * time_ns(),
                      FieldMaximum(abs, model.velocities.u),
                      FieldMaximum(abs, model.velocities.v),
                      FieldMaximum(abs, model.velocities.w),
                      FieldMaximum(abs, model.diffusivities.νₑ),
                      AdvectiveCFL(Δt),
                      DiffusiveCFL(Δt),
                      Δt)

get_Δt(Δt) = Δt
get_Δt(wizard::TimeStepWizard) = wizard.Δt

function (pm::SimulationProgressMessenger)(simulation)
    model = simulation.model

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    current_wall_time = 1e-9 * time_ns() - pm.wall_time₀
    time_since_last_callback = 1e-9 * time_ns() - pm.wall_time⁻
    wall_time_per_step = time_since_last_callback / simulation.iteration_interval
    pm.wall_time⁻ = 1e-9 * time_ns()

    @info @sprintf("[%06.2f%%] iteration: % 6d, time: % 10s, Δt: % 10s, wall time: % 8s (% 8s / time step)",
                    progress, i, prettytime(t), prettytime(get_Δt(pm.Δt)), prettytime(current_wall_time), prettytime(wall_time_per_step))

    @info @sprintf("          └── umax: (%.2e, %.2e, %.2e) m/s, CFL: %.2e, νmax: %.2e m²/s, νCFL: %.2e",
                    pm.umax(model), pm.vmax(model), pm.wmax(model), pm.adv_cfl(model), pm.νmax(model), pm.dif_cfl(model))

    @info ""

    return nothing
end
