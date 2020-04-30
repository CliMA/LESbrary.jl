mutable struct SimulationProgressMessenger{T, U, V, W, N, A, D, Z} <: Function
    wall_time :: T
         umax :: U
         vmax :: V
         wmax :: W
         νmax :: N
      adv_cfl :: A
      dif_cfl :: D
       wizard :: Z
end

SimulationProgressMessenger(model, Δt) =
    SimulationProgressMessenger(
                      time_ns(),
                      FieldMaximum(abs, model.velocities.u),
                      FieldMaximum(abs, model.velocities.v),
                      FieldMaximum(abs, model.velocities.w),
                      FieldMaximum(abs, model.diffusivities.νₑ),
                      AdvectiveCFL(Δt),
                      DiffusiveCFL(Δt),
                      Δt
                     )

function (pm::SimulationProgressMessenger)(simulation)
    model = simulation.model

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    elapsed_wall_time = 1e-9 * (time_ns() - pm.wall_time)
    pm.wall_time = time_ns()

    msg1 = @sprintf("[%06.2f%%] i: % 6d, sim time: % 10s, Δt: % 10s, wall time: % 8s,",
                    progress, i, prettytime(t), prettytime(simulation.Δt.Δt), prettytime(elapsed_wall_time))

    msg2 = @sprintf("umax: (%.2e, %.2e, %.2e) m/s, CFL: %.2e, νmax: %.2e m² s⁻¹, νCFL: %.2e,\n",
                    pm.umax(model), pm.vmax(model), pm.wmax(model), pm.adv_cfl(model), pm.νmax(model),
                    pm.dif_cfl(model))

    @printf("%s %s", msg1, msg2)

    return nothing
end
