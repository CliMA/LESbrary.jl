
struct TurbulentKineticEnergy{E, U, V, W, Ua, Va, Ea}
            e :: E # CellField...
            u :: U
            v :: V
            w :: W
    U_average :: Ua
    V_average :: Va
    e_average :: Ea
end

function TurbulentKineticEnergy(model)
    u, v, w = model.velocities
    e = CellField(model.architecture, model.grid)

    U_average = HorizontalAverage(u)
    V_average = HorizontalAverage(v)
    e_average = HorizontalAverage(e)

    return TurbulentKineticEnergy(e, u, v, w, U_average, V_average, e_average)
end

@inline w²(i, j, k, grid, w) = @inbounds w[i, j, k]^2

function _compute_turbulent_kinetic_energy!(tke, grid, u, v, w, U, V)
    @loop_xyz i j k grid begin
        @inbounds tke[i, j, k] = (   (u[i, j, k] - U[k + grid.Hz])^2 
                                   + (v[i, j, k] - V[k + grid.Hz])^2 
                                   + ℑzᵃᵃᶜ(i, j, k, grid, w², w)
                                 ) / 2
    end
    return nothing
end

function (tke::TurbulentKineticEnergy)(model)
    run_diagnostic(model, tke.U_average)
    run_diagnostic(model, tke.V_average)

    u, v, w = datatuple(model.velocities)

    Tx, Ty = 16, 16 # CUDA threads per block
    Bx, By, Bz = floor(Int, model.grid.Nx/Tx), floor(Int, model.grid.Ny/Ty), model.grid.Nz  # Blocks in grid

    @launch(device(model.architecture), threads=(Tx, Ty), blocks=(Bx, By, Bz), 
            _compute_turbulent_kinetic_energy!(tke.e.data, model.grid, u, v, w, tke.U_average.result, tke.V_average.result))

    # Compute horizontally-averaged turbulent kinetic energy
    return tke.e_average(model)
end

