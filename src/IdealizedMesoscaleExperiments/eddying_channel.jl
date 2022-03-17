using Statistics
using Printf
using Logging
using JLD2
using NCDatasets
using GeoData
using Oceanostics
using Oceananigans
using Oceananigans.Units

using Oceanostics.TKEBudgetTerms: TurbulentKineticEnergy, ZShearProduction

using LESbrary.Utils: SimulationProgressMessenger, fit_cubic, poly
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant
using LESbrary.TurbulenceStatistics: first_order_statistics,
                                     first_through_second_order,
                                     turbulent_kinetic_energy_budget,
                                     subfilter_momentum_fluxes,
                                     subfilter_tracer_fluxes,
                                     ViscousDissipation

try
    using CairoMakie
catch
    using GLMakie
finally
    @warn "Could not load either CairoMakie or GLMakie; animations are not available."
end

Logging.global_logger(OceananigansLogger())

@inline passive_tracer_forcing(x, y, z, t, p) = p.μ⁺ * exp(-(z - p.z₀)^2 / (2 * p.λ^2)) - p.μ⁻

# # Code credit: https://discourse.julialang.org/t/collecting-all-output-from-shell-commands/15592
# function execute(cmd::Cmd)
#     out, err = Pipe(), Pipe()
#     process = run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
#     close(out.in)
#     close(err.in)
#     return (stdout = out |> read |> String, stderr = err |> read |> String, code = process.exitcode)
# end
#test
function eddying_channel_simulation(; τ = 0.2, β = 1e-11, ridge_height = 0.0)
    filepath = "tau_" * string(τ) * "_beta_" * string(β) * "_ridge_height_" * string(ridge_height)
    filename = filepath
    # Domain
    Lx = 4000kilometers # zonal domain length [m]
    Ly = 2000kilometers # meridional domain length [m]
    Lz = 3kilometers    # depth [m]

    # number of grid points
    Nx = 80
    Ny = 40
    Nz = 30

    save_fields_interval = 7days
    stop_time = 20years + 1day
    Δt₀ = 5minutes

    grid = RectilinearGrid(arch;
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (3, 3, 3),
        x = (0, Lx),
        y = (0, Ly),
        z = (-Lz, 0))

    @info "Built a grid: $grid."

    #####
    ##### Boundary conditions
    #####

    α = 2e-4     # [K⁻¹] thermal expansion coefficient
    g = 9.8061   # [m s⁻²] gravitational constant
    cᵖ = 3994.0   # [J K⁻¹] heat capacity
    ρ = 1024.0   # [kg m⁻³] reference density

    parameters = (Ly = Ly,
        Lz = Lz,
        Qᵇ = 10 / (ρ * cᵖ) * α * g,          # buoyancy flux magnitude [m² s⁻³]
        y_shutoff = 5 / 6 * Ly,                # shutoff location for buoyancy flux [m]
        τ = 0.15 / ρ,                          # surface kinematic wind stress [m² s⁻²]
        μ = 2e-3,                            # quadratic bottom drag coefficient []
        ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
        H = Lz,                              # domain depth [m]
        h = 1000.0,                          # exponential decay scale of stable stratification [m]
        y_sponge = 19 / 20 * Ly,               # southern boundary of sponge layer [m]
        λt = 7days,                          # relaxation time scale [s]
        λs = 2e-4,                           # surface relaxation flux velocity [m/s]
    )

    @inline relaxation_profile(y, p) = p.ΔB * (y / p.Ly)
    @inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
        y = ynode(Center(), j, grid)
        return @inbounds p.λs * (model_fields.b[i, j, grid.Nz] - relaxation_profile(y, p))
    end

    buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)

    @inline function u_stress(i, j, grid, clock, model_fields, p)
        y = ynode(Center(), j, grid)
        return -p.τ * sin(π * y / p.Ly)
    end

    u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form = true, parameters = parameters)

    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * model_fields.u[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * model_fields.v[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)

    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

    #####
    ##### Coriolis
    #####

    f = -1e-4     # [s⁻¹]
    β = β    # [m⁻¹ s⁻¹]
    coriolis = BetaPlane(f₀ = f, β = β)

    #####
    ##### Forcing and initial condition
    #####

    @inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
    @inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

    @inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
        timescale = p.λt
        y = ynode(Center(), j, grid)
        z = znode(Center(), k, grid)
        target_b = initial_buoyancy(z, p)
        b = @inbounds model_fields.b[i, j, k]
        return -1 / timescale * mask(y, p) * (b - target_b)
    end

    Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

    # Turbulence closures

    κh = 0.5e-5 # [m²/s] horizontal diffusivity
    νh = 30.0   # [m²/s] horizontal viscocity
    κz = 0.5e-5 # [m²/s] vertical diffusivity
    νz = 3e-4   # [m²/s] vertical viscocity

    vertical_diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = νz, κ = κz)

    horizontal_diffusive_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)

    convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
        convective_νz = 0.0)

    #####
    ##### Model building
    #####

    @info "Building a model..."

    model = HydrostaticFreeSurfaceModel(grid = grid,
        free_surface = ImplicitFreeSurface(),
        momentum_advection = WENO5(),
        tracer_advection = WENO5(),
        buoyancy = BuoyancyTracer(),
        coriolis = coriolis,
        closure = (horizontal_diffusive_closure, vertical_diffusive_closure, convective_adjustment),
        tracers = (:b),
        boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs),
        forcing = (; b = Fb))

    @info "Built $model."

    #####
    ##### Initial conditions
    #####

    # resting initial condition
    ε(σ) = σ * randn()
    bᵢ(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h)) + ε(1e-8)
    uᵢ(x, y, z) = ε(1e-8)
    vᵢ(x, y, z) = ε(1e-8)
    wᵢ(x, y, z) = ε(1e-8)

    Δy = 100kilometers
    Δz = 100

    set!(model, b = bᵢ, u = uᵢ, v = vᵢ, w = wᵢ)

    #####
    ##### Simulation building
    #####

    simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

    # add timestep wizard callback
    wizard = TimeStepWizard(cfl = 0.1, max_change = 1.1, max_Δt = 20minutes)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

    # add progress callback
    wall_clock = [time_ns()]

    function print_progress(sim)
        @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

        wall_clock[1] = time_ns()

        return nothing
    end

    simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(20))


    #####
    ##### Diagnostics
    #####

    u, v, w = model.velocities
    b = model.tracers.b
    η = model.free_surface.η

    ζ = Field(∂x(v) - ∂y(u))

    B = Field(Average(b, dims = 1))
    U = Field(Average(u, dims = 1))
    η̄ = Field(Average(η, dims = 1))
    V = Field(Average(v, dims = 1))
    W = Field(Average(w, dims = 1))

    b′ = b - B
    u′ = u - U
    v′ = v - V
    w′ = w - W

    tke_op = @at (Center, Center, Center) (u′ * u′ + v′ * v′ + w′ * w′) / 2
    tke = Field(Average(tke_op, dims = 1))

    uv_op = @at (Center, Center, Center) u′ * v′
    vw_op = @at (Center, Center, Center) v′ * w′
    uw_op = @at (Center, Center, Center) u′ * w′

    u′v′ = Field(Average(uv_op, dims = 1))
    v′w′ = Field(Average(vw_op, dims = 1))
    u′w′ = Field(Average(uw_op, dims = 1))

    b′b′ = Field(Average(b′ * b′, dims = 1))
    v′b′ = Field(Average(b′ * v′, dims = 1))
    w′b′ = Field(Average(b′ * w′, dims = 1))

    outputs = (; b, ζ, u, v, w)

    zonally_averaged_outputs = (b = B, u = U, v = V, w = W, η = η̄,
        vb = v′b′, wb = w′b′, bb = b′b′,
        tke = tke, uv = u′v′, vw = v′w′, uw = u′w′)

    #####
    ##### Build checkpointer and output writer
    #####

    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = TimeInterval(10years),
        prefix = filename,
        force = true)

    slicers = (west = (1, :, :),
        east = (grid.Nx, :, :),
        south = (:, 1, :),
        north = (:, grid.Ny, :),
        bottom = (:, :, 1),
        top = (:, :, grid.Nz))

    for side in keys(slicers)
        indices = slicers[side]

        simulation.output_writers[side] = JLD2OutputWriter(model, outputs;
            schedule = TimeInterval(save_fields_interval),
            indices,
            prefix = filename * "_$(side)_slice",
            force = true)
    end

    @info "Running the simulation..."

    run!(simulation, pickup = false)

    @info "Simulation completed in " * prettytime(simulation.runjulia_wall_time)

    return filepath
end
