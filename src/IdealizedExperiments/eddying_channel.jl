using Statistics
using Printf
using Logging
using JLD2
using NCDatasets
using Oceananigans
using Oceananigans.Grids: ynode, znode

Logging.global_logger(OceananigansLogger())

default_boundary_layer_closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0, convective_νz = 0.0)

#####
##### Boundary conditions and forcing functions
#####

@inline relaxation_profile(y, p) = p.ΔB * (y / p.Ly)

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return @inbounds p.λs * (model_fields.b[i, j, grid.Nz] - relaxation_profile(y, p))
end

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return -p.τ * sin(π * y / p.Ly)
end

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.Cᵈ * model_fields.u[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.Cᵈ * model_fields.v[i, j, 1] * sqrt(model_fields.u[i, j, 1]^2 + model_fields.v[i, j, 1]^2)

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (p.Ly - p.y_sponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, j, k]
    return -1 / timescale * mask(y, p) * (b - target_b)
end

function eddying_channel_simulation(;
    architecture                      = CPU(),
    size                              = (80, 40, 30),
    extent                            = (4000kilometers, 2000kilometers, 3kilometers),
    max_momentum_flux                 = 1.5e-4,
    max_buoyancy_flux                 = 5e-9,
    drag_coefficient                  = 2e-3,
    vertical_buoyancy_jump            = 0.02,
    background_horizontal_diffusivity = 0.5e-5, # [m²/s] horizontal diffusivity
    background_horizontal_viscosity   = 30.0,   # [m²/s] horizontal viscocity
    background_vertical_diffusivity   = 0.5e-5, # [m²/s] vertical diffusivity
    background_vertical_viscosity     = 3e-4,   # [m²/s] vertical viscocity
    f₀                                = - 1e-4,
    β                                 = 1e-11,
    ridge_height                      = 0.0,
    boundary_layer_closure            = default_boundary_layer_closure,
    save_fields_interval              = 7days
    )

    filepath = "tau_" * string(max_momentum_flux) * "_beta_" * string(β) * "_ridge_height_" * string(ridge_height)
    filename = filepath

    # Domain
    Lx, Ly, Lz = extent
    Nx, Ny, Nz = size
    stop_time = 20years + 1day
    Δt₀ = 5minutes

    grid = RectilinearGrid(architecture; size, extent,
                           topology = (Periodic, Bounded, Bounded),
                           halo = (3, 3, 3))

    @info "Built a grid: $grid."

    #####
    ##### Boundary conditions
    #####

    parameters = (; Ly, Lz,
        Qᵇ = max_buoyancy_flux,      # buoyancy flux magnitude [m² s⁻³]
        y_shutoff = 5 / 6 * Ly,      # shutoff location for buoyancy flux [m]
        τ = max_momentum_flux,       # surface kinematic wind stress [m² s⁻²]
        Cᵈ = drag_coefficient,       # quadratic bottom drag coefficient []
        ΔB = vertical_buoyancy_jump, # surface horizontal buoyancy gradient [s⁻²]
        h = 1000.0,                  # exponential decay scale of stable stratification [m]
        y_sponge = 19 / 20 * Ly,     # southern boundary of sponge layer [m]
        λt = 7days,                  # relaxation time scale [s]
        λs = 2e-4,                   # surface relaxation flux velocity [m/s]
    )

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)
    b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)

    u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form = true, parameters = parameters)
    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

    #####
    ##### Coriolis
    #####

    coriolis = BetaPlane(; f₀, β)

    #####
    ##### Forcing and initial condition
    #####
    
    Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

    #####
    ##### Turbulence closures
    #####

    κh = background_horizontal_diffusivity
    νh = background_horizontal_viscosity
    κz = background_vertical_diffusivity
    νz = background_vertical_viscosity

    vertical_diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = νz, κ = κz)
    horizontal_diffusive_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
    closure = (horizontal_diffusive_closure, vertical_diffusive_closure, boundary_layer_closure)

    #####
    ##### Model building
    #####

    @info "Building a model..."

    model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                        free_surface = ImplicitFreeSurface(),
                                        momentum_advection = WENO5(),
                                        tracer_advection = WENO5(),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs),
                                        forcing = (; b = Fb))

    @info "Built $model."

    #####
    ##### Initial conditions
    #####

    # resting initial condition
    ε(σ) = σ * randn()
    bᵢ(x, y, z) = initial_buoyancy(z, parameters) + ε(1e-8)
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

    slice_indices = (;
        west   = (1, :, :),
        east   = (grid.Nx, :, :),
        south  = (:, 1, :),
        north  = (:, grid.Ny, :),
        bottom = (:, :, 1),
        top    = (:, :, grid.Nz))

    for side in keys(slice_indices)
        indices = slice_indices[side]

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
