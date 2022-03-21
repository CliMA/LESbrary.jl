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

@inline relaxation_profile(y, p) = p.ΔB * y / p.Ly

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

#####
##### Util
#####

wall_clock = Ref(time_ns())

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end

##### 
##### The business
#####

function eddying_channel_simulation(;
    name                              = "",
    architecture                      = CPU(),
    size                              = (80, 40, 30),
    extent                            = (4000kilometers, 2000kilometers, 3kilometers),
    peak_momentum_flux                = 1.5e-4,
    bottom_drag_coefficient           = 2e-3,
    f₀                                = - 1e-4,
    β                                 = 1e-11,
    buoyancy_differential             = 0.02,
    background_horizontal_diffusivity = 0.5e-5, # [m²/s] horizontal diffusivity
    background_horizontal_viscosity   = 30.0,   # [m²/s] horizontal viscocity
    background_vertical_diffusivity   = 0.5e-5, # [m²/s] vertical diffusivity
    background_vertical_viscosity     = 3e-4,   # [m²/s] vertical viscocity
    ridge_height                      = 0.0,
    boundary_layer_closure            = default_boundary_layer_closure,
    slice_save_interval               = 7days,
    zonal_averages_interval           = nothing,
    time_averages_interval            = nothing,
    time_averages_window              = time_averages_interval,
    stop_time                         = 1year,
    pickup                            = false)

    filepath = name * "eddying_channel_tau_" * string(peak_momentum_flux) * "_beta_" * string(β) * "_ridge_height_" * string(ridge_height)
    filename = filepath

    # Domain
    Lx, Ly, Lz = extent
    Nx, Ny, Nz = size
    Δt₀ = 5minutes

    grid = RectilinearGrid(architecture; size, extent,
                           topology = (Periodic, Bounded, Bounded),
                           halo = (3, 3, 3))

    @info "Built a grid: $grid."

    #####
    ##### Boundary conditions
    #####

    parameters = (; Ly, Lz,
        y_shutoff = 5 / 6 * Ly,      # shutoff location for buoyancy flux [m]
        τ = peak_momentum_flux,       # surface kinematic wind stress [m² s⁻²]
        Cᵈ = bottom_drag_coefficient,       # quadratic bottom drag coefficient []
        ΔB = buoyancy_differential, # surface horizontal buoyancy gradient [s⁻²]
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
                                        tracers = (:b, :c),
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
    Δc = 2Δy
    cᵢ(x, y, z) = exp(-(y - Ly/2)^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

    set!(model, b = bᵢ, u = uᵢ, v = vᵢ, w = wᵢ, c = cᵢ)

    #####
    ##### Simulation building
    #####

    simulation = Simulation(model; Δt = Δt₀, stop_time)

    # add timestep wizard callback
    wizard = TimeStepWizard(cfl = 0.1, max_change = 1.1, max_Δt = 20minutes)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

    wall_clock[] = time_ns()
    simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

    #####
    ##### Diagnostics
    #####

    u, v, w = model.velocities
    b = model.tracers.b
    η = model.free_surface.η

    ζ = ∂x(v) - ∂y(u)

    outputs = (; b, ζ, u, v, w)

    zonally_averaged_outputs = (
        b   = Average(b,     dims=1),
        u   = Average(u,     dims=1),
        v   = Average(v,     dims=1),
        w   = Average(w,     dims=1),
        η   = Average(η,     dims=1),
        vb  = Average(v * b, dims=1),
        wb  = Average(w * b, dims=1),
        bb  = Average(b * b, dims=1),
        uv  = Average(u * v, dims=1),
        uw  = Average(u * w, dims=1),
        vw  = Average(v * w, dims=1),
        uu  = Average(u * u, dims=1),
        vv  = Average(v * v, dims=1),
        ww  = Average(w * w, dims=1),
        ζ²  = Average(ζ^2,   dims=1),
    )

    #####
    ##### Build checkpointer and output writer
    #####
    
    if !isnothing(zonal_averages_interval)
        simulation.output_writers[:zonal] =
            JLD2OutputWriter(model, zonally_averaged_outputs;
                             schedule = TimeInterval(zonal_averages_interval),
                             prefix = filename * "_zonal_averages",
                             force = true)
    end

    if !isnothing(time_averages_interval)
        schedule = AveragedTimeInterval(time_averages_interval, window=time_averages_window),
        simulation.output_writers[:time] =
            JLD2OutputWriter(model, zonally_averaged_outputs;
                             schedule = AveragedTimeInterval(zonal_averages_interval),
                             prefix = filename * "_zonal_time_averages",
                             force = true)
    end

    slice_indices = (;
        west   = (1,       :,       :      ),
        east   = (grid.Nx, :,       :      ),
        south  = (:,       1,       :      ),
        north  = (:,       grid.Ny, :      ),
        bottom = (:,       :,       1      ),
        top    = (:,       :,       grid.Nz)
    )

    for side in keys(slice_indices)
        indices = slice_indices[side]

        simulation.output_writers[side] =
            JLD2OutputWriter(model, outputs; indices,
                             schedule = TimeInterval(slice_save_interval),
                             prefix = filename * "_$(side)_slice",
                             force = true)
    end

    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = TimeInterval(10years),
        prefix = filename,
        force = true)

    return simulation
end

