using Statistics
using Printf
using Logging
using JLD2
using NCDatasets
using Oceananigans
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using Oceananigans.Grids: ynode, znode

Logging.global_logger(OceananigansLogger())

default_boundary_layer_closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

#####
##### Boundary conditions and forcing functions
#####

@inline relaxation_profile(y, p) = p.ΔB * y / p.Ly

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return @inbounds p.q★ * (model_fields.b[i, j, grid.Nz] - relaxation_profile(y, p))
end

@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(Center(), j, grid)
    return - p.τ * sin(π * y / p.Ly)
end

@inline u_drag(i, j, grid, clock, fields, p) = @inbounds -p.Cᵈ * fields.u[i, j, 1] * sqrt(fields.u[i, j, 1]^2 + fields.v[i, j, 1]^2)
@inline v_drag(i, j, grid, clock, fields, p) = @inbounds -p.Cᵈ * fields.v[i, j, 1] * sqrt(fields.u[i, j, 1]^2 + fields.v[i, j, 1]^2)

@inline initial_buoyancy(y, z, p) = p.ΔB * (exp(z / p.h) - 1) + p.ΔB * y / p.Ly
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (p.Ly - p.y_sponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λᵇ
    y = ynode(Center(), j, grid)
    z = znode(Center(), k, grid)
    target_b = initial_buoyancy(y, z, p)
    b = @inbounds model_fields.b[i, j, k]
    return -1 / timescale * mask(y, p) * (b - target_b)
end

#####
##### Util
#####

wall_clock = Ref(time_ns())

function print_progress(sim)

    msg = @sprintf("[%05.2f%%] i: %d, t: %s, Δt: %s, wall time: %s, max(u): (%6.2e, %6.2e, %6.2e) m s⁻¹",
                   100 * (time(sim) / sim.stop_time),
                   iteration(sim),
                   prettytime(sim),
                   prettytime(sim.Δt),
                   prettytime(1e-9 * (time_ns() - wall_clock[])),
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   maximum(abs, sim.model.velocities.w))

    if :e ∈ propertynames(sim.model.tracers)
        e = sim.model.tracers.e
        msg *= @sprintf(", max(e): %6.2e", maximum(abs, e))
    end

    println(msg)

    wall_clock[] = time_ns()

    return nothing
end

##### 
##### The business
#####

function eddying_channel_simulation(;
    name                              = "eddying_channel",
    architecture                      = CPU(),
    size                              = (80, 40, 30),
    extent                            = (4000kilometers, 2000kilometers, 3kilometers),
    vertical_grid_refinement          = nothing,
    peak_momentum_flux                = 1.5e-4,
    bottom_drag_coefficient           = 2e-3,
    f₀                                = - 1e-4,
    β                                 = 1e-11,
    max_Δt                            = 20minutes,
    initial_Δt                        = 20minutes,
    buoyancy_increment                = 0.02, # surface N² ~ buoyancy_increment / scale_height
    scale_height                      = 1000,
    biharmonic_horizontal_diffusivity = (extent[1] / size[1])^4 / 30days,
    biharmonic_horizontal_viscosity   = biharmonic_horizontal_diffusivity,
    buoyancy_piston_velocity          = 2e-4,   # [m s⁻¹] piston velocity for surface buoyancy flux
    buoyancy_restoring_time_scale     = 7days,  # [s] Timescale for internal buoyancy restoring
    ridge_height                      = 0.0,
    boundary_layer_closure            = default_boundary_layer_closure,
    slice_save_interval               = 7days,
    zonal_averages_interval           = nothing,
    time_averages_interval            = nothing,
    time_averages_window              = time_averages_interval,
    stop_time                         = 2years,
    pickup                            = false)

    filename = string(name,
                      "_tau_", peak_momentum_flux,
                      "_beta_", β,
                      "_ridge_height_", ridge_height)
                      

    # Domain
    Lx, Ly, Lz = extent
    Nx, Ny, Nz = size

    if isnothing(vertical_grid_refinement)
        z = (-Lz, 0)
    else # we're gonna stretch
        r = 1 - 1 / vertical_grid_refinement
        # Controls relative depth of stretching
        # (smaller concentrates stretching near surface):
        w = 0.5

        # Near-surface stretching, near-constant interior spacing:
        δ(k) = 1 - r * (1 + tanh((k - Nz) / (w * Nz)))
        Δz₁ = Lz / sum(δ.(1:Nz)) # Spacing at k = 1
        Δz(k) = Δz₁ * δ(k)
        z(k) = k == 1 ? -Lz : -Lz + sum(Δz.(1:k-1))
    end

    grid = RectilinearGrid(architecture; size, z,
                           topology = (Periodic, Bounded, Bounded),
                           halo = (3, 3, 3),
                           x = (0, Lx),
                           y = (0, Ly))

    @info "Built a grid: $grid."

    #####
    ##### Boundary conditions
    #####

    parameters = (; Ly, Lz,
        τ        = peak_momentum_flux,       # surface kinematic wind stress [m² s⁻²]
        Cᵈ       = bottom_drag_coefficient,  # quadratic bottom drag coefficient []
        h        = scale_height,             # exponential decay scale of stable stratification [m]
        ΔB       = buoyancy_increment,       # ∂z(b) ~ ΔB / h, ∂y(b) ~ ΔB / Ly
        y_sponge = 19 / 20 * Ly,             # southern boundary of sponge layer [m]
        λᵇ       = 7days,                    # internal buoyancy restoring time scale [s]
        q★       = buoyancy_piston_velocity, # surface buoyancy relaxation flux velocity [m/s]
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

    κh = biharmonic_horizontal_diffusivity
    νh = biharmonic_horizontal_viscosity

    horizontal_diffusive_closure = HorizontalScalarBiharmonicDiffusivity(ν = νh, κ = κh)
    closure = (horizontal_diffusive_closure, boundary_layer_closure)

    #####
    ##### Model building
    #####

    @info "Building a model..."

    tracers = [:b, :c]
    boundary_layer_closure isa CATKEVerticalDiffusivity && push!(tracers, :e)
    tracers = Tuple(tracers)

    model = HydrostaticFreeSurfaceModel(; grid, coriolis, tracers, closure,
                                        momentum_advection = WENO5(), # WENO5(; grid),
                                        tracer_advection = WENO5(), #WENO5(; grid),
                                        free_surface = ImplicitFreeSurface(),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs),
                                        forcing = (; b = Fb))

    @info "Built $model."

    #####
    ##### Initial conditions
    #####

    # Resting initial condition
    ε(σ) = σ * randn()
    bᵢ(x, y, z) = initial_buoyancy(y, z, parameters) + ε(1e-8)
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

    simulation = Simulation(model; Δt=initial_Δt, stop_time)

    # add timestep wizard callback
    wizard = TimeStepWizard(; cfl = 0.2, max_change = 1.01, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

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
                             prefix = filename * "_time_averages",
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

    simulation.output_writers[:checkpointer] =
        Checkpointer(model,
                     schedule = TimeInterval(10years),
                     prefix = filename,
                     force = true)

    return simulation
end

