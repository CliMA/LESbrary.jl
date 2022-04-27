using Statistics
using Printf
using Logging
using JLD2
using NCDatasets
using GeoData
using Oceanostics
using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators: Δzᶜᶜᶜ

using Oceanostics.TKEBudgetTerms: TurbulentKineticEnergy, ZShearProduction

using LESbrary.Utils: SimulationProgressMessenger, fit_cubic, poly
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant
using LESbrary.TurbulenceStatistics: first_order_statistics,
                                     first_through_second_order,
                                     turbulent_kinetic_energy_budget,
                                     subfilter_momentum_fluxes,
                                     subfilter_tracer_fluxes,
                                     ViscousDissipation

Logging.global_logger(OceananigansLogger())

@inline passive_tracer_forcing(x, y, z, t, p) = p.μ⁺ * exp(-(z - p.z₀)^2 / (2 * p.λ^2)) - p.μ⁻

"""
    three_layer_constant_fluxes_simulation(; kw...)

Build a ocean boundary large eddy simulation. The boundary layer is
initially quiescent with a "three layer" stratification structure, and forced
by constant momentum and buoyancy fluxes.
"""
function three_layer_constant_fluxes_simulation(;
    name                            = "",
    size                            = (32, 32, 32),
    passive_tracers                 = true,
    extent                          = (512meters, 512meters, 256meters),
    architecture                    = CPU(),
    stop_time                       = 0.1hours,
    initial_Δt                      = 1.0,
    f                               = 1e-4,
    buoyancy_flux                   = 1e-8,
    momentum_flux                   = -1e-4,
    thermocline_type                = "linear",
    surface_layer_depth             = 48.0,
    thermocline_width               = 24.0,
    surface_layer_buoyancy_gradient = 2e-6,
    thermocline_buoyancy_gradient   = 1e-5,
    deep_buoyancy_gradient          = 2e-6,
    surface_temperature             = 20.0,
    stokes_drift                    = true, # will use ConstantFluxStokesDrift with stokes_drift_peak_wavenumber
    stokes_drift_peak_wavenumber    = 1e-6 * 9.81 / abs(momentum_flux), # severe approximation, it is what it is
    pickup                          = false,
    jld2_output                     = true,
    netcdf_output                   = false,
    statistics                      = "first_order", # or "second_order"
    snapshot_time_interval          = 2minutes,
    averages_time_interval          = 3hours,
    averages_time_window            = 15minutes,
    time_averaged_statistics        = false,
    data_directory                  = joinpath(pwd(), "data"))
    # End kwargs

    Nx, Ny, Nz = size
    Lx, Ly, Lz = extent
    slice_depth = 8.0
    Qᵇ = buoyancy_flux
    Qᵘ = momentum_flux
    stop_hours = stop_time / hour
    
    ## Determine filepath prefix
    prefix = @sprintf("three_layer_constant_fluxes_%s_hr%d_Qu%.1e_Qb%.1e_f%.1e_Nh%d_Nz%d_",
                      thermocline_type, stop_hours, abs(Qᵘ), Qᵇ, f, Nx, Nz)
    
    data_directory = joinpath(data_directory, prefix * name) # save data in /data/prefix
    
    @info "Mapping grid..."

    refinement = 1.3
    stretching = 10
    Nz = size[3]
    Lz = extent[3]

    # Normalized height ranging from 0 to 1
    h(k) = (k - 1) / Nz

    # Linear near-surface generator
    ζ₀(k) = 1 + (h(k) - 1) / refinement

    # Bottom-intensified stretching function
    Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

    # Generating function
    z_faces(k) = Lz * (ζ₀(k) * Σ(k) - 1)

    grid = RectilinearGrid(architecture; size, halo = (3, 3, 3),
                           x = (0, extent[1]),
                           y = (0, extent[2]),
                           z = z_faces)
    
    @show grid
        
    # Buoyancy and boundary conditions
    
    @info "Enforcing boundary conditions..."
    
    equation_of_state = LinearEquationOfState(thermal_expansion=2e-4)
    buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=35.0)

    N²_surface_layer = surface_layer_buoyancy_gradient
    N²_thermocline   = thermocline_buoyancy_gradient
    N²_deep          = deep_buoyancy_gradient
    α = buoyancy.equation_of_state.thermal_expansion
    g = buoyancy.gravitational_acceleration
    
    Qᶿ = Qᵇ / (α * g)
    dθdz_surface_layer = N²_surface_layer / (α * g)
    dθdz_thermocline   = N²_thermocline   / (α * g)
    dθdz_deep          = N²_deep          / (α * g)
    
    θ_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᶿ),
                                    bottom = GradientBoundaryCondition(dθdz_deep))
    
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    
    # Tracer forcing
    
    @info "Forcing and sponging tracers..."
    
    # # Initial condition and sponge layer
    
    ## Fiddle with indices to get a correct discrete profile
    z = CUDA.@allowscalar Array(znodes(Center, grid))
    k_transition = searchsortedfirst(z, -surface_layer_depth)
    k_deep = searchsortedfirst(z, -(surface_layer_depth + thermocline_width))
    
    z_transition = z[k_transition]
    z_deep = z[k_deep]
    
    θ_surface = surface_temperature
    θ_transition = θ_surface + z_transition * dθdz_surface_layer
    θ_deep = θ_transition + (z_deep - z_transition) * dθdz_thermocline
    
    # Passive tracer parameters
    λ = 4.0
    μ⁺ = 1 / 6hour
    μ₀ = √(2π) * λ / grid.Lz * μ⁺ / 2
    μ∞ = √(2π) * λ / grid.Lz * μ⁺
    
    c₀_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=  0.0, λ=λ, μ⁺=μ⁺, μ⁻=μ₀))
    c₁_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=-48.0, λ=λ, μ⁺=μ⁺, μ⁻=μ∞))
    c₂_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=-96.0, λ=λ, μ⁺=μ⁺, μ⁻=μ∞))
    
    # Sponge layer for u, v, w, and T
    gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)
    u_sponge = v_sponge = w_sponge = Relaxation(rate=4/hour, mask=gaussian_mask)
    
    T_sponge = Relaxation(rate = 4/hour,
                          target = LinearTarget{:z}(intercept = θ_deep - z_deep*dθdz_deep, gradient = dθdz_deep),
                          mask = gaussian_mask)
    
    if stokes_drift && momentum_flux != 0.0
        @info "Whipping up the Stokes drift..."

        stokes_drift = ConstantFluxStokesDrift(grid, momentum_flux, stokes_drift_peak_wavenumber)
        uˢ₀ = CUDA.@allowscalar stokes_drift.uˢ[1, 1, grid.Nz]
        kᵖ = stokes_drift_peak_wavenumber
        a★ = stokes_drift.air_friction_velocity
        ρʷ = stokes_drift.water_density
        ρᵃ = stokes_drift.air_density
        u★ = a★ * sqrt(ρᵃ / ρʷ)
        @printf "Air u★: %.4f, water u★: %.4f, λᵖ: %.4f, Surface Stokes drift: %.4f m s⁻¹\n" a★ u★ 2π/kᵖ uˢ₀
    else
        stokes_drift = nothing
    end
    
    @info "Framing the model..."

    Δz = CUDA.@allowscalar grid.Δzᵃᵃᶜ[grid.Nz]
    @show Δz
    Cᴬᴹᴰ = SurfaceEnhancedModelConstant(Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4Δz)

    tracers = passive_tracers ? (:T, :c₀, :c₁, :c₂) : :T
    
    model = NonhydrostaticModel(; grid, buoyancy, tracers, stokes_drift,
                                timestepper = :RungeKutta3,
                                advection = WENO5(; grid),
                                coriolis = FPlane(f=f),
                                closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                                boundary_conditions = (T=θ_bcs, u=u_bcs),
                                forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge,
                                           c₀=c₀_forcing, c₁=c₁_forcing, c₂=c₂_forcing))
    
    # # Set Initial condition
    
    @info "Setting initial conditions..."
    
    ## Noise with 8 m decay scale
    Ξ(z) = rand() * exp(z / 8)
    
    function thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep,
                                            θ_deep, dθdz_surface_layer, dθdz_thermocline, dθdz_deep)

        if thermocline_type == "linear"
            return z -> θ_transition + dθdz_thermocline * (z - z_transition)
    
        elseif thermocline_type == "cubic"
            p1 = (z_transition, θ_transition)
            p2 = (z_deep, θ_deep)
            coeffs = fit_cubic(p1, p2, dθdz_surface_layer, dθdz_deep)
            return z -> poly(z, coeffs)
    
        else
            @error "Invalid thermocline type: $thermocline"
        end
    end
    
    θ_thermocline = thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep, θ_deep,
                                                   dθdz_surface_layer, dθdz_thermocline, dθdz_deep)
    
    """
        initial_temperature(x, y, z)
    
    Returns a three-layer initial temperature distribution. The average temperature varies in z
    and is augmented by three-dimensional, surface-concentrated random noise.
    """
    function initial_temperature(x, y, z)
        noise = 1e-6 * Ξ(z) * dθdz_surface_layer * grid.Lz
    
        if z_transition < z <= 0
            return θ_surface + dθdz_surface_layer * z + noise
        elseif z_deep < z <= z_transition
            return θ_thermocline(z) + noise
        else
            return θ_deep + dθdz_deep * (z - z_deep) + noise
        end
    end
    
    set!(model, T = initial_temperature)
    
    # # Prepare the simulation
    
    @info "Conjuring the simulation..."
    
    simulation = Simulation(model; Δt=initial_Δt, stop_time)
                        
    # Adaptive time-stepping
    wizard = TimeStepWizard(cfl=0.8, max_change=1.1, min_Δt=0.01, max_Δt=30.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    simulation.callbacks[:progress] = Callback(SimulationProgressMessenger(wizard), IterationInterval(100))
    
    # # Prepare Output
    
    @info "Strapping on checkpointer..."
    
    force = !pickup
    
    simulation.output_writers[:checkpointer] =
        Checkpointer(model, schedule = TimeInterval(stop_time/3), prefix = prefix * "_checkpointer", dir = data_directory)
    
    @info "Squeezing out statistics..."
    
    # Prepare turbulence statistics
    zF = CUDA.@allowscalar Array(znodes(Face, grid))
    k_xy_slice = searchsortedfirst(zF, -slice_depth)
    
    b = BuoyancyField(model)
    p = Field(sum(model.pressures))
    
    ccc_scratch = Field{Center, Center, Center}(model.grid)
    ccf_scratch = Field{Center, Center, Face}(model.grid)
    fcf_scratch = Field{Face, Center, Face}(model.grid)
    cff_scratch = Field{Center, Face, Face}(model.grid)
    
    if statistics == "first_order"
        primitive_statistics = first_through_second_order(model, b=b, p=p, w_scratch=ccf_scratch, c_scratch=ccc_scratch)
    elseif statistics == "second_order"
        primitive_statistics = first_order_statistics(model, b=b, p=p, w_scratch=ccf_scratch, c_scratch=ccc_scratch)
    end
    
    subfilter_flux_statistics = merge(
        subfilter_momentum_fluxes(model, uz_scratch=ccf_scratch, vz_scratch=cff_scratch, c_scratch=ccc_scratch),
        subfilter_tracer_fluxes(model, w_scratch=ccf_scratch))
    
    U = primitive_statistics[:u]
    V = primitive_statistics[:v]
    B = primitive_statistics[:b]
    e = Field(Average(TurbulentKineticEnergy(model, U=U, V=V), dims=(1, 2)))

    e = Field(TurbulentKineticEnergy(model, U=U, V=V))
    shear_production = Field(ZShearProduction(model, U=U, V=V))
    dissipation = Field(ViscousDissipation(model))
    tke_budget_statistics = turbulent_kinetic_energy_budget(model,
                                                            b=b, p=p, U=U, V=V, e=e,
                                                            shear_production=shear_production,
                                                            dissipation=dissipation)
    
    dynamics_statistics = Dict(:Ri => Field(∂z(B) / (∂z(U)^2 + ∂z(V)^2)))
    #fields_to_output = merge(model.velocities, model.tracers, (e=e, ϵ=dissipation))
    fields_to_output = merge(model.velocities, model.tracers, (; e=e))

    statistics_to_output = merge(primitive_statistics,
                                 subfilter_flux_statistics,
                                 tke_budget_statistics,
                                 dynamics_statistics)

    pop!(statistics_to_output, :tke_dissipation)

    @info "Garnishing output writers..."
    
    global_attributes = (
        LESbrary_jl_commit_SHA1 = execute(`git rev-parse HEAD`).stdout |> strip,
        name = name,
        thermocline_type = thermocline_type,
        buoyancy_flux = Qᵇ,
        momentum_flux = Qᵘ,
        temperature_flux = Qᶿ,
        coriolis_parameter = f,
        thermal_expansion_coefficient = α,
        gravitational_acceleration = g,
        boundary_condition_θ_top = Qᶿ,
        boundary_condition_θ_bottom = dθdz_deep,
        boundary_condition_u_top = Qᵘ,
        boundary_condition_u_bottom = 0.0,
        surface_layer_depth = surface_layer_depth,
        thermocline_width = thermocline_width,
        N²_surface_layer = N²_surface_layer,
        N²_thermocline = N²_thermocline,
        N²_deep = N²_deep,
        dθdz_surface_layer = dθdz_surface_layer,
        dθdz_thermocline = dθdz_thermocline,
        dθdz_deep = dθdz_deep,
        θ_surface = θ_surface,
        θ_transition = θ_transition,
        θ_deep = θ_deep,
        z_transition = z_transition,
        z_deep = z_deep,
        k_transition = k_transition,
        k_deep = k_deep
    )
    
    function init_save_some_metadata!(file, model)
        for (name, value) in pairs(global_attributes)
            file["parameters/$(string(name))"] = value
        end
        return nothing
    end
    
    ## Add JLD2 output writers
    
    if jld2_output
        simulation.output_writers[:xy] =
            JLD2OutputWriter(model, fields_to_output,
                             dir = data_directory,
                             prefix = name * "_xy_slice",
                             schedule = TimeInterval(snapshot_time_interval),
                             indices = (:, :, k_xy_slice),
                             with_halos = true,
                             force = force,
                             init = init_save_some_metadata!)

        simulation.output_writers[:xz] =
            JLD2OutputWriter(model, fields_to_output,
                             dir = data_directory,
                             prefix = name * "_xz_slice",
                             schedule = TimeInterval(snapshot_time_interval),
                             indices = (:, 1, :),
                             with_halos = true,
                             force = force,
                             init = init_save_some_metadata!)

        simulation.output_writers[:yz] =
            JLD2OutputWriter(model, fields_to_output,
                             dir = data_directory,
                             prefix = name * "_yz_slice",
                             schedule = TimeInterval(snapshot_time_interval),
                             indices = (1, :, :),
                             with_halos = true,
                             force = force,
                             init = init_save_some_metadata!)

        simulation.output_writers[:statistics] =
            JLD2OutputWriter(model, statistics_to_output,
                             dir = data_directory,
                             prefix = name * "_instantaneous_statistics",
                             schedule = TimeInterval(snapshot_time_interval),
                             with_halos = true,
                             force = force,
                             init = init_save_some_metadata!)

        if time_averaged_statistics
            simulation.output_writers[:time_averaged_statistics] =
                JLD2OutputWriter(model, statistics_to_output,
                                 dir = data_directory,
                                 prefix = name * "_time_averaged_statistics",
                                 schedule = AveragedTimeInterval(averages_time_interval,
                                                                 window = averages_time_window),
                                 with_halos = true,
                                 force = force,
                                 init = init_save_some_metadata!)
        end
    end

    if netcdf_output # Add NetCDF output writers
        statistics_to_output = Dict(string(k) => v for (k, v) in statistics_to_output)

        simulation.output_writers[:xy_nc] =
            NetCDFOutputWriter(model, fields_to_output,
                             mode = "c",
                         filepath = joinpath(data_directory, "xy_slice.nc"),
                         schedule = TimeInterval(snapshot_time_interval),
                          indices = (:, :, k_xy_slice),
                global_attributes = global_attributes)

        simulation.output_writers[:xz_nc] =
            NetCDFOutputWriter(model, fields_to_output,
                             mode = "c",
                         filepath = joinpath(data_directory, "xz_slice.nc"),
                         schedule = TimeInterval(snapshot_time_interval),
                          indices = (:, 1, :),
                global_attributes = global_attributes)

        simulation.output_writers[:yz_nc] =
            NetCDFOutputWriter(model, fields_to_output,
                             mode = "c",
                         filepath = joinpath(data_directory, "yz_slice.nc"),
                         schedule = TimeInterval(snapshot_time_interval),
                          indices = (1, :, :),
                global_attributes = global_attributes)

        simulation.output_writers[:stats_nc] =
            NetCDFOutputWriter(model, statistics_to_output,
                             mode = "c",
                         filepath = joinpath(data_directory, "instantaneous_statistics.nc"),
                         schedule = TimeInterval(snapshot_time_interval),
                global_attributes = global_attributes)

        if time_averaged_statistics
            simulation.output_writers[:averaged_stats_nc] =
                NetCDFOutputWriter(model, statistics_to_output,
                                 mode = "c",
                             filepath = joinpath(data_directory, "time_averaged_statistics.nc"),
                             schedule = AveragedTimeInterval(averages_time_interval, window = averages_time_window),
                    global_attributes = global_attributes)
        end
    end

    return simulation
end

