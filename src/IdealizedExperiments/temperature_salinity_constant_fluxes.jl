using Statistics
using Printf
using Logging
using OrderedCollections
using Oceanostics
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: zspacing
using Oceananigans.Utils: WallTimeInterval
using Oceananigans.Operators: Δzᶜᶜᶜ

using Oceanostics.TKEBudgetTerms: TurbulentKineticEnergy

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
    temperature_salinity_constant_fluxes_simulation(; kw...)

Build a ocean boundary large eddy simulation. The boundary layer is
initially quiescent with a "three layer" stratification structure, and forced
by constant momentum and buoyancy fluxes.
"""
function temperature_salinity_constant_fluxes_simulation(;
    name                             = "",
    size                             = (32, 32, 32),
    passive_tracers                  = false,
    explicit_closure                 = false,
    extent                           = (512meters, 512meters, 256meters),
    architecture                     = CPU(),
    stop_time                        = 48hours,
    initial_Δt                       = 1.0,
    f                                = 0.0,
    temperature_flux                 = 5e-4,
    salinity_flux                    = 0.0,
    momentum_flux                    = 0.0,
    tracer_forcing_timescale         = 6hours,
    surface_temperature              = 18.0,
    surface_salinity                 = 36.0,
    temperature_gradient             = 0.014,
    salinity_gradient                = 0.0021,
    stokes_drift                     = true, # will use ConstantFluxStokesDrift with stokes_drift_peak_wavenumber
    stokes_drift_peak_wavenumber     = 1e-6 * 9.81 / abs(momentum_flux), # severe approximation, it is what it is
    pickup                           = false,
    jld2_output                      = true,
    netcdf_output                    = false,
    checkpoint                       = false,
    statistics                       = "first_order", # or "second_order"
    snapshot_time_interval           = 2minutes,
    averages_time_interval           = 2hours,
    fields_time_interval             = 12hours,
    averages_time_window             = 10minutes,
    time_averaged_statistics         = false,
    data_directory                   = joinpath(pwd(), "data"))
    # End kwargs

    Nx, Ny, Nz = size
    Lx, Ly, Lz = extent
    slice_depth = 8.0

    Jᵀ = temperature_flux
    Jˢ = salinity_flux
    τˣ = momentum_flux
    dTdz = temperature_gradient
    dSdz = salinity_gradient
    T₀ = surface_temperature
    S₀ = surface_salinity
    stop_hours = stop_time / hour
    
    ## Determine filepath prefix
    prefix = @sprintf("temperature_salinity_hr%d_tx%.1e_JT%.1e_JS%.1e_T0%d_S0%d_Tz%.1e_Sz%.1e_f%.1e_Nh%d_Nz%d_",
                      stop_hours, abs(τˣ), Jᵀ, Jˢ, T₀, S₀, dTdz, dSdz, abs(f), Nx, Nz)
    
    data_directory = joinpath(data_directory, prefix * name) # save data in /data/prefix
    
    @info "Making the grid..."

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

    grid = RectilinearGrid(architecture; size,
                           halo = (5, 5, 5),
                           x = (0, extent[1]),
                           y = (0, extent[2]),
                           z = z_faces)
    
    # @show grid
        
    # Buoyancy and boundary conditions
    
    @info "Enforcing boundary conditions..."
    
    equation_of_state = TEOS10EquationOfState()
    buoyancy = SeawaterBuoyancy(; equation_of_state)

    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵀ))
    S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jˢ))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τˣ))
    
    # Tracer forcing
    
    @info "Forcing and sponging tracers..."
    
    # Passive tracer parameters
    λ = 8
    μ⁺ = 1 / tracer_forcing_timescale
    μ⁻ = √(2π) * λ / grid.Lz * μ⁺
    z₀ = -96
    c_forcing_func(x, y, z) = μ⁺ * exp(-(z - z₀)^2 / (2 * λ^2)) - μ⁻
    c_forcing_field = CenterField(grid)
    set!(c_forcing_field, c_forcing_func)
    c_forcing = Forcing(c_forcing_field)

    # Sponge layer for u, v, w, and b
    gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)
    u_sponge = v_sponge = w_sponge = Relaxation(rate=4/hour, mask=gaussian_mask)
    
    T_sponge = Relaxation(rate = 4/hour, mask = gaussian_mask,
                          target = LinearTarget{:z}(intercept=T₀, gradient=dTdz))

    S_sponge = Relaxation(rate = 4/hour, mask = gaussian_mask,
                          target = LinearTarget{:z}(intercept=S₀, gradient=dSdz))

    if stokes_drift && momentum_flux != 0.0
        @info "Whipping up the Stokes drift..."

        stokes_drift = ConstantFluxStokesDrift(grid, momentum_flux, stokes_drift_peak_wavenumber)
        uˢ₀ = CUDA.@allowscalar stokes_drift.uˢ[1, 1, grid.Nz]
        kᵖ = stokes_drift_peak_wavenumber
        a★ = stokes_drift.air_friction_velocity
        ρʷ = stokes_drift.water_density
        ρᵃ = stokes_drift.air_density
        u★ = a★ * sqrt(ρᵃ / ρʷ)
        La = sqrt(u★ / uˢ₀)
        @info @sprintf("Air u★: %.4f, water u★: %.4f, λᵖ: %.4f, La: %.3f, Surface Stokes drift: %.4f m s⁻¹",
                       a★, u★, 2π/kᵖ, La, uˢ₀)
    else
        stokes_drift = nothing
    end
    
    @info "Framing the model..."

    tracers = passive_tracers ? (:T, :S, :c) : (:T, :S)

    if explicit_closure
        closure = SmagorinskyLilly()
        advection = CenteredSecondOrder()
    else
        closure = nothing
        advection = WENO(order=9)
    end
    
    model = NonhydrostaticModel(; grid, buoyancy, tracers, stokes_drift, closure, advection,
                                coriolis = FPlane(; f),
                                boundary_conditions = (T=T_bcs, S=S_bcs, u=u_bcs),
                                forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge, S=S_sponge, c=c_forcing))
    
    # # Set Initial condition
    
    @info "Built model:"
    @info "$model"

    @info "Setting initial conditions..."
    
    ## Noise with 8 m decay scale
    Ξ(z) = rand() * exp(z / 8)
    Tᵢ(x, y, z) = T₀ + dTdz * z + 1e-4 * dTdz * grid.Lz * Ξ(z)
    Sᵢ(x, y, z) = S₀ + dSdz * z + 1e-4 * dSdz * grid.Lz * Ξ(z)
    set!(model, T=Tᵢ, S=Sᵢ)
    
    # # Prepare the simulation
    
    @info "Conjuring the simulation..."
    
    simulation = Simulation(model; Δt=initial_Δt, stop_time)
                        
    # Adaptive time-stepping
    wizard = TimeStepWizard(cfl=0.8, max_change=1.1, min_Δt=0.01, max_Δt=60.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    simulation.callbacks[:progress] = Callback(SimulationProgressMessenger(wizard), IterationInterval(100))
    
    # # Prepare Output
    
    overwrite_existing = !pickup
    
    if checkpoint
        @info "Strapping on checkpointer..."
        simulation.output_writers[:checkpointer] =
            Checkpointer(model, schedule = WallTimeInterval(20minutes),
                         prefix = prefix * "_checkpointer", dir = data_directory, cleanup=true)
    else
        pickup && throw(ArgumentError("Cannot pickup when checkpoint=false!"))
    end
    
    @info "Squeezing out statistics..."

    #=
    b = BuoyancyField(model)
    p = sum(model.pressures)

    ccc_scratch = Field{Center, Center, Center}(model.grid)
    ccf_scratch = Field{Center, Center, Face}(model.grid)
    fcf_scratch = Field{Face, Center, Face}(model.grid)
    cff_scratch = Field{Center, Face, Face}(model.grid)
    
    if statistics == "first_order"
        primitive_statistics = first_order_statistics(model, b=b, p=p, w_scratch=ccf_scratch, c_scratch=ccc_scratch)
    elseif statistics == "second_order"
        primitive_statistics = first_through_second_order(model, b=b, p=p, w_scratch=ccf_scratch, c_scratch=ccc_scratch)
    end
    
    U = Field(primitive_statistics[:u])
    V = Field(primitive_statistics[:v])
    B = Field(primitive_statistics[:b])

    e = TurbulentKineticEnergy(model, U=U, V=V)
    fields_to_output = merge(model.velocities, model.tracers, (; e=e))

    additional_statistics = Dict(:e => Average(TurbulentKineticEnergy(model, U=U, V=V), dims=(1, 2)),
                                 :Ri => ∂z(B) / (∂z(U)^2 + ∂z(V)^2))

    # If we change to WENO(order=9) so there are no subfilter fluxes...
    #subfilter_flux_statistics = merge(subfilter_momentum_fluxes(model), subfilter_tracer_fluxes(model))
    #statistics_to_output = merge(primitive_statistics, subfilter_flux_statistics, additional_statistics)
    #statistics_to_output = merge(primitive_statistics, additional_statistics)
    =#

    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))
    e = TurbulentKineticEnergy(model, U=U, V=V)
    fields_to_output = merge(model.velocities, model.tracers, (; e=e))

    u, v, w = model.velocities
    b = BuoyancyField(model)
    c = model.tracers.c
    T = model.tracers.T
    S = model.tracers.S
    
    statistics_to_output = (;
        u = U,
        v = V,
        b = Average(b, dims=(1, 2)),
        T = Average(T, dims=(1, 2)),
        S = Average(S, dims=(1, 2)),
        c = Average(c, dims=(1, 2)),
    )

    @info "Garnishing output writers..."
    @info "    - with fields: $(keys(fields_to_output))"
    @info "    - with statistics: $(keys(statistics_to_output))"
    
    global_attributes = OrderedDict()

    global_attributes[:LESbrary_jl_commit_SHA1]          = execute(`git rev-parse HEAD`).stdout |> strip
    global_attributes[:name]                             = name
    global_attributes[:temperature_flux]                 = Jᵀ
    global_attributes[:salinity_flux]                    = Jˢ
    global_attributes[:temperature_gradient]             = dTdz
    global_attributes[:salinity_gradient]                = dSdz
    global_attributes[:surface_temperature]              = T₀
    global_attributes[:surface_salinity]                 = S₀
    global_attributes[:momentum_flux]                    = τˣ
    global_attributes[:coriolis_parameter]               = f
    global_attributes[:tracer_forcing_timescale]         = tracer_forcing_timescale
    global_attributes[:tracer_forcing_width]             = λ
    global_attributes[:tracer_forcing_depth]             = -z₀

    if !isnothing(stokes_drift)
        global_attributes[:stokes_drift_surface_velocity] = uˢ₀  = CUDA.@allowscalar stokes_drift.uˢ[1, 1, grid.Nz]
        global_attributes[:stokes_drift_peak_wavenumber]         = stokes_drift_peak_wavenumber
        global_attributes[:stokes_drift_air_friction_velocity]   = stokes_drift.air_friction_velocity
        global_attributes[:stokes_drift_water_density]           = stokes_drift.water_density
        global_attributes[:stokes_drift_air_density]             = stokes_drift.air_density
        global_attributes[:stokes_drift_water_friction_velocity] = u★ = a★ * sqrt(ρᵃ / ρʷ)
        global_attributes[:stokes_drift_Langmuir_number]         = sqrt(u★ / uˢ₀)
    end

    global_attributes = NamedTuple(global_attributes)
    
    function init_save_some_metadata!(file, model)
        for (name, value) in pairs(global_attributes)
            file["parameters/$(string(name))"] = value
        end
        return nothing
    end
    
    ## Add JLD2 output writers
    
    # Prepare turbulence statistics
    zF = CUDA.@allowscalar Array(znodes(grid, Face()))
    k_xy_slice = searchsortedfirst(zF, -slice_depth)

    if jld2_output
        simulation.output_writers[:fields] =
            JLD2OutputWriter(model, fields_to_output;
                             dir = data_directory,
                             filename = name * "_fields",
                             schedule = TimeInterval(fields_time_interval),
                             with_halos = true,
                             overwrite_existing,
                             init = init_save_some_metadata!)


        simulation.output_writers[:xy] =
            JLD2OutputWriter(model, fields_to_output;
                             dir = data_directory,
                             filename = name * "_xy_slice",
                             schedule = TimeInterval(snapshot_time_interval),
                             indices = (:, :, k_xy_slice),
                             with_halos = true,
                             overwrite_existing,
                             init = init_save_some_metadata!)

        simulation.output_writers[:xz] =
            JLD2OutputWriter(model, fields_to_output;
                             dir = data_directory,
                             filename = name * "_xz_slice",
                             schedule = TimeInterval(snapshot_time_interval),
                             indices = (:, 1, :),
                             with_halos = true,
                             overwrite_existing,
                             init = init_save_some_metadata!)

        simulation.output_writers[:yz] =
            JLD2OutputWriter(model, fields_to_output;
                             dir = data_directory,
                             filename = name * "_yz_slice",
                             schedule = TimeInterval(snapshot_time_interval),
                             indices = (1, :, :),
                             with_halos = true,
                             overwrite_existing,
                             init = init_save_some_metadata!)

        simulation.output_writers[:statistics] =
            JLD2OutputWriter(model, statistics_to_output;
                             dir = data_directory,
                             filename = name * "_instantaneous_statistics",
                             schedule = TimeInterval(snapshot_time_interval),
                             with_halos = true,
                             overwrite_existing,
                             init = init_save_some_metadata!)

        if time_averaged_statistics
            simulation.output_writers[:time_averaged_statistics] =
                JLD2OutputWriter(model, statistics_to_output;
                                 dir = data_directory,
                                 filename = name * "_time_averaged_statistics",
                                 schedule = AveragedTimeInterval(averages_time_interval,
                                                                 window = averages_time_window),
                                 with_halos = true,
                                 overwrite_existing,
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

