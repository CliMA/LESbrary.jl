# # Turbulent mixing of a three layer boundary layer driven by constant surface fluxes

# This script runs a simulation of a turbulent oceanic boundary layer with an initial
# three-layer temperature stratification. Turbulent mixing is driven by constant fluxes
# of momentum and heat at the surface.
#
# This script is set up to be configurable on the command line --- a useful property
# when launching multiple jobs at on a cluster.

using Pkg
using Statistics
using Printf
using Logging
using ArgParse
using JLD2
using Plots

using LESbrary
using Oceananigans
using Oceananigans.Buoyancy
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Utils

using Oceananigans.Fields: PressureField

using LESbrary.Utils: SimulationProgressMessenger, fit_cubic, poly
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant
using LESbrary.TurbulenceStatistics: first_through_second_order, turbulent_kinetic_energy_budget,
                                     subfilter_momentum_fluxes, subfilter_tracer_fluxes,
                                     TurbulentKineticEnergy, ShearProduction, ViscousDissipation

Logging.global_logger(OceananigansLogger())

# To start, we ensure that all packages in the LESbrary environment are installed:

Pkg.instantiate()

# Next, we parse the command line arguments

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--size"
            help = "The number of grid points in x, y, and z."
            nargs = 3
            default = [32, 32, 32]
            arg_type = Int

        "--extent"
            help = "The length of the x, y, and z dimensions."
            nargs = 3
            default = [512meters, 512meters, 256meters]
            arg_type = Float64

        "--buoyancy-flux"
            help = """The surface buoyancy flux in units of m² s⁻³.
                      A positive buoyancy flux implies cooling.

                      Note:
                          buoyancy-flux = + 1e-7 corresponds to cooling at 208 W / m²
                          buoyancy-flux = + 1e-8 corresponds to cooling at 21 W / m²
                          buoyancy-flux = - 1e-7 corresponds to heating at 208 W / m²"""
            default = 1e-8
            arg_type = Float64

        "--momentum-flux"
            help = """The surface x-momentum flux divided by density in units of m² s⁻².
                      A negative flux drives currents in the positive x-direction.

                      Note:
                        momentum-flux = - 1e-4 corresponds to U₁₀ = +6 m/s, roughly speaking
                        momentum-flux = - 1e-3 corresponds to U₁₀ = +19 m/s, roughly speaking"""
            default = -1e-4
            arg_type = Float64

        "--coriolis"
            help = "The Coriolis parameter, calculated as f=2Ω*sinϕ, where Ω is the rotation rate of the Earth in rad/s and ϕ is the latitude."
            default = 1e-4
            arg_type = Float64

        "--surface-temperature"
            help = """The temperature at the surface in ᵒC."""
            default = 20.0
            arg_type = Float64

        "--surface-layer-depth"
            help = "The depth of the surface layer in units of m."
            default = 48.0
            arg_type = Float64

        "--thermocline"
            help = """Two choices for the thermocline structure:
                        * linear: a thermocline with a linear buoyancy structure (constant stratification)
                        * cubic: a thermocline with a fitted cubic structure"""
            default = "linear"
            arg_type = String

        "--thermocline-width"
            help = "The width of the thermocline in units of m."
            default = 24.0
            arg_type = Float64

        "--surface-layer-buoyancy-gradient"
            help = "The buoyancy gradient in the surface layer in units s⁻²."
            default = 2e-6
            arg_type = Float64

        "--thermocline-buoyancy-gradient"
            help = "The buoyancy gradient in the thermocline in units s⁻²."
            default = 1e-5
            arg_type = Float64

        "--deep-buoyancy-gradient"
            help = "The buoyancy gradient below the thermocline in units s⁻²."
            default = 2e-6
            arg_type = Float64

        "--hours"
            help = "Number of hours to run the simulation for"
            default = 0.1
            arg_type = Float64

        "--animation"
            help = "Make an animation of the horizontal and vertical velocity when the simulation completes."
            default = false
            arg_type = Bool

        "--pickup"
            help = "Whether or not to pick the simulation up from latest checkpoint"
            default = false
            arg_type = Bool

        "--name"
            help = "A name to add to the end of the output filename, e.g. weak_wind_strong_cooling."
            default = ""
            arg_type = String
    end

    return parse_args(settings)
end

# # Setup
#
# We start by parsing the arguments received on the command line, and prescribing
# a few additional output-related arguments.

@info "Parsing command line arguments..."

args = parse_command_line_arguments()

Nx, Ny, Nz = args["size"]
Lx, Ly, Lz = args["extent"]
stop_hours = args["hours"]
f = args["coriolis"]
name = args["name"]

snapshot_time_interval = 10minutes
averages_time_interval = 3hours
averages_time_window = 15minutes

slice_depth = 8.0

# Domain
#
# We use a three-dimensional domain that's twice as wide as it is deep.
# We choose this aspect ratio so that the horizontal scale is 4x larger
# than the boundary layer depth when the boundary layer penetrates half
# the domain depth.

@info "Mapping grid..."

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# Buoyancy and boundary conditions

@info "Enforcing boundary conditions..."

Qᵇ = args["buoyancy-flux"]
Qᵘ = args["momentum-flux"]

thermocline_type = args["thermocline"]

prefix = @sprintf("three_layer_constant_fluxes_%s_hr%d_Qu%.1e_Qb%.1e_f%.1e_Nh%d_Nz%d_",
                  thermocline_type, stop_hours, abs(Qᵘ), Qᵇ, f, grid.Nx, grid.Nz)
data_directory = joinpath(@__DIR__, "..", "data", prefix * name) # save data in /data/prefix

surface_layer_depth = args["surface-layer-depth"]
thermocline_width = args["thermocline-width"]

N²_surface_layer = args["surface-layer-buoyancy-gradient"]
N²_thermocline = args["thermocline-buoyancy-gradient"]
N²_deep = args["deep-buoyancy-gradient"]

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

Qᶿ = Qᵇ / (α * g)
dθdz_surface_layer = N²_surface_layer / (α * g)
dθdz_thermocline   = N²_thermocline   / (α * g)
dθdz_deep          = N²_deep          / (α * g)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz_deep))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

# Tracer forcing

@info "Forcing and sponging tracers..."

# # Initial condition and sponge layer

## Fiddle with indices to get a correct discrete profile
k_transition = searchsortedfirst(grid.zC, -surface_layer_depth)
k_deep = searchsortedfirst(grid.zC, -(surface_layer_depth + thermocline_width))

z_transition = grid.zC[k_transition]
z_deep = grid.zC[k_deep]

θ_surface = args["surface-temperature"]
θ_transition = θ_surface + z_transition * dθdz_surface_layer
θ_deep = θ_transition + (z_deep - z_transition) * dθdz_thermocline

@inline passive_tracer_forcing(x, y, z, t, p) = p.μ⁺ * exp(-(z - p.z₀)^2 / (2 * p.λ^2)) - p.μ⁻

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

# # LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.

@info "Building the wall model..."

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# # Instantiate Oceananigans.IncompressibleModel

@info "Framing the model..."

model = IncompressibleModel(
           architecture = GPU(),
            timestepper = :RungeKutta3,
              advection = WENO5(),
                   grid = grid,
                tracers = (:T, :c₀, :c₁, :c₂),
               buoyancy = buoyancy,
               coriolis = FPlane(f=f),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = (T=θ_bcs, u=u_bcs),
                forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge,
                           c₀=c₀_forcing, c₁=c₁_forcing, c₂=c₂_forcing)
)

# # Set Initial condition

@info "Setting initial conditions..."

## Noise with 8 m decay scale
Ξ(z) = rand() * exp(z / 8)

function thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep, θ_deep, dθdz_surface_layer, dθdz_thermocline, dθdz_deep)
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

@info "Gesticulating mimes..."

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.8, Δt=1.0, max_change=1.1, min_Δt=0.01, max_Δt=30.0)

stop_time = stop_hours * hour

simulation = Simulation(model,
                    Δt = wizard,
             stop_time = stop_time,
    iteration_interval = 10,
              progress = SimulationProgressMessenger(model, wizard)
)

# # Prepare Output

@info "Strapping on checkpointer..."

pickup = args["pickup"]
force = pickup ? false : true

simulation.output_writers[:checkpointer] =
    Checkpointer(model, schedule = TimeInterval(stop_time/3), prefix = prefix * "_checkpointer", dir = data_directory)

@info "Squeezing out statistics..."

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# TODO: Save command line arguments used to a file

# Prepare turbulence statistics
k_xy_slice = searchsortedfirst(grid.zF[:], -slice_depth)

b = BuoyancyField(model)
p = PressureField(model)

ccc_scratch = Field(Cell, Cell, Cell, model.architecture, model.grid)
ccf_scratch = Field(Cell, Cell, Face, model.architecture, model.grid)
fcf_scratch = Field(Face, Cell, Face, model.architecture, model.grid)
cff_scratch = Field(Cell, Face, Face, model.architecture, model.grid)

primitive_statistics = first_through_second_order(model, b=b, p=p, w_scratch=ccf_scratch, c_scratch=ccc_scratch)

subfilter_flux_statistics = merge(
    subfilter_momentum_fluxes(model, uz_scratch=ccf_scratch, vz_scratch=cff_scratch, c_scratch=ccc_scratch),
    subfilter_tracer_fluxes(model, w_scratch=ccf_scratch),
)

U = primitive_statistics[:u]
V = primitive_statistics[:v]

e = TurbulentKineticEnergy(model, U=U, V=V)
shear_production = ShearProduction(model, data=ccc_scratch.data, U=U, V=V)
dissipation = ViscousDissipation(model, data=ccc_scratch.data)

tke_budget_statistics = turbulent_kinetic_energy_budget(model, b=b, p=p, U=U, V=V, e=e,
                                                        shear_production=shear_production, dissipation=dissipation)

fields_to_output = merge(model.velocities, model.tracers, (e=e, ϵ=dissipation))

statistics_to_output = merge(primitive_statistics, subfilter_flux_statistics, tke_budget_statistics)
statistics_to_output = Dict(string(k) => v for (k, v) in statistics_to_output)

@info "Garnishing output writers..."

global_attributes = (
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

simulation.output_writers[:xy_jld2] =
    JLD2OutputWriter(model, fields_to_output,
                              dir = data_directory,
                           prefix = "xy_slice",
                         schedule = TimeInterval(snapshot_time_interval),
                     field_slicer = FieldSlicer(k=k_xy_slice),
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:xz_jld2] =
    JLD2OutputWriter(model, fields_to_output,
                              dir = data_directory,
                           prefix = "xz_slice",
                         schedule = TimeInterval(snapshot_time_interval),
                     field_slicer = FieldSlicer(j=1),
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:yz_jld2] =
    JLD2OutputWriter(model, fields_to_output,
                              dir = data_directory,
                           prefix = "yz_slice",
                         schedule = TimeInterval(snapshot_time_interval),
                     field_slicer = FieldSlicer(i=1),
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:statistics_jld2] =
    JLD2OutputWriter(model, statistics_to_output,
                          dir = data_directory,
                       prefix = "instantaneous_statistics",
                     schedule = TimeInterval(snapshot_time_interval),
                        force = force,
                         init = init_save_some_metadata!)

simulation.output_writers[:averaged_statistics_jld2] =
    JLD2OutputWriter(model, statistics_to_output,
                          dir = data_directory,
                       prefix = "averaged_statistics",
                     schedule = AveragedTimeInterval(averages_time_interval,
                                                     window = averages_time_window),
                        force = force,
                         init = init_save_some_metadata!)

## Add NetCDF output writers

simulation.output_writers[:xy_nc] =
    NetCDFOutputWriter(model, fields_to_output,
                 filepath = joinpath(data_directory, "xy_slice.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
             field_slicer = FieldSlicer(k=k_xy_slice),
        global_attributes = global_attributes)

simulation.output_writers[:xz_nc] =
    NetCDFOutputWriter(model, fields_to_output,
                 filepath = joinpath(data_directory, "xz_slice.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
             field_slicer = FieldSlicer(j=1),
        global_attributes = global_attributes)

simulation.output_writers[:yz_nc] =
    NetCDFOutputWriter(model, fields_to_output,
                 filepath = joinpath(data_directory, "yz_slice.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
             field_slicer = FieldSlicer(i=1),
        global_attributes = global_attributes)

simulation.output_writers[:statistics_nc] =
    NetCDFOutputWriter(model, statistics_to_output,
                 filepath = joinpath(data_directory, "instantaneous_statistics.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
        global_attributes = global_attributes)

simulation.output_writers[:averaged_statistics_nc] =
    NetCDFOutputWriter(model, statistics_to_output,
                 filepath = joinpath(data_directory, "time_averaged_statistics.nc"),
                 schedule = AveragedTimeInterval(averages_time_interval, window = averages_time_window),
        global_attributes = global_attributes)

# # Run

@info "Reticulating splines..."

run!(simulation)

# # Load and plot turbulence statistics

make_animation = args["animation"]

ENV["GKSwstype"] = "100"

using Plots
using GeoData
using NCDatasets
using GeoData: GeoXDim, GeoYDim, GeoZDim

@dim xC GeoXDim "x"
@dim xF GeoXDim "x"
@dim yC GeoYDim "y"
@dim yF GeoYDim "y"
@dim zC GeoZDim "z"
@dim zF GeoZDim "z"

function squeeze(A)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims=singleton_dims)
end

if make_animation
    ds_xy = NCDstack(joinpath(data_directory, "xy_slice.nc"))
    ds_xz = NCDstack(joinpath(data_directory, "xz_slice.nc"))

    _, _, _, times = dims(ds_xy[:u])
    Nt = length(times)

    kwargs = (xlabel="", ylabel="", xticks=[], yticks=[], colorbar=false, framestyle=:box)

    anim = @animate for n in 1:Nt
        @info "Plotting xy/xz movie frame $n/$Nt..."

        xy_xz_plots = (
            plot(ds_xy[:u][Ti=n] |> squeeze; title="u-velocity", color=:balance, clims=(-0.1, 0.1), kwargs...),
            plot(ds_xy[:v][Ti=n] |> squeeze; title="v-velocity", color=:balance, clims=(-0.1, 0.1), kwargs...),
            plot(ds_xy[:w][Ti=n] |> squeeze; title="w-velocity", color=:balance, clims=(-0.1, 0.1), kwargs...),
            plot(ds_xy[:T][Ti=n] |> squeeze; title="temperature", color=:thermal, kwargs...),
            plot(ds_xy[:c₀][Ti=n] |> squeeze; title="tracer 0", color=:ice, kwargs...),
            plot(ds_xy[:c₁][Ti=n] |> squeeze; title="tracer 1", color=:ice, kwargs...),
            plot(ds_xy[:c₂][Ti=n] |> squeeze; title="tracer 2", color=:ice, kwargs...),
            plot(ds_xy[:e][Ti=n] .|> log10 |> squeeze; title="log TKE", color=:deep, clims=(-5, -2), kwargs...),
            plot(ds_xy[:ϵ][Ti=n] .|> log10 |> squeeze; title="log ε", color=:dense, clims=(-10, -5), kwargs...),
            plot(ds_xz[:u][Ti=n] |> squeeze; title="", color=:balance, clims=(-0.1, 0.1), kwargs...),
            plot(ds_xz[:v][Ti=n] |> squeeze; title="", color=:balance, clims=(-0.1, 0.1), kwargs...),
            plot(ds_xz[:w][Ti=n] |> squeeze; title="", color=:balance, clims=(-0.1, 0.1), kwargs...),
            plot(ds_xz[:T][Ti=n] |> squeeze; title="", color=:thermal, kwargs...),
            plot(ds_xz[:c₀][Ti=n] |> squeeze; title="", color=:ice, kwargs...),
            plot(ds_xz[:c₁][Ti=n] |> squeeze; title="", color=:ice, kwargs...),
            plot(ds_xz[:c₂][Ti=n] |> squeeze; title="", color=:ice, kwargs...),
            plot(ds_xz[:e][Ti=n] .|> log10 |> squeeze; title="", color=:deep, clims=(-8, -2), kwargs...),
            plot(ds_xz[:ϵ][Ti=n] .|> log10 |> squeeze; title="", color=:dense, clims=(-15, -5), kwargs...),
        )

        plot(xy_xz_plots..., layout=(2, 9), size=(2000, 400))
    end

    mp4(anim, joinpath(data_directory, "xy_xz_movie.mp4"), fps=15)
    gif(anim, joinpath(data_directory, "xy_xz_movie.gif"), fps=15)
end

if make_animation
    ds = NCDstack(joinpath(data_directory, "statistics.nc"))

    kwargs = (linewidth=3, linealpha=0.7, ylabel="", yticks=[], ylims=(-128, 0), title="",
              grid=false, legend=:bottomright, legendfontsize=12, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    _, times = dims(ds[:u])
    Nt = length(times)

    anim = @animate for n in 1:Nt
        @info "Plotting statistics $n/$Nt..."

        uv_plot = plot(ds[:u][Ti=n]; label="u", xlims=(-0.1, 0.1), xticks=[-0.1, 0, 0.1], kwargs...)
                  plot!(ds[:v][Ti=n]; label="v", xlabel="m/s", kwargs..., yticks=-128:32:0)

        T_plot = plot(ds[:T][Ti=n]; label="T", xlabel="°C", kwargs...)

        c_plot = plot(ds[:c₀][Ti=n]; label="c₀", xlims=(-0.32, 2), xticks=[0, 1, 2], kwargs...)
                 plot!(ds[:c₁][Ti=n]; label="c₁", kwargs...)
                 plot!(ds[:c₂][Ti=n]; label="c₂", xlabel="c", kwargs...)

        ke_plot = plot(ds[:uu][Ti=n]; label="uu", xlims=(-1e-3, 0.01), xticks=[0, 0.01], kwargs...)
                  plot!(ds[:vv][Ti=n]; label="vv", kwargs...)
                  plot!(ds[:ww][Ti=n]; label="ww", xlabel="m²/s²", kwargs...)

        U_Σ_plot = plot(ds[:uv][Ti=n]; label="uv", xlims=(-0.005, 0.002), xticks=[-0.005, 0], kwargs...)
                   plot!(ds[:wu][Ti=n]; label="uw", kwargs...)
                   plot!(ds[:wv][Ti=n]; label="vw", xlabel="m²/s²", kwargs...)

        UT_plot = plot(ds[:uT][Ti=n]; label="uT", xlims=(-2, 2), xticks=[-2, 0, 2], kwargs...)
                  plot!(ds[:vT][Ti=n]; label="vT", kwargs...)
                  plot!(ds[:wT][Ti=n]; label="wT", xlabel="m·K/s", kwargs...)

        Ub_plot = plot(ds[:ub][Ti=n]; label="ub", xlims=(-0.003, 0.003), xticks=[-0.003, 0, 0.003], kwargs...)
                  plot!(ds[:vb][Ti=n]; label="vb", kwargs...)
                  plot!(ds[:wb][Ti=n]; label="wb", xlabel="m²/s³", kwargs...)

        cc_plot = plot(ds[:c₀c₀][Ti=n]; label="c₀c₀", xlims=(-0.5, 10), xticks=[0, 10], kwargs...)
                  plot!(ds[:c₁c₁][Ti=n]; label="c₁c₁", kwargs...)
                  plot!(ds[:c₂c₂][Ti=n]; label="c₂c₂", xlabel="c²", kwargs...)

        Uc₀_plot = plot(ds[:uc₀][Ti=n]; label="uc₀", xlims=(-0.08, 0.08), xticks=[-0.08, 0, 0.08], kwargs...)
                   plot!(ds[:vc₀][Ti=n]; label="vc₀", kwargs...)
                   plot!(ds[:wc₀][Ti=n]; label="wc₀", xlabel="m⋅c/s", kwargs...)

        Uc₁_plot = plot(ds[:uc₁][Ti=n]; label="uc₁", xlims=(-0.04, 0.04), xticks=[-0.04, 0, 0.04], kwargs...)
                   plot!(ds[:vc₁][Ti=n]; label="vc₁", kwargs...)
                   plot!(ds[:wc₁][Ti=n]; label="wc₁", xlabel="m⋅c/s", kwargs...)

        Uc₂_plot = plot(ds[:uc₂][Ti=n]; label="uc₂", xlims=(-0.03, 0.03), xticks=[-0.03, 0, 0.03], kwargs...)
                   plot!(ds[:vc₂][Ti=n]; label="vc₂", kwargs...)
                   plot!(ds[:wc₂][Ti=n]; label="wc₂", xlabel="m⋅c/s", kwargs...)

        bc_plot = plot(ds[:bc₀][Ti=n]; label="bc₀", xlims=(-0.02, 0.15), xticks=[0, 0.1], kwargs...)
                  plot!(ds[:bc₁][Ti=n]; label="bc₁", kwargs...)
                  plot!(ds[:bc₂][Ti=n]; label="bc₂", xlabel="m⋅c/s", kwargs..., yticks=-256:64:0)

        νₑ∂zU_plots = plot(ds[:νₑ_∂z_u][Ti=n]; label="νₑ ∂z(u)", xlims=(-1e-5, 4e-5), xticks=[0, 4e-5], kwargs...)
                      plot!(ds[:νₑ_∂z_v][Ti=n]; label="νₑ ∂z(v)", kwargs...)
                      plot!(ds[:νₑ_∂z_w][Ti=n]; label="νₑ ∂z(w)", xlabel="m²/s²", kwargs...)

        κₑ∂zT_plot = plot(ds[:κₑ_∂z_T][Ti=n]; label="κₑ ∂z(T)", xlabel="m⋅°C/s",
                          xlims=(-2e-6, 2e-6), xticks=[-2e-6, 0, 2e-6], kwargs...)

        κₑ∂zC_plots = plot(ds[:κₑ_∂z_c₀][Ti=n]; label="κₑ ∂z(c₀)", xlims=(-6e-5, 6e-5), xticks=[-6e-5, 0, 6e-5], kwargs...)
                      plot!(ds[:κₑ_∂z_c₁][Ti=n]; label="κₑ ∂z(c₁)", kwargs...)
                      plot!(ds[:κₑ_∂z_c₂][Ti=n]; label="κₑ ∂z(c₂)", xlabel="m⋅c/s", kwargs...)

        e_plot = plot(ds[:e][Ti=n]; xlabel="TKE", xaxis=:log, xlims=(1e-9, 1e-2), xticks=[1e-9, 1e-2], kwargs...)

        ϵ_plot = plot(ds[:tke_dissipation][Ti=n]; label="", xlabel="TKE\ndissipation", xaxis=:log,
                      xlims=(1e-15, 1e-6), xticks=[1e-15, 1e-6], kwargs...)

        Uk_plot = plot(ds[:tke_advective_flux][Ti=n]; label="", xlabel="TKE\nadvective flux",
                      xlims=(-6e-7, 1e-7), xticks=[-6e-7, 0], kwargs...)

        bk_plot = plot(ds[:tke_buoyancy_flux][Ti=n]; label="", xlabel="TKE\nbuoyancy flux",
                      xlims=(-1e-8, 1e-8), xticks=[-1e-8, 0, 1e-8], kwargs...)

        pk_plot = plot(ds[:tke_pressure_flux][Ti=n]; label="", xlabel="TKE\npressure flux",
                      xlims=(-3e-7, 3e-7), xticks=[-3e-7, 0, 3e-7], kwargs...)

        sp_plot = plot(ds[:tke_shear_production][Ti=n]; label="", xlabel="TKE\nshear production",
                      xlims=(-1e-7, 1e-6), xticks=[0, 1e-6], kwargs...)

        plot(uv_plot, c_plot, ke_plot, U_Σ_plot, UT_plot, Ub_plot, cc_plot, Uc₀_plot, Uc₁_plot, Uc₂_plot,
             bc_plot, νₑ∂zU_plots, κₑ∂zT_plot, κₑ∂zC_plots, e_plot, ϵ_plot, Uk_plot, bk_plot, pk_plot, sp_plot,
             layout=(2, 10), size=(1600, 1100))
    end

    mp4(anim, joinpath(data_directory, "statistics.mp4"), fps=15)
    gif(anim, joinpath(data_directory, "statistics.gif"), fps=15)
end
