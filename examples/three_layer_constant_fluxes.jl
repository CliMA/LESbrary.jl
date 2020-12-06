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
using Oceananigans.Grids
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
            default = true
            arg_type = Bool

        "--plot-statistics"
            help = "Plot some turbulence statistics after the simulation is complete."
            default = true
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

prefix = @sprintf("three_layer_constant_fluxes_hr%d_Qu%.1e_Qb%.1e_f%.1e_Nh%d_Nz%d_", stop_hours, abs(Qᵘ), Qᵇ, f, grid.Nx, grid.Nz)*name
data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

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

thermocline_type = args["thermocline"]

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

function init_save_some_metadata!(file, model)
    file["parameters/coriolis_parameter"] = f
    file["parameters/density"] = 1027.0
    file["boundary_conditions/θ_top"] = Qᶿ
    file["boundary_conditions/θ_bottom"] = dθdz_deep
    file["boundary_conditions/u_top"] = Qᵘ
    file["boundary_conditions/u_bottom"] = 0.0
    return nothing
end

@info "Garnishing output writers..."

simulation.output_writers[:xz] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_xz",
                     field_slicer = FieldSlicer(j=1),
                              dir = data_directory,
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:yz] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_yz",
                     field_slicer = FieldSlicer(i=1),
                              dir = data_directory,
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:xy] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_xy",
                     field_slicer = FieldSlicer(k=k_xy_slice),
                              dir = data_directory,
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:statistics] =
    JLD2OutputWriter(model, statistics_to_output,
                     schedule = TimeInterval(snapshot_time_interval),
                       prefix = prefix * "_statistics",
                          dir = data_directory,
                        force = force,
                         init = init_save_some_metadata!)

simulation.output_writers[:averaged_statistics] =
    JLD2OutputWriter(model, statistics_to_output,
                     schedule = AveragedTimeInterval(averages_time_interval,
                                                     window = averages_time_window),
                       prefix = prefix * "_averaged_statistics",
                          dir = data_directory,
                        force = force,
                         init = init_save_some_metadata!)

# # Run

@info "Reticulating splines..."

run!(simulation)

# # Load and plot turbulence statistics

make_animation = args["animation"]
plot_statistics = args["plot-statistics"]

if make_animation
    pyplot()

    xw, yw, zw = nodes(model.velocities.w)
    xc, yc, zc = nodes(model.tracers.c₀)

    file = jldopen(joinpath(data_directory, prefix * "_xz.jld2"))
    statistics_file = jldopen(joinpath(data_directory, prefix * "_statistics.jld2"))

    iterations = parse.(Int, keys(file["timeseries/t"]))

    U, V, E, T, C₀, C₁, C₂ = [zeros(length(iterations), grid.Nz) for i = 1:7]

    for (i, iter) in enumerate(iterations)
        U[i, :] .= statistics_file["timeseries/u/$iter"][1, 1, :]
        V[i, :] .= statistics_file["timeseries/v/$iter"][1, 1, :]
        E[i, :] .= statistics_file["timeseries/e/$iter"][1, 1, :]
        T[i, :] .= statistics_file["timeseries/T/$iter"][1, 1, :]
        C₀[i, :] .= statistics_file["timeseries/c₀/$iter"][1, 1, :]
        C₁[i, :] .= statistics_file["timeseries/c₁/$iter"][1, 1, :]
        C₂[i, :] .= statistics_file["timeseries/c₂/$iter"][1, 1, :]
    end

    c₀min = minimum(C₀) - 1e-9
    c₀max = maximum(C₀) + 1e-9
    c₁min = minimum(C₁) - 1e-9
    c₁max = maximum(C₁) + 1e-9
    c₂min = minimum(C₂) - 1e-9
    c₂max = maximum(C₂) + 1e-9

    umax = max(
               maximum(abs, U),
               maximum(abs, V),
               maximum(abs, sqrt.(E))
              ) + 1e-9

    # Set wlim based on maximum across all time steps
    wlim = maximum([maximum(abs, file["timeseries/w/$(iterations[i])"]) for i=1:length(iterations)])

    # Finally, we're ready to animate.

    @info "Making an animation from the saved data..."

    anim = @animate for (i, iter) in enumerate(iterations)

        @info "Drawing frame $i from iteration $iter \n"

        t = file["timeseries/t/$iter"]

        w = file["timeseries/w/$iter"][:, 1, :]
        u = file["timeseries/u/$iter"][:, 1, :]
        v = file["timeseries/v/$iter"][:, 1, :]
        c₀ = file["timeseries/c₀/$iter"][:, 1, :]
        c₁ = file["timeseries/c₁/$iter"][:, 1, :]

        wlim = 2 * umax
        wlevels = range(-wlim, stop=wlim, length=41)
        c₀levels = range(c₀min, stop=c₀max, length=40)
        c₁levels = range(c₁min, stop=c₁max, length=40)

        T_plot = plot(T[i, :], zc, label=nothing, xlim=(initial_temperature(0, 0, -grid.Lz), θ_surface),
                      xlabel = "T (ᵒC)", ylabel = "z (m)")

        U_plot = plot([U[i, :] V[i, :] sqrt.(E[i, :])], zc,
                      label=["u" "v" "√E"], linewidth=[1 1 2], xlim=(-umax, umax),
                      legend=:bottomleft,
                      xlabel = "Velocities (m s⁻¹)", ylabel = "z (m)")

        C_plot = plot([C₀[i, :] C₁[i, :] C₂[i, :]], zc, label = ["C₀" "C₁" "C₂"], legend=:bottom,
                      xlabel = "Tracers", ylabel = "z (m)")

        wxz_plot = contourf(xw, zw, clamp.(w, -wlim, wlim)';
                                  color = :balance,
                            aspectratio = :equal,
                                  clims = (-wlim, wlim),
                                 levels = wlevels,
                                  xlims = (0, grid.Lx),
                                  ylims = (-grid.Lz, 0),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

        c₀xz_plot = contourf(xc, zc, clamp.(c₀, c₀min, c₀max)';
                                   color = :thermal,
                             aspectratio = :equal,
                                  levels = c₀levels,
                                   clims = (c₀min, c₀max),
                                   xlims = (0, grid.Lx),
                                   ylims = (-grid.Lz, 0),
                                  xlabel = "x (m)",
                                  ylabel = "z (m)")

        c₁xz_plot = contourf(xc, zc, clamp.(c₁, c₁min, c₁max)';
                                   color = :thermal,
                             aspectratio = :equal,
                                  levels = c₁levels,
                                   clims = (c₁min, c₁max),
                                   xlims = (0, grid.Lx),
                                   ylims = (-grid.Lz, 0),
                                  xlabel = "x (m)",
                                  ylabel = "z (m)")

        w_title = @sprintf("w(x, y=0, z, t=%s) (m/s)", prettytime(t))
        T_title = ""
        c₀_title = @sprintf("c₀(x, y=0, z, t=%s)", prettytime(t))
        U_title = ""
        c₁_title = @sprintf("c₁(x, y=0, z, t=%s)", prettytime(t))
        C_title = ""

        plot(wxz_plot, T_plot, c₀xz_plot, U_plot, c₁xz_plot, C_plot,
             layout = Plots.grid(3, 2, widths=(0.7, 0.3)),
             size = (1000, 1000),
             link = :y,
             title = [w_title T_title c₀_title U_title c₁_title C_title])

        iter == iterations[end] && close(file)
    end

    gif(anim, prefix * ".gif", fps = 8)
end
