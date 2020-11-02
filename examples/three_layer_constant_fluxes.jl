# # Turbulent mixing of a three layer boundary layer driven by constant surface fluxes

# This script runs a simulation of a turbulent oceanic boundary layer with an initial
# three-layer temperature stratification. Turbulent mixing is driven by constant fluxes
# of momentum and heat at the surface.
#
# This script is set up to be configurable on the command line --- a useful property
# when launching multiple jobs at on a cluster.

using Pkg
using ArgParse
using Statistics
using Printf
using JLD2
using Plots

using LESbrary
using Oceananigans
using Oceananigans.Buoyancy
using Oceananigans.BoundaryConditions
using Oceananigans.Grids
using Oceananigans.Forcings

using Oceananigans.Fields: AveragedField
using Oceananigans.Advection: WENO5
using Oceananigans.Utils: minute, hour, GiB, prettytime
using Oceananigans.OutputWriters: JLD2OutputWriter, FieldSlicer

using LESbrary.Utils: SimulationProgressMessenger
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant
using LESbrary.TurbulenceStatistics: first_through_second_order
using LESbrary.TurbulenceStatistics: TurbulentKineticEnergy

# To start, we ensure that all packages in the LESbrary environment are installed:

Pkg.instantiate()

# Next, we parse the command line arguments

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--Nh"
            help = "The number of grid points in x, y."
            default = 32
            arg_type = Int

        "--Nz"
            help = "The number of grid points in z."
            default = 32
            arg_type = Int

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

        "--surface-temperature"
            help = """The temperature at the surface in ᵒC."""
            default = 20.0
            arg_type = Float64

        "--surface-layer-depth"
            help = "The depth of the surface layer in units of m."
            default = 48.0
            arg_type = Float64

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
    end

    return parse_args(settings)
end

# # Setup
#
# We start by parsing the arguments received on the command line, and prescribing
# a few additional output-related arguments.

args = parse_command_line_arguments()

snapshot_time_interval = 10minutes
averages_time_interval = 24hours
averages_time_window = inertial_period
averages_stride = 100

slice_depth = 8.0

# Domain
#
# We use a three-dimensional domain that's twice as wide as it is deep.
# We choose this aspect ratio so that the horizontal scale is 4x larger
# than the boundary layer depth when the boundary layer penetrates half
# the domain depth.

Nh = args["Nh"]
Nz = args["Nz"]

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, 512), y=(0, 512), z=(-256, 0))

# Buoyancy and boundary conditions

Qᵇ = args["buoyancy-flux"]
Qᵘ = args["momentum-flux"]

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

# # Initial condition and sponge layer

## Fiddle with indices to get a correct discrete profile
k_transition = searchsortedfirst(grid.zC, -surface_layer_depth)
k_deep = searchsortedfirst(grid.zC, -(surface_layer_depth + thermocline_width))

z_transition = grid.zC[k_transition]
z_deep = grid.zC[k_deep]

θ_surface = args["surface-temperature"]
θ_transition = θ_surface + z_transition * dθdz_surface_layer
θ_deep = θ_transition + (z_deep - z_transition) * dθdz_thermocline

@inline passive_tracer_forcing(x, y, z, t, p) =
    1 / p.τ_source * exp((z - p.z₀)/ p.λ) - 1 / p.τ_damping
 
c_forcing  = Forcing(passive_tracer_forcing, parameters=(z₀=  0.0, λ=24.0, τ_source=12hour, τ_damping=24hour))
d_forcing  = Forcing(passive_tracer_forcing, parameters=(z₀=-96.0, λ=24.0, τ_source=12hour, τ_damping=24hour))

# Sponge layer for u, v, w, and T
gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=4/hour, mask=gaussian_mask)

T_sponge = Relaxation(rate = 4/hour,
                      target = LinearTarget{:z}(intercept = θ_deep - z_deep*dθdz_deep, gradient = dθdz_deep),
                      mask = gaussian_mask)

# # LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# # Instantiate Oceananigans.IncompressibleModel

model = IncompressibleModel(architecture = GPU(),
                             timestepper = :RungeKutta3,
                               advection = WENO5(),
                                    grid = grid,
                                 tracers = (:T, :c, :d),
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(),
                     boundary_conditions = (T=θ_bcs, u=u_bcs),
                                 forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge,
                                            c=c_forcing, d=d_forcing)
                                )

# # Set Initial condition

## Noise with 8 m decay scale
Ξ(z) = rand() * exp(z / 8)
                    
"""
    initial_temperature(x, y, z)

Returns a three-layer initial temperature distribution. The average temperature varies in z
and is augmented by three-dimensional, surface-concentrated random noise.
"""
function initial_temperature(x, y, z)

    noise = 1e-6 * Ξ(z) * dθdz_surface_layer * grid.Lz

    if z > z_transition
        return θ_surface + dθdz_surface_layer * z + noise

    elseif z > z_deep
        return θ_transition + dθdz_thermocline * (z - z_transition) + noise

    else
        return θ_deep + dθdz_deep * (z - z_deep) + noise

    end
end

set!(model, T = initial_temperature)

# # Prepare the simulation

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.8, Δt=1.0, max_change=1.1, max_Δt=30.0)

stop_time = args["hours"] * hour

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, iteration_interval=10,
                        progress=SimulationProgressMessenger(model, wizard))

# # Prepare Output

pickup = args["pickup"]
force = pickup ? false : true

prefix = @sprintf("three_layer_constant_fluxes_Qu%.1e_Qb%.1e_Nh%d_Nz%d", abs(Qᵘ), Qᵇ, grid.Nx, grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

simulation.output_writers[:checkpointer] =
    Checkpointer(model, schedule = TimeInterval(stop_time/3), prefix = prefix * "_checkpointer", dir = data_directory)

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# Prepare turbulence statistics
k_xy_slice = searchsortedfirst(grid.zF[:], -slice_depth)

b = BuoyancyField(model)
p = PressureField(model)
w_scratch = ZFaceField(model.architecture, model.grid)
c_scratch = CellField(model.architecture, model.grid)

primitive_statistics = first_through_second_order(model, b=b, p=p, w_scratch=w_scratch, c_scratch=c_scratch)

U = primitive_statistics[:u]
V = primitive_statistics[:v]

e = TurbulentKineticEnergy(model, U=U, V=V)
shear_production = ShearProduction(model, data=c_scratch.data, U=U, V=V)
dissipation = ViscousDissipation(model, data=c_scratch.data)

tke_budget_statistics = turbulent_kinetic_energy_budget(model, b=b, p=p, U=U, V=V, e=e,
                                                        shear_production=shear_production, dissipation=dissipation)

fields_to_output = merge(model.velocities, model.tracers, (e=e, ϵ=dissipation))

statistics_to_output = merge(primitive_statistics, tke_budget_statistics)

simulation.output_writers[:xz] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_xz",
                     field_slicer = FieldSlicer(j=1),
                              dir = data_directory,
                            force = force)

simulation.output_writers[:yz] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_yz",
                     field_slicer = FieldSlicer(i=1),
                              dir = data_directory,
                            force = force)

simulation.output_writers[:xy] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_xy",
                     field_slicer = FieldSlicer(k=k_xy_slice),
                              dir = data_directory,
                            force = force)

simulation.output_writers[:statistics] =
    JLD2OutputWriter(model, statistics_to_output,
                     schedule = TimeInterval(snapshot_time_interval),
                       prefix = prefix * "_statistics",
                          dir = data_directory,
                        force = force)

simulation.output_writers[:averaged_statistics] =
    JLD2OutputWriter(model, statistics_to_output,
                     schedule = AveragedTimeInterval(averages_time_interval,
                                                     window = averages_time_window,
                                                     stride = averages_stride),
                       prefix = prefix * "_averaged_statistics",
                          dir = data_directory,
                        force = force)

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

# # Load and plot turbulence statistics

pyplot()

make_animation = args["animation"]
plot_statistics = args["plot-statistics"]

if make_animation

    xw, yw, zw = nodes(model.velocities.w)
    xu, yu, zu = nodes(model.velocities.u)
    xc, yc, zc = nodes(model.tracers.c)

    xw, yw, zw = xw[:], yw[:], zw[:]
    xu, yu, zu = xu[:], yu[:], zu[:]
    xc, yc, zc = xc[:], yc[:], zc[:]

    #file = jldopen(simulation.output_writers[:slices].filepath)
    file = jldopen(joinpath(data_directory, prefix * "_slices.jld2"))
    statistics_file = jldopen(joinpath(data_directory, prefix * "_statistics.jld2"))

    iterations = parse.(Int, keys(file["timeseries/t"]))

    # This utility is handy for calculating nice contour intervals:

    function nice_divergent_levels(c, clim)
        levels = range(-clim, stop=clim, length=40)

        cmax = maximum(abs, c)

        if clim < cmax # add levels on either end
            levels = vcat([-cmax], range(-clim, stop=clim, length=40), [cmax])
        end

        return levels
    end

    # Finally, we're ready to animate.

    @info "Making an animation from the saved data..."

    anim = @animate for (i, iter) in enumerate(iterations)

        @info "Drawing frame $i from iteration $iter \n"

        t = file["timeseries/t/$iter"]

        ## Load 3D fields from file
        w = file["timeseries/w/$iter"][:, 1, :]
        u = file["timeseries/u/$iter"][:, 1, :]
        v = file["timeseries/v/$iter"][:, 1, :]
        c = file["timeseries/c/$iter"][:, 1, :]

        U = statistics_file["timeseries/u/$iter"][1, 1, :]
        V = statistics_file["timeseries/v/$iter"][1, 1, :]
        E = statistics_file["timeseries/tke/$iter"][1, 1, :]
        T = statistics_file["timeseries/T/$iter"][1, 1, :]
        C = statistics_file["timeseries/c/$iter"][1, 1, :]
        D = statistics_file["timeseries/d/$iter"][1, 1, :]

        wlim = 0.02
        clim = 0.8

        cmax = maximum(abs, c)

        wlevels = nice_divergent_levels(w, wlim)

        clevels = cmax > clim ? vcat(range(0, stop=clim, length=40), [cmax]) :
                                     range(0, stop=clim, length=40)

        T_plot = plot(T, zc, label="T", xlim=(initial_temperature(0, 0, -grid.Lz), θ_surface), legend=:bottom)

        U_plot = plot([U, V, sqrt.(E)], zc, label=["u" "v" "√E"], linewidth=[1 1 2], legend=:bottom)

        C_plot = plot([C D], zc,
                      label = ["C" "D"],
                      legend=:bottom,
                       xlim = (0, 1))

        wxz_plot = contourf(xw, zw, w';
                                  color = :balance,
                            aspectratio = :equal,
                                  clims = (-wlim, wlim),
                                 levels = wlevels,
                                  xlims = (0, grid.Lx),
                                  ylims = (-grid.Lz, 0),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

        cxz_plot = contourf(xc, zc, c';
                                  color = :thermal,
                            aspectratio = :equal,
                                  clims = (0, clim),
                                 levels = clevels,
                                  xlims = (0, grid.Lx),
                                  ylims = (-grid.Lz, 0),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

        dxz_plot = contourf(xc, zc, d';
                                  color = :thermal,
                            aspectratio = :equal,
                                  clims = (0, clim),
                                 levels = clevels,
                                  xlims = (0, grid.Lx),
                                  ylims = (-grid.Lz, 0),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

        w_title = @sprintf("w(x, y=0, z, t=%s) (m/s)", prettytime(t))
        T_title = "T"
        c_title = @sprintf("c(x, y=0, z, t=%s)", prettytime(t))
        U_title = "U and V"
        d_title = @sprintf("d(x, y=0, z, t=%s)", prettytime(t))
        C_title = "C and D"

        plot(wxz_plot, T_plot, cxz_plot, U_plot, dxz_plot, C_plot, layout=(3, 2),
             size = (1000, 1000),
             link = :y,
             title = [w_title T_title c_title U_title d_title C_title])

        iter == iterations[end] && close(file)
    end

    gif(anim, prefix * ".gif", fps = 8)
end
