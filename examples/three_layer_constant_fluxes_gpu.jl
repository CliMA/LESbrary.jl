# # Turbulent mixing of a three layer boundary layer driven by constant surface fluxes

# This script runs a simulation of a turbulent oceanic boundary layer with an initial
# three-layer temperature stratification. Turbulent mixing is driven by constant fluxes
# of momentum and heat at the surface.
#
# This script is set up to be configurable on the command line --- a useful property
# when launching multiple jobs at on a cluster.

using ArgParse

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--Nh"
            help = "The number of grid points in x, y."
            default = 128
            arg_type = Int

        "--Nz"
            help = "The number of grid points in z."
            default = 128
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
            default = 32.0
            arg_type = Float64

        "--surface-layer-buoyancy-gradient"
            help = "The buoyancy gradient in the surface layer in units s⁻²."
            default = 1e-7
            arg_type = Float64

        "--thermocline-buoyancy-gradient"
            help = "The buoyancy gradient in the thermocline in units s⁻²."
            default = 1e-5
            arg_type = Float64

        "--deep-buoyancy-gradient"
            help = "The buoyancy gradient below the thermocline in units s⁻²."
            default = 1e-6
            arg_type = Float64

        "--device", "-d"
            help = "The CUDA device index on which to run the simulation."
            default = 1
            arg_type = Int

        "--animation"
            help = "Make an animation of the horizontal and vertical velocity when the simulation completes."
            default = false
            arg_type = Bool

        "--plot-statistics"
            help = "Plot some turbulence statistics after the simulation is complete."
            default = false
            arg_type = Bool
    end

    return parse_args(settings)
end

# # Setup
#
# We start by parsing the arguments received on the command line.

args = parse_command_line_arguments()

# Select GPU device
#
# We use a LESbrary tool to select the GPU device specified on the
# command line.

using LESbrary

#LESbrary.Utils.select_device!(args["device"])

# Domain
#
# We use a three-dimensional domain that's twice as wide as it is deep.
# We choose this aspect ratio so that the horizontal scale is 4x larger
# than the boundary layer depth when the boundary layer penetrates half
# the domain depth.

using Oceananigans

Nh = args["Nh"]
Nz = args["Nz"]

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, 512), y=(0, 512), z=(-256, 0))

# Buoyancy and boundary conditions

using Oceananigans.Buoyancy, Oceananigans.BoundaryConditions

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

u_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

c_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Value, 1.0),
                                       bottom = BoundaryCondition(Value, 0.0))

# Tracer forcing

using Oceananigans.Grids
using Oceananigans.Forcings
using Oceananigans.Utils: hour

const λᶜ = 24.0

@inline c_target(z) = exp(-abs(z) / λᶜ)

@inline c_func(i, j, k, grid, clock, model_fields) =
    @inbounds 1/hour * (c_target(znode(Cell, k, grid)) - model_fields.c[i, j, k])

c_forcing = Forcing(c_func, discrete_form=true)

# # Initial condition and sponge layer

## Fiddle with indices to get a correct discrete profile
k_transition = searchsortedfirst(grid.zC, -surface_layer_depth)
k_deep = searchsortedfirst(grid.zC, -(surface_layer_depth + thermocline_width))

z_transition = grid.zC[k_transition]
z_deep = grid.zC[k_deep]

θ_surface = args["surface-temperature"]
θ_transition = θ_surface + z_transition * dθdz_surface_layer
θ_deep = θ_transition + (z_deep - z_transition) * dθdz_thermocline

gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = Relaxation(rate=1/hour, mask=gaussian_mask)

T_sponge = Relaxation(rate = 1/hour,
                      target = LinearTarget{:z}(intercept = θ_deep - z_deep*dθdz_deep, gradient = dθdz_deep),
                      mask = gaussian_mask)

# LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# Instantiate Oceananigans.IncompressibleModel

using Oceananigans.Advection: WENO5

model = IncompressibleModel(architecture = GPU(),
                             timestepper = :RungeKutta3,
                               advection = WENO5(),
                                    grid = grid,
                                 tracers = (:T, :c),
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                     boundary_conditions = (T=θ_bcs, u=u_bcs),
                                 forcing = (u=u_sponge, v=u_sponge, w=u_sponge, T=T_sponge, c=c_forcing,))

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

set!(model, T = initial_temperature, c = (x, y, z) -> c_target(z))

# # Prepare the simulation

using Oceananigans.Utils: minute
using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.8, Δt=10.0, max_change=1.1, max_Δt=30.0)

simulation = Simulation(model, Δt=wizard, stop_time=16minute, iteration_interval=100,
                        progress=SimulationProgressMessenger(model, wizard))

# Prepare Output

using Printf
using Oceananigans.Utils: GiB
using Oceananigans.OutputWriters: JLD2OutputWriter, FieldSlicer

prefix = @sprintf("three_layer_constant_fluxes_Qu%.1e_Qb%.1e_Nh%d_Nz%d", abs(Qᵘ), Qᵇ, grid.Nx, grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers); 
                                                      time_interval = 4hour, # every quarter period
                                                             prefix = prefix * "_fields",
                                                                dir = data_directory,
                                                       max_filesize = 2GiB,
                                                              force = true)

simulation.output_writers[:slices] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      time_interval = 15minute, # every quarter period
                                                             prefix = prefix * "_slices",
                                                       field_slicer = FieldSlicer(j=floor(Int, grid.Ny/2)),
                                                                dir = data_directory,
                                                       max_filesize = 2GiB,
                                                              force = true)
    
# Horizontally-averaged turbulence statistics
using LESbrary.TurbulenceStatistics: turbulent_kinetic_energy_budget
using LESbrary.TurbulenceStatistics: first_order_statistics, first_through_second_order

# Create scratch space for online calculations
b = BuoyancyField(model)
c_scratch = CellField(model.architecture, model.grid)
u_scratch = XFaceField(model.architecture, model.grid)
v_scratch = YFaceField(model.architecture, model.grid)
w_scratch = ZFaceField(model.architecture, model.grid)

# Build output dictionaries
turbulence_statistics = first_through_second_order(model, c_scratch = c_scratch,
                                                          u_scratch = u_scratch,
                                                          v_scratch = v_scratch,
                                                          w_scratch = w_scratch,
                                                                  b = b)

tke_budget_statistics = turbulent_kinetic_energy_budget(model, c_scratch = c_scratch,
                                                               w_scratch = w_scratch,
                                                                       b = b)

#statistics = merge(turbulence_statistics, tke_budget_statistics)
statistics = turbulence_statistics

simulation.output_writers[:statistics] = JLD2OutputWriter(model, statistics,
                                                          time_averaging_window = 2minute,
                                                                  time_interval = 15minute,
                                                                         prefix = prefix * "_statistics",
                                                                            dir = data_directory,
                                                                          force = true)

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

# # Load and plot turbulence statistics

using JLD2, Plots, Oceananigans.Grids

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

        ## Load 3D fields from file
        w = file["timeseries/w/$iter"][:, 1, :]
        u = file["timeseries/u/$iter"][:, 1, :]
        c = file["timeseries/c/$iter"][:, 1, :]

        wlim = 0.02
        ulim = 0.05

        cmax = maximum(abs, c)
        clim = 0.8 * cmax

        wlevels = nice_divergent_levels(w, wlim)
        ulevels = nice_divergent_levels(u, ulim)
        clevels = vcat(range(0, stop=clim, length=40), [cmax])

        wxz_plot = contourf(xw, zw, w';
                                  color = :balance,
                            aspectratio = :equal,
                                  clims = (-wlim, wlim),
                                 levels = wlevels,
                                  xlims = (0, grid.Lx),
                                  ylims = (-grid.Lz, 0),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

        uxz_plot = contourf(xu, zu, u';
                                  color = :balance,
                            aspectratio = :equal,
                                  clims = (-ulim, ulim),
                                 levels = ulevels,
                                  xlims = (0, grid.Lx),
                                  ylims = (-grid.Lz, 0),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

        cxz_plot = contourf(xu, zu, c';
                                  color = :balance,
                            aspectratio = :equal,
                                  clims = (0, clim),
                                 levels = clevels,
                                  xlims = (0, grid.Lx),
                                  ylims = (-grid.Lz, 0),
                                 xlabel = "x (m)",
                                 ylabel = "z (m)")

        plot(wxz_plot, uxz_plot, cxz_plot, layout=(3, 1), size=(1000, 1000),
             title = ["w(x, y=0, z, t) (m/s)" "u(x, y=0, z, t) (m/s)" "c(x, y = 0, z, t)"])

        iter == iterations[end] && close(file)
    end

    gif(anim, "three_layer_constant_fluxes.gif", fps = 15)
end

if plot_statistics

    ## Some plot parameters
    linewidth = 3
    ylim = (-256, 0)
    plot_size = (500, 500)
    zC = znodes(Cell, grid)
    zF = znodes(Face, grid)

    ## Load data
    #file = jldopen(simulation.output_writers[:statistics].filepath)
    file = jldopen(joinpath(data_directory, prefix * "_statistics.jld2"))

    iterations = parse.(Int, keys(file["timeseries/t"]))
    iter = iterations[end] # plot final iteration

    ## First-order quantities
    u = file["timeseries/u/$iter"][1, 1, :]
    v = file["timeseries/v/$iter"][1, 1, :]
    T = file["timeseries/T/$iter"][1, 1, :]
    c = file["timeseries/c/$iter"][1, 1, :]

    ## Velocity variances
    w²  = file["timeseries/ww/$iter"][1, 1, :]
    tke = file["timeseries/turbulent_kinetic_energy/$iter"][1, 1, :]

    ## Fluxes
    wu = file["timeseries/wu/$iter"][1, 1, :]
    wv = file["timeseries/wv/$iter"][1, 1, :]
    wc = file["timeseries/wc/$iter"][1, 1, :]
    wT = file["timeseries/wT/$iter"][1, 1, :]

    ## Terms in the TKE budget
       buoyancy_flux =   file["timeseries/buoyancy_flux/$iter"][1, 1, :]
    shear_production = - file["timeseries/shear_production/$iter"][1, 1, :]
         dissipation = - file["timeseries/dissipation/$iter"][1, 1, :]
     pressure_flux_divergence = - file["timeseries/pressure_flux_divergence/$iter"][1, 1, :]
    advective_flux_divergence = - file["timeseries/advective_flux_divergence/$iter"][1, 1, :]

    transport = pressure_flux_divergence .+ advective_flux_divergence

    close(file)

    # Plot data

    velocities = plot([u v], zC, size = plot_size,
                         linewidth = linewidth,
                            xlabel = "Velocity (m s⁻¹)",
                            ylabel = "z (m)",
                              ylim = ylim,
                             label = ["u" "v"],
                            legend = :bottom)

    temperature = plot(T, zC, size = plot_size,
                         linewidth = linewidth,
                            xlabel = "Temperature (ᵒC)",
                            ylabel = "z (m)",
                              ylim = ylim,
                             label = nothing)

    tracer = plot(c, zC, size = plot_size,
                         linewidth = linewidth,
                            xlabel = "Tracer",
                            ylabel = "z (m)",
                              ylim = ylim,
                             label = nothing)

    variances = plot(tke, zC, size = plot_size,
                         linewidth = linewidth,
                            xlabel = "Velocity variances (m² s⁻²)",
                            ylabel = "z (m)",
                              ylim = ylim,
                             label = "(u² + v² + w²) / 2",
                            legend = :bottom)

    normalize(wϕ) = wϕ ./ maximum(abs, wϕ)

    fluxes = plot([normalize(wu) normalize(wv) normalize(wc) normalize(wT)], zF,
                              size = plot_size,
                         linewidth = linewidth,
                            xlabel = "Normalized fluxes",
                            ylabel = "z (m)",
                              ylim = ylim,
                             label = ["wu" "wv" "wc" "wT"],
                            legend = :bottom)

    plot!(variances, 1/2 .* w², zF, linewidth = linewidth,
                                        label = "w² / 2")

    budget = plot([buoyancy_flux dissipation transport], zC, size = plot_size,
                  linewidth = linewidth,
                     xlabel = "TKE budget terms",
                     ylabel = "z (m)",
                       ylim = ylim,
                      label = ["buoyancy flux" "dissipation" "kinetic energy transport"],
                     legend = :bottom)

    plot(velocities, temperature, tracer, variances, fluxes, budget, layout=(1, 6), size=(1000, 500))

    savefig("three_layer_constant_fluxes_statistics.png")
end
