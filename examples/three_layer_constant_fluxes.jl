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
using Oceananigans.Utils: minute, hour, GiB
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
            default = 32.0
            arg_type = Float64

        "--thermocline-width"
            help = "The width of the thermocline in units of m."
            default = 32.0
            arg_type = Float64

        "--surface-layer-buoyancy-gradient"
            help = "The buoyancy gradient in the surface layer in units s⁻²."
            default = 1e-6
            arg_type = Float64

        "--thermocline-buoyancy-gradient"
            help = "The buoyancy gradient in the thermocline in units s⁻²."
            default = 1e-5
            arg_type = Float64

        "--deep-buoyancy-gradient"
            help = "The buoyancy gradient below the thermocline in units s⁻²."
            default = 1e-6
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
            default = false
            arg_type = Bool
    end

    return parse_args(settings)
end

# # Setup
#
# We start by parsing the arguments received on the command line.

args = parse_command_line_arguments()

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

u_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

c_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Value, 1.0),
                                       bottom = BoundaryCondition(Value, 0.0))

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

const λᶜ = 24.0

#=
# Note: ContinuousForcing doesn't currently work on the GPU on Oceananigans master,
# so we resort to the "old" style of implementing a sponge layer with tons of const's
# and hand-written sponge forcing funcs.

const z_mask = -grid.Lz
const h_mask = grid.Lz / 10
const μ₀ = 1/hour

const θ̃ = θ_deep - z_deep * dθdz_deep
const θ̃z = dθdz_deep

""" Gaussian mask + damping at the rate μ₀. """
@inline μ(z) = μ₀ * exp(-(z - z_mask)^2 / (2 * h_mask^2))

@inline c_target(z) = exp(-abs(z) / λᶜ)

""" Relax the tracer `c` back to an exponential profile with decay rate λᶜ. """
@inline c_func(i, j, k, grid, clock, model_fields) =
    @inbounds 1/hour * (c_target(znode(Cell, k, grid)) - model_fields.c[i, j, k])

c_forcing = Forcing(c_func, discrete_form=true)

T_sponge_func(i, j, k, grid, clock, model_fields) = @inbounds   μ(grid.zC[k]) * (θ̃ + θ̃z * grid.zC[k] - model_fields.T[i, j, k])
u_sponge_func(i, j, k, grid, clock, model_fields) = @inbounds - μ(grid.zC[k]) * model_fields.u[i, j, k]
v_sponge_func(i, j, k, grid, clock, model_fields) = @inbounds - μ(grid.zC[k]) * model_fields.v[i, j, k]
w_sponge_func(i, j, k, grid, clock, model_fields) = @inbounds - μ(grid.zF[k]) * model_fields.w[i, j, k]
    
T_sponge = Forcing(T_sponge_func, discrete_form=true)
u_sponge = Forcing(u_sponge_func, discrete_form=true)
v_sponge = Forcing(v_sponge_func, discrete_form=true)
w_sponge = Forcing(w_sponge_func, discrete_form=true)
=#

# How we would like to implement the sponge layer and tracer forcing.
@inline c_target(x, y, z, t) = exp(-abs(z) / λᶜ)

c_forcing = Relaxation(rate = 1/hour,
                       target = c_target)

gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=1/hour, mask=gaussian_mask)

T_sponge = Relaxation(rate = 1/hour,
                      target = LinearTarget{:z}(intercept = θ_deep - z_deep*dθdz_deep, gradient = dθdz_deep),
                      mask = gaussian_mask)

# LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# Instantiate Oceananigans.IncompressibleModel

model = IncompressibleModel(architecture = GPU(),
                             timestepper = :RungeKutta3,
                               advection = WENO5(),
                                    grid = grid,
                                 tracers = (:T, :c),
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                     boundary_conditions = (T=θ_bcs, u=u_bcs),
                                 forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge, c=c_forcing))

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

set!(model, T = initial_temperature, c = (x, y, z) -> c_target(x, y, z, 0))

# # Prepare the simulation

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.8, Δt=1.0, max_change=1.1, max_Δt=30.0)

stop_hours = args["hours"]

simulation = Simulation(model, Δt=wizard, stop_time=stop_hours * hour, iteration_interval=100,
                        progress=SimulationProgressMessenger(model, wizard))

# Prepare Output

prefix = @sprintf("three_layer_constant_fluxes_Qu%.1e_Qb%.1e_Nh%d_Nz%d", abs(Qᵘ), Qᵇ, grid.Nx, grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers); 
                                                      time_interval = 24hour, # every quarter period
                                                             prefix = prefix * "_fields",
                                                                dir = data_directory,
                                                       max_filesize = 2GiB,
                                                              force = true)

simulation.output_writers[:slices] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      time_interval = 5minute, # every quarter period
                                                             prefix = prefix * "_slices",
                                                       field_slicer = FieldSlicer(j=floor(Int, grid.Ny/2)),
                                                                dir = data_directory,
                                                              force = true)
    
# Horizontally-averaged turbulence statistics

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

#=
using LESbrary.TurbulenceStatistics: turbulent_kinetic_energy_budget

tke_budget_statistics = turbulent_kinetic_energy_budget(model, c_scratch = c_scratch,
                                                               u_scratch = u_scratch,
                                                               v_scratch = v_scratch,
                                                               w_scratch = w_scratch,
                                                                       b = b)

turbulence_statistics = merge(turbulence_statistics, tke_budget_statistics)

usq = turbulence_statistics[:turbulent_u_variance].operand

using Oceananigans.Utils: work_layout
using Oceananigans.Fields: _compute!
using Oceananigans.Architectures: device
using KernelAbstractions: @ka_code_typed

workgroup, worksize = work_layout(grid, :xyz, include_right_boundaries=true, location=(Cell, Cell, Cell))
compute_kernel! = _compute!(device(GPU()), workgroup, worksize)
@ka_code_typed compute_kernel!(usq.data, usq.operand; dependencies=Event(device(GPU())))
=#

turbulent_kinetic_energy = TurbulentKineticEnergy(model,
                                                  data = c_scratch.data,
                                                  U = turbulence_statistics[:u],
                                                  V = turbulence_statistics[:v])

turbulence_statistics[:tke] = AveragedField(turbulent_kinetic_energy, dims=(1, 2))

simulation.output_writers[:statistics] = JLD2OutputWriter(model, turbulence_statistics,
                                                          time_interval = 5minute,
                                                                 prefix = prefix * "_statistics",
                                                                    dir = data_directory,
                                                                  force = true)

simulation.output_writers[:averaged_statistics] = JLD2OutputWriter(model, turbulence_statistics,
                                                                   time_averaging_window = 15minute,
                                                                           time_interval = 1hour,
                                                                                  prefix = prefix * "_averaged_statistics",
                                                                                     dir = data_directory,
                                                                                   force = true)

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

        ## Load 3D fields from file
        w = file["timeseries/w/$iter"][:, 1, :]
        u = file["timeseries/u/$iter"][:, 1, :]
        v = file["timeseries/v/$iter"][:, 1, :]
        c = file["timeseries/c/$iter"][:, 1, :]

        U = statistics_file["timeseries/u/$iter"][1, 1, :]
        V = statistics_file["timeseries/v/$iter"][1, 1, :]
        T = statistics_file["timeseries/T/$iter"][1, 1, :]
        C = statistics_file["timeseries/c/$iter"][1, 1, :]

        wlim = 0.02
        ulim = 0.05
        clim = 0.8

        cmax = maximum(abs, c)

        wlevels = nice_divergent_levels(w, wlim)
        ulevels = nice_divergent_levels(u, ulim)

        clevels = cmax > clim ? vcat(range(0, stop=clim, length=40), [cmax]) :
                                     range(0, stop=clim, length=40)

        T_plot = plot(T, zc, label="T", xlim=(initial_temperature(0, 0, -grid.Lz), θ_surface))
        U_plot = plot([U, V], zc, label=["u" "v"])
        C_plot = plot(C, zc, label="C", xlim=(0, 1))

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

        plot(wxz_plot, T_plot, uxz_plot, U_plot, cxz_plot, C_plot, layout=(3, 2),
             size = (1000, 1000),
             link = :y,
             title = ["w(x, y=0, z, t) (m/s)" "T" "u(x, y=0, z, t) (m/s)" "U and V" "c(x, y = 0, z, t)" "C"])

        iter == iterations[end] && close(file)
    end

    gif(anim, prefix * ".gif", fps = 8)
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

    savefig(prefix * "_statistics.png")
end
