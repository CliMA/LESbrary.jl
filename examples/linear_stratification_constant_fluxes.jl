# # Turbulent mixing of a linearly stratified boundary layer driven by constant surface fluxes

# This script runs a simulation of a turbulent oceanic boundary layer with an initially
# linear temperature stratification. Turbulent mixing is driven by constant fluxes
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
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations

using Oceananigans.Fields: AveragedField, PressureField
using Oceananigans.Advection: WENO5
using Oceananigans.Utils: minute, hour, GiB, prettytime

using LESbrary.Utils: SimulationProgressMessenger
using LESbrary.TurbulenceStatistics: first_through_second_order
using LESbrary.TurbulenceStatistics: turbulent_kinetic_energy_budget
using LESbrary.TurbulenceStatistics: TurbulentKineticEnergy
using LESbrary.TurbulenceStatistics: ShearProduction

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

        "--buoyancy-gradient"
            help = "The initial buoyancy gradient in units s⁻²."
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

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, 256), y=(0, 256), z=(-128, 0))

# Buoyancy and boundary conditions

Qᵘ = args["momentum-flux"]
Qᵇ = args["buoyancy-flux"]
N² = args["buoyancy-gradient"]

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

dθdz = N² / (α * g)
Qᶿ = Qᵇ / (α * g)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

# Tracer forcing

# # Initial condition and sponge layer

@inline c_forcing_func(x, y, z, t, p) = 1/p.growth * exp(z / p.scale) - p.decay
@inline d_forcing_func(x, y, z, t, p) = 1/p.growth * exp(-(z + 96.0) / p.scale) - p.decay

c_forcing = Forcing(c_forcing_func, parameters=(growth=1hour, decay=12hour, scale=24.0))
d_forcing = Forcing(d_forcing_func, parameters=(growth=1hour, decay=12hour, scale=24.0))

# Sponge layer for u, v, w, and T
gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=1/hour, mask=gaussian_mask)

θ_surface = args["surface-temperature"]

T_sponge = Relaxation(rate = 1/hour,
                      target = LinearTarget{:z}(intercept=θ_surface, gradient=dθdz),
                      mask = gaussian_mask)

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

initial_temperature(x, y, z) = θ_surface + dθdz * z + 1e-6 * Ξ(z) * dθdz * grid.Lz
                   
set!(model, T = initial_temperature)

# # Prepare the simulation

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.8, Δt=1.0, max_change=1.1, max_Δt=30.0)

stop_hours = args["hours"]

simulation = Simulation(model, Δt=wizard, stop_time=stop_hours * hour, iteration_interval=100,
                        progress=SimulationProgressMessenger(model, wizard))

# Prepare Output

prefix = @sprintf("simple_constant_fluxes_Qu%.1e_Qb%.1e_Nh%d_Nz%d", abs(Qᵘ), Qᵇ, grid.Nx, grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

slice_interval = 15minute

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers); 
                                                           schedule = TimeInterval(12hour),
                                                             prefix = prefix * "_fields",
                                                                dir = data_directory,
                                                       max_filesize = 2GiB,
                                                              force = true)

simulation.output_writers[:slices] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                           schedule = TimeInterval(slice_interval),
                                                             prefix = prefix * "_slices",
                                                       field_slicer = FieldSlicer(j=floor(Int, grid.Ny/2)),
                                                                dir = data_directory,
                                                              force = true)
# Horizontally-averaged turbulence statistics

# Create scratch space for online calculations
b = BuoyancyField(model)
c_scratch = CenterField(model.architecture, model.grid)
u_scratch = XFaceField(model.architecture, model.grid)
v_scratch = YFaceField(model.architecture, model.grid)
w_scratch = ZFaceField(model.architecture, model.grid)

# Build output dictionaries
turbulence_statistics = first_through_second_order(model, c_scratch = c_scratch,
                                                          u_scratch = u_scratch,
                                                          v_scratch = v_scratch,
                                                          w_scratch = w_scratch,
                                                                  b = b)

tke_budget_statistics = turbulent_kinetic_energy_budget(model;
                                                        b = b,
                                                        w_scratch = w_scratch,
                                                        c_scratch = c_scratch,
                                                        U = turbulence_statistics[:u],
                                                        V = turbulence_statistics[:v])

statistics = merge(turbulence_statistics, tke_budget_statistics) 

simulation.output_writers[:statistics] = JLD2OutputWriter(model, statistics,
                                                          schedule = TimeInterval(slice_interval),
                                                            prefix = prefix * "_statistics",
                                                               dir = data_directory,
                                                             force = true)

simulation.output_writers[:averaged_statistics] = JLD2OutputWriter(model, statistics,
                                                                   schedule = AveragedTimeInterval(3hour, window=30minute),
                                                                     prefix = prefix * "_averaged_statistics",
                                                                        dir = data_directory,
                                                                      force = true)

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

# # Load and plot turbulence statistics

pyplot()

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)
xc, yc, zc = nodes(model.tracers.c)

xw, yw, zw = xw[:], yw[:], zw[:]
xu, yu, zu = xu[:], yu[:], zu[:]
xc, yc, zc = xc[:], yc[:], zc[:]

file = jldopen(joinpath(data_directory, prefix * "_slices.jld2"))
statistics_file = jldopen(joinpath(data_directory, prefix * "_statistics.jld2"))

@show keys(statistics_file["timeseries"])

iterations = parse.(Int, keys(file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function divergent_levels(c, clim, nlevels=31)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
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
    d = file["timeseries/d/$iter"][:, 1, :]

    U = statistics_file["timeseries/u/$iter"][1, 1, :]
    V = statistics_file["timeseries/v/$iter"][1, 1, :]
    E = statistics_file["timeseries/e/$iter"][1, 1, :]
    T = statistics_file["timeseries/T/$iter"][1, 1, :]
    C  = statistics_file["timeseries/c/$iter"][1, 1, :]
    D  = statistics_file["timeseries/d/$iter"][1, 1, :]

    cmax = maximum(abs, c) + 1e-9
    clim = 0.8 * cmax

    clevels = cmax > clim ? vcat(range(0, stop=clim, length=40), [cmax]) :
                                 range(0, stop=clim, length=40)

    wlim = 0.02
    ulim = 0.8 * maximum(abs, u) + 1e-9

    wlevels = divergent_levels(w, wlim)
    ulevels = divergent_levels(u, ulim)

    T_plot = plot(T, zc, label="T", legend=:bottom)

    U_plot = plot([U, V, sqrt.(E)], zc, label=["u" "v" "√E"], linewidth=[1 1 2], legend=:bottom)

    C_plot = plot([C D], zc,
                  label = ["C" "D"],
                  legend=:bottom)

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

    cxz_plot = contourf(xc, zc, c';
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
    u_title = @sprintf("u(x, y=0, z, t=%s)", prettytime(t))
    U_title = "U and V"
    c_title = @sprintf("c(x, y=0, z, t=%s)", prettytime(t))
    C_title = "C's and D's"

    plot(wxz_plot, T_plot, uxz_plot, U_plot, cxz_plot, C_plot, layout=(3, 2),
         size = (1000, 1000),
         link = :y,
         title = [w_title T_title u_title U_title c_title C_title])

    if iter == iterations[end]
        close(file)
        close(statistics_file)
    end
end

gif(anim, prefix * ".gif", fps = 8)
