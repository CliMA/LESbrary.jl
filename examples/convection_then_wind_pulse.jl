# # Ocean convection, followed by a wind pulse

using Oceananigans, LESbrary, Printf

# ## Model set-up
#
# ### Domain specification and Grid construction
#
# We create a grid with modest resolution.

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(64, 64, 64), extent=(512, 512, 256))

# ### Boundary conditions
#
# First, we impose the linear startification

N² = 5e-6 # s⁻²

# Next we impose a cooling with buoyancy flux

const cooling_flux = 5e-7 # m² s⁻³

# which corresponds to an upward heat flux of ≈ 1000 W m⁻².
# We cool just long enough to deepen the boundary layer to 100 m.

target_depth = 100

const tᶜ = target_depth^2 * N² / (3 * cooling_flux) # the 3 is empirical

@inline Qᵇ(x, y, t) = ifelse(t < tᶜ, cooling_flux, 0.0)

using Oceananigans.Utils: hour

@info @sprintf("Convection time: %.2f hours", tᶜ / hour)

# A bit after convection has done its work, we impose a strong pulse of wind
# for 8 hours,

const τᵐᵃˣ = 1e-3 # m² s⁻²
const tᵖ = 1.2tᶜ
const Tᵖ = 12hour

@inline wind_pulse(t, T, τ) = - τ * t / T * exp( - t^2 / (2 * T^2))

@inline Qᵘ(x, y, t) = ifelse(t < tᵖ, 0.0, wind_pulse(t - tᵖ, Tᵖ, τᵐᵃˣ))
                             

# Oceananigans uses "positive upward" conventions for all fluxes. In consequence,
# a negative flux at the surface drives positive velocities, and a positive flux of
# buoyancy drives cooling.

# To summarize,

using Oceananigans.BoundaryConditions

u_boundary_conditions = UVelocityBoundaryConditions(grid, top = UVelocityBoundaryCondition(Flux, :z, Qᵘ))

# and a surface flux and bottom linear gradient on buoyancy, $b$,

b_boundary_conditions = TracerBoundaryConditions(grid, top = TracerBoundaryCondition(Flux, :z, Qᵇ),
                                                       bottom = BoundaryCondition(Gradient, N²))
nothing # hide

# ### Coriolis parameter

f = 1e-4 # s⁻¹

# ## Model instantiation
#
# Finally, we are ready to build the model. We use the AnisotropicMinimumDissipation
# model for large eddy simulation. Because our Stokes drift does not vary in $x, y$,
# we use `UniformStokesDrift`, which expects Stokes drift functions of $z, t$ only.

using Oceananigans.Buoyancy: BuoyancyTracer 

model = IncompressibleModel(        architecture = CPU(),
                                            grid = grid,
                                         tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = FPlane(f=f),
                                         closure = AnisotropicMinimumDissipation(),
                             boundary_conditions = (u=u_boundary_conditions, 
                                                    b=b_boundary_conditions),
                            )

# ## Initial conditions
#
# Our initial condition for buoyancy consists of a linear stratification, plus noise,

Ξ(z) = randn() * exp(z / 8)

bᵢ(x, y, z) = N² * z + 1e-3 * Ξ(z) * N² * model.grid.Lz

set!(model, b=bᵢ)

# ## Setting up the simulation
#
# We use the `TimeStepWizard` for adaptive time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2,

wizard = TimeStepWizard(cfl=0.2, Δt=5.0, max_change=1.1, max_Δt=20.0)
nothing # hide

# ### Nice progress messaging
#
# We define a function that prints a helpful message with
# maximum absolute value of $u, v, w$ and the current wall clock time.

using Oceananigans.Diagnostics, Printf

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   umax(), vmax(), wmax(),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )       

    @info msg

    return nothing
end

# Now we create the simulation,

using Oceananigans.Utils: hour # correpsonds to "1 hour", in units of seconds

simulation = Simulation(model, progress_frequency = 100,
                                               Δt = wizard,
                                        stop_time = 36hour,
                                         progress = print_progress)
                        
# ## Output
#
# We set up an output writer for the simulation that saves all velocity fields, 
# tracer fields, and the subgrid turbulent diffusivity every 2 minutes.

using Oceananigans.OutputWriters
using Oceananigans.Utils: minute

prefix = "convection_then_wind_pulse"

fields = merge(model.velocities, model.tracers)

fields_writer = JLD2OutputWriter(model, FieldOutputs(fields), interval = 2hour,
                                                                 force = true,
                                                                prefix = prefix * "_fields")

averages_writer = JLD2OutputWriter(model, LESbrary.Statistics.horizontal_averages(model);
                                      force = true,
                                   interval = 15minute,
                                     prefix = prefix * "_averages")

slices_writer = JLD2OutputWriter(model, LESbrary.Statistics.XZSlices(fields, y=128);
                                      force = true,
                                   interval = 2minute,
                                     prefix = prefix * "_xz_slices")

simulation.output_writers[:fields] = fields_writer
simulation.output_writers[:averages] = averages_writer
simulation.output_writers[:slices] = slices_writer

# ## Running the simulation
#
# This part is easy,

run!(simulation)

# # Making a neat movie
#
# Making the coordinate arrays takes a few lines of code,

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)

xw, yw, zw = xw[:], yw[:], zw[:]
xu, yu, zu = xu[:], yu[:], zu[:]
nothing # hide

# Next, we open the JLD2 file, and extract the iterations we ended up saving at,

using JLD2, Plots

file = jldopen(simulation.output_writers[:slices].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

function nice_divergent_levels(c, clim)
    levels = range(-clim, stop=clim, length=10)

    cmax = maximum(abs, c)

    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=10), [cmax])
    end

    return levels
end

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load slices from file, omitting halo regions
    wxz = file["timeseries/w/$iter"][2:end-1, 1, 2:end-1]
    uxz = file["timeseries/u/$iter"][2:end-1, 1, 2:end-1]

    wlim = 0.05
    ulim = 0.1
    wlevels = nice_divergent_levels(wxz, wlim)
    ulevels = nice_divergent_levels(uxz, ulim)

    wxz_plot = heatmap(xw, zw, wxz';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-wlim, wlim),
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    uxz_plot = heatmap(xu, zu, uxz';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-ulim, ulim),
                             levels = ulevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")
                        
    t = @sprintf("%.2f hrs", file["timeseries/t/$iter"] / hour)
    plot(wxz_plot, uxz_plot, layout=(1, 2), size=(1000, 400),
         title = ["w(x, y=0, z, t=$t) (m/s)" "u(x, y = 0, z, t=$t) (m/s)"])

    iter == iterations[end] && close(file)
end

mp4(anim, "convection_then_wind_pulse.mp4", fps = 15) # hide
