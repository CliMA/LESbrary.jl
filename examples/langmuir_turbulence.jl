# # Langmuir turbulence example
#
# This example implements the Langmuir turbulence simulation reported in section
# 4 of
#
# [McWilliams, J. C. et al., "Langmuir Turbulence in the ocean," Journal of Fluid Mechanics (1997)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/langmuir-turbulence-in-the-ocean/638FD0E368140E5972144348DB930A38).

using Oceananigans
using Oceananigans.Grids
using Oceananigans.BoundaryConditions
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters

using Printf
using JLD2
using Plots

using Oceananigans.Buoyancy: g_Earth
using Oceananigans.Buoyancy: BuoyancyTracer
using Oceananigans.SurfaceWaves: UniformStokesDrift
using Oceananigans.Utils: prettytime, minute, hour

grid = RegularCartesianGrid(size=(32, 32, 48), extent=(128, 128, 96))

# ### The Stokes Drift profile
#
# The surface wave Stokes drift profile used in McWilliams et al. (1997)
# corresponds to a 'monochromatic' (that is, single-frequency) wave field with
# wavenumber = 2π/60 m⁻¹ and amplitude 0.8 m.
#
# Note that `Oceananigans.jl` implements the Lagrangian-mean form of the Craik-Leibovich
# equations. This means that our model takes the *vertical derivative* as an input,
# rather than the Stokes drift profile itself.

const wavenumber = 2π / 60 # m⁻¹
const amplitude = 0.8 # m
const Uˢ = amplitude^2 * wavenumber * sqrt(g_Earth * wavenumber) # m s⁻¹

uˢ(z) = Uˢ * exp(2wavenumber * z)
∂z_uˢ(z, t) = 2wavenumber * Uˢ * exp(2wavenumber * z)

# ### Boundary conditions
#
# At the surface at $z=0$, McWilliams et al. (1997) impose wind stress and destabilising buoyancy flux.
# At the bottom we relax the buoyancy back to its initial profile, $b = N² * z$, where $N² = 1.936e-5 s⁻²$.

Qᵘ = -3.72e-5 # m² s⁻²
Qᵇ = 2.307e-9 # m³ s⁻²
N² = 1.936e-5 # s⁻²

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))
                                               

b_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵇ),
                                       bottom = BoundaryCondition(Gradient, N²))

model = IncompressibleModel(        architecture = CPU(),
                                     timestepper = :RungeKutta3,
                                       advection = UpwindBiasedFifthOrder(),
                                            grid = grid,
                                         tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = FPlane(f=1e-4),
                                         closure = AnisotropicMinimumDissipation(),
                                   surface_waves = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                             boundary_conditions = (u=u_boundary_conditions,
                                                    b=b_boundary_conditions),
                            )

# ## Initial conditions

Ξ(z) = randn() * exp(z / 4)

bᵢ(x, y, z) = N² * z                + 1e-2 * Ξ(z) * N² * model.grid.Lz
uᵢ(x, y, z) = uˢ(z) + sqrt(abs(Qᵘ)) * 1e-2 * Ξ(z)
wᵢ(x, y, z) =         sqrt(abs(Qᵘ)) * 1e-2 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

wizard = TimeStepWizard(cfl=1.0, Δt=5.0, max_change=1.1, max_Δt=10.0)

# ### Nice progress messaging
#
# We define a function that prints a helpful message with
# maximum absolute value of $u, v, w$ and the current wall clock time.

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

simulation = Simulation(model, iteration_interval = 100,
                                               Δt = wizard,
                                        stop_time = 4hour,
                                         progress = print_progress)

# ## Output
#
# We set up an output writer for the simulation that saves all velocity fields,
# tracer fields, and the subgrid turbulent diffusivity every 2 minutes.


fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,))

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields_to_output,
                                                      schedule = TimeInterval(2minute),
                                                        prefix = "langmuir_turbulence",
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

tke_budget_statistics = turbulent_kinetic_energy_budget(model; w_scratch = w_scratch,
                                                               c_scratch = c_scratch,
                                                                       b = b,
                                                                       U = turbulence_statistics[:u],
                                                                       V = turbulence_statistics[:v])

turbulence_statistics = merge(turbulence_statstics, tke_budget_statistics) 

simulation.output_writers[:statistics] = JLD2OutputWriter(model, turbulence_statistics,
                                                          schedule = IterationInterval(10),
                                                          #schedule = TimeInterval(slice_interval),
                                                            prefix = prefix * "_statistics",
                                                               dir = data_directory,
                                                             force = true)

simulation.output_writers[:averaged_statistics] = JLD2OutputWriter(model, turbulence_statistics,
                                                                   schedule = AveragedTimeInterval(3hour, window=30minute),
                                                                     prefix = prefix * "_averaged_statistics",
                                                                        dir = data_directory,
                                                                      force = true)

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

# # Making a neat movie
#
# We look at the results by plotting vertical slices of $u$ and $w$, and a horizontal
# slice of $w$ to look for Langmuir cells.

k = searchsortedfirst(grid.zF[:], -8)

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

function nice_divergent_levels(c, clim, nlevels=30)
    levels = range(-clim, stop=clim, length=nlevels)

    cmax = maximum(abs, c)

    if clim < cmax # add levels on either end
        levels = vcat([-cmax], range(-clim, stop=clim, length=nlevels), [cmax])
    end

    return levels
end

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from file
    w = file["timeseries/w/$iter"]
    u = file["timeseries/u/$iter"]

    ## Extract slices
    wxy = w[:, :, k]
    wxz = w[:, 1, :]
    uxz = u[:, 1, :]

    wlim = 0.02
    ulim = 0.05
    wlevels = nice_divergent_levels(w, wlim)
    ulevels = nice_divergent_levels(w, ulim)

    wxy_plot = contourf(xw, yw, wxy';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-wlim, wlim),
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (0, grid.Ly),
                             xlabel = "x (m)",
                             ylabel = "y (m)")

    wxz_plot = contourf(xw, zw, wxz';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-wlim, wlim),
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    uxz_plot = contourf(xu, zu, uxz';
                              color = :balance,
                        aspectratio = :equal,
                              clims = (-ulim, ulim),
                             levels = ulevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    plot(wxy_plot, wxz_plot, uxz_plot, layout=(1, 3), size=(1000, 400),
         title = ["w(x, y, z=-8, t) (m/s)" "w(x, y=0, z, t) (m/s)" "u(x, y = 0, z, t) (m/s)"])

    iter == iterations[end] && close(file)
end

gif(anim, "langmuir_turbulence.gif", fps = 15) # hide
