# # Ocean convection, followed by a wind pulse

using Oceananigans, LESbrary, Printf

using Oceananigans: @hascuda

using LESbrary.Utils

@hascuda select_device!(3)

# ## Model set-up
#
# ### Domain specification and Grid construction
#
# We create a grid with modest resolution.

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(128, 128, 128), extent=(512, 512, 256))

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

using Oceananigans.Utils: hour, day

@info @sprintf("Convection time: %.2f hours", tᶜ / hour)

# A bit after convection has done its work, we impose a strong pulse of wind
# for 8 hours,

const τᵐᵃˣ = 1e-4 # m² s⁻²
const tᵖ = 1.2tᶜ
const Tᵖ = 12hour

@inline wind_pulse(t, T, τ) = - τ * t / T * exp(- t^2 / (2 * T^2))

@inline Qᵘ(x, y, t) = ifelse(t < tᵖ, 0.0, - τᵐᵃˣ) #wind_pulse(t - tᵖ, Tᵖ, τᵐᵃˣ))

## Uncomment to run convection followed by a wind *pulse*, rather than constant wind
#@inline Qᵘ(x, y, t) = ifelse(t < tᵖ, 0.0, wind_pulse(t - tᵖ, Tᵖ, τᵐᵃˣ))

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

# # Sponge layer specification

using Oceananigans.Forcing

using LESbrary.SpongeLayers: Fu, Fv, Fw, Fb

τ = 120  # [s] Sponge layer damping time-scale
δ = 8   # [m] Sponge layer width

u_forcing = ParameterizedForcing(Fu, (δ=δ, τ=τ))
v_forcing = ParameterizedForcing(Fv, (δ=δ, τ=τ))
w_forcing = ParameterizedForcing(Fw, (δ=δ, τ=τ))
b_forcing = ParameterizedForcing(Fb, (δ=δ, τ=τ, dbdz=N²))

# We use a wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# This is necessary to obtain smooth buoyancy profiles near the boundary.

using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant

Δz = grid.Lz / grid.Nz

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(Δz, C₀=1/12, enhancement=3, decay_scale=4Δz)

# ## Model instantiation
#
# Finally, we are ready to build the model. We use the AnisotropicMinimumDissipation
# model for large eddy simulation. Because our Stokes drift does not vary in $x, y$,
# we use `UniformStokesDrift`, which expects Stokes drift functions of $z, t$ only.

using Oceananigans.Buoyancy: BuoyancyTracer 

model = IncompressibleModel(        architecture = GPU(),
                                            grid = grid,
                                         tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = FPlane(f=f),
                                         closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                             boundary_conditions = (u=u_boundary_conditions, 
                                                    b=b_boundary_conditions),
                                         forcing = ModelForcing(u=u_forcing, v=v_forcing, w=w_forcing, b=b_forcing)
                            )

# ## Initial conditions
#
# Our initial condition for buoyancy consists of a linear stratification, plus noise,

Ξ(z) = randn() * exp(z / 8)

bᵢ(x, y, z) = N² * z + 1e-6 * Ξ(z) * N² * model.grid.Lz

set!(model, b=bᵢ)

# ## Setting up the simulation
#
# We use the `TimeStepWizard` for adaptive time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2,

wizard = TimeStepWizard(cfl=0.2, Δt=5.0, max_change=1.1, max_Δt=20.0)
nothing # hide

# Now we create the simulation,

using Oceananigans.Utils: hour # correpsonds to "1 hour", in units of seconds

simulation = Simulation(model,
                        Δt = wizard,
        progress_frequency = 100,
                 stop_time = 120hour,
                  progress = SimulationProgressMessenger(model, wizard)
)

# ## Output
#
# We set up an output writer for the simulation that saves all velocity fields, 
# tracer fields, and the subgrid turbulent diffusivity every 2 minutes.

using Oceananigans.OutputWriters
using Oceananigans.Utils: minute
using LESbrary.Statistics

prefix = "convection_then_wind_Nx$(grid.Nx)"

fields = merge(model.velocities, model.tracers)

fields_writer = JLD2OutputWriter(model, FieldOutputs(fields), interval = 4hour,
                                                                 force = true,
                                                                prefix = prefix * "_fields")

averages_writer = JLD2OutputWriter(model, LESbrary.Statistics.horizontal_averages(model);
                                      force = true,
                                   interval = 15minute,
                                     prefix = prefix * "_averages")

slices_writer = JLD2OutputWriter(model, XZSlices(model.velocities, y=128);
                                      force = true,
                                   interval = minute / 4,
                                     prefix = prefix * "_xz_slices")

  simulation.output_writers[:slices] = slices_writer
  simulation.output_writers[:fields] = fields_writer
simulation.output_writers[:averages] = averages_writer

# # Run the simulation

run!(simulation)

exit() # don't hang onto GPU memory
