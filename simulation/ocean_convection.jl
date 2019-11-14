using Statistics, Printf

using Oceananigans
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.AbstractOperations

using Oceananigans: Cell, Face, cell_advection_timescale

# Float type and architecture
FT = Float64
arch = GPU()

# Resolution
Nx = 256
Ny = 256
Nz = 256

# Domain size
Lx = 100
Ly = 100
Lz = 100

# Initial stratification
∂θ∂z = 0.01

# Simulation time
days = 8
end_time = day * days

# Data directory
base_dir = "ocean_convection_data"
mkpath(base_dir)

# Physical constants.
const f₀ = 1e-4  # Coriolis parameter [s⁻¹]
const ρ₀ = 1027  # Density of seawater [kg/m³]
const cₚ = 4000  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# Surface heat flux (negative values imply cooling)
@inline Q(x, y, t) = -100
Q_str = "-100"

# Surface wind stress along the x-direction
@inline τx(x, y, t) = 0
τx_str = "0"

# Convert heat flux and wind stress to boundary conditions.
@inline Fu(x, y, t) = τx(x, y, t) / ρ₀
@inline Fθ(x, y, t) =  Q(x, y, t) / (ρ₀ * cₚ)

Fu_str = "τx(x, y, t) / ρ₀"
Fθ_str = "Q(x, y, t) / (ρ₀ * cₚ)"

u_top_bc = BoundaryFunction{:z, Face, Cell}(Fu)
θ_top_bc = BoundaryFunction{:z, Cell, Cell}(Fθ)

# Define boundary conditions
ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, u_top_bc))
θbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, θ_top_bc),
                               bottom = BoundaryCondition(Gradient, ∂θ∂z))
Sbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, 0),
                               bottom = BoundaryCondition(Gradient, ∂θ∂z))

# Create model
model = Model(float_type = FT,
            architecture = arch,
                    grid = RegularCartesianGrid(FT; size=(Nx, Ny, Nz), length=(Lx, Ly, Lz)),
                coriolis = FPlane(FT; f=f₀),
                buoyancy = SeawaterBuoyancy(FT; equation_of_state=LinearEquationOfState(β=0)),
                 closure = AnisotropicMinimumDissipation(FT),
     boundary_conditions = BoundaryConditions(u=ubcs, T=θbcs, S=Sbcs))

# Set initial condition.
ε(μ) = μ * randn() # noise
θ₀(x, y, z) = 20 + ∂θ∂z * z + ε(1e-10) * exp(4z/Lz)
S₀(x, y, z) = θ₀(x, y, z)

set_ic!(model, T=θ₀, S=S₀)

# Function that saves metadata with every output file.
function init_save_parameters_and_bcs(file, model)
    file["parameters/coriolis_parameter"] = f₀
    file["parameters/density"] = ρ₀
    file["parameters/specific_heat_capacity"] = cₚ
    file["parameters/viscosity"] = model.closure.ν
    file["parameters/diffusivity"] = model.closure.κ.T
    file["parameters/diffusivity_T"] = model.closure.κ.T
    file["parameters/diffusivity_S"] = model.closure.κ.S
    file["parameters/surface_cooling"] = Q_str
    file["parameters/temperature_gradient"] = ∂θ∂z
    file["parameters/wind_stress_x"] = τx_str
    file["boundary_conditions/top/FT"] = Fθ_str
    file["boundary_conditions/top/Fu"] = Fu_str
    file["boundary_conditions/bottom/dTdz"] = ∂θ∂z
end

# Saving 3D fields to JLD2 output files.
fields = Dict(
     :u => model -> Array(model.velocities.u.data.parent),
     :v => model -> Array(model.velocities.v.data.parent),
     :w => model -> Array(model.velocities.w.data.parent),
     :T => model -> Array(model.tracers.T.data.parent),
     :S => model -> Array(model.tracers.S.data.parent),
    :nu => model -> Array(model.diffusivities.νₑ.data.parent),
:kappaT => model -> Array(model.diffusivities.κₑ.T.data.parent),
:kappaS => model -> Array(model.diffusivities.κₑ.S.data.parent)
)

field_writer = JLD2OutputWriter(model, fields; dir=base_dir, prefix="ocean_convection_fields",
                                init=init_save_parameters_and_bcs,
                                max_filesize=100GiB, interval=6hour, force=true, verbose=true)
push!(model.output_writers, field_writer)

####
#### Set up diagnostics.
####

# NaN checker will abort simulation if NaNs are produced.
push!(model.diagnostics, NaNChecker(model; frequency=1000, fields=Dict(:w => model.velocities.w)))

# Time interval for computing and saving profiles.
Δtₚ = 10minute

# Define horizontal average diagnostics.
 Up = HorizontalAverage(model.velocities.u;       return_type=Array)
 Vp = HorizontalAverage(model.velocities.v;       return_type=Array)
 Wp = HorizontalAverage(model.velocities.w;       return_type=Array)
 Tp = HorizontalAverage(model.tracers.T;          return_type=Array)
 Sp = HorizontalAverage(model.tracers.S;          return_type=Array)
 νp = HorizontalAverage(model.diffusivities.νₑ;   return_type=Array)
κTp = HorizontalAverage(model.diffusivities.κₑ.T; return_type=Array)
κSp = HorizontalAverage(model.diffusivities.κₑ.S; return_type=Array)
dTp = HorizontalAverage(model.timestepper.Gⁿ.T;   return_type=Array)

u = model.velocities.u
v = model.velocities.v
w = model.velocities.w
T = model.tracers.T
S = model.tracers.S

uu = HorizontalAverage(u*u, model; return_type=Array)
vv = HorizontalAverage(v*v, model; return_type=Array)
ww = HorizontalAverage(w*w, model; return_type=Array)
uv = HorizontalAverage(u*v, model; return_type=Array)
uw = HorizontalAverage(u*w, model; return_type=Array)
vw = HorizontalAverage(v*w, model; return_type=Array)
wT = HorizontalAverage(w*T, model; return_type=Array)
wS = HorizontalAverage(w*S, model; return_type=Array)

# Create output writer that writes vertical profiles to JLD2 output files.
profiles = Dict(
     :u => model -> Up(model),
     :v => model -> Vp(model),
     :w => model -> Wp(model),
     :T => model -> Tp(model),
     :S => model -> Sp(model),
    :nu => model -> νp(model),
:kappaT => model -> κTp(model),
:kappaS => model -> κSp(model),
  :dTdt => model -> dTp(model),
    :uu => model -> uu(model),
    :vv => model -> vv(model),
    :ww => model -> ww(model),
    :uv => model -> uv(model),
    :uw => model -> uw(model),
    :vw => model -> vw(model),
    :wT => model -> wT(model),
    :wS => model -> wS(model)
)

profile_writer = JLD2OutputWriter(model, profiles; dir=base_dir, prefix="ocean_convection_profiles",
                                  init=init_save_parameters_and_bcs,
                                  interval=Δtₚ, max_filesize=25GiB, force=true, verbose=true)

push!(model.output_writers, profile_writer)

# Wizard utility that calculates safe adaptive time steps.
wizard = TimeStepWizard(cfl=0.25, Δt=3.0, max_change=1.2, max_Δt=30.0)

# Number of time steps to perform at a time before printing a progress
# statement and updating the adaptive time step.
Ni = 50

while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)

    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / end_time)

    # Calculate advective CFL number.
    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = wizard.Δt / cell_advection_timescale(model)

    # Calculate diffusive CFL number.
    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)

    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    νCFL = wizard.Δt / (Δ^2 / νmax)
    κCFL = wizard.Δt / (Δ^2 / κmax)

    # Calculate a new adaptive time step.
    update_Δt!(wizard, model)

    # Print progress statement.
    @printf("[%06.2f%%] i: %d, t: %5.2f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, νκmax: (%6.3g, %6.3g), νκCFL: (%6.4g, %6.4g), next Δt: %8.5g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time / day,
            umax, vmax, wmax, CFL, νmax, κmax, νCFL, κCFL,
            wizard.Δt, prettytime(walltime / Ni))
end
