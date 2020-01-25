using Statistics
using Printf
using ArgParse

using Oceananigans
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.AbstractOperations
using Oceananigans.Utils

#####
##### Parse command line arguments
#####

description =
"""
Run simulations of a stratified fluid forced by surface heat fluxes and wind
stresses, simulating an oceanic boundary layer that develops a deepening mixed layer.
"""

args = ArgParseSettings(description=description)

@add_arg_table args begin
    "--horizontal-resolution", "-N"
        arg_type=Int
        required=true
        dest_name="Nh"
        help="Number of grid points in the horizontal (Nx, Ny) = (N, N)."
    "--vertical-resolution", "-V"
        arg_type=Int
        required=true
        dest_name="Nz"
        help="Number of grid points in the vertical Nz."
    "--length", "-L"
        arg_type=Float64
        required=true
        dest_name="L"
        help="Horizontal size of the domain (Lx, Ly) = (L, L) [meters] ."
    "--height", "-H"
        arg_type=Float64
        required=true
        dest_name="H"
        help="Vertical height (or depth) of the domain Lz [meters]."
    "--dTdz"
        arg_type=Float64
        required=true
        dest_name="∂T∂z"
        help="Temperature gradient (stratification) to impose [K/m]."
    "--heat-flux", "-Q"
        arg_type=Float64
        required=true
        dest_name="Q"
        help="Heat flux to impose at the surface [W/m²]. Negative values imply a cooling flux."
    "--wind-stress"
        arg_type=Float64
        required=true
        dest_name="τ"
        help="Wind stress to impose at the surface in the x-direction [N/m²]."
    "--days"
        arg_type=Float64
        required=true
        dest_name="days"
        help="Number of days to run the simulation."
    "--output-dir", "-d"
        arg_type=AbstractString
        required=true
        dest_name="base_dir"
        help="Base directory to save output to."
end

parsed_args = parse_args(args)
parse_int(n) = isinteger(n) ? Int(n) : n
Nh, Nz, L, H, Q, τ, ∂T∂z, days = [parsed_args[k] for k in ["Nh", "Nz", "L", "H", "Q", "τ", "∂T∂z", "days"]]
Nh, Nz, L, H, Q, τ, ∂T∂z, days = parse_int.([Nh, Nz, L, H, Q, τ, ∂T∂z, days])

base_dir = parsed_args["base_dir"]
if !isdir(base_dir)
    @info "Creating directory: $base_dir"
    mkpath(base_dir)
end

# Filename prefix for output files.
prefix = @sprintf("boundary_layer_turbulence_Q%d_dTdz%.3f_tau%.2f", Q, ∂T∂z, τ)

#####
##### Model setup
#####

# Physical constants.
ρ₀ = 1027  # Density of seawater [kg/m³]
cₚ = 4000  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# We impose the wind stress Fu as a flux at the surface.
# To impose a flux boundary condition, the top flux imposed should be negative
# for a heating flux and positive for a cooling flux, thus the minus sign on Fθ.
Fu = τ / ρ₀
Fθ = -Q / (ρ₀*cₚ)

# Model parameters
FT = Float64
arch = GPU()
Nx, Ny, Nz = Nh, Nh, Nz
Lx, Ly, Lz = L, L, H
end_time = days * day

ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Fu))
Tbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Fθ),
                               bottom = BoundaryCondition(Gradient, ∂T∂z))
Sbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Gradient, 0),
                               bottom = BoundaryCondition(Gradient, ∂T∂z))

model = Model(
             float_type = FT,
           architecture = arch,
                   grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(Lx, Ly, Lz)),
               coriolis = FPlane(FT, f=1e-4),
               buoyancy = SeawaterBuoyancy(FT),
                closure = AnisotropicMinimumDissipation(FT),
    boundary_conditions = HorizontallyPeriodicSolutionBCs(u=ubcs, T=Tbcs, S=Sbcs)
)

# Set initial condition.
ε(μ) = μ * randn() # Gaussian noise with zero mean and standard deviation μ.
T₀(x, y, z) = 20 + ∂T∂z * z + ε(1e-10) * exp(z/25)
S₀(x, y, z) = T₀(x, y, z)

set!(model, T=T₀, S=S₀)

#####
##### Set up output writers and diagnostics
#####

model.diagnostics[:nan_checker] =
    NaNChecker(model, frequency=1000, fields=Dict(:w => model.velocities.w))

function init_save_parameters_and_bcs(file, model)
    file["parameters/density"] = ρ₀
    file["parameters/specific_heat_capacity"] = cₚ
    file["parameters/viscosity"] = model.closure.ν
    file["parameters/diffusivity"] = model.closure.κ
    file["parameters/surface_cooling"] = Q
    file["parameters/temperature_gradient"] = ∂T∂z
    file["parameters/wind_stress"] = τ
    file["boundary_conditions/top/FT"] = Fθ
    file["boundary_conditions/top/Fu"] = Fu
    file["boundary_conditions/bottom/dTdz"] = ∂T∂z
end

fields = Dict(
         :u => model -> Array(model.velocities.u.data.parent),
         :v => model -> Array(model.velocities.v.data.parent),
         :w => model -> Array(model.velocities.w.data.parent),
         :T => model -> Array(model.tracers.T.data.parent),
         :S => model -> Array(model.tracers.S.data.parent),
    :kappaT => model -> Array(model.diffusivities.κₑ.T.data.parent),
    :kappaS => model -> Array(model.diffusivities.κₑ.S.data.parent),
        :nu => model -> Array(model.diffusivities.νₑ.data.parent)
)

model.output_writers[:field_writer] =
    JLD2OutputWriter(model, fields, dir=base_dir, prefix=prefix * "_fields",
                     init=init_save_parameters_and_bcs,
                     max_filesize=100GiB, interval=6hour, force=true, verbose=true)

Δtₚ = 10minute  # Time interval for computing and saving profiles.

Up = HorizontalAverage(model.velocities.u,     return_type=Array)
Vp = HorizontalAverage(model.velocities.v,     return_type=Array)
Wp = HorizontalAverage(model.velocities.w,     return_type=Array)
Tp = HorizontalAverage(model.tracers.T,        return_type=Array)
Sp = HorizontalAverage(model.tracers.S,        return_type=Array)
νp = HorizontalAverage(model.diffusivities.νₑ, return_type=Array)

κTp = HorizontalAverage(model.diffusivities.κₑ.T, return_type=Array)
κSp = HorizontalAverage(model.diffusivities.κₑ.S, return_type=Array)

u = model.velocities.u
v = model.velocities.v
w = model.velocities.w
T = model.tracers.T
S = model.tracers.S

uu = HorizontalAverage(u*u, model, return_type=Array)
vv = HorizontalAverage(v*v, model, return_type=Array)
ww = HorizontalAverage(w*w, model, return_type=Array)
uv = HorizontalAverage(u*v, model, return_type=Array)
uw = HorizontalAverage(u*w, model, return_type=Array)
vw = HorizontalAverage(v*w, model, return_type=Array)
wT = HorizontalAverage(w*T, model, return_type=Array)
wS = HorizontalAverage(w*S, model, return_type=Array)

profiles = Dict(
     :u => model -> Up(model),
     :v => model -> Vp(model),
     :w => model -> Wp(model),
     :T => model -> Tp(model),
     :S => model -> Sp(model),
    :nu => model -> νp(model),
:kappaT => model -> κTp(model),
:kappaS => model -> κSp(model),
    :uu => model -> uu(model),
    :vv => model -> vv(model),
    :ww => model -> ww(model),
    :uv => model -> uv(model),
    :uw => model -> uw(model),
    :vw => model -> vw(model),
    :wT => model -> wT(model),
    :wS => model -> wS(model)
)

model.output_writers[:profile_writer] =
    JLD2OutputWriter(model, profiles, dir=base_dir, prefix=prefix * "_profiles",
                     init=init_save_parameters_and_bcs,
                     interval=Δtₚ, max_filesize=25GiB, force=true, verbose=true)

#####
##### Time step!
#####

# Wizard utility that calculates safe adaptive time steps.
wizard = TimeStepWizard(cfl=0.25, Δt=3.0, max_change=1.2, max_Δt=30.0)

# CFL utilities for reporting stability criterions.
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)

    update_Δt!(wizard, model)

    @printf("[%06.2f%%] i: %d, t: %.2f days, umax: (%.2e, %.2e, %.2e) m/s, CFL: %.2e, νκmax: (%.2e, %.2e), dCFL: %.2e, next Δt: %.2e, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time / day, umax, vmax, wmax, cfl(model), νmax, κmax, dcfl(model), wizard.Δt, prettytime(walltime / Ni))
end
