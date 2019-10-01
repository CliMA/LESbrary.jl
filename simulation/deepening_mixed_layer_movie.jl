using Statistics, Printf
using ArgParse

using Oceananigans
using Oceananigans: fill_halo_regions!, zero_halo_regions!

s = ArgParseSettings(description="Run simulations of a stratified fluid forced by surface heat fluxes and wind" *
                     "stresses, simulating an oceanic boundary layer that develops a deepening mixed layer.")

@add_arg_table s begin
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

parsed_args = parse_args(s)
parse_int(n) = isinteger(n) ? Int(n) : n
Nh, Nz, L, H, Q, τ, ∂T∂z, days = [parsed_args[k] for k in ["Nh", "Nz", "L", "H", "Q", "τ", "∂T∂z", "days"]]
Nh, Nz, L, H, Q, τ, ∂T∂z, days = parse_int.([Nh, Nz, L, H, Q, τ, ∂T∂z, days])

base_dir = parsed_args["base_dir"]
if !isdir(base_dir)
    @info "Creating directory: $base_dir"
    mkpath(base_dir)
end

# Filename prefix for output files.
prefix = @sprintf("mixed_layer_simulation_Q%d_dTdz%.3f_tau%.2f", Q, ∂T∂z, τ)

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
Nx, Ny, Nz = Nh, Nh, Nz
Lx, Ly, Lz = L, L, H
end_time = days * day

ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Fu))
Tbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Fθ),
                               bottom = BoundaryCondition(Gradient, ∂T∂z))

model = Model(float_type = FT,
            architecture = GPU(),
                    grid = RegularCartesianGrid(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz)),
                coriolis = FPlane(FT; f=1e-4),
                buoyancy = SeawaterBuoyancy(FT; equation_of_state=LinearEquationOfState(β=0)),
                 closure = AnisotropicMinimumDissipation(FT),
     boundary_conditions = BoundaryConditions(u=ubcs, T=Tbcs))

# Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
ε(μ) = μ * randn() # noise
T₀(x, y, z) = 20 + ∂T∂z * z + ε(1e-10) * exp(z/25)

# Noise is needed so that AMD does not blow up due to dividing by ∇u or ∇S.
u₀(x, y, z) = ε(1e-10) * exp(z/25)
v₀(x, y, z) = ε(1e-10) * exp(z/25)
w₀(x, y, z) = ε(1e-10) * exp(z/25)
S₀(x, y, z) = ε(1e-10) * exp(z/25)

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

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

xy_slice(field; k=1) = Array(field.data.parent[:, :, k])
xz_slice(field; j=1) = Array(field.data.parent[:, j, :])
yz_slice(field; i=1) = Array(field.data.parent[i, :, :])

movie_slices = Dict(
    :u_xy_slice => model -> xy_slice(model.velocities.u),
    :u_xz_slice => model -> xz_slice(model.velocities.u),
    :u_yz_slice => model -> yz_slice(model.velocities.u),
    :v_xy_slice => model -> xy_slice(model.velocities.v),
    :v_xz_slice => model -> xz_slice(model.velocities.v),
    :v_yz_slice => model -> yz_slice(model.velocities.v),
    :w_xy_slice => model -> xy_slice(model.velocities.w),
    :w_xz_slice => model -> xz_slice(model.velocities.w),
    :w_yz_slice => model -> yz_slice(model.velocities.w),
    :T_xy_slice => model -> xy_slice(model.tracers.T),
    :T_xz_slice => model -> xz_slice(model.tracers.T),
    :T_yz_slice => model -> yz_slice(model.tracers.T))

Δtₛ = 5second  # Time interval for saving slices and profiles.

slice_writer = JLD2OutputWriter(model, movie_slices; dir=base_dir, prefix=prefix * "_slices",
                                init=init_save_parameters_and_bcs,
                                max_filesize=10GiB, interval=Δtₛ, force=true, verbose=false)
push!(model.output_writers, slice_writer)

Up = HorizontalAverage(model, model.velocities.u;       return_type=Array)
Vp = HorizontalAverage(model, model.velocities.v;       return_type=Array)
Wp = HorizontalAverage(model, model.velocities.w;       return_type=Array)
Tp = HorizontalAverage(model, model.tracers.T;          return_type=Array)
νp = HorizontalAverage(model, model.diffusivities.νₑ;   return_type=Array)
κp = HorizontalAverage(model, model.diffusivities.κₑ.T; return_type=Array)
wT = HorizontalAverage(model, [model.velocities.w, model.tracers.T]; return_type=Array)

profiles = Dict(
    :u => Up(model),
    :v => Vp(model),
    :w => Wp(model),
    :T => Tp(model),
   :nu => νp(model),
:kappa => κp(model),
   :wT => wT(model))

profile_writer = JLD2OutputWriter(model, profiles; dir=base_dir, prefix=prefix * "_profiles",
                                  init=init_save_parameters_and_bcs,
                                  interval=Δtₛ, max_filesize=25GiB, force=true, verbose=true)

push!(model.output_writers, profile_writer)

# Set up diagnostics.
push!(model.diagnostics, NaNChecker(model; frequency=1000, fields=Dict(:w => model.velocities.w)))

# Wizard utility that calculates safe adaptive time steps.
wizard = TimeStepWizard(cfl=0.2, Δt=0.5second, max_change=1.2, max_Δt=5second)

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=wizard.Δt)

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = wizard.Δt / cell_advection_timescale(model)

    νmax = maximum(model.diffusivities.νₑ.data.parent)
    κmax = maximum(model.diffusivities.κₑ.T.data.parent)

    Δ = min(model.grid.Δx, model.grid.Δy, model.grid.Δz)
    νCFL = wizard.Δt / (Δ^2 / νmax)
    κCFL = wizard.Δt / (Δ^2 / κmax)

    update_Δt!(wizard, model)

    @printf("[%06.2f%%] i: %d, t: %5.2f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, νκmax: (%6.3g, %6.3g), νκCFL: (%6.4g, %6.4g), next Δt: %8.5g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time / day,
            umax, vmax, wmax, CFL, νmax, κmax, νCFL, κCFL,
            wizard.Δt, prettytime(walltime / Ni))
end
