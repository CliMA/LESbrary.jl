using ArgParse, Printf
using Oceananigans

s = ArgParseSettings(description="Run simulations of a mixed layer over an idealized seasonal cycle.")

@add_arg_table s begin
    "--resolution", "-N"
        arg_type=Int
        required=true
        dest_name="N"
        help="Number of grid points in each dimension (Nx, Ny, Nz) = (N, N, N)."
    "--dTdz"
        arg_type=Float64
        required=true
        dest_name="dTdz"
        help="Temperature gradient (and stratification) to impose in the initial condition and bottom [K/m]."
    "--cycles"
        arg_type=Int
        required=true
        dest_name="c"
        help="Number of idealized seasonal cycles."
    "--days"
        arg_type=Float64
        required=true
        dest_name="days"
        help="Simulation length in days."
    "--output-dir", "-d"
        arg_type=AbstractString
        required=true
        dest_name="base_dir"
        help="Base directory to save output to."
end


# Parse command line arguments.
parsed_args = parse_args(s)
parse_int(n) = isinteger(n) ? Int(n) : n
N, ∂T∂z, c, days = [parsed_args[k] for k in ["N", "dTdz", "c", "days"]]
N, ∂T∂z, c, days = parse_int.([N, ∂T∂z, c, days])

base_dir = parsed_args["base_dir"]
if !isdir(base_dir)
    @info "Creating directory: $base_dir"
    mkpath(base_dir)
end

# Filename prefix for output files.
prefix = @sprintf("idealized_seasonal_cycle_dTdz%.2f_c%d_days%d", ∂T∂z, c, days)

L = 200
end_time = days * day

# Physical constants.
ρ₀ = 1027  # Density of seawater [kg/m³]
cₚ = 4000  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

const ωs = 2π / end_time                # Seasonal frequency [s⁻¹]
const Φavg = ∂T∂z * L^2 / (8end_time)  # Average heat flux.
const a = 1.5 * Φavg             # Asymmetry factor.
const C = c

# Seasonal cycle forcing.
# @inline Qsurface(i, j, grid, c, Gc, κ_bottom, t, iter, U, Φ) = (Φavg + a*sin(C*ωs*t))
@inline Qsurface(t) = (Φavg + a*sin(C*ωs*t))

Tbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Qsurface(0)),
                               bottom = BoundaryCondition(Gradient, ∂T∂z))

"""
Add a sponge layer to the bottom layer of the `model`. The purpose of this
sponge layer is to effectively dampen out waves reaching the bottom of the
domain and avoid having waves being continuously generated and reflecting from
the bottom, some of which may grow unrealistically large in amplitude.

Numerically the sponge layer acts as an extra source term in the momentum
equations. It takes on the form Gu[i, j, k] += -u[i, j, k]/τ for each momentum
source term where τ is a damping timescale. Typially, Δt << τ, otherwise
it's easy to find yourself not satisfying the diffusive stability criterion.
"""
const τ⁻¹ = 1 / 60  # Damping/relaxation time scale [s⁻¹].
const Δμ = 0.01L    # Sponge layer width [m] set to 1% of the domain height.
@inline μ(z, Lz) = τ⁻¹ * exp(-(z+Lz) / Δμ)

@inline Fu(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.u[i, j, k]
@inline Fv(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * U.v[i, j, k]
@inline Fw(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zF[k], grid.Lz) * U.w[i, j, k]

const Tₛ = 20.0  # Surface temperature [°C].
const dTdz = ∂T∂z
@inline T₀(z) = Tₛ + dTdz * z 
@inline FT(grid, U, Φ, i, j, k) = @inbounds -μ(grid.zC[k], grid.Lz) * (Φ.T[i, j, k] - T₀(grid.zC[k]))

forcing = Forcing(Fu=Fu, Fv=Fv, Fw=Fw, FT=FT)

# Create the model.
model = Model(N = (N, N, N),
              L = (L, L, L),
           arch = GPU(),
     float_type = Float64,
            eos = LinearEquationOfState(βS=0.0),  # Turn off salinity for now.
            bcs = BoundaryConditions(T=Tbcs),
        closure = AnisotropicMinimumDissipation(),  # Use AMD with molecular viscosities.
        forcing = forcing)

@printf("""
    Simulating an idealized seasonal cycle
        N : %d, %d, %d
        L : %.3g, %.3g, %.3g [m]
        Δ : %.3g, %.3g, %.3g [m]
     ∂T∂z : %.3g [K/m]
       ωs : %.3g [s⁻¹]
     Φavg : %.3g [W/m²]
        a : %.3g [W/m²]
     Φmin : %.3g [W/m²]
     Φmax : %.3g [W/m²]
     days : %d
  closure : %s
    """, model.grid.Nx, model.grid.Ny, model.grid.Nz,
         model.grid.Lx, model.grid.Ly, model.grid.Lz,
         model.grid.Δx, model.grid.Δy, model.grid.Δz,
         ∂T∂z, ωs, Φavg, a, Φavg-a, Φavg+a, days, typeof(model.closure))

# Set initial conditions.
# AMD requires noise on initial conditions everywhere as it divides by velocity
# and tracer gradients, so there can't be a zero gradient anywhere.

ε(σ) = σ * randn()  # Gaussian noise with mean 0 and standard deviation σ.

u₀(x, y, z) = ε(1e-12)
v₀(x, y, z) = ε(1e-12)
w₀(x, y, z) = ε(1e-12)
T₀(x, y, z) = 20 + ∂T∂z * z + ε(1e-12)
S₀(x, y, z) = ε(1e-12)
set!(model; u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

# Saving fields.
function init_save_parameters_and_bcs(file, model)
    file["parameters/density"] = ρ₀
    file["parameters/specific_heat_capacity"] = cₚ
    file["parameters/temperature_gradient"] = ∂T∂z
    file["parameters/seasonal_frequency"] = ωs
    file["parameters/Phi_avg"] = Φavg
    file["parameters/a"] = a
    file["parameters/simulation_days"] = days
end

fields = Dict(
    :u => model -> Array(model.velocities.u.data.parent),
    :v => model -> Array(model.velocities.v.data.parent),
    :w => model -> Array(model.velocities.w.data.parent),
    :T => model -> Array(model.tracers.T.data.parent),
   :nu => model -> Array(model.diffusivities.νₑ.data.parent),
:kappaT=> model -> Array(model.diffusivities.κₑ.T.data.parent))

field_writer = JLD2OutputWriter(model, fields; dir=base_dir, prefix=prefix * "_fields",
                                init=init_save_parameters_and_bcs,
                                interval=6hour, max_filesize=25GiB, force=true, verbose=true)
push!(model.output_writers, field_writer)

# Checkpoint infrequently in case we need to pick up.
checkpointer = Checkpointer(model; interval=30day, dir=base_dir, prefix=prefix*"_checkpoint", force=true)
push!(model.output_writers, checkpointer)

# Set up diagnostics.
push!(model.diagnostics, NaNChecker(model))

Δtₚ = 15minute  # Time interval for computing and saving profiles.

Up = HorizontalAverage(model, model.velocities.u; interval=Δtₚ)
Vp = HorizontalAverage(model, model.velocities.v; interval=Δtₚ)
Wp = HorizontalAverage(model, model.velocities.w; interval=Δtₚ)
Tp = HorizontalAverage(model, model.tracers.T;    interval=Δtₚ)
wT = HorizontalAverage(model, [model.velocities.w, model.tracers.T]; interval=Δtₚ)
νp = HorizontalAverage(model, model.diffusivities.νₑ; interval=Δtₚ)
κp = HorizontalAverage(model, model.diffusivities.κₑ.T; interval=Δtₚ)
UUp = HorizontalAverage(model, [model.velocities.u, model.velocities.u]; interval=Δtₚ)
VVp = HorizontalAverage(model, [model.velocities.v, model.velocities.v]; interval=Δtₚ)
WWp = HorizontalAverage(model, [model.velocities.w, model.velocities.w]; interval=Δtₚ)
UVp = HorizontalAverage(model, [model.velocities.u, model.velocities.v]; interval=Δtₚ)
UWp = HorizontalAverage(model, [model.velocities.u, model.velocities.w]; interval=Δtₚ)
VWp = HorizontalAverage(model, [model.velocities.v, model.velocities.w]; interval=Δtₚ)

append!(model.diagnostics, [Up, Vp, Wp, Tp, wT, UUp, VVp, WWp, UVp, UWp, VWp])

profiles = Dict(
     :u => model -> Array(Up.profile),
     :v => model -> Array(Vp.profile),
     :w => model -> Array(Wp.profile),
     :T => model -> Array(Tp.profile),
    :wT => model -> Array(wT.profile),
    :nu => model -> Array(νp.profile),
:kappaT => model -> Array(κp.profile),
    :uu => model -> Array(UUp.profile),
    :vv => model -> Array(VVp.profile),
    :ww => model -> Array(WWp.profile),
    :uv => model -> Array(UVp.profile),
    :uw => model -> Array(UWp.profile),
    :vw => model -> Array(VWp.profile))

profile_writer = JLD2OutputWriter(model, profiles; dir=base_dir, prefix=prefix * "_profiles",
                                  init=init_save_parameters_and_bcs,
                                  interval=Δtₚ, max_filesize=25GiB, force=true, verbose=true)

push!(model.output_writers, profile_writer)

# Wizard utility that calculates safe adaptive time steps.
wizard = TimeStepWizard(cfl=0.30, Δt=0.1, max_change=1.2, max_Δt=30.0)

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

while model.clock.time < end_time
    model.boundary_conditions.T.z.top = BoundaryCondition(Flux, Qsurface(model.clock.time))
    
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
