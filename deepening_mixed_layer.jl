using ArgParse

s = ArgParseSettings(description="Run simulations of a stratified fluid forced by surface heat fluxes and wind" *
                     "stresses, simulating an oceanic boundary layer that develops a deepening mixed layer.")

@add_arg_table s begin
    "--horizontal-resolution", "-N"
        arg_type=Int
        required=true
        help="Number of grid points in the horizontal (Nx, Ny) = (N, N)."
    "--vertical-resolution", "-V"
        arg_type=Int
        required=true
        help="Number of grid points in the vertical Nz."
    "--length", "-L"
        arg_type=Float64
        required=true
        help="Horizontal size of the domain (Lx, Ly) = (L, L) [meters] ."
    "--height", "-H"
        arg_type=Float64
        required=true
        help="Vertical height (or depth) of the domain Lz [meters]."
    "--dTdz"
        arg_type=Float64
        required=true
        help="Temperature gradient (stratification) to impose [K/m]."
    "--heat-flux", "-Q"
        arg_type=Float64
        required=true
        help="Heat flux to impose at the surface [W/m²]. Negative values imply a cooling flux."
    "--wind-stress"
        arg_type=Float64
        required=true
        help="Wind stress to impose at the surface in the x-direction [N/m²]."
    "--days"
        arg_type=Float64
        required=true
        help="Number of Europa days to run the model."
    "--output-dir", "-d"
        arg_type=AbstractString
        required=true
        help="Base directory to save output to."
end

parsed_args = parse_args(s)
Nh = parsed_args["horizontal-resolution"]
Nz = parsed_args["vertical-resolution"]
L  = parsed_args["length"]
H  = parsed_args["height"]
Q  = parsed_args["heat-flux"]
τ  = parsed_args["wind-stress"]
∂T∂z = parsed_args["dTdz"]

parse_int(n) = isinteger(n) ? Int(n) : n
Nh, Nz, L, H, Q, days = parse_int.([Nh, Nz, L, H, Q, days])

base_dir = parsed_args["output-dir"]
filename_prefix = "mixed_layer_simulation" * "_Q" * str(Q) * "_dTdz" * str(∂T∂z) * "_tau" * str(τ)

if !isdir(base_dir)
    @info "Creating directory: $base_dir"
    mkpath(output_dir)
end

using Statistics, Printf
using Oceananigans

function horizontal_avg(model, field)
    function havg(model)
        f = Array(ardata(field))
        return mean(f, dims=[1, 2])[:]
    end
    return havg
end

function horizontal_avg(model, f1, f2)
    function havg(model)
        prod = Array(ardata(f1)) .* Array(ardata(f2))
        return mean(prod, dims=[1, 2])[:]
    end
    return havg
end

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]

# We impose the wind stress Fu as a flux at the surface.
# To impose a flux boundary condition, the top flux imposed should be negative
# for a heating flux and positive for a cooling flux, thus the minus sign on Fθ.
Fu = τ / ρ₀
Fθ = -Q / (ρ₀*cₚ)

# Model parameters
FT = Float64
arch = HAVE_CUDA ? GPU() : CPU()
Nx, Ny, Nz = N, N, Nz
Lx, Ly, Lz = L, L, H
end_time = days * day
ν, κ = 1e-5, 1e-5

ubcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Fu))
Tbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Fθ),
                               bottom = BoundaryCondition(Gradient, ∂T∂z))

model = Model(float_type = FT,
                    arch = arch,
                       N = (Nx, Ny, Nz),
                       L = (Lx, Ly, Lz),
                     eos = LinearEquationOfState(βS=0.0),
                 closure = AnisotropicMinimumDissipation(FT; ν = ν, κ = κ),
                     bcs = BoundaryConditions(u=ubcs, T=Tbcs))

# Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
ε(μ) = μ * randn() # noise
T₀(x, y, z) = 20 + ∂T∂z * z + ε(1e-4)

# Noise is needed so that AMD does not blow up due to dividing by ∇u or ∇S.
u₀(x, y, z) = ε(1e-4)
v₀(x, y, z) = ε(1e-4)
w₀(x, y, z) = ε(1e-4)
S₀(x, y, z) = ε(1e-4)

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

function init_save_parameters_and_bcs(file, model)
    file["parameters/density"] = ρ₀
    file["parameters/specific_heat_capacity"] = cₚ
    file["parameters/viscosity"] = ν
    file["parameters/diffusivity"] = κ
    file["parameters/surface_cooling"] = Q
    file["parameters/temperature_gradient"] = ∂T∂z
    file["parameters/wind_stress"] = τ
    file["boundary_conditions/top/FT"] = Fθ
    file["boundary_conditions/top/Fu"] = Fu
    file["boundary_conditions/bottom/dTdz"] = ∂T∂z
end

u(model) = Array(model.velocities.u.data.parent)
v(model) = Array(model.velocities.v.data.parent)
w(model) = Array(model.velocities.w.data.parent)
θ(model) = Array(model.tracers.T.data.parent)
κTe(model) = Array(model.diffusivities.κₑ.T.data.parent)
νe(model) = Array(model.diffusivities.νₑ.data.parent)

fields = Dict(:u=>u, :v=>v, :w=>w, :T=>θ, :kappaT=>κTe, :nu=>νe)

filename = @sprintf("mixed_layer_simulation_fields_Q%d_Tz%.2f", Q, ∂T∂z)
field_writer = JLD2OutputWriter(model, fields; dir="data", prefix=filename,
                                init=init_save_parameters_and_bcs, interval=6hour, force=true)

profiles = Dict{Symbol, Function}()

# Horizontal profiles for u, v, w, T, S.
for fs in [:velocities, :tracers]
    for f in propertynames(getproperty(model, fs))
        field = getproperty(getproperty(model, fs), f)
        profiles[f] = horizontal_avg(model, field)
    end
end

profiles[:wT] = horizontal_avg(model, model.velocities.w, model.tracers.T)
profiles[:kappaT] = horizontal_avg(model, model.diffusivities.κₑ.T)
profiles[:nu] = horizontal_avg(model, model.diffusivities.νₑ)

# Horizontal profiles for velocity covariances.
U = model.velocities
for i in propertynames(U)
    for j in propertynames(U)
        f = Symbol(string(i) * string(j))
        ui = getproperty(U, i)
        uj = getproperty(U, j)
        profiles[f] = horizontal_avg(model, ui, uj)
    end
end

filename = @sprintf("mixed_layer_simulation_profiles_Q%d_Tz%.2f", Q, ∂T∂z)
profile_writer = JLD2OutputWriter(model, profiles; dir="data", prefix=filename,
                                  init=init_save_parameters_and_bcs, interval=10minute, force=true)

push!(model.output_writers, field_writer)
push!(model.output_writers, profile_writer)

Ni = 100  # Number of intermediate time steps to take before printing a progress message.
while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=Δt)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = Δt / cell_advection_timescale(model)

    progress = 100 * (model.clock.time / end_time)
    @printf("[%06.2f%%] i: %d, t: %8.5g, umax: (%6.3g, %6.3g, %6.3g), CFL: %6.4g, ⟨wall time⟩: %s\n",
            progress, model.clock.iteration, model.clock.time, umax, vmax, wmax, CFL, prettytime(1e9*walltime / Ni))
end

