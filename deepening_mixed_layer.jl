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

# Simulation parameters.
Q    = -50
∂T∂z = 0.01
τ    = 0

# We impose the wind stress as a flux at the surface.
# To impose a flux boundary condition, the top flux imposed should be negative
# for a heating flux and positive for a cooling flux, thus the minus sign.
Fu = τ / ρ₀
Fθ = -Q / (ρ₀*cₚ)

# Model parameters
FT = Float64
arch = HAVE_CUDA ? GPU() : CPU()
Nx, Ny, Nz = 256, 256, 256
Lx, Ly, Lz = 100, 100, 100
end_time = 8day
Δt = 3
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

