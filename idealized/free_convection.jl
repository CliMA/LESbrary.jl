using Oceananigans, Oceananigans.Diagnostics, Oceananigans.OutputWriters, Oceananigans.Coriolis, 
      Oceananigans.BoundaryConditions, Oceananigans.Forcing, Oceananigans.TurbulenceClosures, 
      Oceananigans.Buoyancy

using Random, Printf, Statistics, CUDAapi

if has_cuda() 
    include("../utils/cuda.jl")
    select_device!(2)
end

makeplot = false

include("../utils/setup.jl")

#####
##### Parameters and such
#####

Nh = 32       # Number of grid points in x, y
Nz = 32       # Number of grid points in z
Lh = 256      # [m] Grid spacing in x, y (meters)
Lz = 128      # [m] Grid spacing in z (meters)
Qᵇ = 1e-7     # [m² s⁻³] Buoyancy flux at surface
N² = 2e-6     # [s⁻²] Initial buoyancy gradient
θ₀ = 20.0     # [ᵒC] Surface temperature
 f = 1e-4     # [s⁻¹] Coriolis parameter

# Create the grid 
grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(0, Lh), y=(0, Lh), z=(-Lz, 0))

# Calculate stop time as time when boundary layer depth is h = Lz/2.
# Uses a conservative estimate based on 
#
#   h ∼ √(2 * Qᵇ * stop_time / N²)

h = Lz/2 # end boundary layer depth is half domain depth
stop_time = 1/2 * h^2 * N² / Qᵇ 

#####
##### Buoyancy, equation of state, temperature flux, and initial temperature gradient
#####

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

   Qᶿ = Qᵇ / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)
dθdz₀ = N² / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)
 dθdz = dθdz₀

#####
##### Near-wall LES diffusivity modification and temperature gradient
#####

# Wall-aware AMD model constant
const Δz = Lz / Nz
@inline Cᴬᴹᴰ(x, y, z) = 1/12 * (1 + 2 * exp((z + Δz/2) / 8Δz))

κₑ_bcs = SurfaceFluxDiffusivityBoundaryConditions(grid, Qᵇ; Cʷ=0.1)

κ₀ = κₑ_bcs.z.top.condition # surface diffusivity
dθdz_surface = - Qᶿ / κ₀

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Gradient, dθdz_surface),
                                       bottom = BoundaryCondition(Gradient, dθdz₀))

#####
##### Sponge layer
#####

δ = 8   # [m] Sponge layer width
τ = 60  # [s] Sponge layer damping time-scale

u_forcing = ParameterizedForcing(Fu, (δ=δ, τ=τ))
v_forcing = ParameterizedForcing(Fv, (δ=δ, τ=τ))
w_forcing = ParameterizedForcing(Fw, (δ=δ, τ=τ))
θ_forcing = ParameterizedForcing(Fθ, (δ=δ, τ=τ, dθdz=dθdz₀, θ₀=θ₀))

#####
##### Model instantiation, initial condition, and model run
#####

prefix = @sprintf("free_convection_Qb%.1e_Nsq%.1e_Nh%d_Nz%d", Qᵇ, N², Nh, Nz)

model = IncompressibleModel(       architecture = has_cuda() ? GPU() : CPU(),
                                           grid = grid,
                                        tracers = :T,
                                       buoyancy = buoyancy,
                                       coriolis = FPlane(f=f),
                                        closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                            boundary_conditions = (T=θ_bcs, κₑ=(T=κₑ_bcs,)),
                                        forcing = ModelForcing(u=u_forcing, v=v_forcing, w=w_forcing, T=θ_forcing)
                           )

# Initial condition
ε₀, Δθ, w★ = 1e-6, dθdz₀ * Lz, (Qᵇ * Lz)^(1/3)
Ξ(ε₀, z) = ε₀ * randn() * z / Lz * exp(4z / Lz) # noise
θᵢ(x, y, z) = θ₀ + dθdz₀ * z + Ξ(ε₀ * Δθ, z)
uᵢ(x, y, z) = Ξ(ε₀ * w★, z)

Oceananigans.set!(model, T=θᵢ, u=uᵢ, v=uᵢ, w=uᵢ)

function init(file, model; kwargs...)
    save_global!(file, "sponge_layer", :δ)
    save_global!(file, "sponge_layer", :τ)
    save_global!(file, "initial_conditions", :dθdz)
    save_global!(file, "initial_conditions", :θ₀)
    save_global!(file, "boundary_conditions", :Qᵇ)
    save_global!(file, "boundary_conditions", :Qᶿ)
    return nothing
end

#####
##### Progress messenger
#####

# Just a nice message to print while the simulation runs.

mutable struct ProgressMessenger{T, U, V, W, N, K, A, D, Z} <: Function
    wall_time :: T
         umax :: U
         vmax :: V
         wmax :: W
         νmax :: N
         κmax :: K
      adv_cfl :: A
      dif_cfl :: D
       wizard :: Z
end

function (pm::ProgressMessenger)(simulation)
    model = simulation.model

    i, t = model.clock.iteration, model.clock.time

    progress = 100 * (t / simulation.stop_time)

    elapsed_wall_time = 1e-9 * (time_ns() - pm.wall_time)
    pm.wall_time = time_ns()

    msg1 = @sprintf("[%06.2f%%] i: % 6d, sim time: % 10s, Δt: % 10s, wall time: % 8s,",
                    progress, i, prettytime(t), prettytime(simulation.Δt.Δt), prettytime(elapsed_wall_time))

    msg2 = @sprintf("umax: (%.2e, %.2e, %.2e) m/s, CFL: %.2e, νκmax: (%.2e, %.2e), νκCFL: %.2e,\n",
                    pm.umax(model), pm.vmax(model), pm.wmax(model), pm.adv_cfl(model), pm.νmax(model), 
                    pm.κmax(model), pm.dif_cfl(model))

    @printf("%s %s", msg1, msg2)

    return nothing
end

#####
##### Create the simulation
#####

wizard = TimeStepWizard(       cfl = 0.2,
                                Δt = 1e-1,
                        max_change = 1.1,
                            max_Δt = 10.0)


messenger = ProgressMessenger(time_ns(), 
                              FieldMaximum(abs, model.velocities.u),
                              FieldMaximum(abs, model.velocities.v),
                              FieldMaximum(abs, model.velocities.w),
                              FieldMaximum(abs, model.diffusivities.νₑ),
                              FieldMaximum(abs, model.diffusivities.κₑ.T),
                              AdvectiveCFL(wizard),
                              DiffusiveCFL(wizard),
                              wizard)

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, progress_frequency=100, progress=messenger)

#####
##### Three-dimensional field output
#####

field_interval = (stop_time - wizard.max_Δt) / 10 # output 10 fields

fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,),
                         prefix_tuple_names(:κₑ, model.diffusivities.κₑ))

field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=field_interval, max_filesize=1GiB,
                                dir="data", prefix=prefix*"_fields", init=init, force=true)

simulation.output_writers[:fields] = field_writer

#####
##### One-dimensional averages
#####

averages = horizontal_statistics(model)

averages_writer = JLD2OutputWriter(model, averages; interval=15minute, force=true,
                                   dir="data", prefix=prefix*"_averages", init=init)

simulation.output_writers[:averages] = averages_writer

#=
#####
##### Downdraft
#####

downdraft_statistics = UpdraftStatistics(model; downdraft_quantile=0.1)
downdrafts_writer = JLD2OutputWriter(model, (downdrafts=downdraft_statistics,); interval=15minute, force=true,
                                   dir="data", prefix=prefix*"_downdrafts", init=init)

simulation.output_writers[:downdrafts] = downdrafts_writer
=#

#####
##### Run it
#####

print_banner(simulation)

run!(simulation)

exit() # Release GPU memory
