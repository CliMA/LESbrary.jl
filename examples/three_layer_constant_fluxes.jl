# # Turbulent mixing of a three layer boundary layer driven by constant surface fluxes

# This script runs a simulation of a turbulent oceanic boundary layer with an initial
# three-layer temperature stratification. Turbulent mixing is driven by constant fluxes
# of momentum and heat at the surface.
#
# This script is set up to be configurable on the command line --- a useful property
# when launching multiple jobs at on a cluster.

using Pkg
using Statistics
using Printf
using Logging

using ArgParse
using JLD2

using LESbrary
using Oceanostics
using Oceananigans
using Oceananigans.Grids
using Oceananigans.BuoyancyModels
using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Utils
using Oceananigans.Units

using Oceananigans.Grids: Face, Center
using Oceananigans.Fields: PressureField
using Oceanostics.FlowDiagnostics: richardson_number_ccf!
using Oceanostics.TurbulentKineticEnergyTerms: TurbulentKineticEnergy, ShearProduction_z

using LESbrary.Utils: SimulationProgressMessenger, fit_cubic, poly
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant
using LESbrary.TurbulenceStatistics: first_through_second_order, turbulent_kinetic_energy_budget,
                                     subfilter_momentum_fluxes, subfilter_tracer_fluxes,
                                     ViscousDissipation

Logging.global_logger(OceananigansLogger())

# To start, we ensure that all packages in the LESbrary environment are installed:

Pkg.instantiate()

# Next, we parse the command line arguments

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--size"
            help = "The number of grid points in x, y, and z."
            nargs = 3
            default = [32, 32, 32]
            arg_type = Int

        "--extent"
            help = "The length of the x, y, and z dimensions."
            nargs = 3
            default = [512meters, 512meters, 256meters]
            arg_type = Float64

        "--buoyancy-flux"
            help = """The surface buoyancy flux in units of m² s⁻³.
                      A positive buoyancy flux implies cooling.

                      Note:
                          buoyancy-flux = + 1e-7 corresponds to cooling at 208 W / m²
                          buoyancy-flux = + 1e-8 corresponds to cooling at 21 W / m²
                          buoyancy-flux = - 1e-7 corresponds to heating at 208 W / m²"""
            default = 1e-8
            arg_type = Float64

        "--momentum-flux"
            help = """The surface x-momentum flux divided by density in units of m² s⁻².
                      A negative flux drives currents in the positive x-direction.

                      Note:
                        momentum-flux = - 1e-4 corresponds to U₁₀ = +6 m/s, roughly speaking
                        momentum-flux = - 1e-3 corresponds to U₁₀ = +19 m/s, roughly speaking"""
            default = -1e-4
            arg_type = Float64

        "--coriolis"
            help = "The Coriolis parameter, calculated as f=2Ω*sinϕ, where Ω is the rotation rate of the Earth in rad/s and ϕ is the latitude."
            default = 1e-4
            arg_type = Float64

        "--surface-temperature"
            help = """The temperature at the surface in ᵒC."""
            default = 20.0
            arg_type = Float64

        "--surface-layer-depth"
            help = "The depth of the surface layer in units of m."
            default = 48.0
            arg_type = Float64

        "--thermocline"
            help = """Two choices for the thermocline structure:
                        * linear: a thermocline with a linear buoyancy structure (constant stratification)
                        * cubic: a thermocline with a fitted cubic structure"""
            default = "linear"
            arg_type = String

        "--thermocline-width"
            help = "The width of the thermocline in units of m."
            default = 24.0
            arg_type = Float64

        "--surface-layer-buoyancy-gradient"
            help = "The buoyancy gradient in the surface layer in units s⁻²."
            default = 2e-6
            arg_type = Float64

        "--thermocline-buoyancy-gradient"
            help = "The buoyancy gradient in the thermocline in units s⁻²."
            default = 1e-5
            arg_type = Float64

        "--deep-buoyancy-gradient"
            help = "The buoyancy gradient below the thermocline in units s⁻²."
            default = 2e-6
            arg_type = Float64

        "--hours"
            help = "Number of hours to run the simulation for"
            default = 0.1
            arg_type = Float64

        "--time-averaged-statistics"
            help = "Compute and output time-averaged statistics."
            action = :store_true

        "--animation"
            help = "Make an animation of the horizontal and vertical velocity when the simulation completes."
            action = :store_true

        "--pickup"
            help = "Whether or not to pick the simulation up from latest checkpoint"
            default = false
            arg_type = Bool

        "--name"
            help = "A name to add to the end of the output filename, e.g. weak_wind_strong_cooling."
            default = ""
            arg_type = String
    end

    return parse_args(settings)
end

# # Setup
#
# We start by parsing the arguments received on the command line, and prescribing
# a few additional output-related arguments.

@info "Parsing command line arguments..."

args = parse_command_line_arguments()

Nx, Ny, Nz = args["size"]
Lx, Ly, Lz = args["extent"]
stop_hours = args["hours"]
f = args["coriolis"]
name = args["name"]

snapshot_time_interval = 10minutes
averages_time_interval = 3hours
averages_time_window = 15minutes

slice_depth = 8.0

## Determine filepath prefix

Qᵇ = args["buoyancy-flux"]
Qᵘ = args["momentum-flux"]

thermocline_type = args["thermocline"]

prefix = @sprintf("three_layer_constant_fluxes_%s_hr%d_Qu%.1e_Qb%.1e_f%.1e_Nh%d_Nz%d_",
                  thermocline_type, stop_hours, abs(Qᵘ), Qᵇ, f, Nx, Nz)
data_directory = joinpath(@__DIR__, "..", "data", prefix * name) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# Save command line arguments used to an executable bash script
open(joinpath(data_directory, "run_three_layer_constant_fluxes.sh"), "w") do io
    write(io, "#!/bin/sh\n")
    write(io, "julia " * basename(@__FILE__) * " " * join(ARGS, " ") * "\n")
end

# Domain
#
# We use a three-dimensional domain that's twice as wide as it is deep.
# We choose this aspect ratio so that the horizontal scale is 4x larger
# than the boundary layer depth when the boundary layer penetrates half
# the domain depth.

@info "Mapping grid..."

grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# Buoyancy and boundary conditions

@info "Enforcing boundary conditions..."

surface_layer_depth = args["surface-layer-depth"]
thermocline_width = args["thermocline-width"]

N²_surface_layer = args["surface-layer-buoyancy-gradient"]
N²_thermocline = args["thermocline-buoyancy-gradient"]
N²_deep = args["deep-buoyancy-gradient"]

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

Qᶿ = Qᵇ / (α * g)
dθdz_surface_layer = N²_surface_layer / (α * g)
dθdz_thermocline   = N²_thermocline   / (α * g)
dθdz_deep          = N²_deep          / (α * g)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz_deep))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

# Tracer forcing

@info "Forcing and sponging tracers..."

# # Initial condition and sponge layer

## Fiddle with indices to get a correct discrete profile
k_transition = searchsortedfirst(grid.zC, -surface_layer_depth)
k_deep = searchsortedfirst(grid.zC, -(surface_layer_depth + thermocline_width))

z_transition = grid.zC[k_transition]
z_deep = grid.zC[k_deep]

θ_surface = args["surface-temperature"]
θ_transition = θ_surface + z_transition * dθdz_surface_layer
θ_deep = θ_transition + (z_deep - z_transition) * dθdz_thermocline

@inline passive_tracer_forcing(x, y, z, t, p) = p.μ⁺ * exp(-(z - p.z₀)^2 / (2 * p.λ^2)) - p.μ⁻

λ = 4.0
μ⁺ = 1 / 6hour
μ₀ = √(2π) * λ / grid.Lz * μ⁺ / 2
μ∞ = √(2π) * λ / grid.Lz * μ⁺

c₀_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=  0.0, λ=λ, μ⁺=μ⁺, μ⁻=μ₀))
c₁_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=-48.0, λ=λ, μ⁺=μ⁺, μ⁻=μ∞))
c₂_forcing = Forcing(passive_tracer_forcing, parameters=(z₀=-96.0, λ=λ, μ⁺=μ⁺, μ⁻=μ∞))

# Sponge layer for u, v, w, and T
gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=4/hour, mask=gaussian_mask)

T_sponge = Relaxation(rate = 4/hour,
                      target = LinearTarget{:z}(intercept = θ_deep - z_deep*dθdz_deep, gradient = dθdz_deep),
                      mask = gaussian_mask)

# # LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.

@info "Building the wall model..."

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# # Instantiate Oceananigans.IncompressibleModel

@info "Framing the model..."

model = IncompressibleModel(
           architecture = GPU(),
            timestepper = :RungeKutta3,
              advection = WENO5(),
                   grid = grid,
                tracers = (:T, :c₀, :c₁, :c₂),
               buoyancy = buoyancy,
               coriolis = FPlane(f=f),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = (T=θ_bcs, u=u_bcs),
                forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge,
                           c₀=c₀_forcing, c₁=c₁_forcing, c₂=c₂_forcing)
)

# # Set Initial condition

@info "Setting initial conditions..."

## Noise with 8 m decay scale
Ξ(z) = rand() * exp(z / 8)

function thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep, θ_deep, dθdz_surface_layer, dθdz_thermocline, dθdz_deep)
    if thermocline_type == "linear"
        return z -> θ_transition + dθdz_thermocline * (z - z_transition)

    elseif thermocline_type == "cubic"
        p1 = (z_transition, θ_transition)
        p2 = (z_deep, θ_deep)
        coeffs = fit_cubic(p1, p2, dθdz_surface_layer, dθdz_deep)
        return z -> poly(z, coeffs)

    else
        @error "Invalid thermocline type: $thermocline"
    end
end

θ_thermocline = thermocline_structure_function(thermocline_type, z_transition, θ_transition, z_deep, θ_deep,
                                               dθdz_surface_layer, dθdz_thermocline, dθdz_deep)

"""
    initial_temperature(x, y, z)

Returns a three-layer initial temperature distribution. The average temperature varies in z
and is augmented by three-dimensional, surface-concentrated random noise.
"""
function initial_temperature(x, y, z)

    noise = 1e-6 * Ξ(z) * dθdz_surface_layer * grid.Lz

    if z_transition < z <= 0
        return θ_surface + dθdz_surface_layer * z + noise

    elseif z_deep < z <= z_transition
        return θ_thermocline(z) + noise

    else
        return θ_deep + dθdz_deep * (z - z_deep) + noise

    end
end

set!(model, T = initial_temperature)

# # Prepare the simulation

@info "Conjuring the simulation..."

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.8, Δt=1.0, max_change=1.1, min_Δt=0.01, max_Δt=30.0)

stop_time = stop_hours * hour

simulation = Simulation(model,
                    Δt = wizard,
             stop_time = stop_time,
    iteration_interval = 10,
              progress = SimulationProgressMessenger(wizard)
)

# # Prepare Output

@info "Strapping on checkpointer..."

pickup = args["pickup"]
force = pickup ? false : true

simulation.output_writers[:checkpointer] =
    Checkpointer(model, schedule = TimeInterval(stop_time/3), prefix = prefix * "_checkpointer", dir = data_directory)

@info "Squeezing out statistics..."

# Prepare turbulence statistics
k_xy_slice = searchsortedfirst(grid.zF[:], -slice_depth)

b = BuoyancyField(model)
p = PressureField(model)

ccc_scratch = Field(Center, Center, Center, model.architecture, model.grid)
ccf_scratch = Field(Center, Center, Face, model.architecture, model.grid)
fcf_scratch = Field(Face, Center, Face, model.architecture, model.grid)
cff_scratch = Field(Center, Face, Face, model.architecture, model.grid)

primitive_statistics = first_through_second_order(model, b=b, p=p, w_scratch=ccf_scratch, c_scratch=ccc_scratch)

subfilter_flux_statistics = merge(
    subfilter_momentum_fluxes(model, uz_scratch=ccf_scratch, vz_scratch=cff_scratch, c_scratch=ccc_scratch),
    subfilter_tracer_fluxes(model, w_scratch=ccf_scratch),
)

U = primitive_statistics[:u]
V = primitive_statistics[:v]
B = primitive_statistics[:b]

e = TurbulentKineticEnergy(model, U=U, V=V)
shear_production = ShearProduction_z(model, U=U, V=V)
dissipation = ViscousDissipation(model)

tke_budget_statistics = turbulent_kinetic_energy_budget(model, b=b, p=p, U=U, V=V, e=e,
                                                        shear_production=shear_production, dissipation=dissipation)

# FIXME: This 3D kernel actually wastes a lot of computation since we just need a 1D kernel.
# See: https://github.com/CliMA/LESbrary.jl/issues/114
Ri_kcf = KernelComputedField(Center, Center, Face, richardson_number_ccf!, model,
                               computed_dependencies=(U, V, B), parameters=(dUdz_bg=0, dVdz_bg=0, N2_bg=0))

Ri = AveragedField(Ri_kcf, dims=(1, 2))

dynamics_statistics = Dict(:Ri => Ri)

fields_to_output = merge(model.velocities, model.tracers, (e=e, ϵ=dissipation))

statistics_to_output = merge(primitive_statistics, subfilter_flux_statistics, tke_budget_statistics, dynamics_statistics)

@info "Garnishing output writers..."

# Code credit: https://discourse.julialang.org/t/collecting-all-output-from-shell-commands/15592
function execute(cmd::Cmd)
    out, err = Pipe(), Pipe()

    process = run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
    close(out.in)
    close(err.in)

    return (stdout = out |> read |> String, stderr = err |> read |> String, code = process.exitcode)
end

global_attributes = (
    LESbrary_jl_commit_SHA1 = execute(`git rev-parse HEAD`).stdout |> strip,
    name = name,
    thermocline_type = thermocline_type,
    buoyancy_flux = Qᵇ,
    momentum_flux = Qᵘ,
    temperature_flux = Qᶿ,
    coriolis_parameter = f,
    thermal_expansion_coefficient = α,
    gravitational_acceleration = g,
    boundary_condition_θ_top = Qᶿ,
    boundary_condition_θ_bottom = dθdz_deep,
    boundary_condition_u_top = Qᵘ,
    boundary_condition_u_bottom = 0.0,
    surface_layer_depth = surface_layer_depth,
    thermocline_width = thermocline_width,
    N²_surface_layer = N²_surface_layer,
    N²_thermocline = N²_thermocline,
    N²_deep = N²_deep,
    dθdz_surface_layer = dθdz_surface_layer,
    dθdz_thermocline = dθdz_thermocline,
    dθdz_deep = dθdz_deep,
    θ_surface = θ_surface,
    θ_transition = θ_transition,
    θ_deep = θ_deep,
    z_transition = z_transition,
    z_deep = z_deep,
    k_transition = k_transition,
    k_deep = k_deep
)

function init_save_some_metadata!(file, model)
    for (name, value) in pairs(global_attributes)
        file["parameters/$(string(name))"] = value
    end

    file["timeseries/u/serialized/location"] = (Nothing, Nothing, Center)
    file["timeseries/v/serialized/location"] = (Nothing, Nothing, Center)
    file["timeseries/T/serialized/location"] = (Nothing, Nothing, Center)
    file["timeseries/uw/serialized/location"] = (Nothing, Nothing, Face)
    file["timeseries/vw/serialized/location"] = (Nothing, Nothing, Face)
    file["timeseries/wT/serialized/location"] = (Nothing, Nothing, Face)

    file["timeseries/u/serialized/boundary_conditions"] = nothing
    file["timeseries/v/serialized/boundary_conditions"] = nothing
    file["timeseries/T/serialized/boundary_conditions"] = nothing
    file["timeseries/uw/serialized/boundary_conditions"] = nothing
    file["timeseries/vw/serialized/boundary_conditions"] = nothing
    file["timeseries/wT/serialized/boundary_conditions"] = nothing

    return nothing
end

## Add JLD2 output writers

using Oceananigans.Fields: interior_copy

struct HackyHorizontalAverage{F}
    field :: F
end

function (hha::HackyHorizontalAverage)(model)
    f = hha.field
    F = AveragedField(f, dims=(1, 2))
    interior(F) .= mean(interior_copy(f), dims=(1, 2))
    return F.data.parent
end

struct HackyProductHorizontalAverage{F, G}
    field1 :: F
    field2 :: G
end

function (hpha::HackyProductHorizontalAverage)(model)
    f = hpha.field1
    g = hpha.field2

    fg = ComputedField(f * g)
    compute!(fg)

    FG = AveragedField(f * g, dims=(1, 2))
    interior(FG) .= mean(interior_copy(fg), dims=(1, 2))
    return FG.data.parent
end

u, v, w = model.velocities
T = model.tracers.T

statistics_to_output = (
    u = HackyHorizontalAverage(u),
    v = HackyHorizontalAverage(v),
    T = HackyHorizontalAverage(T),
    uw = HackyProductHorizontalAverage(u, w),
    vw = HackyProductHorizontalAverage(v, w),
    wT = HackyProductHorizontalAverage(w, T)
)

simulation.output_writers[:stats_with_halos] =
    JLD2OutputWriter(model, statistics_to_output,
                              dir = data_directory,
                           prefix = "instantaneous_statistics_with_halos",
                         schedule = TimeInterval(snapshot_time_interval),
                     field_slicer = FieldSlicer(with_halos=true),
                            force = force,
                             init = init_save_some_metadata!)

# # Run

@info "Teaching simulation to run!..."

run!(simulation)
