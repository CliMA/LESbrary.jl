# # Turbulent mixing of a linearly stratified boundary layer driven by constant surface fluxes

# This script runs a simulation of a turbulent oceanic boundary layer with an initially
# linear temperature stratification. Turbulent mixing is driven by constant fluxes
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
using NCDatasets
using GeoData
using CairoMakie

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

using Oceananigans.Grids: Face, Center
using Oceananigans.Fields: PressureField
using Oceanostics.FlowDiagnostics: richardson_number_ccf!

using LESbrary.Utils: SimulationProgressMessenger, fit_cubic, poly
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant
using LESbrary.TurbulenceStatistics: first_through_second_order, turbulent_kinetic_energy_budget,
                                     subfilter_momentum_fluxes, subfilter_tracer_fluxes,
                                     TurbulentKineticEnergy, ShearProduction, ViscousDissipation

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

        "--buoyancy-gradient"
            help = "The initial buoyancy gradient in units s⁻²."
            default = 1e-6
            arg_type = Float64

        "--hours"
            help = "Number of hours to run the simulation for"
            default = 0.1
            arg_type = Float64

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
N² = args["buoyancy-gradient"]

prefix = @sprintf("linear_stratification_constant_fluxes_hr%d_Qu%.1e_Qb%.1e_f%.1e_Nh%d_Nz%d_",
                  stop_hours, abs(Qᵘ), Qᵇ, f, Nx, Nz)
data_directory = joinpath(@__DIR__, "..", "data", prefix * name) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# Save command line arguments used to an executable bash script
open(joinpath(data_directory, "run_linear_stratification_constant_fluxes.sh"), "w") do io
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

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))

# Buoyancy and boundary conditions

@info "Enforcing boundary conditions..."

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

dθdz = N² / (α * g)
Qᶿ = Qᵇ / (α * g)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

# Tracer forcing

@info "Forcing and sponging tracers..."

# # Initial condition and sponge layer

# # Set Initial condition
θ_surface = args["surface-temperature"]

## Noise with 8 m decay scale

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
                      target = LinearTarget{:z}(intercept=θ_surface, gradient=dθdz),
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
                # closure = IsotropicDiffusivity(ν=1e-2, κ=1e-5),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = (T=θ_bcs, u=u_bcs),
                forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge,
                           c₀=c₀_forcing, c₁=c₁_forcing, c₂=c₂_forcing)
)

# # Set Initial condition

@info "Setting initial conditions..."

Ξ(z) = rand() * exp(z / 8)
initial_temperature(x, y, z) = θ_surface + dθdz * z + 1e-6 * Ξ(z) * dθdz * grid.Lz     
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
              progress = SimulationProgressMessenger(model, wizard)
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

e = TurbulentKineticEnergy(model, U=U, V=V)
shear_production = ShearProduction(model, data=ccc_scratch.data, U=U, V=V)
dissipation = ViscousDissipation(model, data=ccc_scratch.data)

tke_budget_statistics = turbulent_kinetic_energy_budget(model, b=b, p=p, U=U, V=V, e=e,
                                                        shear_production=shear_production, dissipation=dissipation)

Ri_local = KernelComputedField(Center, Center, Face, richardson_number_ccf!, model,
                               field_dependencies=(U, V, b), parameters=(dUdz_bg=0, dVdz_bg=0, N2_bg=0))

Ri = AveragedField(Ri_local, dims=(1, 2))

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
    buoyancy_flux = Qᵇ,
    momentum_flux = Qᵘ,
    temperature_flux = Qᶿ,
    coriolis_parameter = f,
    thermal_expansion_coefficient = α,
    gravitational_acceleration = g,
    boundary_condition_θ_top = Qᶿ,
    boundary_condition_θ_bottom = dθdz,
    boundary_condition_u_top = Qᵘ,
    boundary_condition_u_bottom = 0.0,
    dθdz = dθdz,
    θ_surface = θ_surface,
)

function init_save_some_metadata!(file, model)
    for (name, value) in pairs(global_attributes)
        file["parameters/$(string(name))"] = value
    end
    return nothing
end

## Add JLD2 output writers

simulation.output_writers[:xy_jld2] =
    JLD2OutputWriter(model, fields_to_output,
                              dir = data_directory,
                           prefix = "xy_slice",
                         schedule = TimeInterval(snapshot_time_interval),
                     field_slicer = FieldSlicer(k=k_xy_slice),
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:xz_jld2] =
    JLD2OutputWriter(model, fields_to_output,
                              dir = data_directory,
                           prefix = "xz_slice",
                         schedule = TimeInterval(snapshot_time_interval),
                     field_slicer = FieldSlicer(j=1),
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:yz_jld2] =
    JLD2OutputWriter(model, fields_to_output,
                              dir = data_directory,
                           prefix = "yz_slice",
                         schedule = TimeInterval(snapshot_time_interval),
                     field_slicer = FieldSlicer(i=1),
                            force = force,
                             init = init_save_some_metadata!)

simulation.output_writers[:statistics_jld2] =
    JLD2OutputWriter(model, statistics_to_output,
                          dir = data_directory,
                       prefix = "instantaneous_statistics",
                     schedule = TimeInterval(snapshot_time_interval),
                        force = force,
                         init = init_save_some_metadata!)

simulation.output_writers[:averaged_statistics_jld2] =
    JLD2OutputWriter(model, statistics_to_output,
                          dir = data_directory,
                       prefix = "time_averaged_statistics",
                     schedule = AveragedTimeInterval(averages_time_interval,
                                                     window = averages_time_window),
                        force = force,
                         init = init_save_some_metadata!)

## Add NetCDF output writers

statistics_to_output = Dict(string(k) => v for (k, v) in statistics_to_output)

simulation.output_writers[:xy_nc] =
    NetCDFOutputWriter(model, fields_to_output,
                     mode = "c",
                 filepath = joinpath(data_directory, "xy_slice.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
             field_slicer = FieldSlicer(k=k_xy_slice),
        global_attributes = global_attributes)

simulation.output_writers[:xz_nc] =
    NetCDFOutputWriter(model, fields_to_output,
                     mode = "c",
                 filepath = joinpath(data_directory, "xz_slice.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
             field_slicer = FieldSlicer(j=1),
        global_attributes = global_attributes)

simulation.output_writers[:yz_nc] =
    NetCDFOutputWriter(model, fields_to_output,
                     mode = "c",
                 filepath = joinpath(data_directory, "yz_slice.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
             field_slicer = FieldSlicer(i=1),
        global_attributes = global_attributes)

simulation.output_writers[:statistics_nc] =
    NetCDFOutputWriter(model, statistics_to_output,
                     mode = "c",
                 filepath = joinpath(data_directory, "instantaneous_statistics.nc"),
                 schedule = TimeInterval(snapshot_time_interval),
        global_attributes = global_attributes)

simulation.output_writers[:averaged_statistics_nc] =
    NetCDFOutputWriter(model, statistics_to_output,
                     mode = "c",
                 filepath = joinpath(data_directory, "time_averaged_statistics.nc"),
                 schedule = AveragedTimeInterval(averages_time_interval, window = averages_time_window),
        global_attributes = global_attributes)

# # Run

@info "Teaching simulation to run!..."

run!(simulation)

# # Load and plot turbulence statistics

make_animation = args["animation"]

function squeeze(A)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims=singleton_dims)
end

if make_animation
    ds_xy = NCDstack(joinpath(data_directory, "xy_slice.nc"))
    ds_xz = NCDstack(joinpath(data_directory, "xz_slice.nc"))

    times = ds_xy[:time]
    Nt = length(times)

    xc = ds_xy[:xC].data
    xf = ds_xy[:xF].data
    yc = ds_xy[:yC].data
    yf = ds_xy[:yF].data
    zc = ds_xz[:zC].data
    zf = ds_xz[:zF].data

    fig = Figure(resolution=(3000, 750))

    u_max = max(maximum(abs, ds_xy[:u]), maximum(abs, ds_xz[:u]))
    v_max = max(maximum(abs, ds_xy[:v]), maximum(abs, ds_xz[:v]))
    w_max = max(maximum(abs, ds_xy[:w]), maximum(abs, ds_xz[:w]))
    U_max = max(u_max, v_max, w_max)
    U_lims = 0.5 .* (-U_max, +U_max)

    frame = Node(1)

    plot_title = @lift "LESbrary.jl linear stratification constant fluxes $name: time = $(prettytime(times[$frame]))"

    u_xy = @lift ds_xy[:u][Ti=$frame].data |> squeeze
    v_xy = @lift ds_xy[:v][Ti=$frame].data |> squeeze
    w_xy = @lift ds_xy[:w][Ti=$frame].data |> squeeze
    T_xy = @lift ds_xy[:T][Ti=$frame].data |> squeeze
    c₀_xy = @lift ds_xy[:c₀][Ti=$frame].data |> squeeze
    c₁_xy = @lift ds_xy[:c₁][Ti=$frame].data |> squeeze
    c₂_xy = @lift ds_xy[:c₂][Ti=$frame].data |> squeeze
    e_xy = @lift ds_xy[:e][Ti=$frame].data |> squeeze .|> log10
    ϵ_xy = @lift ds_xy[:ϵ][Ti=$frame].data |> squeeze .|> log10

    u_xz = @lift ds_xz[:u][Ti=$frame].data |> squeeze
    v_xz = @lift ds_xz[:v][Ti=$frame].data |> squeeze
    w_xz = @lift ds_xz[:w][Ti=$frame].data |> squeeze
    T_xz = @lift ds_xz[:T][Ti=$frame].data |> squeeze
    c₀_xz = @lift ds_xz[:c₀][Ti=$frame].data |> squeeze
    c₁_xz = @lift ds_xz[:c₁][Ti=$frame].data |> squeeze
    c₂_xz = @lift ds_xz[:c₂][Ti=$frame].data |> squeeze
    e_xz = @lift ds_xz[:e][Ti=$frame].data |> squeeze .|> log10
    ϵ_xz = @lift ds_xz[:ϵ][Ti=$frame].data |> squeeze .|> log10

    ax_u_xy = fig[1, 1] = Axis(fig, title="u-velocity")
    hm_u_xy = heatmap!(ax_u_xy, xf, yc, u_xy, colormap=:balance, colorrange=U_lims)
    hidedecorations!(ax_u_xy)

    ax_v_xy = fig[1, 2] = Axis(fig, title="v-velocity")
    hm_v_xy = heatmap!(ax_v_xy, xc, yf, v_xy, colormap=:balance, colorrange=U_lims)
    hidedecorations!(ax_v_xy)

    ax_w_xy = fig[1, 3] = Axis(fig, title="w-velocity")
    hm_w_xy = heatmap!(ax_w_xy, xc, yc, w_xy, colormap=:balance, colorrange=U_lims)
    hidedecorations!(ax_w_xy)

    ax_T_xy = fig[1, 4] = Axis(fig, title="temperature")
    hm_T_xy = heatmap!(ax_T_xy, xc, yc, T_xy, colormap=:thermal, colorrange=extrema(ds_xy[:T]))
    hidedecorations!(ax_T_xy)

    ax_c₀_xy = fig[1, 5] = Axis(fig, title="tracer 0")
    hm_c₀_xy = heatmap!(ax_c₀_xy, xc, yc, c₀_xy, colormap=:ice, colorrange=extrema(ds_xy[:c₀]))
    hidedecorations!(ax_c₀_xy)

    ax_c₁_xy = fig[1, 6] = Axis(fig, title="tracer 1")
    hm_c₁_xy = heatmap!(ax_c₁_xy, xc, yc, c₁_xy, colormap=:ice, colorrange=extrema(ds_xy[:c₁]))
    hidedecorations!(ax_c₁_xy)

    ax_c₂_xy = fig[1, 7] = Axis(fig, title="tracer 2")
    hm_c₂_xy = heatmap!(ax_c₂_xy, xc, yc, c₂_xy, colormap=:ice, colorrange=extrema(ds_xy[:c₂]))
    hidedecorations!(ax_c₂_xy)

    ax_e_xy = fig[1, 8] = Axis(fig, title="log TKE")
    hm_e_xy = heatmap!(ax_e_xy, xc, yc, e_xy, colormap=:deep, colorrange=(-5, -2))
    hidedecorations!(ax_e_xy)

    ax_ϵ_xy = fig[1, 9] = Axis(fig, title="log TKE dissipation")
    hm_ϵ_xy = heatmap!(ax_ϵ_xy, xc, yc, ϵ_xy, colormap=:dense, colorrange=(-10, -5))
    hidedecorations!(ax_ϵ_xy)

    ax_u_xz = fig[2, 1] = Axis(fig)
    hm_u_xz = heatmap!(ax_u_xz, xf, zc, u_xz, colormap=:balance, colorrange=U_lims)
    hidedecorations!(ax_u_xz)

    ax_v_xz = fig[2, 2] = Axis(fig)
    hm_v_xz = heatmap!(ax_v_xz, xc, zc, v_xz, colormap=:balance, colorrange=U_lims)
    hidedecorations!(ax_v_xz)

    ax_w_xz = fig[2, 3] = Axis(fig)
    hm_w_xz = heatmap!(ax_w_xz, xc, zf, w_xz, colormap=:balance, colorrange=U_lims)
    hidedecorations!(ax_w_xz)

    ax_T_xz = fig[2, 4] = Axis(fig)
    hm_T_xz = heatmap!(ax_T_xz, xc, zc, T_xz, colormap=:thermal, colorrange=extrema(ds_xz[:T]))
    hidedecorations!(ax_T_xz)

    ax_c₀_xz = fig[2, 5] = Axis(fig)
    hm_c₀_xz = heatmap!(ax_c₀_xz, xc, zc, c₀_xz, colormap=:ice, colorrange=extrema(ds_xz[:c₀]))
    hidedecorations!(ax_c₀_xz)

    ax_c₁_xz = fig[2, 6] = Axis(fig)
    hm_c₁_xz = heatmap!(ax_c₁_xz, xc, zc, c₁_xz, colormap=:ice, colorrange=extrema(ds_xz[:c₁]))
    hidedecorations!(ax_c₁_xz)

    ax_c₂_xz = fig[2, 7] = Axis(fig)
    hm_c₂_xz = heatmap!(ax_c₂_xz, xc, zc, c₂_xz, colormap=:ice, colorrange=extrema(ds_xz[:c₂]))
    hidedecorations!(ax_c₂_xz)

    ax_e_xz = fig[2, 8] = Axis(fig)
    hm_e_xz = heatmap!(ax_e_xz, xc, zc, e_xz, colormap=:deep, colorrange=(-8, -2))
    hidedecorations!(ax_e_xz)

    ax_ϵ_xz = fig[2, 9] = Axis(fig)
    hm_ϵ_xz = heatmap!(ax_ϵ_xz, xc, zc, ϵ_xz, colormap=:dense, colorrange=(-15, -5))
    hidedecorations!(ax_ϵ_xz)

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

    filepath = joinpath(data_directory, "xy_xz_movie.mp4")
    record(fig, filepath, 1:Nt, framerate=15) do n
        @info "Animating xy/xz movie frame $n/$Nt..."
        frame[] = n
    end

    @info "Movie saved: $filepath"
end

if make_animation
    ds = NCDstack(joinpath(data_directory, "instantaneous_statistics.nc"))

    times = ds[:time]
    Nt = length(times)

    zc = ds[:zC].data
    zf = ds[:zF].data

    fig = Figure(resolution=(3000, 2000))

    frame = Node(1)

    plot_title = @lift "LESbrary.jl three-layer constant fluxes $name: time = $(prettytime(times[$frame]))"

    u = @lift ds[:u][Ti=$frame]
    v = @lift ds[:v][Ti=$frame]
    T = @lift ds[:T][Ti=$frame]
    c₀ = @lift ds[:c₀][Ti=$frame]
    c₁ = @lift ds[:c₁][Ti=$frame]
    c₂ = @lift ds[:c₂][Ti=$frame]
    uu = @lift ds[:uu][Ti=$frame]
    vv = @lift ds[:vv][Ti=$frame]
    ww = @lift ds[:ww][Ti=$frame]
    uv = @lift ds[:uv][Ti=$frame]
    wu = @lift ds[:wu][Ti=$frame]
    wv = @lift ds[:wv][Ti=$frame]
    uT = @lift ds[:uT][Ti=$frame]
    vT = @lift ds[:vT][Ti=$frame]
    wT = @lift ds[:wT][Ti=$frame]
    Ri = @lift ds[:Ri][Ti=$frame]
    νₑ_∂z_u = @lift ds[:νₑ_∂z_u][Ti=$frame]
    νₑ_∂z_v = @lift ds[:νₑ_∂z_v][Ti=$frame]
    νₑ_∂z_w = @lift ds[:νₑ_∂z_w][Ti=$frame]
    κₑ_∂z_T = @lift ds[:κₑ_∂z_T][Ti=$frame]
    κₑ_∂z_c₀ = @lift ds[:κₑ_∂z_c₀][Ti=$frame]
    κₑ_∂z_c₁ = @lift ds[:κₑ_∂z_c₁][Ti=$frame]
    κₑ_∂z_c₂ = @lift ds[:κₑ_∂z_c₂][Ti=$frame]
    e = @lift ds[:e][Ti=$frame]
    tke_dissipation = @lift ds[:tke_dissipation][Ti=$frame]
    tke_advective_flux = @lift ds[:tke_advective_flux][Ti=$frame]
    tke_buoyancy_flux = @lift ds[:tke_buoyancy_flux][Ti=$frame]
    tke_pressure_flux = @lift ds[:tke_pressure_flux][Ti=$frame]
    tke_shear_production = @lift ds[:tke_shear_production][Ti=$frame]

    colors = ["dodgerblue2", "crimson", "forestgreen"]

    ax_uv = fig[1, 1] = Axis(fig, xlabel="m/s", ylabel="z (m)")
    line_u = lines!(ax_uv, u, zc, label="U", linewidth=3, color=colors[1])
    line_v = lines!(ax_uv, v, zc, label="V", linewidth=3, color=colors[2])
    axislegend(ax_uv, position=:rb, framevisible=false)
    xlims!(ax_uv, extrema([extrema(ds[:u])..., extrema(ds[:v])...]))
    ylims!(ax_uv, extrema(zf))

    ax_T = fig[1, 2] = Axis(fig, xlabel="°C")
    line_T = lines!(ax_T, T, zc, linewidth=3, color=colors[1])
    xlims!(ax_T, extrema(ds[:T]))
    ylims!(ax_T, extrema(zf))
    hideydecorations!(ax_T, grid=false)

    ax_c = fig[1, 3] = Axis(fig, xlabel="c")
    line_c₀ = lines!(ax_c, c₀, zc, linewidth=3, label="c₀", color=colors[1])
    line_c₁ = lines!(ax_c, c₁, zc, linewidth=3, label="c₁", color=colors[2])
    line_c₂ = lines!(ax_c, c₂, zc, linewidth=3, label="c₂", color=colors[3])
    axislegend(ax_c, position=:rb, framevisible=false)
    xlims!(ax_c, extrema([extrema(ds[:c₀])..., extrema(ds[:c₁])..., extrema(ds[:c₂])...]))
    ylims!(ax_c, extrema(zf))
    hideydecorations!(ax_c, grid=false)

    ax_ke = fig[1, 4] = Axis(fig, xlabel="m²/s²")
    line_uu = lines!(ax_ke, uu, zc, linewidth=3, label="uu", color=colors[1])
    line_vv = lines!(ax_ke, vv, zc, linewidth=3, label="vv", color=colors[2])
    line_ww = lines!(ax_ke, ww, zf, linewidth=3, label="ww", color=colors[3])
    axislegend(ax_ke, position=:rb, framevisible=false)
    xlims!(ax_ke, extrema([extrema(ds[:c₀])..., extrema(ds[:c₁])..., extrema(ds[:c₂])...]))
    ylims!(ax_ke, extrema(zf))
    hideydecorations!(ax_ke, grid=false)

    ax_UΣ = fig[1, 5] = Axis(fig, xlabel="m²/s²")
    line_uv = lines!(ax_UΣ, uv, zc, linewidth=3, label="uv", color=colors[1])
    line_wu = lines!(ax_UΣ, wu, zf, linewidth=3, label="wu", color=colors[2])
    line_wv = lines!(ax_UΣ, wv, zf, linewidth=3, label="wv", color=colors[3])
    axislegend(ax_UΣ, position=:rb, framevisible=false)
    xlims!(ax_UΣ, extrema([extrema(ds[:uv])..., extrema(ds[:wu])..., extrema(ds[:wv])...]))
    ylims!(ax_UΣ, extrema(zf))
    hideydecorations!(ax_UΣ, grid=false)

    ax_UT = fig[1, 6] = Axis(fig, xlabel="m·K/s")
    line_uT = lines!(ax_UT, uT, zc, linewidth=3, label="uT", color=colors[1])
    line_vT = lines!(ax_UT, vT, zc, linewidth=3, label="vT", color=colors[2])
    line_wT = lines!(ax_UT, wT, zf, linewidth=3, label="wT", color=colors[3])
    axislegend(ax_UT, position=:rb, framevisible=false)
    xlims!(ax_UT, extrema([extrema(ds[:uT])..., extrema(ds[:vT])..., extrema(ds[:wT])...]))
    ylims!(ax_UT, extrema(zf))
    hideydecorations!(ax_UT, grid=false)

    ax_νₑ = fig[1, 7] = Axis(fig, xlabel="m²/s²")
    line_νₑ_∂z_u = lines!(ax_νₑ, νₑ_∂z_u, zf, linewidth=3, label="νₑ ∂z(u)", color=colors[1])
    line_νₑ_∂z_v = lines!(ax_νₑ, νₑ_∂z_v, zf, linewidth=3, label="νₑ ∂z(v)", color=colors[2])
    line_νₑ_∂z_w = lines!(ax_νₑ, νₑ_∂z_w, zc, linewidth=3, label="νₑ ∂z(w)", color=colors[3])
    axislegend(ax_νₑ, position=:rb, framevisible=false)
    xlims!(ax_νₑ, extrema([extrema(ds[:νₑ_∂z_u])..., extrema(ds[:νₑ_∂z_v])..., extrema(ds[:νₑ_∂z_w])...]))
    ylims!(ax_νₑ, extrema(zf))
    hideydecorations!(ax_νₑ, grid=false)

    ax_κₑ∂zT = fig[1, 8] = Axis(fig, xlabel="m⋅K/s")
    line_κₑ∂zT = lines!(ax_κₑ∂zT, κₑ_∂z_T, zf, label="κₑ ∂z(T)", linewidth=3, color=colors[1])
    axislegend(ax_κₑ∂zT, position=:rb, framevisible=false)
    xlims!(ax_κₑ∂zT, extrema(ds[:κₑ_∂z_T]))
    ylims!(ax_κₑ∂zT, extrema(zf))
    hideydecorations!(ax_κₑ∂zT, grid=false)

    ax_κₑ∂zC = fig[2, 1] = Axis(fig, xlabel="m²/s²")
    line_κₑ_∂z_c₀ = lines!(ax_κₑ∂zC, κₑ_∂z_c₀, zf, linewidth=3, label="κₑ ∂z(c₀)", color=colors[1])
    line_κₑ_∂z_c₁ = lines!(ax_κₑ∂zC, κₑ_∂z_c₂, zf, linewidth=3, label="κₑ ∂z(c₁)", color=colors[2])
    line_κₑ_∂z_c₂ = lines!(ax_κₑ∂zC, κₑ_∂z_c₂, zf, linewidth=3, label="κₑ ∂z(c₂)", color=colors[3])
    axislegend(ax_κₑ∂zC, position=:rb, framevisible=false)
    xlims!(ax_κₑ∂zC, extrema([extrema(ds[:κₑ_∂z_c₀])..., extrema(ds[:κₑ_∂z_c₂])..., extrema(ds[:κₑ_∂z_c₂])...]))
    ylims!(ax_κₑ∂zC, extrema(zf))

    ax_Ri = fig[2, 2] = Axis(fig, xlabel="Ri")
    line_Ri = lines!(ax_Ri, Ri, zf, linewidth=3, color=colors[1])
    xlims!(ax_Ri, (-0.5, 4))
    ylims!(ax_Ri, extrema(zf))
    hideydecorations!(ax_Ri, grid=false)

    ax_e = fig[2, 3] = Axis(fig, xlabel="TKE")
    line_e = lines!(ax_e, e, zc, linewidth=3, color=colors[1])
    xlims!(ax_e, extrema(ds[:e]))
    ylims!(ax_e, extrema(zf))
    hideydecorations!(ax_e, grid=false)

    ax_ϵ = fig[2, 4] = Axis(fig, xlabel="TKE dissipation")
    line_ϵ = lines!(ax_ϵ, tke_dissipation, zc, linewidth=3, color=colors[1])
    xlims!(ax_ϵ, extrema(ds[:tke_dissipation]))
    ylims!(ax_ϵ, extrema(zf))
    hideydecorations!(ax_ϵ, grid=false)

    ax_tke_a = fig[2, 5] = Axis(fig, xlabel="TKE advective flux")
    line_tke_a = lines!(ax_tke_a, tke_advective_flux, zf, linewidth=3, color=colors[1])
    xlims!(ax_tke_a, extrema(ds[:tke_advective_flux]))
    ylims!(ax_tke_a, extrema(zf))
    hideydecorations!(ax_tke_a, grid=false)

    ax_tke_b = fig[2, 6] = Axis(fig, xlabel="TKE buoyancy flux")
    line_tke_b = lines!(ax_tke_b, tke_buoyancy_flux, zc, linewidth=3, color=colors[1])
    xlims!(ax_tke_b, extrema(ds[:tke_buoyancy_flux]))
    ylims!(ax_tke_b, extrema(zf))
    hideydecorations!(ax_tke_b, grid=false)

    ax_tke_p = fig[2, 7] = Axis(fig, xlabel="TKE pressure flux")
    line_tke_p = lines!(ax_tke_p, tke_pressure_flux, zf, linewidth=3, color=colors[1])
    xlims!(ax_tke_p, extrema(ds[:tke_pressure_flux]))
    ylims!(ax_tke_p, extrema(zf))
    hideydecorations!(ax_tke_p, grid=false)

    ax_tke_sp = fig[2, 8] = Axis(fig, xlabel="TKE shear production")
    line_tke_sp = lines!(ax_tke_sp, tke_shear_production, zc, linewidth=3, color=colors[1])
    xlims!(ax_tke_sp, extrema(ds[:tke_shear_production]))
    ylims!(ax_tke_sp, extrema(zf))
    hideydecorations!(ax_tke_sp, grid=false)

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
    
    filepath = joinpath(data_directory, "instantaneous_statistics.mp4")
    record(fig, filepath, 1:Nt, framerate=15) do n
        @info "Animating instantaneous statistics movie frame $n/$Nt..."
        frame[] = n
    end

    @info "Movie saved: $filepath"
end
