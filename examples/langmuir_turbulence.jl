# # Langmuir turbulence
#
# This script reproduces the Langmuir turbulence simulation reported in section 4 of
#
# [McWilliams, J. C. et al., "Langmuir Turbulence in the ocean," Journal of Fluid Mechanics (1997)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/langmuir-turbulence-in-the-ocean/638FD0E368140E5972144348DB930A38).


using Pkg

Pkg.instantiate()

using Printf
using JLD2

## https://discourse.julialang.org/t/unable-to-display-plot-using-the-repl-gks-errors/12826/18
ENV["GKSwstype"] = "nul"
using Plots

using ArgParse

using Oceananigans
using Oceananigans.Units
using Oceananigans.BuoyancyModels: g_Earth

using LESbrary
using LESbrary.TurbulenceStatistics: TurbulentKineticEnergy, ViscousDissipation
using LESbrary.TurbulenceStatistics: first_through_second_order, turbulent_kinetic_energy_budget
using Oceanostics.TKEBudgetTerms: ZShearProduction

"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--Nh"
            help = "The number of grid points in x, y."
            default = 32
            arg_type = Int

        "--Nz"
            help = "The number of grid points in z."
            default = 32
            arg_type = Int

        "--Lh"
            help = "The physical extent of the domain in x, y."
            default = 128.0
            arg_type = Float64

        "--Lz"
            help = "The physical extent of the domain in z."
            default = 96.0
            arg_type = Float64

        "--stop-hours"
            help = "Number of hours to run the simulation for"
            default = 12.0
            arg_type = Float64

        "--advection-scheme"
            help = "Advection scheme: CenteredSecondOrder, WENO5, etc"
            default = :WENO5
            arg_type = Symbol

        "--pickup"
            help = "Whether or not to pick the simulation up from latest checkpoint"
            default = false
            arg_type = Bool
    end

    return parse_args(settings)
end

args = parse_command_line_arguments()

#####
##### Parameters
#####

Nh = args["Nh"]
Nz = args["Nz"]
Lh = args["Lh"]
Lz = args["Lz"]

f = 1e-4
Qᵘ = -3.72e-5 # m² s⁻²
Qᵇ = 2.307e-9 # m³ s⁻²
N² = 1.936e-5 # s⁻²

initial_mixed_layer_depth = 33 # m

inertial_period = 2π / f

stop_time = args["stop-hours"] * hours
snapshot_time_interval = 10minutes
averages_time_interval = 24hours
averages_time_window = inertial_period
averages_stride = 100
slice_depth = 8.0

amplitude = 0.8 # m

const wavenumber = 2π / 60 # m⁻¹
const Uˢ = amplitude^2 * wavenumber * sqrt(g_Earth * wavenumber) # m s⁻¹

uˢ(z) = Uˢ * exp(2wavenumber * z)
@inline ∂z_uˢ(z, t) = 2wavenumber * Uˢ * exp(2wavenumber * z)

#####
##### Grid
#####

grid = RectilinearGrid(CPU(), size=(Nh, Nh, Nz), extent=(Lh, Lh, Lz))

#####
##### Boundary conditions
#####

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ),
                                bottom = GradientBoundaryCondition(N²))

#####
##### Sponge layer
#####

gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=1/hour, mask=gaussian_mask)

b_sponge = Relaxation(rate = 1/hour,
                      target = LinearTarget{:z}(intercept=0.0, gradient=N²),
                      mask = gaussian_mask)

#####
##### Model setup
#####

advection = eval(args["advection-scheme"])()

model = NonhydrostaticModel(; advection, grid,
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            coriolis = FPlane(f=f),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_bcs, b=b_bcs),
                            forcing = (u=u_sponge, v=v_sponge, w=w_sponge, b=b_sponge))

#####
##### Initial conditions: Stokes drift + stratification + noise
#####

Ξ(z) = randn() * exp(z / 4)

stratification(z) = z < - initial_mixed_layer_depth ? N² * z : - N² * initial_mixed_layer_depth

bᵢ(x, y, z) = stratification(z)     + 1e-2 * Ξ(z) * N² * model.grid.Lz
uᵢ(x, y, z) = uˢ(z) + sqrt(abs(Qᵘ)) * 1e-2 * Ξ(z)
wᵢ(x, y, z) =         sqrt(abs(Qᵘ)) * 1e-2 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

#####
##### Simulation setup
#####

simulation = Simulation(model; Δt=1.0, stop_time)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=30.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(simulation)
    model = simulation.model

    umax = maximum(abs, model.velocities.u)
    vmax = maximum(abs, model.velocities.v)
    wmax = maximum(abs, model.velocities.w)

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(simulation),
                   prettytime(simulation.Δt),
                   umax, vmax, wmax,
                   prettytime(1e-9 * (time_ns() - wall_clock[1]))
                  )

    wall_clock[1] = time_ns()

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(10))

#####
##### Output setup
#####

prefix = @sprintf("langmuir_turbulence_Nx%d_Nz%d", grid.Nx, grid.Nz)
data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# "Primitive" statistics

b = BuoyancyField(model)
p = model.pressures.pHY′ + model.pressures.pNHS
w_scratch = ZFaceField(model.grid)
c_scratch = CenterField(model.grid)

primitive_statistics = first_through_second_order(model, b=b, p=p, w_scratch=w_scratch, c_scratch=c_scratch)

U = primitive_statistics[:u]
V = primitive_statistics[:v]

# Turbulent kinetic energy budget terms

e = TurbulentKineticEnergy(model, U=U, V=V)
shear_production = ZShearProduction(model, U=U, V=V)
dissipation = ViscousDissipation(model)

tke_budget_statistics = turbulent_kinetic_energy_budget(model, b=b, p=p, U=U, V=V, e=e,
                                                        shear_production=shear_production, dissipation=dissipation)

statistics_to_output = merge(primitive_statistics, tke_budget_statistics)

fields_to_output = merge(model.velocities, model.tracers,
                         (νₑ = model.diffusivity_fields.νₑ,
                           e = Field(e),
                          sp = Field(shear_production),
                           ϵ = Field(dissipation)))

# Output configured for pickup

pickup = args["pickup"]
force = pickup ? false : true

k_xy_slice = searchsortedfirst(znodes(Face, grid), -slice_depth)

simulation.output_writers[:checkpointer] =
    Checkpointer(model, schedule = TimeInterval(26hour), prefix = prefix * "_checkpointer", dir = data_directory)

simulation.output_writers[:xz] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_xz",
                     field_slicer = FieldSlicer(j=1),
                              dir = data_directory,
                            force = force)

simulation.output_writers[:yz] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_yz",
                     field_slicer = FieldSlicer(i=1),
                              dir = data_directory,
                            force = force)

simulation.output_writers[:xy] =
    JLD2OutputWriter(model, fields_to_output,
                         schedule = TimeInterval(snapshot_time_interval),
                           prefix = prefix * "_xy",
                     field_slicer = FieldSlicer(k=k_xy_slice),
                              dir = data_directory,
                            force = force)

simulation.output_writers[:statistics] =
    JLD2OutputWriter(model, statistics_to_output,
                     schedule = TimeInterval(snapshot_time_interval),
                       prefix = prefix * "_statistics",
                          dir = data_directory,
                        force = force)

simulation.output_writers[:averaged_statistics] =
    JLD2OutputWriter(model, statistics_to_output,
                     schedule = AveragedTimeInterval(averages_time_interval,
                                                     window = averages_time_window,
                                                     stride = averages_stride),
                       prefix = prefix * "_averaged_statistics",
                          dir = data_directory,
                        force = force)

#####
##### Run
#####

run!(simulation)

#####
##### Animations and analysis
#####

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)
xc, yc, zc = nodes((Center, Center, Center), grid)

xyfile = jldopen(joinpath(data_directory, prefix * "_xy.jld2"))
xzfile = jldopen(joinpath(data_directory, prefix * "_xz.jld2"))

iterations = parse.(Int, keys(xyfile["timeseries/t"]))

function divergent_levels(c, clim, nlevels=31)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function sequential_levels(c, clims, nlevels=31)
    levels = collect(range(clims[1], stop=clims[2], length=nlevels))
    cmin = minimum(c)
    cmax = maximum(c)
    cmin < clims[1] && pushfirst!(levels, cmin)
    cmax > clims[2] && push!(levels, cmax)
    return clims, levels
end

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    ## Load 3D fields from file
    t = xyfile["timeseries/t/$iter"]
    wxy = xyfile["timeseries/w/$iter"][:, :, 1]
    uxz = xzfile["timeseries/u/$iter"][:, 1, :]
    exz = xzfile["timeseries/e/$iter"][:, 1, :]

    wlims, wlevels = divergent_levels(wxy, 0.02)
    ulims, ulevels = divergent_levels(uxz, 0.1)
    elims, elevels = sequential_levels(exz, (0, 1e-4))

    wxy_plot = contourf(xw, yw, clamp.(wxy, wlims[1], wlims[2])';
                              color = :balance,
                          linewidth = 0,
                        aspectratio = :equal,
                              clims = wlims,
                             levels = wlevels,
                              xlims = (0, grid.Lx),
                              ylims = (0, grid.Ly),
                             xlabel = "x (m)",
                             ylabel = "y (m)")

    uxz_plot = contourf(xu, zu, clamp.(uxz, ulims[1], ulims[2])';
                              color = :balance,
                          linewidth = 0,
                        aspectratio = :equal,
                              clims = ulims,
                             levels = ulevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    exz_plot = contourf(xc, zc, clamp.(exz, elims[1], elims[2])';
                              color = :thermal,
                          linewidth = 0,
                        aspectratio = :equal,
                              clims = elims,
                             levels = elevels,
                              xlims = (0, grid.Lx),
                              ylims = (-grid.Lz, 0),
                             xlabel = "x (m)",
                             ylabel = "z (m)")

    w_title = @sprintf("w(z=0, t=%s) (m s⁻¹)", prettytime(t))
    u_title = @sprintf("u(y=0, t=%s) (m s⁻¹)", prettytime(t))
    e_title = @sprintf("e(y=0, t=%s) (m s⁻¹)", prettytime(t))

    plot(wxy_plot, uxz_plot, exz_plot, layout=(3, 1), size=(1200, 1000),
         title = [w_title u_title e_title])

    if iter == iterations[end]
        close(xyfile)
        close(xzfile)
    end
end

mp4(anim, prefix * ".mp4", fps = 8)

##### Turbulent kinetic energy budget analysis

xu, yu, zu = nodes(model.velocities.u)

statistics_file = jldopen(joinpath(data_directory, prefix * "_averaged_statistics.jld2"))
statistics_iterations = parse.(Int, keys(statistics_file["timeseries/t"]))

last_iteration = statistics_iterations[end]

U = statistics_file["timeseries/u/$last_iteration"][1, 1, :]
V = statistics_file["timeseries/v/$last_iteration"][1, 1, :]
E = statistics_file["timeseries/e/$last_iteration"][1, 1, :]

# Budget terms
shear_production = statistics_file["timeseries/tke_shear_production/$last_iteration"][1, 1, :]
dissipation = statistics_file["timeseries/tke_dissipation/$last_iteration"][1, 1, :]
buoyancy_flux = statistics_file["timeseries/tke_buoyancy_flux/$last_iteration"][1, 1, :]
advective_flux = statistics_file["timeseries/tke_advective_flux/$last_iteration"][1, 1, :]
pressure_flux = statistics_file["timeseries/tke_pressure_flux/$last_iteration"][1, 1, :]

close(statistics_file)

Δz = grid.Δzᵃᵃᶜ
pressure_flux_divergence = @. (pressure_flux[2:end] - pressure_flux[1:end-1]) ./ Δz
advective_flux_divergence = @. (advective_flux[2:end] - advective_flux[1:end-1]) ./ Δz

tendency = @. - pressure_flux_divergence - advective_flux_divergence + shear_production + buoyancy_flux - dissipation

kwargs = (linewidth=2, alpha=0.8, ylims=(-48, 0))

velocities_plot = plot(U, zu; label="\$ U \$", legend=:bottomleft, xlabel="Velocities (m s⁻¹)", ylabel="z (m)", kwargs...)
plot!(velocities_plot, V, zu; label="\$ V \$", kwargs...)
plot!(velocities_plot, sqrt.(E), zu; label="\$ \\sqrt{E} \$", kwargs...)

budget_plot = plot(-dissipation, zu; label="dissipation", legend=:bottomleft, xlabel="Turbulent kinetic energy budget", ylabel="z (m)", kwargs...)
plot!(budget_plot, shear_production, zu; label="shear production", kwargs...)
plot!(budget_plot, buoyancy_flux, zu; label="buoyancy flux", kwargs...)
plot!(budget_plot, -pressure_flux_divergence, zu; label="pressure flux divergence", kwargs...)
plot!(budget_plot, -advective_flux_divergence, zu; label="advective flux divergence", kwargs...)
plot!(budget_plot, tendency, zu; label="tendency", linestyle=:dash, kwargs...)

tendency_plot = plot(tendency, zu; xlabel="TKE tendency", ylabel="z (m)", legend=nothing, linestyle=:dash, kwargs...)

statistics_plot = plot(velocities_plot, budget_plot, tendency_plot,
                       layout = (1, 3), size = (1000, 600))

savefig(statistics_plot, prefix * "_statistics.png")
