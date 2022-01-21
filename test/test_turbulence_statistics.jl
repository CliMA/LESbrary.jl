include("runtests_preamble.jl")

using LESbrary.TurbulenceStatistics:
    horizontally_averaged_tracers,
    velocity_covariances,
    tracer_covariances,
    third_order_velocity_statistics,
    third_order_tracer_statistics,
    first_order_statistics,
    second_order_statistics,
    third_order_statistics,
    first_through_second_order,
    first_through_third_order,
    subfilter_momentum_fluxes,
    subfilter_tracer_fluxes,
    turbulent_kinetic_energy_budget

using Oceananigans.Fields: XYReducedField

function output_works(simulation, output, output_name="")
    model = simulation.model
    model.clock.time = 0
    model.clock.iteration = 0
    simulation.stop_iteration = 1

    simulation.output_writers[:test] = JLD2OutputWriter(model, output,
                                                        schedule = IterationInterval(1),
                                                        prefix = "test",
                                                        dir = ".")

    success = try
        run!(simulation)
        true
    catch err
        @warn "Output test for $output_name failed with " * sprint(showerror, err)
        false
    finally
        rm("test.jld2")
        pop!(simulation.output_writers, :test)
    end

    return success
end

for arch in architectures
    @testset "Turbulence Statistics [$(typeof(arch))]" begin
        @info "Testing turbulence statistics [$(typeof(arch))]..."

        model = NonhydrostaticModel(grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1)),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    closure = AnisotropicMinimumDissipation())

        simulation = Simulation(model, Δt=1.0, stop_iteration=1)

        C  = horizontally_averaged_tracers(model)
        u² = velocity_covariances(model)
        c² = tracer_covariances(model)
        u³ = third_order_velocity_statistics(model)
        u³ = third_order_tracer_statistics(model)

        ψ¹ = first_order_statistics(model)
        ψ² = second_order_statistics(model)
        ψ³ = third_order_statistics(model)

        ψ¹_ψ² = first_through_second_order(model)
        ψ¹_ψ³ = first_through_third_order(model)

        Qᵘ = subfilter_momentum_fluxes(model)
        Qᶜ = subfilter_tracer_fluxes(model)

        tke_budget = turbulent_kinetic_energy_budget(model)

        @test all(ϕ isa XYReducedField for ϕ in values( C     ))
        @test all(ϕ isa XYReducedField for ϕ in values( u²    ))
        @test all(ϕ isa XYReducedField for ϕ in values( c²    ))
        @test all(ϕ isa XYReducedField for ϕ in values( u³    ))
        @test all(ϕ isa XYReducedField for ϕ in values( u³    ))

        @test all(ϕ isa XYReducedField for ϕ in values( ψ¹    ))
        @test all(ϕ isa XYReducedField for ϕ in values( ψ²    ))
        @test all(ϕ isa XYReducedField for ϕ in values( ψ³    ))

        @test all(ϕ isa XYReducedField for ϕ in values( ψ¹_ψ² ))
        @test all(ϕ isa XYReducedField for ϕ in values( ψ¹_ψ³ ))

        @test all(ϕ isa XYReducedField for ϕ in values( Qᵘ    ))
        @test all(ϕ isa XYReducedField for ϕ in values( Qᶜ    ))

        @test output_works(simulation, C, "Horizontally averaged tracers")
        @test output_works(simulation, u², "Velocity covariances")
        @test output_works(simulation, c², "Tracer covariances")
        @test output_works(simulation, ψ¹, "First-order statistics")

        # Individually test outputs for better failure messages
        for (name, output) in ψ³
            @test output_works(simulation, Dict(name => output), name)
        end

        for (name, output) in tke_budget
            @test output_works(simulation, Dict(name => output), name)
        end

        for (name, output) in Qᵘ
            @test output_works(simulation, Dict(name => output), name)
        end

        for (name, output) in Qᶜ
            @test output_works(simulation, Dict(name => output), name)
        end
    end
end
