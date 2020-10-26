using Test
using Printf
using LESbrary
using Oceananigans
using Oceananigans.Fields
using Oceananigans.OutputWriters

architectures = (CPU(), GPU())

function run_script(replace_strings, script_name, script_filepath, module_suffix="")
    file_content = read(script_filepath, String)
    test_script_filepath = script_name * "_test.jl"

    for strs in replace_strings
        new_file_content = replace(file_content, strs[1] => strs[2])

        if new_file_content == file_content
            @warn "$(strs[1]) => $(strs[2]) replacement not found in $script_filepath."
            return false
        else
            file_content = new_file_content
        end

    end

    open(test_script_filepath, "w") do f
        # Wrap test script inside module to avoid polluting namespaces
        write(f, "module _Test_$script_name" * "_$module_suffix\n")
        write(f, file_content)
        write(f, "\nend # module")
    end

    try
        include(test_script_filepath)
    catch err
        @warn "Error while testing script: " * sprint(showerror, err)

        # Print the content of the file to the test log, with line numbers, for debugging
        test_file_content = read(test_script_filepath, String)
        delineated_file_content = split(test_file_content, '\n')
        for (number, line) in enumerate(delineated_file_content)
            @printf("% 3d %s\n", number, line)
        end

        rm(test_script_filepath)

        return false
    end

    # Delete the test script (if it hasn't been deleted already)
    rm(test_script_filepath)

    return true
end

function output_works(simulation, output, output_name="")
    model = simulation.model
    model.clock.time = 0
    model.clock.iteration = 0
    simulation.stop_iteration = 1

    simulation.output_writers[:test] = JLD2OutputWriter(model, output,
                                                        schedule=IterationInterval(1),
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

@testset "Turbulence Statistics" begin

    using LESbrary.TurbulenceStatistics: pressure
    using LESbrary.TurbulenceStatistics: subfilter_viscous_dissipation
    using LESbrary.TurbulenceStatistics: horizontally_averaged_tracers
    using LESbrary.TurbulenceStatistics: velocity_covariances
    using LESbrary.TurbulenceStatistics: tracer_covariances
    using LESbrary.TurbulenceStatistics: third_order_velocity_statistics
    using LESbrary.TurbulenceStatistics: third_order_tracer_statistics
    using LESbrary.TurbulenceStatistics: first_order_statistics
    using LESbrary.TurbulenceStatistics: second_order_statistics
    using LESbrary.TurbulenceStatistics: third_order_statistics
    using LESbrary.TurbulenceStatistics: first_through_second_order
    using LESbrary.TurbulenceStatistics: first_through_third_order

    for arch in (GPU(),) #architectures
        model = IncompressibleModel(architecture = arch,
                                    grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1)),
                                    closure = AnisotropicMinimumDissipation())

        @test pressure(model) isa Oceananigans.AbstractOperations.BinaryOperation
        @test subfilter_viscous_dissipation(model) isa Oceananigans.AbstractOperations.AbstractOperation

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

        @test all(ϕ isa AveragedField for ϕ in values( C     ))
        @test all(ϕ isa AveragedField for ϕ in values( u²    ))
        @test all(ϕ isa AveragedField for ϕ in values( c²    ))
        @test all(ϕ isa AveragedField for ϕ in values( u³    ))
        @test all(ϕ isa AveragedField for ϕ in values( u³    ))
                                                          
        @test all(ϕ isa AveragedField for ϕ in values( ψ¹    ))
        @test all(ϕ isa AveragedField for ϕ in values( ψ²    ))
        @test all(ϕ isa AveragedField for ϕ in values( ψ³    ))
                                                          
        @test all(ϕ isa AveragedField for ϕ in values( ψ¹_ψ² ))
        @test all(ϕ isa AveragedField for ϕ in values( ψ¹_ψ³ ))

        simulation = Simulation(model, Δt=1.0, stop_iteration=1)

        @test output_works(simulation, C, "Horizontally averaged tracers")
        @test output_works(simulation, u², "Velocity covariances")
        @test output_works(simulation, c², "Tracer covariances")
        @test output_works(simulation, ψ¹, "First-order statistics")

        for (name, output) in ψ³
            @test output_works(simulation, Dict(name => output), name)
        end

        tke_budget = LESbrary.TurbulenceStatistics.turbulent_kinetic_energy_budget(model)

        for (name, output) in tke_budget
            @test output_works(simulation, Dict(name => output), name)
        end
    end
end

@testset "Examples" begin

    #####
    ##### Free convection example
    #####
    
    free_convection_example = joinpath(@__DIR__, "..", "examples", "free_convection.jl")

    replace_strings = [
                       ("size=(32, 32, 32)", "size=(1, 1, 1)"),
                       ("TimeInterval(4hour)", "TimeInterval(2minute)"),
                       ("AveragedTimeInterval(1hour, window=15minute)", "AveragedTimeInterval(2minute, window=1minute)"),
                       ("iteration_interval=100", "iteration_interval=1"),
                       ("stop_time=8hour", "stop_time=2minute")
                      ]

    @test run_script(replace_strings, "free_convection", free_convection_example)

    push!(replace_strings, ("CPU()", "GPU()"))

    @test_skip run_script(replace_strings, "free_convection", free_convection_example)

    #####
    ##### Three layer constant fluxes example
    #####
    
    three_layer_constant_fluxes_example = joinpath(@__DIR__, "..", "examples", "three_layer_constant_fluxes.jl")

    replace_strings = [
                       ("""Nh = args["Nh"]""", "Nh = 1"),
                       ("TimeInterval(4hour)", "TimeInterval(2minute)"),
                       ("schedule = AveragedTimeInterval(3hour, window=30minute)", "schedule = AveragedTimeInterval(2minute, window=1minute)"),
                       ("stop_time=4hour", "stop_time=2minute")
                      ]

    @test_skip run_script(replace_strings, "three_layer_constant_fluxes", three_layer_constant_fluxes_example)
end
