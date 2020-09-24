using Test
using Printf
using LESbrary
using Oceananigans
using Oceananigans.Fields

function run_script(replace_strings, script_name, script_filepath, module_suffix="")
    file_content = read(script_filepath, String)
    test_script_filepath = script_name * "_test.jl"

    for strs in replace_strings
        new_file_content = replace(file_content, strs[1] => strs[2])

        if new_file_content == file_content
            error("$(strs[1]) => $(strs[2]) replacement not found in $script_filepath. " *
                  "Make sure the script has not changed, otherwise the test might take a long time.")
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
        @error sprint(showerror, err)

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

@testset "Turbulence Statistics" begin

    model = IncompressibleModel(grid=RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1)))

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
    using LESbrary.TurbulenceStatistics: first_through_second_order_statistics
    using LESbrary.TurbulenceStatistics: first_through_third_order_statistics

    @test pressure(model) isa Oceananigans.AbstractOperations.BinaryOperation
    @test subfilter_viscous_dissipation(model) isa Oceananigans.AbstractOperations.AbstractOperation

    @test all(ϕ for ϕ in values( horizontally_averaged_tracers(model)   ) isa AveragedField)
    @test all(ϕ for ϕ in values( velocity_covariances(model)            ) isa AveragedField)
    @test all(ϕ for ϕ in values( tracer_covariances(model)              ) isa AveragedField)
    @test all(ϕ for ϕ in values( third_order_velocity_statistics(model) ) isa AveragedField)
    @test all(ϕ for ϕ in values( third_order_tracer_statistics(model)   ) isa AveragedField)

    @test all(ϕ for ϕ in values( first_order_statistics(model)  ) isa AveragedField)
    @test all(ϕ for ϕ in values( second_order_statistics(model) ) isa AveragedField)
    @test all(ϕ for ϕ in values( third_order_statistics(model)  ) isa AveragedField)

    @test all(ϕ for ϕ in values( first_through_second_order_statistics(model) ) isa AveragedField)
    @test all(ϕ for ϕ in values( first_through_third_order_statistics(model)  ) isa AveragedField)

end

@testset "Examples" begin

    #####
    ##### Free convection example
    #####
    
    free_convection_example = joinpath(@__DIR__, "..", "examples", "free_convection.jl")

    replace_strings = [("size=(64, 64, 64)", "size=(1, 1, 1)")]

    @test run_script(replace_strings, "free_convection", free_convection_example)

    #####
    ##### Three layer constant fluxes example
    #####
    
    three_layer_constant_fluxes_example = joinpath(@__DIR__, "..", "examples", "three_layer_constant_fluxes.jl")

    replace_strings = [
                       ("""Nh = args["Nh"]""", "Nh = 1"), 
                       ("""Nz = args["Nz"]""", "Nz = 1"), 
                      ]

    @test run_script(replace_strings, "three_layer_constant_fluxes", three_layer_constant_fluxes_example)
end
