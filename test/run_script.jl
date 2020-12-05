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

    try
        open(test_script_filepath, "w") do f
            # Wrap test script inside module to avoid polluting namespaces
            write(f, "module _Test_$script_name" * "_$module_suffix\n")
            write(f, file_content)
            write(f, "\nend # module")
        end

        include(test_script_filepath)
    catch err
        @warn "Error while testing script: " * sprint(showerror, err)

        # Print the content of the file to the test log, with line numbers, for debugging
        test_file_content = read(test_script_filepath, String)
        delineated_file_content = split(test_file_content, '\n')
        for (number, line) in enumerate(delineated_file_content)
            @printf("% 3d %s\n", number, line)
        end

        return false
    finally
        # Delete the test script
        rm(test_script_filepath, force=true)
    end

    return true
end
