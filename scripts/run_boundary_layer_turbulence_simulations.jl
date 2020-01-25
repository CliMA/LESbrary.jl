Qs = (-25, -50, -100, -200) .|> string
τs = (0, 0.01, 0.02, 0.05) .|> string
∂T∂zs = (0.01, 0.02, 0.03) .|> string

output_dir = "/central/groups/esm/alirama/ocean_turbulence_training_data"

for Q in Qs, τ in τs, ∂T∂z in ∂T∂zs
    simulation_name = "Oceananigans.jl boundary layer turbulence Q=$Q, tau=$τ, dTdz=$∂T∂z"
    cmd = "julia --project simulation/boundary_layer_turbulence.jl -N 256 -V 256 -L 100 -H 100 --dTdz $∂T∂z -Q $Q --wind-stress $τ --days 8 --output-dir $output_dir"

    slurm_script = read("boundary_layer_turbulence.slurm", String)

    replace_strings = [
        "#SBATCH -J \"Oceananigans\"" => "#SBATCH -J \"$simulation_name\"",
        "julia --project simulation/boundary_layer_turbulence_simple.jl" => cmd
    ]

    for strs in replace_strings
        slurm_script = replace(slurm_script, strs[1] => strs[2])
    end

    slurm_script_filename = "boundary_layer_turbulence_Q$(Q)_tau$(τ)_dTdz$(∂T∂z).slurm"
    open(slurm_script_filename, "w") do f
        write(f, slurm_script)
    end

    run(`sbatch $slurm_script_filename`)
end
