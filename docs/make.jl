push!(LOAD_PATH, "..")

using
  Documenter,
  Literate,
  Plots,  # to not capture precompilation output
  LESbrary

# Gotta set this environment variable when using the GR run-time on Travis CI.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const EXPERIMENTS_DIR = joinpath(@__DIR__, "..", "experiments")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
            #"langmuir_turbulence.jl",
           ]

for example in examples
  example_filepath = joinpath(EXAMPLES_DIR, example)
  withenv("GITHUB_REPOSITORY" => "LESbrary.jl") do
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
    #Literate.notebook(example_filepath, OUTPUT_DIR, documenter=true)
    Literate.script(example_filepath, OUTPUT_DIR, documenter=true)
  end
end

#####
##### Build and deploy docs
#####

# Set up a timer to print a space ' ' every 240 seconds. This is to avoid Travis CI
# timing out when building demanding Literate.jl examples.
Timer(t -> println(" "), 0, interval=240)

format = Documenter.HTML(
  collapselevel = 2,
     prettyurls = get(ENV, "CI", nothing) == "true",
      canonical = "https://github.com/CliMA/LESbrary.jl.git"
)

makedocs(
 modules = [LESbrary],
 doctest = false,
   clean = true,
checkdocs = :all,
  format = format,
 authors = "Ali Ramadhan, Gregory L. Wagner, and Adeline Hillier",
sitename = "LESbrary.jl",
   pages = Any["Home" => "index.md",
               "Examples" => Any[
                                 "generated/langmuir_turbulence.md"
                                ],
               "Experiments" => Any[
                                    # "generated/three_layer_constant_fluxes.md",
                                   ],
               "DocStrings" => Any[
                                   #"man/functions.md"
                                   ],
              ],
)

withenv("GITHUB_REPOSITORY" => "LESbrary.jl") do
  deploydocs(       repo = "https://github.com/CliMA/LESbrary.jl.git",
                versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
            push_preview = false
            )
end
