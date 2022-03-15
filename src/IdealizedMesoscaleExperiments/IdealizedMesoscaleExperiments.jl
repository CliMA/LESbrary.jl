module IdealizedMesoscaleExperiments

using Oceananigans.Units

include("eddying_channel.jl")

flat_bottom_parameters = Dict{Symbol,Any}(
    :flux_weak_wind_beta => Dict{Symbol,Any}(:wind_stress => 0.1, β = 1e-11),
    :flux_regular_wind_beta => Dict{Symbol,Any}(:wind_stress => 0.2, β = 1e-11),
    :flux_strong_wind_beta => Dict{Symbol,Any}(:wind_stress => 0.4, β = 1e-11),
    :flux_regular_wind_no_beta => Dict{Symbol,Any}(:wind_stress => 0.2, β = 0e-11),
)

# default :f => -1e-4, 

for (name, set) in flat_bottom_parameters
    set[:name] = string(name)
    set[:stop_time] = 200years
end

end # module
