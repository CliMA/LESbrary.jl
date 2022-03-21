module IdealizedExperiments

using Oceananigans.Units

include("three_layer_constant_fluxes.jl")
include("eddying_channel.jl")

two_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 1.2e-7, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -1e-3,   :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -7e-4,   :buoyancy_flux => 6e-8,   :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -3.3e-4, :buoyancy_flux => 1.1e-7, :f => 1e-4),
    :strong_wind_weak_heating => Dict{Symbol, Any}(:momentum_flux => -1e-3,   :buoyancy_flux => -4e-8,  :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -2e-4,   :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in two_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 2days
end

four_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 7.0e-8, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -8e-4,   :buoyancy_flux => 0.0,    :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -6.5e-4, :buoyancy_flux => 4e-8,   :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -3e-4,   :buoyancy_flux => 7e-8,   :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -1e-4,   :buoyancy_flux => 0.0,    :f => 0.0),
)

for (name, set) in four_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 4days
end

six_day_suite_parameters = Dict{Symbol, Any}(
    :free_convection          => Dict{Symbol, Any}(:momentum_flux => 0.0,     :buoyancy_flux => 5e-8, :f => 1e-4),
    :strong_wind              => Dict{Symbol, Any}(:momentum_flux => -7e-4,   :buoyancy_flux => 0.0,  :f => 1e-4),
    :strong_wind_weak_cooling => Dict{Symbol, Any}(:momentum_flux => -5.5e-4, :buoyancy_flux => 3e-8, :f => 1e-4),
    :weak_wind_strong_cooling => Dict{Symbol, Any}(:momentum_flux => -2.2e-4, :buoyancy_flux => 5e-8, :f => 1e-4),
    :strong_wind_no_rotation  => Dict{Symbol, Any}(:momentum_flux => -7e-5,   :buoyancy_flux => 0.0,  :f => 0.0),
)

for (name, set) in six_day_suite_parameters
    set[:name] = string(name)
    set[:stop_time] = 6days
end

#####
##### Mesoscale parameters
#####

eddying_channel_parameters = Dict{Symbol, Any}()

eddying_channel_parameters[:flat_bottom] = Dict{Symbol, Any}(
    :flux_weak_wind_beta       => Dict{Symbol, Any}(:momentum_flux => 0.1, :β => 1e-11),
    :flux_regular_wind_beta    => Dict{Symbol, Any}(:momentum_flux => 0.2, :β => 1e-11),
    :flux_strong_wind_beta     => Dict{Symbol, Any}(:momentum_flux => 0.4, :β => 1e-11),
    :flux_regular_wind_no_beta => Dict{Symbol, Any}(:momentum_flux => 0.2, :β => 0e-11),
)

# default :f => -1e-4, 

for (name, set) in eddying_channel_parameters[:flat_bottom]
    set[:name] = string(name)
    set[:stop_time] = 200years
end


end # module
