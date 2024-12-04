# Example script for generating LESbrary data

using Oceananigans
using Oceananigans.Units
using Printf

using LESbrary.IdealizedExperiments: forty_eight_hour_suite_parameters
using LESbrary.IdealizedExperiments: twenty_four_hour_suite_parameters
using LESbrary.IdealizedExperiments: twelve_hour_suite_parameters
using LESbrary.IdealizedExperiments: six_hour_suite_parameters
using LESbrary.IdealizedExperiments: seventy_two_hour_suite_parameters

cases = [
    :free_convection,
    :weak_wind_strong_cooling,
    :med_wind_med_cooling,
    :strong_wind_weak_cooling,
    :strong_wind,
    :strong_wind_no_rotation
]

suites = [
    six_hour_suite_parameters,
    twelve_hour_suite_parameters,
    twenty_four_hour_suite_parameters,
    forty_eight_hour_suite_parameters,
    seventy_two_hour_suite_parameters,
]

for suite in suites
    for case in cases
        suite_parameters = suite[case]

        α = 2e-4
        g = 9.81
        ρₒ = 1024
        ρₐ = 1.2
        cp = 3991
        κ = 0.4
        g = 9.81
        Cg = 0.011

        # Heat flux
        Jb = suite_parameters[:buoyancy_flux]
        Q = ρₒ * cp * Jb / (α * g)

        # Deduce ua(z=10m)
        τₒ = suite_parameters[:momentum_flux]
        τₐ = ρₒ * τₒ / ρₐ

        # Atmos friction velocity
        u★ = sqrt(abs(τₐ))

        if τₐ == 0
            u₁₀ = 0.0
            c₁₀ = 0.0
        else
            ℓ = Cg * abs(τₐ) / g
            c₁₀ = (κ / log(10 / ℓ))^2

            # τₐ = c10 * u10^2
            u₁₀ = sqrt(abs(τₐ) / c₁₀)
        end

        @info string(@sprintf("% 24s:", case),
                     @sprintf(" c₁₀: %.2e", c₁₀),
                     @sprintf(" u₁₀: %2d", u₁₀),
                     @sprintf(" Q: %d", Q))
    end
end
