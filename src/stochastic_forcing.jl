using DifferentialEquations
using Polynomials
using Interpolations

"""
    stochastic_forcing(τ::Number; σ, Δt, T)

Generate stochastic forcing with mean `τ`, standard deviation `σ`, time step `Δt`,
and length `T`.
"""
function stochastic_wind_forcing(τ::Number; σ, Δt, T)
    W = WienerProcess(0.0, τ)
    prob = NoiseProblem(W, (0.0, T))
    sol = solve(prob, dt=Δt)
    linear_trend = polyfit(sol.t, sol.u, 1)
    τ′ = sol.u .- linear_trend.(sol.t)
    τ′ = σ/std(τ′) * τ′
    return sol.t, τ′
end

"""
    stochastic_forcing(τ::AbstractArray, times; σ, Δt, T)

Generate stochastic forcing on top of time series `τ` with standard deviation `σ`, time
step `Δt`, and length `T`.
"""
function stochastic_wind_forcing(τ::AbstractArray, times; σ, Δt, T)
    # FIXME: Gotta ensure that mean(τ′) = mean(τ).
    t, τ′ = stochastic_wind_forcing(τ[1], σ=σ, Δt=Δt, T=T)
    ℑτ = LinearInterpolation(times, τ, extrapolation_bc=Flat())
    @. τ′ = ℑτ(t) + τ′ 
    return t, τ′
end

