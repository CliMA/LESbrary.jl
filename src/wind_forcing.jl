using DifferentialEquations
using Polynomials

"""
    stochastic_wind_forcing(τ::Number; σ, Δt, T)

Generate stochastic wind forcing with mean τ, standard deviation σ, time step Δt,
and length T.
"""
function stochastic_wind_forcing(τ::Number; σ, Δt, T)
    W = WienerProcess(0, τ)
    prob = NoiseProblem(W, (0.0, T))
    sol = solve(prob, dt=Δt)
    linear_trend = polyfit(sol.t, sol.u, 1)
    τ′ = sol.u .- linear_trend.(sol.t)
    τ′ = σ/std(τ′) * τ′
    return sol.t, τ′
end


