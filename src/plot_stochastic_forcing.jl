using Plots
pyplot()

include("stochastic_forcing.jl")

day = 86400
t = collect(0:10) .* day
τ = 1e-2 .* [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]

t′, τ′ = stochastic_forcing(τ, t, σ=1e-2, Δt=0.01day, T=10day)

plot(t/day, τ, label="", xlabel="days", ylabel="wind stress")
plot!(t′/day, τ′, label="")

