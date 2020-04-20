module SpongeLayers

export 
    μ,
    Ψᵢ,
    Fu,
    Fv,
    Fw,
    Fθ,
    Fs,
    Fb

"z-dependent damping time scale."
@inline μ(z, L, δ, τ) = 1/τ * exp(-(z + L)^2 / 2δ^2)

"Generic initial profile of a variable Ψ with gradient Γ and surface value Ψ₀."
@inline Ψᵢ(z, Γ, Ψ₀=-0) = Ψ₀ + Γ * z

"Damping functions for a velocity field."
@inline Fu(i, j, k, grid, clock, state, p) = @inbounds -μ(grid.zC[k], grid.Lz, p.δ, p.τ) * state.velocities.u[i, j, k]
@inline Fv(i, j, k, grid, clock, state, p) = @inbounds -μ(grid.zC[k], grid.Lz, p.δ, p.τ) * state.velocities.v[i, j, k]
@inline Fw(i, j, k, grid, clock, state, p) = @inbounds -μ(grid.zF[k], grid.Lz, p.δ, p.τ) * state.velocities.w[i, j, k]

"Damping functions for scalars with initial profiles."
@inline Fθ(i, j, k, grid, clock, state, p) =
    @inbounds -μ(grid.zC[k], grid.Lz, p.δ, p.τ) * (state.tracers.T[i, j, k] - Ψᵢ(grid.zC[k], p.dθdz, p.θ₀))

@inline Fs(i, j, k, grid, clock, state, p) =
    @inbounds -μ(grid.zC[k], grid.Lz, p.δ, p.τ) * (state.tracers.S[i, j, k] - Ψᵢ(grid.zC[k], p.dsdz, p.s₀))

@inline Fb(i, j, k, grid, clock, state, p) =
    @inbounds -μ(grid.zC[k], grid.Lz, p.δ, p.τ) * (state.tracers.b[i, j, k] - Ψᵢ(grid.zC[k], p.dbdz))

end # module
