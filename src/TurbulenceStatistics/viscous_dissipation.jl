using KernelAbstractions

using Oceananigans.Fields

using Oceananigans.Operators: ∂xᶜᵃᵃ, ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶜ, ∂zᵃᵃᶠ, ℑxyᶜᶜᵃ, ℑxzᶜᵃᶜ, ℑyzᵃᶜᶜ

struct ViscousDissipation{A, G, K, U, V, W} <: AbstractField{Cell, Cell, Cell, A, G}
    data :: A
    grid :: G
      νₑ :: K
       u :: U
       v :: V
       w :: W
end

"""
    ViscousDissipation(model)

Returns an `AbstractField` representing the viscous dissipation of turbulent
kinetic energy of `model`.

Calling `compute!(ϵ::ViscousDissipation)` computes the viscous dissipation of
turbulent kinetic energy associated with `model` and stores it in `ϵ.data`.

The viscous dissipation ``\\epsilon`` is defined

```math
ϵ ≡ ∫ uⱼ ∂ⱼ τᵢⱼ \\, \\rm{d} V
```

where ``τᵢⱼ`` is the subfilter substress tensor, defined by

```math
τᵢⱼ = 2 νₑ Σᵢⱼ
```

where ``νₑ`` is the eddy viscosity and ``Σᵢⱼ = (∂ᵢ uⱼ + ∂ⱼ uⱼ) / 2``
is the rate-of-strain tensor.
"""
function ViscousDissipation(model; data = nothing)

    if isnothing(data)
        data = new_data(model.architecture, model.grid, (Cell, Cell, Cell))
    end

    u, v, w = model.velocities
    νₑ = model.diffusivities.νₑ

    return ViscousDissipation(data, model.grid, νₑ, u, v, w)
end

function compute!(ϵ::ViscousDissipation)

    arch = architecture(ϵ.data)

    workgroup, worksize = work_layout(ϵ.grid, :xyz, location=(Cell, Cell, Cell))

    compute_kernel! = compute_viscous_dissipation!(device(arch), workgroup, worksize)

    event = compute_kernel!(ϵ.data, ϵ.grid, ϵ.νₑ, ϵ.u, ϵ.v, ϵ.w; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

@kernel function compute_viscous_dissipation!(ϵ, grid, νₑ, u, v, w)
    i, j, k = @index(Global, NTuple)

    Σˣˣ = ∂xᶜᵃᵃ(i, j, k, grid, u)
    Σʸʸ = ∂yᵃᶜᵃ(i, j, k, grid, v)
    Σᶻᶻ = ∂zᵃᵃᶜ(i, j, k, grid, w)

    Σˣʸ = (ℑxyᶜᶜᵃ(i, j, k, grid, ∂yᵃᶠᵃ, u) + ℑxyᶜᶜᵃ(i, j, k, grid, ∂xᶠᵃᵃ, v)) / 2
    Σˣᶻ = (ℑxzᶜᵃᶜ(i, j, k, grid, ∂zᵃᵃᶠ, u) + ℑxzᶜᵃᶜ(i, j, k, grid, ∂xᶠᵃᵃ, w)) / 2
    Σʸᶻ = (ℑyzᵃᶜᶜ(i, j, k, grid, ∂zᵃᵃᶠ, v) + ℑyzᵃᶜᶜ(i, j, k, grid, ∂yᵃᶠᵃ, w)) / 2

    @inbounds ϵ[i, j, k] = νₑ[i, j, k] * 2 * (Σˣˣ^2 + Σʸʸ^2 + Σᶻᶻ^2 + Σˣʸ^2 + Σˣᶻ^2 + Σʸᶻ^2)
end
