module NearSurfaceTurbulenceModels

export
    SurfaceEnhancedModelConstant,
    SurfaceFluxDiffusivityBoundaryConditions,
    save_closure_parameters!

using Oceananigans,
      Oceananigans.BoundaryConditions,
      Oceananigans.TurbulenceClosures

using JLD2

struct SurfaceEnhancedModelConstant{T} <: Function
             C₀ :: T
             Δz :: T
             z₀ :: T
    enhancement :: T
    decay_scale :: T
end

"""
    SurfaceEnhancedModelConstant(Δz; FT=Float64, C₀=1/12, z₀=-Δz/2,
                                 enhancement=2, decay_scale=8Δz)

Returns a callable object representing a spatially-variable model constant
for an LES eddy diffusivity model with the surface-enhanced form

    ``C(z) = C₀ * (1 + enhancement * exp((z - z₀) / decay_scale)``

"""
function SurfaceEnhancedModelConstant(Δz; FT=Float64, C₀=1/12, z₀=-Δz/2,
                                      enhancement=2, decay_scale=8Δz)

    return SurfaceEnhancedModelConstant{FT}(C₀, Δz, z₀, enhancement, decay_scale)
end

@inline (C::SurfaceEnhancedModelConstant)(x, y, z) =
    C.C₀ * (1 + C.enhancement * exp((z - C.z₀) / C.decay_scale))

"""
    SurfaceEnhancedModelConstant(filename)

Reconstruct a `SurfaceEnhancedModelConstant` from file.
"""
function SurfaceEnhancedModelConstant(filename::String)
    file = jldopen(filename)

    Lz = file["grid/Lz"]
    Nz = file["grid/Nz"]
    Δz = Lz / Nz

    C₀ = file["closure/C₀"]
    z₀ = file["closure/z₀"]
    enhancement = file["closure/enhancement"]
    decay_scale = file["closure/decay_scale"]

    close(file)

    Cᴬᴹᴰ = SurfaceEnhancedModelConstant(Δz; C₀=1/12, z₀=-Δz/2, enhancement=2, decay_scale=8Δz)

    return Cᴬᴹᴰ
end

#####
##### Saving utilities for surface enhanced model constant
#####

save_closure_parameters!(args...) = nothing # fallback

const EnhancedAMD = AnisotropicMinimumDissipation{FT, PK, <:SurfaceEnhancedModelConstant} where {FT, PK}

function save_closure_parameters!(file, closure::EnhancedAMD)
    file["closure/C₀"] = closure.Cν.C₀
    file["closure/z₀"] = closure.Cν.z₀
    file["closure/enhancement"] = closure.Cν.enhancement
    file["closure/decay_scale"] = closure.Cν.decay_scale
    return nothing
end

#####
##### Utility for obtaining surface flux diffusivity boundary conditions
#####

function SurfaceFluxDiffusivityBoundaryConditions(grid, Qᵇ; Cʷ=0.1)
    w★ = (Qᵇ * grid.Lz)^(1/3) # surface turbulent velocity scaling
    κ₀ = Cʷ * grid.Δz * w★
    return DiffusivityBoundaryConditions(grid, top = BoundaryCondition(Value, κ₀))
end

end # module
