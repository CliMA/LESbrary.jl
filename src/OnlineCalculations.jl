module OnlineCalculations

export
    horizontal_averages,
    FieldSlice,
    FieldSlices,
    XYSlice,
    XYSlices,
    YZSlice,
    YZSlices,
    XZSlice,
    XZSlices

using Oceananigans,
      Oceananigans.AbstractOperations,
      Oceananigans.Diagnostics,
      Oceananigans.Fields,
      Oceananigans.Operators,
      Oceananigans.Grids,

using GPUifyLoops: @loop, @launch

#####
##### Functionality
#####

struct TurbulentKineticEnergy{E, U, V, W, Ua, Va, Ea}
            e :: E # CellField...
            u :: U
            v :: V
            w :: W
    U_average :: Ua
    V_average :: Va
    e_average :: Ea
end

function TurbulentKineticEnergy(model)
    u, v, w = model.velocities
    e = CellField(model.architecture, model.grid)

    U_average = HorizontalAverage(u)
    V_average = HorizontalAverage(v)
    e_average = HorizontalAverage(e)

    return TurbulentKineticEnergy(e, u, v, w, U_average, V_average, e_average)
end

@inline w²(i, j, k, grid, w) = @inbounds w[i, j, k]^2

function _compute_turbulent_kinetic_energy!(tke, grid, u, v, w, U, V)
    @loop_xyz i j k grid begin
        @inbounds tke[i, j, k] = ((u[i, j, k] - U[k+grid.Hz])^2 + (v[i, j, k] - V[k+grid.Hz])^2 + ℑzᵃᵃᶜ(i, j, k, grid, w², w)) / 2
    end
    return nothing
end

function (tke::TurbulentKineticEnergy)(model)
    run_diagnostic(model, tke.U_average)
    run_diagnostic(model, tke.V_average)

    u, v, w = datatuple(model.velocities)

    Tx, Ty = 16, 16 # CUDA threads per block
    Bx, By, Bz = floor(Int, model.grid.Nx/Tx), floor(Int, model.grid.Ny/Ty), model.grid.Nz  # Blocks in grid

    @launch(device(model.architecture), threads=(Tx, Ty), blocks=(Bx, By, Bz), 
            _compute_turbulent_kinetic_energy!(tke.e.data, model.grid, u, v, w, tke.U_average.result, tke.V_average.result))

    # Compute horizontally-averaged turbulent kinetic energy
    return tke.e_average(model)
end

function horizontal_averages(model)

    # Create scratch space for calculations
    scratch = CellField(model.architecture, model.grid)

    # Extract short field names
    u, v, w = model.velocities

    # Define horizontal averages
    U = HorizontalAverage(u)
    V = HorizontalAverage(v)
    e = TurbulentKineticEnergy(model)

    U² = HorizontalAverage(u^2, scratch)
    V² = HorizontalAverage(v^2, scratch)
    W² = HorizontalAverage(w^2, scratch)
    W³ = HorizontalAverage(w^3, scratch)
    wu = HorizontalAverage(w*u, scratch)
    wv = HorizontalAverage(w*v, scratch)

    primitive_averages = (
                  U = model -> U(model),
                  V = model -> V(model),
                  E = model -> e(model),

                 U² = model -> U²(model),
                 V² = model -> V²(model),
                 W² = model -> W²(model),
                 W³ = model -> W³(model),

                 wu = model -> wu(model),
                 wv = model -> wv(model),
               )

    # Add subfilter stresses (if they exist)
    average_stresses = Dict()

    νₑ = model.diffusivities.νₑ

    NU = HorizontalAverage(νₑ)

    τ₁₃ = @at (Face, Cell, Face) νₑ * (-∂z(u) - ∂x(w))
    τ₂₃ = @at (Cell, Face, Face) νₑ * (-∂z(v) - ∂y(w))
    τ₃₃ = @at (Cell, Cell, Face) νₑ * (-∂z(w) - ∂z(w))

    T₁₃ = HorizontalAverage(τ₁₃, scratch)
    T₂₃ = HorizontalAverage(τ₂₃, scratch)
    T₃₃ = HorizontalAverage(τ₃₃, scratch)

    average_stresses[:τ₁₃] = model -> T₁₃(model)
    average_stresses[:τ₂₃] = model -> T₂₃(model)
    average_stresses[:τ₃₃] = model -> T₃₃(model)
    average_stresses[:νₑ] = model -> NU(model)

    average_stresses = (; zip(keys(average_stresses), values(average_stresses))...)

    average_tracers = Dict()
    average_fluxes = Dict()
    average_diffusivities = Dict()

    for tracer in keys(model.tracers)

        advective_flux_key = Symbol(:w, tracer)
        subfilter_flux_key = Symbol(:q₃_, tracer)
           diffusivity_key = Symbol(:κₑ_, tracer)

        w = model.velocities.w
        c = getproperty(model.tracers, tracer)

        # Average tracer
        average_tracer = HorizontalAverage(c)
        average_tracers[tracer] = model -> average_tracer(model)

        # Advective flux
        advective_flux = w * c
        average_advective_flux = HorizontalAverage(advective_flux, scratch)
        average_fluxes[advective_flux_key] = model -> average_advective_flux(model)

        # Subfilter diffusivity (if it exists)
        try
            κₑ = getproperty(model.diffusivities.κₑ, tracer)
            average_diffusivity = HorizontalAverage(κₑ)
            average_diffusivities[diffusivity_key] = model -> average_diffusivity(model)

            subfilter_flux = @at (Cell, Cell, Face) -∂z(c) * κₑ
            average_subfilter_flux = HorizontalAverage(subfilter_flux, scratch)
            average_fluxes[subfilter_flux_key] = model -> average_subfilter_flux(model)
        catch
        end
    end

    average_tracers = (; zip(keys(average_tracers), values(average_tracers))...)
    average_fluxes = (; zip(keys(average_fluxes), values(average_fluxes))...)
    average_diffusivities = (; zip(keys(average_diffusivities), values(average_diffusivities))...)

    return merge(primitive_averages, average_tracers, average_stresses, average_fluxes, average_diffusivities)
end

#####
##### Slices! (useful for pretty movies)
#####

struct FieldSlice{F, X, Y, Z, RT}
          field :: F
              i :: X
              j :: Y
              k :: Z
    return_type :: RT
end

rangify(range) = range
rangify(i::Int) = i:i

function FieldSlice(field; i=Colon(), j=Colon(), k=Colon(), return_type=Array)
    i, j, k = rangify.([i, j, k])
    return FieldSlice(field, i, j, k, return_type)
end

(fs::FieldSlice)(model) = fs.return_type(fs.field.data.parent[fs.i, fs.j, fs.k])

function FieldSlices(fields::NamedTuple; kwargs...)
    names = propertynames(fields)
    return NamedTuple{names}(Tuple(FieldSlice(f; kwargs...) for f in fields))
end

function XYSlice(field; z, return_type=Array)
    i = Colon()
    j = Colon()
    k = searchsortedfirst(znodes(field)[:], z) + field.grid.Hz

    return FieldSlice(field, i=i, j=j, k=k, return_type=return_type)
end

function XZSlice(field; y, return_type=Array)
    i = Colon()
    j = searchsortedfirst(ynodes(field)[:], y) + field.grid.Hy
    k = Colon()

    return FieldSlice(field, i=i, j=j, k=k, return_type=return_type)
end

function YZSlice(field; x, return_type=Array)
    i = searchsortedfirst(xnodes(field)[:], x) + field.grid.Hx
    j = Colon()
    k = Colon()

    return FieldSlice(field, i=i, j=j, k=k, return_type=return_type)
end

function XYSlices(fields; suffix="", kwargs...)
    names = Tuple(Symbol(name, suffix) for name in propertynames(fields))
    NamedTuple{names}(Tuple(XYSlice(f; kwargs...) for f in fields))
end

function XZSlices(fields; suffix="", kwargs...)
    names = Tuple(Symbol(name, suffix) for name in propertynames(fields))
    NamedTuple{names}(Tuple(XZSlice(f; kwargs...) for f in fields))
end

function YZSlices(fields; suffix="", kwargs...)
    names = Tuple(Symbol(name, suffix) for name in propertynames(fields))
    return NamedTuple{names}(Tuple(YZSlice(f; kwargs...) for f in fields))
end

end # module
