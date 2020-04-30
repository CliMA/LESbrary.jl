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

