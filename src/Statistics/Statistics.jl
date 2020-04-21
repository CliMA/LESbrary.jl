module OnlineCalculations

export
    TurbulentKineticEnergy,
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

end # module
