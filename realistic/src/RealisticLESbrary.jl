module RealisticLESbrary

export InterpolatedProfile, InterpolatedProfileTimeSeries, ∂z, ∂t
export validate_sose_dates, dates_to_offset

using Dates

include("interpolated_profiles.jl")

const sose_start_date = Date(2013, 1, 1)
const sose_end_date = Date(2018, 1, 1)

function validate_sose_dates(start_date, end_date)
    @assert start_date < end_date
    @assert start_date >= sose_start_date
    @assert end_date <= sose_end_date
end

function dates_to_offset(start_date, end_date; step=1)
    offset = (start_date - sose_start_date).value / step + 1
    n_dates = (end_date - start_date).value / step
    return floor(Int, offset), ceil(Int, n_dates)
end

end # module
