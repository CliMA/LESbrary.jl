using ArgParse
using PyCall

using Dates: Date

include("load_sose_data.jl")

sose_site_analysis = pyimport("sose_site_analysis")

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--latitude"
            help = "Site latitude in degrees North (°N). Note that the SOSE dataset only goes up to -30°N."
            arg_type = Float64

        "--longitude"
            help = "Site longitude in degrees East (°E) between 0°E and 360°E."
            arg_type = Float64

        "--start"
            help = "Start date (format YYYY-mm-dd) between 2013-01-01 and 2018-01-01."
            arg_type = String

        "--end"
            help = "End date (format YYYY-mm-dd) between 2013-01-01 and 2018-01-01. Must be after the start date."
            arg_type = String

        "--sose-dir"
            help = "Directory containing the SOSE datasets."
            arg_type = String
    end

    return parse_args(settings)
end

function dates_to_offset(start_date, end_date, sose_start_date; step=1)
    offset = (start_date - sose_start_date).value / step + 1
    n_dates = (end_date - start_date).value / step
    return floor(Int, offset), ceil(Int, n_dates)
end

@info "Parsing command line arguments..."

args = parse_command_line_arguments()

lat = args["latitude"]
lon = args["longitude"]

start_date = Date(args["start"])
end_date = Date(args["end"])
sose_dir = args["sose-dir"]

sose_start_date = Date(2013, 1, 1)
sose_end_date = Date(2018, 1, 1)

@assert start_date < end_date
@assert start_date >= sose_start_date
@assert end_date <= sose_end_date

@info "Performing ocean site analysis from $start_date to $end_date..."

ds2 = sose.open_sose_2d_datasets(sose_dir)
offset, n_dates = dates_to_offset(start_date, end_date, sose_start_date)
sose_site_analysis.plot_surface_forcing_site_analysis(ds2, lat, lon, offset-1, n_dates)

ds_fluxes = sose.open_sose_advective_flux_datasets(sose_dir)
offset, n_dates = dates_to_offset(start_date, end_date, sose_start_date, step=5)
sose_site_analysis.plot_lateral_flux_site_analysis(ds_fluxes, lat, lon, offset-1, n_dates)
