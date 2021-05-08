using ArgParse
using PyCall
using RealisticLESbrary

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

@info "Parsing command line arguments..."

args = parse_command_line_arguments()

lat = args["latitude"]
lon = args["longitude"]

start_date = Date(args["start"])
end_date = Date(args["end"])
sose_dir = args["sose-dir"]

validate_sose_dates(start_date, end_date)

@info "Performing ocean site analysis from $start_date to $end_date..."

ds2 = sose.open_sose_2d_datasets(sose_dir)
offset, n_dates = dates_to_offset(start_date, end_date)
sose_site_analysis.plot_surface_forcing_site_analysis(ds2, lat, lon, offset-1, n_dates)

ds_fluxes = sose.open_sose_advective_flux_datasets(sose_dir)
offset, n_dates = dates_to_offset(start_date, end_date, step=5)
sose_site_analysis.plot_lateral_flux_site_analysis(ds_fluxes, lat, lon, offset-1, n_dates)
