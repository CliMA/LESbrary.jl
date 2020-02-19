using PyCall

# Needed to import local modules
# See: https://github.com/JuliaPy/PyCall.jl/issues/48
py"""
import sys
sys.path.insert(0, ".")
"""

sose = pyimport("sose_data")

ds2 = sose.open_sose_2d_datasets("/home/alir/cnhlab004/bsose_i122/")
ds3 = sose.open_sose_3d_datasets("/home/alir/cnhlab004/bsose_i122/")

t = sose.get_times(ds2)

lat, lon = 190, -55
τx = sose.get_time_series(ds2, "oceTAUX", lat, lon)
τy = sose.get_time_series(ds2, "oceTAUY", lat, lon)

ds.close()
