import os
import re
import logging
import argparse

import h5py
import matplotlib
import matplotlib.pyplot as plt

from numpy import reshape, linspace, diff, mean, amax

logging.basicConfig(level=logging.INFO)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Needed on Supercloud =/

matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('dark_background')

# Human sorting: https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def plot_vertical_profiles(field, Is, save=True):
    rotation_period = 86400

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 16))
    for (i, file) in Is:
        i = str(i)
        t = file["timeseries/t/" + i][()]
        Nx, Lx = file["grid/Nx"][()], file["grid/Lx"][()]
        Ny, Ly = file["grid/Ny"][()], file["grid/Ly"][()]
        Nz, Lz, dz = file["grid/Nz"][()], file["grid/Lz"][()], file["grid/Δz"][()]
        z = linspace(0, -Lz, Nz)

        if field == "K * dT/dz":
           T = file["timeseries/T/" + i][()][1:Nz+1, 1:Ny+1, 1:Nx+1]
           K = file["timeseries/kappaT/" + i][()][1:Nz+1, 1:Ny+1, 1:Nx+1]

           T = T.take(arange(-1, Nx), axis=0, mode="wrap")
           dTdz = diff(T, n=1, axis=0) / dz

           diffusive_flux_z = K * dTdz
           Fp = mean(diffusive_flux_z, axis=[1, 2])

        Fp = reshape(Fp, Nz)

        ax1.plot(Fp, z, label="{:.2f} days".format(t / rotation_period))

    ax1.set_title("Horizontally averaged {:s} profiles".format(field))
    ax1.set_ylim([-Lz, 0])
    ax1.set_xlabel(f"{field}")
    ax1.set_ylabel("z (m)")
    ax1.legend(frameon=False)

    if save:
        filename = f"{field}_profiles.png"
        logging.info(f"Saving: {filename:s}")
        plt.savefig(filename, dpi=300, format="png", transparent=False)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot vertical profiles from Oceananigans.jl JLD2 output fields.")
    parser.add_argument("-f", "--files", type=str, nargs='+', required=True, help="JLD2 output filepaths")
    args = parser.parse_args()
    filepaths = args.files

    sort_nicely(filepaths)
    files = [h5py.File(fp, 'r') for fp in filepaths]

    Is = []  # We'll generate a list of iterations with output across all files.
    for file in files:
        new_Is = sorted(list(map(int, list(file["timeseries/t"].keys()))))
        tuplified_Is = list(map(lambda i: (i, file), new_Is))
        Is.extend(tuplified_Is)
    
    logging.info(f"Found {len(Is):d} snapshots per field across {len(files):d} files: i={Is[0][0]}->{Is[-1][0]}")
    
    Ip = [Is[n] for n in [1, 10, 30, -1]]  # Iterations to plot.
    plot_vertical_profiles("K * dT/dz", Ip)

