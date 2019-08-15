import os
import re
import logging
import argparse

import h5py
import matplotlib
import matplotlib.pyplot as plt

from numpy import reshape, linspace, amax

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

def plot_horizontal_slice(file, field, i, level, sym=False, save=True):
    rotation_period = file["parameters/rotation_period"][()]

    i = str(i)
    t = file["timeseries/t/" + i]
    Nx, Lx = file["grid/Nx"][()], file["grid/Lx"][()]
    Ny, Ly = file["grid/Ny"][()], file["grid/Ly"][()]
    x, y = linspace(0, Lx, Nx), linspace(0, Ly, Ny)

    Nz, Lz, dz = file["grid/Nz"][()], file["grid/Lz"][()], file["grid/Δz"][()]
    if field == "w":
        z = linspace(0, -Lz, Nz+1)
    else:
        z = linspace(-dz/2, -Lz+dz/2, Nz)

    F_k = file["timeseries/" + field + "/" + i][()][level, 1:Ny+1, 1:Nx+1]

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))

    if sym:
        vmax = amax(F_k)
        im = ax1.contourf(x / 1000, y / 1000, F_k, 30, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    else:
        im = ax1.contourf(x / 1000, y / 1000, F_k, 30, cmap="inferno")

    fig.colorbar(im, ax=ax1)
    ax1.set_title("{:s} @ z={:.1f} km, t={:.0f} rotation periods".format(field, z[k] / 1000, t / rotation_period))
    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("y (km)")

    filename = f"{field}_horizontal_slice_i{i}_k{k:d}.png"
    logging.info(f"Saving: {filename:s}")
    plt.savefig(filename, dpi=300, format="png", transparent=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot horizontal slices from Oceananigans.jl JLD2 output.")
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
    
    k = 700
    Ip = [Is[n] for n in [5, 10, 20, 50]]  # Iterations to plot.
    for (i, file) in Ip:
        plot_horizontal_slice(file, "T", i, k)
        plot_horizontal_slice(file, "w", i, k, sym=True)

