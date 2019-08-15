import os
import re
import logging
import argparse

import h5py
import matplotlib
import matplotlib.pyplot as plt

from numpy import reshape, linspace, amin, amax
from matplotlib.ticker import LogLocator
from matplotlib.colors import LogNorm

from human_sorting import sort_nicely

logging.basicConfig(level=logging.INFO)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Needed on Supercloud =/

matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('dark_background')

def plot_horizontal_slice(file, field, i, level, sym=False, log=False, save=True):
    rotation_period = 86400

    i = str(i)
    t = file["timeseries/t/" + i][()]
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
        vmax = max(abs(amin(F_k)), abs(amax(F_k)))
        im = ax1.contourf(x, y, F_k, 30, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    elif log:
         im = ax1.pcolormesh(x, y, F_k, cmap="inferno", norm=LogNorm(vmin=F_k.min(), vmax=F_k.max()),)
    else:
        im = ax1.contourf(x, y, F_k, 30, cmap="inferno")

    fig.colorbar(im, ax=ax1)
    ax1.set_title("{:s} @ z={:.1f} m, t={:.2f} days".format(field, z[k], t / rotation_period))
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
    
    k = 20
    Ip = [Is[n] for n in [5, 10, 20, -1]]  # Iterations to plot.
    for (i, file) in Ip:
        plot_horizontal_slice(file, "T", i, k)
        plot_horizontal_slice(file, "u", i, k, sym=True)
        plot_horizontal_slice(file, "w", i, k, sym=True)
        plot_horizontal_slice(file, "nu", i, k)
        plot_horizontal_slice(file, "kappaT", i, k)

