import os
import glob
import argparse
import logging

import h5py
import matplotlib
import matplotlib.pyplot as plt
import joblib
import ffmpeg

from numpy import reshape, linspace
from matplotlib.colors import Normalize, LogNorm

from human_sorting import sort_nicely

logging.basicConfig(level=logging.INFO)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Needed on Supercloud =/

matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('dark_background')

def plot_horizontal_slice_frame(filepath, field, i, level, frame_number, vmin, vmax, sym=False, log=False, save=True):
    plt.style.use('dark_background')  # Needs to be set again for each thread.
    
    file = h5py.File(filepath, 'r')
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
        im = ax1.contourf(x, y, F_k, levels=linspace(vmin, vmax, 30), cmap="coolwarm")
    elif log:
         im = ax1.pcolormesh(x, y, F_k, cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax1.contourf(x, y, F_k, 30, levels=linspace(vmin, vmax, 30), cmap="inferno")
   
    fig.colorbar(im)
    im.set_clim(vmin=vmin, vmax=vmax)

    ax1.set_title("{:s} @ z={:.1f} m, t={:.2f} days".format(field, z[level], t / rotation_period))
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    
    if save:
        filename = f"{field}_horizontal_slice_k{level:d}_{frame_number:06d}.png"
        print(f"Saving: {filename:s}")
        plt.savefig(filename, dpi=300, format="png", transparent=False)
        plt.close(fig)

def make_horizontal_slice_movie(Is, field, level, vmin, vmax, sym=False, log=False):
    joblib.Parallel(n_jobs=48)(
         joblib.delayed(plot_horizontal_slice_frame)(I[1], field, I[0], level, frame_number, vmin, vmax, sym=sym, log=log)
         for frame_number, I in enumerate(Is))

    (
        ffmpeg
        .input(f"{field}_horizontal_slice_k{level:d}_%06d.png", framerate=30)
        .output(f"{field}_horizontal_slice_k{level:d}_movie.mp4", crf=15, pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )

    for fn in glob.glob(f"{field}_horizontal_slice_k{level:d}_*.png"):
        os.remove(fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make horizontal slice movies from Oceananigans.jl JLD2 output.")
    parser.add_argument("-f", "--files", type=str, nargs='+', required=True, help="JLD2 output filepaths")
    args = parser.parse_args()
    filepaths = args.files
    sort_nicely(filepaths)

    Is = []  # We'll generate a list of iterations with output across all files.
    for fp in filepaths:
        file = h5py.File(fp, 'r')
        new_Is = sorted(list(map(int, list(file["timeseries/t"].keys()))))
        tuplified_Is = list(map(lambda i: (i, fp), new_Is))
        Is.extend(tuplified_Is)

    logging.info(f"Found {len(Is):d} snapshots per field across {len(filepaths):d} files: i={Is[0][0]}->{Is[-1][0]}")

    k = 20
    make_horizontal_slice_movie(Is, "T", k, 19.5, 20)
    make_horizontal_slice_movie(Is, "w", k, -0.02, 0.02, sym=True)
    make_horizontal_slice_movie(Is, "nu", k, 1e-5, 1e-2, log=True)
    make_horizontal_slice_movie(Is, "kappaT", k, 1e-5, 1e-2, log=True)
    make_horizontal_slice_movie(Is, "nu", 225, 1e-5, 1e-2, log=True)
    make_horizontal_slice_movie(Is, "kappaT", 225, 1e-5, 1e-2, log=True)
    
