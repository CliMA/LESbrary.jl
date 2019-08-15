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

def plot_vertical_slice_frame(filepath, field, i, slice_idx, frame_number, vmin, vmax, sym=False, log=False, save=True):
    plt.style.use('dark_background')  # Needs to be set again for each thread.
    
    file = h5py.File(filepath, 'r')
    rotation_period = 86400

    i = str(i)
    t = file["timeseries/t/" + i][()]
    Nx, Lx = file["grid/Nx"][()], file["grid/Lx"][()]
    Ny, Ly = file["grid/Ny"][()], file["grid/Ly"][()]
    Nz, Lz, dz = file["grid/Nz"][()], file["grid/Lz"][()], file["grid/Δz"][()]
    x, y, z = linspace(0, Lx, Nx), linspace(0, Ly, Ny), linspace(0, Lz, Nz)

    F_slice = file["timeseries/" + field + "/" + i][()][1:Nz+1, 1:Ny+1, slice_idx]
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 16))
    
    if sym:
        im = ax1.contourf(y, z, F_slice, levels=linspace(vmin, vmax, 30), cmap="coolwarm")
    elif log:
         im = ax1.pcolormesh(y, z, F_slice, cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax1.contourf(y, z, F_slice, 30, levels=linspace(vmin, vmax, 30), cmap="inferno")
   
    fig.colorbar(im)
    im.set_clim(vmin=vmin, vmax=vmax)

    ax1.set_title("{:s} @ x={:.1f} m, t={:.2f} days".format(field, x[slice_idx], t / rotation_period))
    ax1.set_xlabel("y (m)")
    ax1.set_ylabel("z (m)")
    
    if save:
        filename = f"{field}_vertical_slice_x{level:d}_{frame_number:06d}.png"
        print(f"Saving: {filename:s}")
        plt.savefig(filename, dpi=300, format="png", transparent=False)
        plt.close(fig)

def make_vertical_slice_movie(Is, field, slice_idx, vmin, vmax, sym=False, log=False):
    joblib.Parallel(n_jobs=48)(
         joblib.delayed(plot_vertical_slice_frame)(I[1], field, I[0], slice_idx, frame_number, vmin, vmax, sym=sym, log=log)
         for frame_number, I in enumerate(Is))

    (
        ffmpeg
        .input(f"{field}_vertical_slice_x{level:d}_%06d.png", framerate=30)
        .output(f"{field}_vertical_slice_x{level:d}_movie.mp4", crf=15, pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )

    for fn in glob.glob(f"{field}_vertical_slice_x{level:d}_*.png"):
        os.remove(fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make vertical slice movies from Oceananigans.jl JLD2 output.")
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

    slice_idx = 128
    make_vertical_slice_movie(Is, "T", slice_idx, 19, 20)
    make_vertical_slice_movie(Is, "w", slice_idx, -0.03, 0.03, sym=True)
    make_vertical_slice_movie(Is, "nu", slice_idx, 1e-5, 1e-2, log=True)
    make_vertical_slice_movie(Is, "kappaT", slice_idx, 1e-5, 1e-2, log=True)
    
