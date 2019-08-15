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

logging.basicConfig(level=logging.INFO)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Needed on Supercloud =/

matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('dark_background')

def plot_vertical_profile_frame(filepath, field, i, xlims, frame_number, save=True):
    plt.style.use('dark_background')  # Needs to be set again for each thread.
    
    file = h5py.File(filepath, "r") 
    rotation_period = 86400
    
    i = str(i)
    t = file["timeseries/t/" + i][()]

    Nz, Lz = file["grid/Nz"][()], file["grid/Lz"][()]
    z = linspace(0, -Lz, Nz)

    Fp = file["timeseries/" + field + "/" + i][()]
    Fp = reshape(Fp, Nz)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 16))
    if field in ["nu", "kappaT"]:
        ax1.semilogx(Fp, z)
    else:
        ax1.plot(Fp, z)
    
    ax1.set_title(f"{field} profile @ t = {t/rotation_period:.2f} days")
    ax1.set_xlim(xlims)
    ax1.set_ylim([-100, 0])
    ax1.set_xlabel(f"{field}")
    ax1.set_ylabel("z (m)")
    
    if save:
        filename = f"{field}_profile_{frame_number:06d}.png"
        print(f"Saving: {filename:s}")
        plt.savefig(filename, dpi=300, format="png", transparent=False)
        plt.close(fig)

def make_vertical_profile_movie(filepath, field, Is, xlims):
    joblib.Parallel(n_jobs=48)(
         joblib.delayed(plot_vertical_profile_frame)(filepath, field, i, xlims, frame_number)
         for frame_number, i in enumerate(Is))

    (
        ffmpeg
        .input(f"{field}_profile_%06d.png", framerate=30)
        .output(f"{field}_profile_movie.mp4", crf=15, pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )

    for fn in glob.glob(f"{field}_profile_*.png"):
        os.remove(fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make movies of vertical profiles from Oceananigans.jl JLD2 output.")
    parser.add_argument("-f", "--file", type=str, required=True, help="JLD2 output filepath")
    args = parser.parse_args()
    filepath = args.file

    file = h5py.File(filepath, "r")
    Is = sorted(list(map(int, list(file["timeseries/t"].keys()))))  # List of iterations with output.
    file.close()
    logging.info(f"{len(Is):d} vertical profiles found: i={min(Is)}->{max(Is)}")

    make_vertical_profile_movie(filepath, "T", Is, [19, 20])
    make_vertical_profile_movie(filepath, "u", Is, [-1e-3, 1e-3])
    make_vertical_profile_movie(filepath, "wT", Is, [-0.5e-5, 1.5e-5])
    make_vertical_profile_movie(filepath, "nu", Is, [0.8e-6, 2e-2])
    make_vertical_profile_movie(filepath, "kappaT", Is, [0.8e-6, 2e-2])

