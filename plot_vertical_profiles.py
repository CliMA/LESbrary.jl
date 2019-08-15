import os
import logging
import argparse

import h5py
import matplotlib
import matplotlib.pyplot as plt

from numpy import reshape, linspace

logging.basicConfig(level=logging.INFO)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Needed on Supercloud =/

matplotlib.rcParams['figure.dpi'] = 200
plt.style.use('dark_background')

def plot_vertical_profiles(file, field, Is, save=True):
    rotation_period = file["parameters/rotation_period"][()]

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 16))
    for i in Is:
        i = str(i)
        t = file["timeseries/t/" + i]
        
        Nz, Lz = file["grid/Nz"][()], file["grid/Lz"][()]
        z = linspace(0, -Lz, Nz)
        
        Fp = file["timeseries/" + field + "/" + i][()]
        Fp = reshape(Fp, Nz)
        
        ax1.plot(Fp, z / 1000, label="{:.0f} days".format(t / rotation_period))
    
    ax1.set_title("Horizontally averaged {:s} profiles".format(field))
    ax1.set_ylim([-100, 0])
    ax1.set_xlabel(f"{field}")
    ax1.set_ylabel("z (km)")
    ax1.legend(frameon=False)

    if save:
        filename = f"{field}_profiles.png"
        logging.info(f"Saving: {filename:s}")
        plt.savefig(filename, dpi=300, format="png", transparent=False)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot vertical profiles from Oceananigans.jl JLD2 output.")
    parser.add_argument("-f", "--file", type=str, required=True, help="JLD2 output filepath")
    args = parser.parse_args()
    filepath = args.file

    file = h5py.File(filepath, "r")
    Is = sorted(list(map(int, list(file["timeseries/t"].keys()))))  # List of iterations with output.
    logging.info(f"{len(Is):d} vertical profiles found: i={min(Is)}->{max(Is)}")

    Ip = [Is[n] for n in [0, 10, 30, 50, 80]]  # Iterations to plot.
    plot_vertical_profiles(file, "T", Ip)
    plot_vertical_profiles(file, "u", Ip)
    plot_vertical_profiles(file, "v", Ip)
    plot_vertical_profiles(file, "wT", Ip)


