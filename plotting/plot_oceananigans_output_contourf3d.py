import matplotlib

matplotlib.use("Agg")

import os
import logging
import argparse

import h5py
import ffmpeg
import cmocean
import numpy as np
import matplotlib.pyplot as plt

from numpy import ones, meshgrid, linspace, square, mean
from mpl_toolkits.mplot3d import axes3d

from human_sorting import sort_nicely

plt.switch_backend("Agg")

import joblib

def plot_contourf3d_from_jld2(slice_filepath, profile_filepath, png_filepath, field, i, vmin, vmax, n_contours, cmap="inferno"):
    # For some reason I need to do the import here so it shows up on all joblib workers.
    from mpl_toolkits.mplot3d import Axes3D

    # Enforcing style in here so that it's applied to all workers launched by joblib.
    plt.style.use("dark_background")

    sfile = h5py.File(slice_filepath, "r")
    pfile = h5py.File(profile_filepath, "r")

    i = str(i)
    t = sfile["timeseries/t/" + i][()]
    Nx, Lx = sfile["grid/Nx"][()], sfile["grid/Lx"][()]
    Ny, Ly = sfile["grid/Ny"][()], sfile["grid/Ly"][()]
    Nz, Lz = sfile["grid/Nz"][()], sfile["grid/Lz"][()]

    x, y, z = linspace(0, Lx, Nx), linspace(0, Ly, Ny), linspace(0, -Lz, Nz)

    xy_slice = sfile["timeseries/" + field + "_xy_slice/" + i][()][1:Nx+1, 1:Ny+1]
    xz_slice = sfile["timeseries/" + field + "_xz_slice/" + i][()][1:Nx+1, 1:Nz+1]
    yz_slice = sfile["timeseries/" + field + "_yz_slice/" + i][()][1:Ny+1, 1:Nz+1]

    T = pfile["timeseries/T/" + i][()][1:Nz+1, 0, 0]

    if field == "T":
        T_avg = T[:, np.newaxis]
        xy_slice = xy_slice - T_avg[0]
        xz_slice = xz_slice - T_avg
        yz_slice = yz_slice - T_avg

    fig = plt.figure(figsize=(16, 9))

    ax = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3, projection="3d")
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.25)

    XC_z, YC_z = meshgrid(linspace(0, Lx, Nx), linspace(0, Ly, Ny))
    YC_x, ZC_x = meshgrid(linspace(0, Ly, Ny), linspace(0, -Lz, Nz))
    XC_y, ZC_y = meshgrid(linspace(0, Lx, Nx), linspace(0, -Lz, Nz))

    x_offset, y_offset, z_offset = Lx/1000, 0, 0

    contour_spacing = (vmax - vmin) / n_contours
    levels = np.arange(vmin, vmax, contour_spacing)

    cf1 = ax.contourf(XC_z, YC_z, xy_slice, zdir="z", offset=z_offset, levels=levels, cmap=cmap)
    cf2 = ax.contourf(yz_slice, YC_x, ZC_x, zdir="x", offset=x_offset, levels=levels, cmap=cmap)
    cf3 = ax.contourf(XC_y, xz_slice, ZC_y, zdir="y", offset=y_offset, levels=levels, cmap=cmap)

    clb = fig.colorbar(cf1, ticks=[-0.1, -0.05, 0, 0.05, 0.1], shrink=0.9)
    clb.ax.set_title(r"$T^\prime$ (°C)")

    ax.set_xlim3d(0, Lx)
    ax.set_ylim3d(0, Ly)
    ax.set_zlim3d(-Lz, 0)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    ax.view_init(elev=30, azim=-135)

    ax.set_title("Convecting wind stress: temperature anomaly @ t = {:05d} s ({:02.2f} hours)".format(int(t), t / 3600), y=1.05)

    ax.set_xticks(linspace(0, Lx, num=5))
    ax.set_yticks(linspace(0, Ly, num=5))
    ax.set_zticks(linspace(0, -Lz, num=5))

    ax2 = plt.subplot2grid((3, 4), (0, 3))
    ax2.plot(T, z)
    ax2.set_title(r"$\overline{T}(z)$ (°C)")
    ax2.set_ylabel("z (m)")
    ax2.set_xlim(19.75, 20)
    ax2.set_ylim(-25, 0)

    uu = np.square(pfile["timeseries/u/" + i][()][1:Nz+1, 0, 0])
    vv = np.square(pfile["timeseries/v/" + i][()][1:Nz+1, 0, 0])

    ax3 = plt.subplot2grid((3, 4), (1, 3))
    ax3.plot(uu + vv, z, color="tab:orange", label=r"$u^2 + v^2$")
    ax3.set_title("Kinetic energy (m$^2$/s$^2$)")
    ax3.set_ylabel("z (m)")
    ax3.set_xlim(-0.001, 0.05)
    ax3.set_ylim(-25, 0)
    ax3.legend(loc="lower right", frameon=False)

    alpha = 2.07e-4
    g = 9.80665
    wT = pfile["timeseries/wT/" + i][()][1:Nz+1, 0, 0]

    ax4 = plt.subplot2grid((3, 4), (2, 3))
    ax4.plot(alpha * g * wT, z, color="tab:red")
    ax4.set_title(r"Buoyancy flux $\alpha g \overline{w' T'}$ (m$^2$/s$^3$)")
    ax4.set_ylabel("z (m)")
    ax4.set_xlim(-10e-8, 10e-8)
    ax4.set_ylim(-25, 0)

    # plt.show()

    plt.savefig(png_filepath, dpi=300, format="png", transparent=False)
    print("Saving: {:s}".format(png_filepath))

    plt.close("all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make horizontal slice movies from Oceananigans.jl JLD2 output.")
    parser.add_argument("-s", "--slices", type=str, nargs='+', required=True, help="JLD2 slice output filepaths")
    parser.add_argument("-p", "--profiles", type=str, required=True, help="JLD2 profile output filepath")
    args = parser.parse_args()
    slice_filepaths, profile_filepath = args.slices, args.profiles
    sort_nicely(slice_filepaths)

    Is = []  # We'll generate a list of iterations with output across all files.
    for fp in slice_filepaths:
        file = h5py.File(fp, 'r')
        new_Is = sorted(list(map(int, list(file["timeseries/t"].keys()))))
        tuplified_Is = list(map(lambda i: (i, fp), new_Is))
        Is.extend(tuplified_Is)

    logging.info(f"Found {len(Is):d} snapshots per field across {len(slice_filepaths):d} files: i={Is[0][0]}->{Is[-1][0]}")

    # Plot many frames in parallel.
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(plot_contourf3d_from_jld2)(slice_filepath=I[1], profile_filepath=profile_filepath, i=I[0], png_filepath="convecting_wind_stress_{:05d}.png".format(frame_number),
                                                  field="T", vmin=-0.105, vmax=0.105, n_contours=100, cmap=cmocean.cm.balance)
                                                  for frame_number, I in enumerate(Is[1:-1]))

    # for frame_number, I in enumerate(Is[0:15000:1500]):
    #     plot_contourf3d_from_jld2(slice_filepath=I[1], profile_filepath=profile_filepath, i=I[0], png_filepath="convecting_wind_stress_{:05d}.png".format(frame_number),
    #                               field="T", vmin=-0.082, vmax=0.082, n_contours=100, cmap=cmocean.cm.balance)

    (
        ffmpeg
        .input("convecting_wind_stress_%05d.png", framerate=30)
        .output("convecting_wind_stress.mp4", crf=15, pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )
