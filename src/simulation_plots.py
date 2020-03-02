import subprocess
import logging
logging.getLogger().setLevel(logging.INFO)

import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import ffmpeg

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def make_movie(filename_pattern, movie_filename, fps=15):
    (
        ffmpeg
        .input(filename_pattern, framerate=fps)
        .output(movie_filename, crf=15, pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )

def plot_fields(ds):
    Nt = ds.time.size
    Nx = ds.xC.size

    for n in range(0, Nt, 1):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=300)

        t = ds.time.values[n] / 86400
        fig.suptitle(f"t = {t:.3f} days", fontsize=16)

        u = ds.u.isel(time=n, xF=0).squeeze()
        u.plot.pcolormesh(ax=axes[0, 0], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 0].set_title("")

        w = ds.w.isel(time=n, xC=0).squeeze()
        w.plot.pcolormesh(ax=axes[0, 1], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 1].set_title("")

        T = ds.T.isel(time=n, xC=0).squeeze()
        T.plot.pcolormesh(ax=axes[1, 0], vmin=15, vmax=20, cmap=cmocean.cm.thermal, extend="both")
        axes[1, 0].set_title("")

        S = ds.S.isel(time=n, xC=0).squeeze()
        S.plot.pcolormesh(ax=axes[1, 1], vmin=0, vmax=1, levels=21, cmap=cmocean.cm.haline)
        axes[1, 1].set_title("")

        png_filename = f"yz_slice_{n:05d}.png"
        logging.info(f"Saving: {png_filename}...")
        plt.savefig(png_filename)

        plt.close("all")

dsf = xr.open_dataset("lesbrary_lat190_lon-55_days10_fields.nc")
dsp = xr.open_dataset("lesbrary_lat190_lon-55_days10_profiles.nc")
dsl = xr.open_dataset("lesbrary_lat190_lon-55_days10_large_scale.nc")

plot_fields(dsf)
make_movie("yz_slice_%05d.png", "yz_slice.mp4")

