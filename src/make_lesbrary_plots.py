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

def plot_forcing(ds, filename="forcing_time_series.png"):
    fig, axes = plt.subplots(nrows=3, figsize=(16, 9), dpi=300)
    lat, lon = ds.attrs["lat"], dsl.attrs["lon"]
    fig.suptitle(f"Forcing at {lat}°N, {lon}°E")

    ds.τx.plot(ax=axes[0])
    ds.τy.plot(ax=axes[0])
    ds.QT.plot(ax=axes[1])
    ds.QS.plot(ax=axes[2])

    logging.info(f"Saving: {filename}...")
    plt.savefig(filename)

def plot_slices(ds):
    for n in range(0, ds.time.size, 1):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=300)

        t = ds.time.values[n] / 86400
        lat, lon = ds.attrs["lat"], dsl.attrs["lon"]
        fig.suptitle(f"Slices at {lat}°N, {lon}°E, t = {t:.3f} days", fontsize=16)

        u = ds.u.isel(time=n).squeeze()
        u.plot.pcolormesh(ax=axes[0, 0], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 0].set_title("")

        w = ds.w.isel(time=n).squeeze()
        w.plot.pcolormesh(ax=axes[0, 1], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 1].set_title("")

        T = ds.T.isel(time=n).squeeze()
        T.plot.pcolormesh(ax=axes[1, 0], vmin=15, vmax=20, cmap=cmocean.cm.thermal, extend="both")
        axes[1, 0].set_title("")

        S = ds.S.isel(time=n).squeeze()
        S.plot.pcolormesh(ax=axes[1, 1], vmin=33.8, vmax=34.8, cmap=cmocean.cm.haline, extend="both")
        axes[1, 1].set_title("")

        png_filename = f"slice_{n:05d}.png"
        logging.info(f"Saving: {png_filename}...")
        plt.savefig(png_filename)

        plt.close("all")

def make_lesbrary_plots(lat, lon, days):
    dsf = xr.open_dataset(f"lesbrary_lat{lat}_lon{lon}_days{days}_fields.nc")
    dss = xr.open_dataset(f"lesbrary_lat{lat}_lon{lon}_days{days}_surface.nc")
    dsx = xr.open_dataset(f"lesbrary_lat{lat}_lon{lon}_days{days}_slice.nc")
    dsp = xr.open_dataset(f"lesbrary_lat{lat}_lon{lon}_days{days}_profiles.nc")
    dsl = xr.open_dataset(f"lesbrary_lat{lat}_lon{lon}_days{days}_large_scale.nc")

    plot_forcing(dsl)
    plot_slices(dsx)
    make_movie("slice_%05d.png", "slice.mp4")

make_lesbrary_plots(-60, 275, 10)

