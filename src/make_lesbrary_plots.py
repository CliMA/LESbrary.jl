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
    lat, lon = ds.attrs["lat"], ds.attrs["lon"]
    fig.suptitle(f"Forcing at {lat}°N, {lon}°E")

    time = ds.time.values / 86400
    τx = ds.τx.values
    τy = ds.τy.values
    QT = ds.QT.values
    QS = ds.QS.values

    axes[0].plot(time, τx, label="τx")
    axes[0].plot(time, τy, label="τy")
    axes[0].legend(loc="best", frameon=False)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Wind stress ()N/m²)")

    axes[1].plot(time, QT)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("QT ()W/m²)")

    axes[2].plot(time, QS)
    axes[2].set_xlabel("time (days)")
    axes[2].set_ylabel("QS (kg/m²/s)")

    logging.info(f"Saving: {filename}...")
    plt.savefig(filename)

def plot_large_scale(ds):
    for n in range(0, 10, 1):
        fig, axes = plt.subplots(ncols=3, figsize=(16, 9), dpi=300)

        t = ds.time.values[n] / 86400
        lat, lon = ds.attrs["lat"], ds.attrs["lon"]
        fig.suptitle(f"Large scale at {lat}°N, {lon}°E, t = {t:.3f} days", fontsize=16)

        u = ds.u.isel(time=n).squeeze()
        v = ds.u.isel(time=n).squeeze()
        u.plot(ax=axes[0], y="zC")
        v.plot(ax=axes[0], y="zC")

        T = ds.T.isel(time=n).squeeze()
        T.plot(ax=axes[1])

        S = ds.S.isel(time=n).squeeze()
        S.plot(ax=axes[2])

        png_filename = f"large_scale_{n:05d}.png"
        logging.info(f"Saving: {png_filename}...")
        plt.savefig(png_filename)

        plt.close("all")

def plot_slices(ds):
    for n in range(0, ds.time.size, 1):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=300)

        t = ds.time.values[n] / 86400
        lat, lon = ds.attrs["lat"], ds.attrs["lon"]
        fig.suptitle(f"Slices at {lat}°N, {lon}°E, t = {t:.3f} days", fontsize=16)

        u = ds.u.isel(time=n).squeeze()
        u.plot.pcolormesh(ax=axes[0, 0], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 0].set_title("")

        w = ds.w.isel(time=n).squeeze()
        w.plot.pcolormesh(ax=axes[0, 1], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 1].set_title("")

        T = ds.T.isel(time=n).squeeze()
        T.plot.pcolormesh(ax=axes[1, 0], cmap=cmocean.cm.thermal, extend="both")
        axes[1, 0].set_title("")

        S = ds.S.isel(time=n).squeeze()
        S.plot.pcolormesh(ax=axes[1, 1], cmap=cmocean.cm.haline, extend="both")
        axes[1, 1].set_title("")

        png_filename = f"slice_{n:05d}.png"
        logging.info(f"Saving: {png_filename}...")
        plt.savefig(png_filename)

        plt.close("all")

def plot_surfaces(ds):
    for n in range(0, ds.time.size, 1):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=300)

        t = ds.time.values[n] / 86400
        lat, lon = ds.attrs["lat"], ds.attrs["lon"]
        fig.suptitle(f"Surface at {lat}°N, {lon}°E, t = {t:.3f} days", fontsize=16)

        u = ds.u.isel(time=n).squeeze()
        u.plot.pcolormesh(ax=axes[0, 0], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 0].set_title("")

        w = ds.w.isel(time=n).squeeze()
        w.plot.pcolormesh(ax=axes[0, 1], vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance, extend="both")
        axes[0, 1].set_title("")

        T = ds.T.isel(time=n).squeeze()
        T.plot.pcolormesh(ax=axes[1, 0], cmap=cmocean.cm.thermal, extend="both")
        axes[1, 0].set_title("")

        S = ds.S.isel(time=n).squeeze()
        S.plot.pcolormesh(ax=axes[1, 1], cmap=cmocean.cm.haline, extend="both")
        axes[1, 1].set_title("")

        png_filename = f"surface_{n:05d}.png"
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
    plot_large_scale(dsl)
    make_movie("large_scale_%05d.png", "large_scale.mp4")
    # plot_slices(dsx)
    # make_movie("slice_%05d.png", "slice.mp4")
    # plot_surfaces(dss)
    # make_movie("surface_%05d.png", "surface.mp4")

make_lesbrary_plots(-60, 275, 10)

