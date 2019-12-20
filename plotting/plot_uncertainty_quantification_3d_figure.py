import matplotlib

matplotlib.use("Agg")

import os

import h5py
import cmocean
import numpy as np
import matplotlib.pyplot as plt

from numpy import ones, meshgrid, linspace, square, mean, flip
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import axes3d

plt.switch_backend("Agg")

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

png_filepath = "free_convection_figure.png"
vmin, vmax = 19.49, 19.725
n_contours = 100
cmap = "YlGnBu"

file = h5py.File("ocean_convection_fields.jld2", "r")

i = "58333"

t = file["timeseries/t/" + i][()]
Nx, Lx = file["grid/Nx"][()], file["grid/Lx"][()]
Ny, Ly = file["grid/Ny"][()], file["grid/Ly"][()]
Nz, Lz = file["grid/Nz"][()], file["grid/Lz"][()]

print(f"i = {i}, t = {t/86400} days")

x, y, z = linspace(0, Lx, Nx), linspace(0, Ly, Ny), linspace(0, -Lz, Nz)

Nzh = int(Nz/2)
xy_slice = file["timeseries/T/" + i][()][Nz, 1:Ny+1, 1:Nx+1][()]
xz_slice = flip(file["timeseries/T/" + i][()][Nzh+1:Nz+1, 1, 1:Nx+1][()], axis=0)
yz_slice = flip(file["timeseries/T/" + i][()][Nzh+1:Nz+1, 1:Ny+1, 1][()], axis=0)

T_3D = flip(file["timeseries/T/" + i][()][1:Nz+1, 1:Ny+1, 1:Nx+1][()], axis=0)
T_prof = mean(T_3D, axis=(1, 2))

fig = plt.figure(figsize=(16, 9))

ax = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3, projection="3d")
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.25)

XC_z, YC_z = meshgrid(linspace(0, Lx, Nx), linspace(0, Ly, Ny))
YC_x, ZC_x = meshgrid(linspace(0, Ly, Ny), linspace(0, -Lz/2, Nzh))
XC_y, ZC_y = meshgrid(linspace(0, Lx, Nx), linspace(0, -Lz/2, Nzh))

x_offset, y_offset, z_offset = Lx/1000, 0, 0

contour_spacing = (vmax - vmin) / n_contours
levels = np.arange(vmin, vmax, contour_spacing)

cf1 = ax.contourf(XC_z, YC_z, xy_slice, zdir="z", offset=z_offset, levels=levels, cmap=cmap, norm=PowerNorm(gamma=2))
cf2 = ax.contourf(yz_slice, YC_x, ZC_x, zdir="x", offset=x_offset, levels=levels, cmap=cmap, norm=PowerNorm(gamma=2))
cf3 = ax.contourf(XC_y, xz_slice, ZC_y, zdir="y", offset=y_offset, levels=levels, cmap=cmap, norm=PowerNorm(gamma=2))

clb = fig.colorbar(cf1, ticks=[19.5, 19.55, 19.6, 19.65, 19.7], shrink=0.9)
clb.ax.set_title(r"$T$ (°C)")

ax.set_xlim3d(0, Lx)
ax.set_ylim3d(0, Ly)
ax.set_zlim3d(-Lz/2, 0)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")

ax.view_init(elev=30, azim=-135)

ax.set_xticks(linspace(0, Lx, num=5))
ax.set_yticks(linspace(0, Ly, num=5))
ax.set_zticks(linspace(0, -Lz/2, num=6))

ax2 = plt.subplot2grid((3, 4), (0, 3), rowspan=3)
ax2.plot(T_prof, z)
ax2.set_xlabel(r"$\overline{T}(z)$ (°C)")
ax2.set_ylabel("z (m)")
ax2.set_xlim(19.5, 19.75)
ax2.set_ylim(-Lz/2, 0)

plt.savefig(png_filepath, dpi=300, format="png", transparent=False)
print("Saving: {:s}".format(png_filepath))

