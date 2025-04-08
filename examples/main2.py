'''
This is the main script
'''

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import sys

# # sys.path.insert(0, '../src')
sys.path.insert(0, __file__.replace('main2.py', '../src'))

import WRA


# child class of DataSites used to interpolate for a input of x,y
# attributes: x, y, z, name, year


path_main = Path(__file__)  # main file
path_dir = path_main.parent  # main file folder: 'examples'
path_package = path_dir.parent  # package path: 'final-project-grustlers'

data_set = path_package / 'inputs/2009-2011.nc'

ds = xr.load_dataset(data_set)

velocity_keys = list(ds.variables.keys())[-4:]

ds1 = WRA.DataSites(ds, "ds1")
ds1.v_x100
ds1.get_velocity()['100']
ds1.get_angle()['10']
ds1.get_surface_roughness()

LAT=2
LON=2

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for i in range(LAT):
    for j in range(LON):
        # for v_k in velocity_keys:
        ds1.get_velocity()['100'][:, i, j].plot(ax=ax[i, j], label='100 m')
        ax[i, j].tick_params(axis='x', rotation=20)
        ax[i, j].set_ylabel("Wind speed [m/s]")
        ax[i, j].set_title(f"Latitude: {ds1.lat[i]}, "
                           + f"Longitude: {ds1.lon[j]}")
        # ax[i, j].legend()
plt.suptitle("Velocity: 100 m")
plt.tight_layout()
plt.show()
