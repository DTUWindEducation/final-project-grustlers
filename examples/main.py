'''
This is the main script
'''

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

path_main = Path(__file__)  # main file
path_dir = path_main.parent  # main file folder: 'examples'
path_package = path_dir.parent  # package path: 'final-project-grustlers'

data_set = path_package / 'inputs/1997-1999.nc'

ds = xr.load_dataset(data_set)

# print(ds.sizes)

for k in ds.variables.keys():
    print(k)

T, LAT, LON = ds['u10'].shape
velocity_keys = list(ds.variables.keys())[-4:]
colors = ['red', 'blue', 'orange', 'purple']

v_k = velocity_keys[0]

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for i in range(LAT):
    for j in range(LON):
        # for v_k in velocity_keys:
        ds[v_k][:, i, j].plot(ax=ax[i, j], label=v_k)
        ax[i, j].tick_params(axis='x', rotation=20)
        ax[i, j].set_ylabel("Wind speed [m/s]")
        ax[i, j].set_title(f"Latitude: {ds['latitude'].values[i]}, "
                           + f"Longitude: {ds['longitude'].values[j]}")
        # ax[i, j].legend()
plt.suptitle(f"Velocity: {v_k}")
plt.tight_layout()
plt.show()
