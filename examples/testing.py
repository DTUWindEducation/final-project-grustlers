import numpy as np

from scipy.interpolate import interpn

'''
This is the main script
'''

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import sys


# child class of DataSites used to interpolate for a input of x,y
# attributes: x, y, z, name, year


path_main = Path(__file__)  # main file
path_dir = path_main.parent  # main file folder: 'examples'
path_package = path_dir.parent  # package path: 'final-project-grustlers'

data_set = path_package / 'inputs/2009-2011.nc'

ds = xr.load_dataset(data_set)

u10 = ds['u10'].values

 
# Example grid definitions

lats = [0, 1]

lons = [0, 1]

times = np.arange(26280)  # time steps
 
# Example wind arrays: shape = (time, len(lats), len(lons))

# These should already be defined in your environment:

# u10, v10, u100, v100
 
# Generalized interpolation function

u10 = ds['u10'].values
u100 = ds['u100'].values
v10 = ds['v10'].values
v100 = ds['v100'].values

def interpolate_wind_components(lat, lon, u10, v10, u100, v100, lats, lons):

    grid = (times, lats, lons)

    points = np.column_stack((

        times, 

        np.full_like(times, lat, dtype=float), 

        np.full_like(times, lon, dtype=float)

    ))
 
    u10_interp = interpn(grid, u10, points, bounds_error=True)

    v10_interp = interpn(grid, v10, points, bounds_error=True)

    u100_interp = interpn(grid, u100, points, bounds_error=True)

    v100_interp = interpn(grid, v100, points, bounds_error=True)
 
    return u10_interp, v10_interp, u100_interp, v100_interp
 
# Example usage

lat = 0.5

lon = 0.5

u10_res, v10_res, u100_res, v100_res = interpolate_wind_components(lat, lon, u10, v10, u100, v100, lats, lons)

print(u10_res)

lat = 0
lon = 1
u10_res, v10_res, u100_res, v100_res = interpolate_wind_components(lat, lon, u10, v10, u100, v100, lats, lons)

print(u10_res)
print(u10[:,0,1])

lat = 2

lon = 3

u10_res, v10_res, u100_res, v100_res = interpolate_wind_components(lat, lon, u10, v10, u100, v100, lats, lons)

print(u10_res)