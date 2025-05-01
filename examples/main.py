'''
This is the main script
'''
import sys
import os
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#  import dask -- used inside xarray.open_mfdataset

sys.path.insert(0, __file__.replace('main.py', '../src'))

from WRA import classes
from WRA import functions

path_main = Path(__file__)  # main file
path_dir = path_main.parent  # main file folder: 'examples'
path_package = path_dir.parent  # package path: 'final-project-grustlers'

'''
data_set = path_package / 'inputs/2009-2011.nc'
ds = xr.load_dataset(data_set)
velocity_keys = list(ds.variables.keys())[-4:]
'''
# Lat: [8.   7.75]
# Lon: [55.5  55.75]
folder_name = 'inputs'
folder_path = path_package / folder_name
ds = functions.load_nc_folder_to_dataset(folder_path)


ds1 = classes.DataSite(ds, latitude=8, longitude=55.5, ref_heights=[10, 100], name="P1")
print(ds1.v_x)
print("test")
ds1.get_velocity_site()  # 2) ws time series at the 2 heights at 1 of 4 locs
ds1.get_angle_rad()  # 2) wd time series at the 2 heights at 1 of 4 locs
ds1.get_angle_deg()
ds1.get_alpha()

'''Added from previous "Testing"-file to demonstrate 
the InterpolatedSite class'''

# Grid definition:
lats = [0, 1]
lons = [0, 1]
time_steps = np.arange(26280)

# Velocity components at the two heights
u10 = ds['u10'].values
u100 = ds['u100'].values
v10 = ds['v10'].values
v100 = ds['v100'].values

# Example usage
lat = 0.5
lon = 0.5

#u10_res, v10_res, u100_res, v100_res = classes.interpolate_wind_components(lat, lon, u10, v10, u100, v100, time_steps, lats, lons)

# print(u10_res)

lat = 0
lon = 1
#u10_res, v10_res, u100_res, v100_res = classes.interpolate_wind_components(lat, lon, u10, v10, u100, v100, time_steps, lats, lons)

# print(u10_res)
# print(u10[:,0,1])
lat = 7.9
lon = 55.6
is1 = classes.InterpolatedSite(ds, lat, lon, 50)
#print(is1.v_x_sites)
#print(list(is1.v_x_sites.keys()))
#print(is1.v_x_sites['100'])
#print(is1.v_x_sites['100']['(7.75,55.5)'])
#print(len(is1.v_x_sites['100']['(7.75,55.5)']))
print(is1.interpolate_wind_components())  # Preparation for 3) Velocity components

#print(is1.load_components_at_sites())
print(is1.get_velocity_site())  # 3) Compute ws and wd at 10, 100 m inside box
test_data = is1.calculate_wind_at_height()  # 4) Compute ws at given height
is1.weibull_distribution()  # 5, 6) Fit Weib dist @ height z inside box w/ hist
print('test')
print(is1.get_angle_rad())
print(is1.angle_point_rad)

is1.show_wind_rose()  # 7) Show wind rose inside the box at a given height
# there's barely any effect of hieght on wind direction
# illegal_height=101
# is2 = classes.InterpolatedSite(ds, lat, lon, illegal_height)

power_curve_path1 = path_package / 'inputs/NREL_Reference_5MW_126.csv'
power_curve_path2 = path_package / 'inputs/NREL_Reference_15MW_240.csv'
power_curve_data = pd.read_csv(power_curve_path1)
AEP, CF = is1.get_AEP(power_curve=power_curve_data,
                      year=2000, show_curve=False)

print(f"At {is1.lat_point:.1f}°N, {is1.lon_point:.1f}°E, {is1.h_point:.1f} m:")
print(f"AEP = {AEP:.2f} MWh")
print(f"CF = {CF*100:.2f}%")

# NEW
AEPs, CFs = is1.compare_AEPs(power_curve=power_curve_data)
