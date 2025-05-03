# pylint: disable=C0103  # Disable warnings about UPPER-CASE variable naming
'''
All of the functions provided by the WRA package are demonstrated in this file.
WRA is short for Wind Resource Assessment.

It is explicitly shown how the package can be used to provide the answers
required by the "Functional requirements" of this WRA project.
'''
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

# Change the working directory to the package path
sys.path.insert(0, __file__.replace('main.py', '../src'))

# Import the package modules
from WRA import classes
from WRA import functions

# Start the timer
start_time = time.time()

# Define the path to the folder containing the ERA5 weather data (.nc files).
path_main = Path(__file__)  # main file
path_dir = path_main.parent  # main file folder: 'examples'
path_package = path_dir.parent  # package path: 'final-project-grustlers'
folder_name = 'inputs'
folder_path = path_package / folder_name

# 1. Load and parse multiple provided netCDF4 files.
data = functions.load_nc_folder_to_dataset(folder_path)

# 2. Compute wind speed and wind direction time series at 10 m and 100 m
# heights for the four provided locations.
ref_lats = data.latitude.values  # 55.75, 55.5
ref_longs = data.longitude.values  # 7.75, 8
ref_heights = np.unique([int(k.replace('u', '').replace('v', ''))
                         for k in list(data.keys())])  # 10, 100 m

z = 50  # [m], height, used later

ds1 = classes.DataSite(data, latitude=ref_lats[0], longitude=ref_longs[0],
                       height_point=z, ref_heights=ref_heights, name="DS1")
ds2 = classes.DataSite(data, latitude=ref_lats[1], longitude=ref_longs[0],
                       height_point=z, ref_heights=ref_heights, name="DS2")
ds3 = classes.DataSite(data, latitude=ref_lats[0], longitude=ref_longs[1],
                       height_point=z, ref_heights=ref_heights, name="DS3")
ds4 = classes.DataSite(data, latitude=ref_lats[1], longitude=ref_longs[1],
                       height_point=z, ref_heights=ref_heights, name="DS4")

# The following two lines can be called for each of the four locations to
# the wind speed and wind direction time series at 10 m and 100 m.
ds1.get_velocity_site()
ds1.get_angle_rad()  # or ds1.get_angle_deg()

# 3. Compute wind speed and wind direction time series at 10 m and 100 m
# heights for a given location inside the box bounded by the four locations,
# such as the Horns Rev 1 site, using interpolation.
lat = 55.6
lon = 7.9
is1 = classes.InterpolatedSite(dataset=data, latitude_point=lat,
                               longitude_point=lon, height_point=z,
                               ref_heights=ref_heights, name="IS1")
is1.get_velocity_site()
is1.get_angle_rad()  # or is1.get_angle_deg()

# 4. Compute wind speed time series at height z for the four provided
# locations using power law profile.

# The following line demonstrates how to do it for a single location and
# should be repeated for each of the other three locations to determine
# their velocities at z.
ds1.calculate_wind_at_height()

# 5. Fit Weibull distribution for wind speed at a given location (inside the
# box) and a given height.
# 6. Plot wind speed distribution (histogram vs. fitted Weibull distribution)
# at a given location (inside the box) and a given height.
k, A = is1.weibull_distribution()

# 7. Plot wind rose diagram that showes the frequencies of different wind
# direction at a given location (inside the box) and a given height.
is1.show_wind_rose()

# 8. Compute AEP of a specifed wind turbine (NREL 5 MW or NREL 15 MW) at a
# given location inside the box for a given year in the period we have
# provided the wind data.

# Provide power curve data as a pandas dataframe. See the module for details.
power_curve_path1 = path_package / 'inputs/NREL_Reference_5MW_126.csv'
power_curve_path2 = path_package / 'inputs/NREL_Reference_15MW_240.csv'
power_curve_data = pd.read_csv(power_curve_path2)
AEP, CF = is1.get_AEP(power_curve=power_curve_data, year=2000)

print(f"At {is1.lat_point:.1f}°N, {is1.lon_point:.1f}°E, {is1.h_point:.1f} m:")
print(f"AEP = {AEP:.2f} MWh")
print(f"CF = {CF*100:.2f}%")


# The two additional functions include:
# - plot_velocity_site() : (method) easy visualization of time series at a site
# - compare_AEPs_years() : (method) compares yearly AEPs given a power curve
ds1_ws = ds1.plot_velocity_site(year=1998)

AEPs, CFs = is1.compare_AEPs_years(power_curve=power_curve_data)

# Stop the timer
print(f"\nElapsed time: {time.time() - start_time:.2f} seconds")
print('Note if not running in an interactive window this time will include'
      ' the time it takes to close the figures.')
