'''Classes and functions used in the package'''

import numpy as np


class DataSites:
    ''' This class is used to handle raw data and
    to compute fundamental values.'''
    def __init__(self, data_set, latitude, longitude, name=None):
        '''Initializes the class with attributes for each
        velocity component in east- and northward directions.'''
        lat_dict = {lat: i for i, lat in enumerate(data_set['latitude'].values)}
        lon_dict = {lon: i for i, lon in enumerate(data_set['longitude'].values)}
        
        self.name = name

        self.lat_coord = latitude
        self.lon_coord = longitude

        lat_iter = lat_dict[self.lat_coord]
        lon_iter = lon_dict[self.lon_coord]

        self.v_x10 = data_set['u10'][:, lat_iter, lon_iter].values
        self.v_y10 = data_set['v10'][:, lat_iter, lon_iter].values
        self.v_x100 = data_set['u100'][:, lat_iter, lon_iter].values
        self.v_y100 = data_set['v100'][:, lat_iter, lon_iter].values

    def get_velocity(self):
        '''This is a method'''
        v_10 = np.sqrt(self.v_x10 ** 2 + self.v_y10 ** 2)
        v_100 = np.sqrt(self.v_x100 ** 2 + self.v_y100 ** 2)
        return {'10': v_10, '100': v_100}

    def get_angle(self):
        '''This is a method'''
        theta_10 = np.arctan(self.v_x10 / self.v_y10)
        theta_100 = np.arctan(self.v_x100 / self.v_y100)
        return {'10': theta_10, '100': theta_100}

    def get_surface_roughness(self):
        '''This is a method'''
        z = 100
        z_ref = 10
        u_z = self.get_velocity()['100']
        u_z_ref = self.get_velocity()['10']
        r = u_z / u_z_ref
        z_0 = np.exp((r * np.log(z_ref) - np.log(z)) / (r - 1))
        # unrealistically high roughness set to 10 below
        z_0_mean = np.mean(z_0[z_0 < 10])
        return z_0_mean

class InterpolatedSites(DataSites):

    def __init__(self, data_set, latitude, longitude, name=None):
        super().__init__(data_set, latitude, longitude, name)