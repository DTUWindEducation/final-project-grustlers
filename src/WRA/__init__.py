'''Classes and functions used in the package'''

import numpy as np


class DataSites:
    ''' This class is used to handle raw data and
    to compute fundamental values.'''
    def __init__(self, data_set, name=None):
        '''Initializes the class with attributes for each
        velocity component in east- and northward directions.'''
        self.lat = data_set['latitude'].values
        self.lon = data_set['longitude'].values
        self.v_x10 = data_set['u10']
        self.v_y10 = data_set['v10']
        self.v_x100 = data_set['u100']
        self.v_y100 = data_set['v100']

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
        z_0_mean = [np.mean(
            z_0.values[:, lat, lon][z_0.values[:, lat, lon] < 10])
            for lat in range(2) for lon in range(2)]
        return z_0_mean
