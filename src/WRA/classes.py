import numpy as np

class WindCalculation():
    def __init__(self, v_x, v_y):
        self.v_x = v_x
        self.v_y = v_y
    
    def get_velocity(self):
        return np.sqrt(self.v_x**2 + self.v_y**2)

    def get_angle(self):
        return np.arctan2(self.v_y, self.v_x)
    
    def get_surface_roughness(self, z=10, z_ref=100):
        return "???"


class DataSite(WindCalculation):
    def __init__(self, dataset, latitude, longitude, height, name=None):
        self.data = dataset
        self.lat = latitude
        self.lon = longitude
        self.h = height
        self.name = name

        self.v_x, self.v_y = self.load_components()
        super().__init__(self.v_x, self.v_y)
    
    def load_components(self):
        lat_dict = {lat: i for i, lat in
                    enumerate(self.data['latitude'].values)}
        lon_dict = {lon: i for i, lon in
                    enumerate(self.data['longitude'].values)}
        
        lat_iter = lat_dict[self.lat]
        lon_iter = lon_dict[self.lon]

        u_str = 'u'+str(self.h)
        v_str = 'v'+str(self.h)

        v_x = self.data[u_str][:, lat_iter, lon_iter].values
        v_y = self.data[v_str][:, lat_iter, lon_iter].values

        return v_x, v_y



# class InterpolatedSite(WindCalculation):
