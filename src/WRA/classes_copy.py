import numpy as np

class WindCalculation():
    def __init__(self, v_x, v_y):
        self.v_x = v_x
        self.v_y = v_y
    
    def get_velocity_site(self):
        return np.sqrt(self.v_x**2 + self.v_y**2)

    def get_angle(self):
        return np.arctan2(self.v_y, self.v_x)
    
    def get_alpha(self, z=10, z_ref=100):
        u_z = self.get_velocity_site()[0]
        u_z_ref = self.get_velocity_site()[1]
        return "???"


class DataSite(WindCalculation):
    def __init__(self, dataset, latitude, longitude, height=[10, 100], name=None):
        self.data = dataset
        self.lat = latitude
        self.lon = longitude
        self.h = height
        self.name = name
        self.data_length = self.data.sizes['valid_time']

        self.v_x, self.v_y = self.load_components()
        super().__init__(self.v_x, self.v_y)
    
    def load_components(self):
        lat_dict = {lat: i for i, lat in
                    enumerate(self.data['latitude'].values)}
        lon_dict = {lon: i for i, lon in
                    enumerate(self.data['longitude'].values)}
        
        lat_iter = lat_dict[self.lat]
        lon_iter = lon_dict[self.lon]

        v_x = np.empty((2, self.data_length))
        v_y = np.empty((2, self.data_length))

        for i, h in enumerate(self.h):
            u_str = 'u'+str(h)
            v_str = 'v'+str(h)

            v_x[i] = self.data[u_str][:, lat_iter, lon_iter].values
            v_y[i] = self.data[v_str][:, lat_iter, lon_iter].values

        return v_x, v_y



# class InterpolatedSite(WindCalculation):
