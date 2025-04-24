import numpy as np

class WindCalculation():  # simple wind calculations for a single point
    def __init__(self, v_x, v_y):
        self.v_x = v_x
        self.v_y = v_y
    
    def get_velocity(self):
        return np.sqrt(self.v_x**2 + self.v_y**2)

    def get_angle(self):
        return np.arctan2(self.v_y, self.v_x)
    

class DataSite(WindCalculation):
    def __init__(self, dataset, latitude, longitude, height, height_ref, name=None):
        self.data = dataset
        self.lat = latitude
        self.lon = longitude
        self.h = height
        self.name = name
        self.h_ref = height_ref

        self.v_x, self.v_y = self.load_components()
        super().__init__(self.v_x, self.v_y)

        self.v_x_ref, self.v_y_ref = self.load_components(self.h_ref)
        self.ref = WindCalculation(self.v_x_ref, self.v_y_ref)
    
    def load_components(self, h=None):
        if h is None:
            h = self.h
        lat_dict = {lat: i for i, lat in
                    enumerate(self.data['latitude'].values)}
        lon_dict = {lon: i for i, lon in
                    enumerate(self.data['longitude'].values)}
        
        lat_iter = lat_dict[self.lat]
        lon_iter = lon_dict[self.lon]

        u_str = 'u'+str(h)
        v_str = 'v'+str(h)

        v_x = self.data[u_str][:, lat_iter, lon_iter].values
        v_y = self.data[v_str][:, lat_iter, lon_iter].values

        return v_x, v_y

    # def load_reference_components(self, h_ref):
    #     v_x_ref, v_y_ref = self.load_components(h=h_ref)
    #     return v_x_ref, v_y_ref  # either for 10 or 100 m

    def get_alpha(self, z_ref=100):  # used for the power law
        u_z = self.get_velocity()  # object is by default the point of interest
        u_z_ref = self.ref.get_velocity()  # reference point is another object
        z = self.h
        z_r = self.h_ref
        alpha = np.log(u_z/u_z_ref) / np.log(z/z_r)
        return alpha



# class InterpolatedSite(WindCalculation):
