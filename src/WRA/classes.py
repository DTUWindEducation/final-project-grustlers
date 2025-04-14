import numpy as np
from scipy.interpolate import interpn
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

class WindCalculation():
    def __init__(self, v_x, v_y, heights):
        self.v_x = v_x
        self.v_y = v_y
        self.h = heights

    def get_velocity_site(self):
        v = np.sqrt(self.v_x**2 + self.v_y**2)
        return v

    def get_angle_rad(self):
        theta = np.arctan2(self.v_y, self.v_x)
        # 270 - theta, 180=E, 270=N
        theta = 1.5*np.pi - theta  # adhere to wind convention
        theta = theta % (2*np.pi)  # output as [0, 2 pi] rad/s
        return theta

    def get_angle_deg(self):
        theta_deg = self.get_angle_rad() * 180/np.pi
        return theta_deg

    def get_alpha(self):
        z, z_r = self.h
        u_z, u_z_ref = self.get_velocity_site()
        alpha = np.log(u_z/u_z_ref) / np.log(z/z_r)
        return alpha


class DataSite(WindCalculation):
    def __init__(self, dataset, latitude, longitude, heights=[10, 100], name=None):
        self.data = dataset
        self.lat = latitude
        self.lon = longitude
        self.h = heights
        self.name = name
        self.data_length = self.data.sizes['valid_time']

        self.v_x, self.v_y = self.load_components()
        super().__init__(self.v_x, self.v_y, self.h)

    def load_components(self):
        v_x = np.empty((2, self.data_length))
        v_y = np.empty((2, self.data_length))

        for i, h in enumerate(self.h):  # idx 1 is reference point

            v_x[i] = self.data[f'u{h}'].sel(latitude=self.lat,
                                            longitude=self.lon).values
            v_y[i] = self.data[f'v{h}'].sel(latitude=self.lat,
                                            longitude=self.lon).values

        return v_x, v_y

### Skal den her bare fjernes?
def interpolate_wind_components(lat, lon, u10, v10, u100, v100, time_steps, lats, lons):
    grid = (time_steps, lats, lons)
    points = np.column_stack((
        time_steps, 
        np.full_like(time_steps, lat, dtype=float),
        np.full_like(time_steps, lon, dtype=float)
    ))
    u10_interp = interpn(grid, u10, points, bounds_error=True)
    v10_interp = interpn(grid, v10, points, bounds_error=True)
    u100_interp = interpn(grid, u100, points, bounds_error=True)
    v100_interp = interpn(grid, v100, points, bounds_error=True)
    return u10_interp, v10_interp, u100_interp, v100_interp


class InterpolatedSite(WindCalculation):
    def __init__(self, dataset, latitude_point, longitude_point, height_point, ref_heights=[10, 100], name=None):
        ### Noget med en exception error hvis height_point ikke er inden for ref_heights. Måske ikke her, men et andet sted?
        self.data = dataset
        self.lat_point = latitude_point
        self.lon_point = longitude_point
        self.h_point = height_point  ### Jeg ændrede navnet på den her fordi dens værdi blev overwritet når vi initializer parent classen.
        self.h_ref = ref_heights
        self.name = name
        self.data_length = self.data.sizes['valid_time']
        self.lats = self.data['latitude'].values
        self.lons = self.data['longitude'].values

        self.v_x_sites, self.v_y_sites = self.load_components_at_sites()

        interp_dict = self.interpolate_wind_components2()
        self.v_x_point = np.vstack([interp_dict[f'u{h}'] for h in self.h_ref])
        self.v_y_point = np.vstack([interp_dict[f'v{h}'] for h in self.h_ref])

        super().__init__(self.v_x_point, self.v_y_point, self.h_ref)  # Initializing the parent class with the desired wind speed components from the point as well as the refence heights for alpha calculation

    def load_components_at_sites(self):
        v_x_sites = {}
        v_y_sites = {}

        for h in self.h_ref:
            v_x_sites[f'u{h}'] = self.data[f'u{h}'].values

            v_y_sites[f'v{h}'] = self.data[f'v{h}'].values

        return v_x_sites, v_y_sites

    def interpolate_wind_components2(self):
        time_steps = np.arange(self.data_length)
        grid = (time_steps, self.lats, self.lons)
        points = np.column_stack((
            time_steps,
            np.full_like(time_steps,
                         self.lat_point,
                         dtype=float),
            np.full_like(time_steps,
                         self.lon_point,
                         dtype=float)
        ))
        interp_dict = {}

        for h in self.h_ref:
            interp_dict[f'u{h}'] = interpn(grid, self.v_x_sites[f'u{h}'], 
                                           points, bounds_error=True)
            interp_dict[f'v{h}'] = interpn(grid, self.v_y_sites[f'v{h}'], 
                                           points, bounds_error=True)

        return interp_dict

    def calculate_wind_at_height(self):
        alpha = self.get_alpha()
        u_ref = self.get_velocity_site()[0]
        u_z = u_ref * (self.h_point/self.h_ref[0])**alpha
        return u_z

    def weibull_distribution(self):
        wind_speeds = self.calculate_wind_at_height()
        params = weibull_min.fit(wind_speeds, floc=0)
        A = params[2]
        k = params[0]

        u = np.linspace(0, wind_speeds.max()+5, 100) ### Måske vi bare skal sætte den her til (0, 25) det er rimelig typisk
        pdf = (k / A) * (u / A)**(k - 1) * np.exp(- (u / A)**k)

        plt.figure(figsize=(8, 5))
        plt.hist(wind_speeds, bins=30, density=True, alpha=0.6, color='skyblue', label='Wind speed histogram')
        plt.plot(u, pdf, 'b-', lw=2, label=f'Weibull fit (k={k:.2f}, A={A:.2f})')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Density [-]')
        plt.title(f'Weibull distribution (Latitude: {self.lat_point}, longitude: {self.lon_point}, and height: {self.h_point}m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return k, A