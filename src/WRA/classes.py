import numpy as np
from scipy.interpolate import interpn
import pandas as pd
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from windrose import WindroseAxes
from scipy.integrate import quad  # for AEP


def pdf_weib(k, A, u):
    return (k / A) * (u / A)**(k - 1) * np.exp(- (u / A)**k)


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


class InterpolatedSite(WindCalculation):
    def __init__(self, dataset, latitude_point, longitude_point, height_point, ref_heights=[10, 100], name=None):
        if not (ref_heights[0] <= height_point <= ref_heights[1]):
            raise ValueError(
                f"Height '{height_point} m' is not between the two "
                + f"reference heights: ({ref_heights} m). Try another value.")
        ### Noget med en exception error hvis height_point ikke er inden for ref_heights. Måske ikke her, men et andet sted? -- Synes det passer godt ind her. Har sat en ind nedenunder.
        self.data = dataset
        self.lat_point = latitude_point
        self.lon_point = longitude_point
        self.h_point = height_point  ### Jeg ændrede navnet på den her fordi dens værdi blev overwritet når vi initializer parent classen. - Alright, godt fanget
        self.h_ref = ref_heights
        self.name = name
        self.data_length = self.data.sizes['valid_time']
        self.lats = self.data['latitude'].values
        self.lons = self.data['longitude'].values

        self.v_x_sites, self.v_y_sites = self.load_components_at_sites()

        interp_dict = self.interpolate_wind_components()
        self.v_x_point = np.vstack([interp_dict[f'u{h}'] for h in self.h_ref])
        self.v_y_point = np.vstack([interp_dict[f'v{h}'] for h in self.h_ref])

        super().__init__(self.v_x_point, self.v_y_point, self.h_ref)
        # Initializing the parent class with the desired wind speed components
        # from the point as well as the refence heights for alpha calculation
        self.angle_point_rad, self.angle_point_deg = self.interpolate_angle()

    def load_components_at_sites(self):
        v_x_sites = {}
        v_y_sites = {}

        for h in self.h_ref:
            v_x_sites[f'u{h}'] = self.data[f'u{h}'].values

            v_y_sites[f'v{h}'] = self.data[f'v{h}'].values

        return v_x_sites, v_y_sites

    def interpolate_wind_components(self):
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
    
    def interpolate_angle(self):
        thetas_h_ref = np.unwrap(self.get_angle_rad(), axis=0)
        theta_rad = (thetas_h_ref[0] + (self.h_point - self.h_ref[0]) / (self.h_ref[1] - self.h_ref[0]) * (thetas_h_ref[1] - thetas_h_ref[0])) % (2 * np.pi)
        theta_deg = theta_rad * 180 / np.pi
        return theta_rad, theta_deg

    def weibull_distribution(self, year=None, show_plot=True):
        wind_speeds = self.calculate_wind_at_height()

        if year is not None:
            #  Find the first year
            year_init=pd.to_datetime(self.data.valid_time.values[0]).year
            #  Find the last year
            year_fin=pd.to_datetime(self.data.valid_time.values[-1]).year
            #  Mask the hours in all years to the wind speeds
            hours_in_years = pd.date_range(f'{year_init}-01-01 00:00Z',
                                           f'{year_fin}-12-31 23:00Z',
                                           freq='h')
            wind_speeds = pd.Series(wind_speeds, index=hours_in_years)
            
            if not (year_init <= year <= year_fin):  # avoid invalid years
                raise ValueError(
                    f"Year '{year}' is not included in the loaded datasets "
                    + f"({year_init}--{year_fin}). Please try another year.")

            wind_speeds = wind_speeds[wind_speeds.index.year == year]
            wind_speeds = wind_speeds.values

        params = weibull_min.fit(wind_speeds, floc=0)
        A = params[2]
        k = params[0]

        u = np.linspace(0, wind_speeds.max()+5, 100)  ### Måske vi bare skal sætte den her til (0, 25) det er rimelig typisk -- Det kan vi godt gøre for min skyld
        pdf = pdf_weib(k, A, u)
        if show_plot:
            plt.figure(figsize=(8, 5))
            plt.hist(wind_speeds, bins=30, density=True, alpha=0.6,
                    color='skyblue', label='Wind speed histogram')
            plt.plot(u, pdf, 'b-', lw=2,
                    label=f'Weibull fit (k={k:.2f}, A={A:.2f})')
            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Density [-]')
            plt.title(f'Weibull distribution (Latitude: {self.lat_point}, longitude: {self.lon_point}, and height: {self.h_point}m)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return k, A

    def show_wind_rose(self):
        wind_speed = self.calculate_wind_at_height()
        wind_direction = self.angle_point_deg  # now using the linearly interpolated angle at the height from the point
        max_ws = int(np.ceil(
            np.sort(wind_speed)[-int(wind_speed.shape[0]/20)]
            ))  # the last bin is for ws that are less frequent than 5%
        wind_range = np.append(np.arange(0, max_ws, 2), max_ws)

        ax = WindroseAxes.from_ax()
        ax.set_facecolor('white')
        ax.bar(wind_direction,
               wind_speed,
               bins=wind_range,
               normed=True,
               opening=0.9,
               edgecolor='white',
               cmap=plt.cm.winter)
        ax.set_legend(title='Wind speed [m/s]', loc='best', fontsize=16)
        yticks = mtick.FormatStrFormatter('%.1f%%')
        ax.yaxis.set_major_formatter(yticks)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_title(f'Wind rose (Latitude: {self.lat_point}, longitude: {self.lon_point}, and height: {self.h_point}m)',
                     pad=35)
        plt.show()

        return wind_speed, wind_direction

    def get_AEP(self, power_curve, year=1997, show_curve=False):
        ### include an exception for is the year is not included in the dataset
        # Defining constants
        eta = 1.0  # turbine availability
        u_in = power_curve['Wind Speed [m/s]'][0]  # cut-in wind speed
        u_out = power_curve['Wind Speed [m/s]'][power_curve.index[-1]]
        k, A = self.weibull_distribution(year=year, show_plot=False)

        # Function for interpolating the power curve at wind speed u
        def p_u(u):
            power_curve_interp = np.interp(u,
                                           power_curve['Wind Speed [m/s]'],
                                           power_curve['Power [kW]'])
            return power_curve_interp
 
        # Combining power curve and weibull functions to a single integrand
        def integrand(u):
            return p_u(u) * pdf_weib(k, A, u)

        hourly_avg, _ = quad(integrand, u_in, u_out,
                             limit=100, epsabs=5e-06, epsrel=5e-06)
        # "_" is simply the estimated absolute integration error

        AEP = eta * 8760 * hourly_avg

        p_rated = max(power_curve['Power [kW]'])
        CF = hourly_avg / p_rated

        if show_curve:
            wind_speeds = np.arange(u_in, u_out+0.1, 0.1)
            plt.plot(wind_speeds, p_u(wind_speeds), label="Power curve")
            plt.xlabel('Wind speed [m/s]')
            plt.ylabel('Power [kW]')
            plt.hlines(y=hourly_avg, xmin=u_in, xmax=u_out,
                       color='orange', linestyle='--', label='Hourly average')
            plt.grid()
            plt.legend()
            plt.title(f"{p_rated/10**3:.0f} MW reference turbine")
            plt.show()

        return AEP, CF
