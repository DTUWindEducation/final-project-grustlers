# pylint: disable=C0103  # Disable warnings about snake case variable naming
'''
This module contains classes useful for accelerating wind resource
assessments (WRA). There are three classes of which the latter two
are subclasses of the first one. They are summarized below along
with their methods:

1. WindCalculation: used for fundamental wind calculations.
Methods:
    - __init__(self, v_x, v_y, height_point, ref_heights)
    - get_velocity_site(self)
    - get_angle_rad(self)
    - get_angle_deg(self)
    - get_alpha(self)
    - calculate_wind_at_height(self)

2. DataSite(WindCalculation): used to perform wind calculations at a
specific data site from where measurements have been taken.
Methods:
    - __init__(self, dataset, latitude, longitude, height_point=10,
                 ref_heights=[10, 100], name=None)
    - load_components(self)
    - plot_velocity_site(self, year=1997)

3. InterpolatedSite(WindCalculation): used to perform wind calculations at a
any given point inside the box defined by the boundaries - latitude,
longitude, and height - of the data sites.
Methods:
    - __init__(self, dataset, latitude_point, longitude_point,
                 height_point, ref_heights=[10, 100], name=None)
    - load_components_at_sites(self)
    - interpolate_wind_components(self)
    - interpolate_angle(self)
    - weibull_distribution(self, year=None, show_plot=True)
    - show_wind_rose(self)
    - get_AEP(self, power_curve, year=1997, show_curve=False)
    - compare_AEPs(self, power_curve, show_comparison=True)

In the explanations below 't_samples' refers to the number of observations
in the wind speeds time series.
'''
import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from scipy.stats import weibull_min
from scipy.integrate import quad  # for AEP
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from windrose import WindroseAxes


# import functions -> wrong because of:
# "sys.path.insert(0, __file__.replace('main.py', '../src'))" in main.py...
from WRA import functions


class WindCalculation():
    """
    Class for calculating wind velocity and direction based on wind
    components and estimating the wind speed at different heights using
    the power law.

    Parameters:
    ----------
    v_x : numpy.ndarray of shape (2, t_samples)
        Time series of wind velocity component in the x-direction (eastward)
        at both of the reference heights.
    v_y : numpy.ndarray of shape (2, t_samples)
        Time series of wind velocity component in the y-direction (northward)
        at both of the reference heights.
    height_point : float
        Height at which the wind speed is to be calculated.
    ref_heights : list of float
        A list [z, z_ref] containing two reference heights (in meters) at
        which velocities have been measured.

    Raises:
    ------
    ValueError
        If the space provided by the reference heights does not span the
        desired 'height_point'.
    """
    def __init__(self, v_x, v_y, height_point, ref_heights):
        if not ref_heights[0] <= height_point <= ref_heights[1]:
            raise ValueError(
                f"Height '{height_point} m' is not between the two "
                + f"reference heights: ({ref_heights} m). Try another value.")
        self.v_x = v_x
        self.v_y = v_y
        self.h_point = height_point
        self.h_ref = ref_heights

    def get_velocity_site(self):
        """
        Calculate the wind speed based on the east- (v_x) and northward (v_y)
        components.

        Returns:
        -------
        v : numpy.ndarray of shape (2, t_samples)
            Time series of the wind speed (magnitude) at both reference
            heights in m/s.
        """
        v = np.sqrt(self.v_x**2 + self.v_y**2)
        return v

    def get_angle_rad(self):
        """
        Calculate the wind direction in radians according to the wind
        industry convention: Wind from the North is 0°.

        Returns:
        -------
        theta : numpy.ndarray of shape (2, t_samples)
            Wind direction in radians, measured clockwise from north (0 to 2π).
        """
        theta = np.arctan2(self.v_y, self.v_x)
        # 270 - theta, 180=E, 270=N
        theta = 1.5*np.pi - theta  # adhere to wind convention
        theta = theta % (2*np.pi)  # output as [0, 2 pi] rad/s
        return theta

    def get_angle_deg(self):
        """
        Convert the wind direction from radians to degrees.

        Returns:
        -------
        theta_deg : numpy.ndarray of shape (2, t_samples)
            Wind direction in degrees, measured clockwise from north (0° to
            360°).
        """
        theta_deg = self.get_angle_rad() * 180/np.pi
        return theta_deg

    def get_alpha(self):
        """
        Calculate the power law exponent using the power law between two
        reference heights.

        Returns:
        -------
        alpha : numpy.ndarray of shape (t_samples,)
            Power law exponent used to estimate wind at any height between
            reference heights.
        """
        z, z_r = self.h_ref
        u_z, u_z_ref = self.get_velocity_site()
        alpha = np.log(u_z/u_z_ref) / np.log(z/z_r)
        return alpha

    def calculate_wind_at_height(self):
        """
        Estimate the wind speed at a specified height using the power law.

        Returns:
        -------
        u_z : numpy.ndarray of shape (t_samples,)
            Estimated wind speed at 'height_point'.
        """
        alpha = self.get_alpha()
        u_ref = self.get_velocity_site()[0]
        u_z = u_ref * (self.h_point/self.h_ref[0])**alpha
        return u_z


class DataSite(WindCalculation):
    """
    Load wind data from a dataset at a given location and initialize
    the superclass WindCalculation with the given velocity components,
    height, and reference heights.

    Parameters:
    ----------
    dataset : xarray.Dataset
        Dataset containing wind components (e.g., 'u10', 'v10').
    latitude : float
        Latitude of the site.
    longitude : float
        Longitude of the site.
    height_point : float, optional
        Target height for wind speed calculation (default 10 m).
    ref_heights : list of float, optional
        Reference heights for power law calculation (default [10, 100]).
    name : str, optional
        Name of the site.
    """
    def __init__(self, dataset, latitude, longitude, height_point=10,
                 ref_heights=[10, 100], name=None):
        self.data = dataset
        self.lat = latitude
        self.lon = longitude
        self.h_point = height_point
        self.h_ref = ref_heights
        self.name = name
        self.data_length = self.data.sizes['time']

        self.v_x, self.v_y = self.load_components()
        super().__init__(self.v_x, self.v_y, self.h_point, self.h_ref)

    def load_components(self):
        """
        Load the wind velocity components from the dataset at the site for
        the two reference heights.

        Returns:
        -------
        (v_x, v_y) : tuple of numpy.ndarray
            v_x and v_y have shape (2, t_samples).
        """
        v_x = np.empty((2, self.data_length))
        v_y = np.empty((2, self.data_length))

        for i, h in enumerate(self.h_ref):

            v_x[i] = self.data[f'u{h}'].sel(latitude=self.lat,
                                            longitude=self.lon).values
            v_y[i] = self.data[f'v{h}'].sel(latitude=self.lat,
                                            longitude=self.lon).values

        return v_x, v_y

    def plot_velocity_site(self, year=1997):
        """
        Plot wind velocity time series at the site for both reference heights
        in a given year.

        Parameters:
        ----------
        year : int, optional
            Year to plot. Must be within the dataset range. Default is 1997.

        Returns:
        -------
        wind_speeds : pandas.DataFrame
            Hourly wind speeds for the selected year at each reference height.
        """
        wind_speeds = self.get_velocity_site()

        if year is not None:
            #  Find the first year
            year_init = pd.to_datetime(self.data.time.values[0]).year
            #  Find the last year
            year_fin = pd.to_datetime(self.data.time.values[-1]).year
            #  Mask the hours in all years to the wind speeds
            hours_in_years = pd.date_range(f'{year_init}-01-01 00:00Z',
                                           f'{year_fin}-12-31 23:00Z',
                                           freq='h')
            wind_speeds = pd.DataFrame(wind_speeds.T,
                                       index=hours_in_years,
                                       columns=[f"{h} m" for h in self.h_ref])

            if not year_init <= year <= year_fin:  # stop if invalid year
                raise ValueError(
                    f"Year '{year}' is not included in the loaded datasets "
                    + f"({year_init}--{year_fin}). Please try another year.")

            # Reduce the dataframe to only include the selected year
            wind_speeds = wind_speeds[wind_speeds.index.year == year]

        _, ax = plt.subplots(figsize=(8, 4))
        for i, col in enumerate(wind_speeds.columns):
            wind_speeds[col].plot(ax=ax,
                                  alpha=1-0.3*i,
                                  label=col)
        plt.legend()
        plt.title(f'Wind speed time series in {year} (Latitude: {self.lat}'
                  + f', longitude: {self.lon})')
        plt.show()

        return wind_speeds


class InterpolatedSite(WindCalculation):
    """
    Interpolate wind data between locations (latitude, longitude) and upwards
    at heights between the reference heights.

    Parameters:
    ----------
    dataset : xarray.Dataset
        Dataset containing wind components.
    latitude_point : float
        Latitude of the interpolation point.
    longitude_point : float
        Longitude of the interpolation point.
    height_point : float
        Target height for wind speed calculation.
    ref_heights : list of float, optional
        Reference heights for power law calculation (default [10, 100]).
    name : str, optional
        Name of the site.
    """
    def __init__(self, dataset, latitude_point, longitude_point,
                 height_point, ref_heights=[10, 100], name=None):
        self.data = dataset
        self.lat_point = latitude_point
        self.lon_point = longitude_point
        self.h_point = height_point
        self.h_ref = ref_heights
        self.name = name
        self.data_length = self.data.sizes['time']
        self.lats = self.data['latitude'].values
        self.lons = self.data['longitude'].values

        self.v_x_sites, self.v_y_sites = self.load_components_at_sites()

        interp_dict = self.interpolate_wind_components()
        self.v_x_point = np.vstack([interp_dict[f'u{h}'] for h in self.h_ref])
        self.v_y_point = np.vstack([interp_dict[f'v{h}'] for h in self.h_ref])

        super().__init__(self.v_x_point, self.v_y_point,
                         self.h_point, self.h_ref)
        # Initializing the parent class with the desired wind speed components
        # from the point as well as the refence heights for alpha calculation
        self.angle_point_rad, self.angle_point_deg = self.interpolate_angle()

    def load_components_at_sites(self):
        """
        Load the wind velocity components from the dataset at all four
        measurement sites and for both reference heights.

        Returns:
        -------
        (v_x_sites, v_y_sites) : tuple of dict
            Dictionaries with east- and northward components, respectively,
            at each reference height.
            Each dictionary has two keys: one for each reference height.
            Each dictionary value is then a 3D numpy.ndarray of shape:
            (t_samples, 2, 2). Where the latter two refer to the coordinates.

            
            They can be accessed like this:
            load_components_at_sites()[0]['u10'][:, 0, 1]
                - In this case the first index [0] chooses to consider eastward
                velocity components.
                - Secondly, 'u10' chooses the data at 10 m.
                - Finally, '[:, 0, 1]' slices the entire time series at
                latitude[0] and longitude[1].
        """
        v_x_sites = {}
        v_y_sites = {}

        for h in self.h_ref:
            v_x_sites[f'u{h}'] = self.data[f'u{h}'].values

            v_y_sites[f'v{h}'] = self.data[f'v{h}'].values

        return v_x_sites, v_y_sites

    def interpolate_wind_components(self):
        """
        Interpolate wind components on a 2D grid at the target point:
        (lat_point, lon_point).

        Returns:
        -------
        interp_dict : dict
            Interpolated wind velocity components for each reference height.
            The dict has four keys: 'u10', 'u100', 'v10', and 'v100'.
            Each key returns a numpy.ndarray of shape (t_samples,).
        """
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

    def interpolate_angle(self):
        """
        Linearly interpolates wind direction angle at the target height from
        the first reference height.

        Returns:
        -------
        (theta_rad, theta_deg) : tuple of numpy.ndarray
            Angle in radians and degrees. Both arrays have shape (t_samples,).
        """
        thetas_h_ref = np.unwrap(self.get_angle_rad(), axis=0)
        theta_rad = ((thetas_h_ref[0]
                      + (self.h_point - self.h_ref[0])
                      / (self.h_ref[1] - self.h_ref[0])
                      * (thetas_h_ref[1] - thetas_h_ref[0])
                      ) % (2 * np.pi)
                     )
        theta_deg = theta_rad * 180 / np.pi
        return theta_rad, theta_deg

    def weibull_distribution(self, year=None, show_plot=True):
        """
        Fit a Weibull distribution to the wind speed data and plot it along
        with the histogram of wind speeds.

        Parameters:
        ----------
        year : int, optional
            Specific year for which to fit Weibull distribution.
        show_plot : bool, optional
            Whether to display the Weibull fit plot and histogram (default
            True).

        Returns:
        -------
        (k, A) : tuple of float
            Shape (k) and scale (A) parameters of Weibull distribution.
        """
        wind_speeds = self.calculate_wind_at_height()

        if year is not None:
            #  Find the first year
            year_init = pd.to_datetime(self.data.time.values[0]).year
            #  Find the last year
            year_fin = pd.to_datetime(self.data.time.values[-1]).year
            #  Mask the hours in all years to the wind speeds
            hours_in_years = pd.date_range(f'{year_init}-01-01 00:00Z',
                                           f'{year_fin}-12-31 23:00Z',
                                           freq='h')
            wind_speeds = pd.Series(wind_speeds, index=hours_in_years)

            if not year_init <= year <= year_fin:  # stop if invalid year
                raise ValueError(
                    f"Year '{year}' is not included in the loaded datasets "
                    + f"({year_init}--{year_fin}). Please try another year.")

            wind_speeds = wind_speeds[wind_speeds.index.year == year]
            wind_speeds = wind_speeds.values

        params = weibull_min.fit(wind_speeds, floc=0)
        A = params[2]
        k = params[0]

        u = np.linspace(0, wind_speeds.max()+5, 100)
        pdf = functions.pdf_weib(k, A, u)
        if show_plot:
            plt.figure(figsize=(8, 5))
            plt.hist(wind_speeds, bins=30, density=True, alpha=0.6,
                     color='skyblue', label='Wind speed histogram')
            plt.plot(u, pdf, 'b-', lw=2,
                     label=f'Weibull fit (k={k:.2f}, A={A:.2f})')
            # Make the plot cleaner, more informational, and prettier
            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Density [-]')
            plt.title(f'Weibull distribution (Latitude: {self.lat_point}'
                      + f', longitude: {self.lon_point}'
                      + f', and height: {self.h_point}m)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return k, A

    def show_wind_rose(self):
        """
        Plot the wind rose at the target location and height.

        Returns:
        -------
        (wind_speed, wind_direction) : tuple of numpy.ndarray
            The arrays correspond exactly to the ones returned by:
            calculate_wind_at_height() and get_angle_deg(), respectively.
        """
        wind_speed = self.calculate_wind_at_height()
        wind_direction = self.angle_point_deg
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
        # Make the plot cleaner, more informational, and prettier:
        ax.set_legend(title='Wind speed [m/s]', loc='best', fontsize=16)
        yticks = mtick.FormatStrFormatter('%.1f%%')
        ax.yaxis.set_major_formatter(yticks)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_title(f'Wind rose (Latitude: {self.lat_point},'
                     + f' longitude: {self.lon_point},'
                     + f' and height: {self.h_point}m)',
                     pad=35)
        plt.show()

        return wind_speed, wind_direction

    def get_AEP(self, power_curve, year=1997, show_power_curve=False):
        """
        Estimate Annual Energy Production (AEP) and capacity factor for a
        given year.

        Parameters:
        ----------
        power_curve : pandas.DataFrame
            Power curve with 'Wind Speed [m/s]' and 'Power [kW]' columns.
        year : int, optional
            Year for which AEP is calculated (default 1997).
        show_curve : bool, optional
            Whether to display the power curve plot (default False).

        Returns:
        -------
        (AEP, CF) : tuple of float
            AEP in [MWh] and capacity factor (CF).
        """
        # Defining constants
        eta = 1.0  # turbine availability
        u_in = power_curve['Wind Speed [m/s]'][0]  # cut-in wind speed
        u_out = power_curve['Wind Speed [m/s]'][power_curve.index[-1]]
        k, A = self.weibull_distribution(year=year, show_plot=False)

        def p_u(u):
            '''
            Function for interpolating the power curve at wind speed u
            '''
            power_curve_interp = np.interp(u,
                                           power_curve['Wind Speed [m/s]'],
                                           power_curve['Power [kW]'])
            return power_curve_interp

        def integrand(u):
            '''
            Combining power curve and weibull functions to a single integrand
            '''
            return p_u(u) * functions.pdf_weib(k, A, u)

        hourly_avg, _ = quad(integrand, u_in, u_out,
                             limit=100, epsabs=5e-06, epsrel=5e-06)
        # "_" is simply the estimated absolute integration error

        AEP = eta * 8760 * hourly_avg / 10**3  # MWh

        p_rated = max(power_curve['Power [kW]'])
        CF = hourly_avg / p_rated

        if show_power_curve:
            wind_speeds = np.arange(u_in, u_out+0.1, 0.1)
            plt.plot(wind_speeds, p_u(wind_speeds), label="Power curve")
            # Make the plot cleaner, more informational, and prettier:
            plt.xlabel('Wind speed [m/s]')
            plt.ylabel('Power [kW]')
            plt.hlines(y=hourly_avg, xmin=u_in, xmax=u_out,
                       color='orange', linestyle='--', label='Hourly average')
            plt.grid()
            plt.legend()
            plt.title(f"{p_rated/10**3:.0f} MW reference turbine")
            plt.show()

        return AEP, CF

    def compare_AEPs_years(self, power_curve, show_comparison=True):
        """
        Compares AEP and CF values across available years. By default the AEPs
        are compared in a bar chart. It is assumed that no years are missing
        between the first year and last year in the dataset.

        Parameters:
        ----------
        power_curve : pandas.DataFrame
            Power curve with 'Wind Speed [m/s]' and 'Power [kW]' columns.
        show_comparison : bool, optional
            Whether to display a bar chart of AEPs (default True).

        Returns:
        -------
        (AEPs, CFs) : tuple of dict
            AEPs and capacity factors (CFs) per year. Each year is a key
            returning a float.
        """
        p_rated = max(power_curve['Power [kW]'])
        years_col = pd.to_datetime(self.data.time.values).year
        years = np.unique(years_col)
        AEPs = {}
        CFs = {}
        for year in years:
            AEPs[f"{year}"], CFs[f"{year}"] = self.get_AEP(power_curve,
                                                           year=year)

        if show_comparison:
            # For plot design:
            linestyles = ['dotted', 'dashdot', 'dashed']
            metric_labels = ['min', 'mean', 'max']

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(x=years, height=list(AEPs.values()),
                          color=plt.cm.tab10.colors[1])

            # Make the plot cleaner, more informational, and prettier:
            ax.set_ylabel("AEP [MWh]", fontsize=13)
            # Set alpha for every other bar to easily distinguish between years
            for i, bar in enumerate(bars):
                if i % 2 == 0:
                    bar.set_alpha(0.75)
                else:
                    bar.set_alpha(1)
            # To see every year on the x-axis:
            ax.set_xticks(np.arange(years[0], years[-1]+1))
            ax.set_xticklabels(years, rotation=45, fontsize=13)
            # To remove the frame and ticks::
            ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
            ax.tick_params(bottom=False, left=False)
            # To only see the gridlines on the y-axis:
            ax.yaxis.grid(True, color="#EEEEEE")
            ax.xaxis.grid(False)
            # To plot the min, mean, and max AEP:
            for j, metric in enumerate([min(list(AEPs.values())),
                                        np.mean(list(AEPs.values())),
                                        max(list(AEPs.values()))]):
                ax.hlines(metric, years[0]-0.5, years[-1]+0.5,
                          color='k', alpha=0.85,
                          linestyles=linestyles[j],
                          label=metric_labels[j])
                # To annotate the values of the metrics:
                ax.text(years[-1]+1.2, metric,
                        f'{metric:.0f} MWh',  # format the number (1 decimal)
                        va='center',
                        ha='left',
                        fontsize=12,
                        color='k',
                        alpha=0.85)

            #plt.subplots_adjust(right=1.2)  # Purely visual: elongates the plot
            ax.set_title(f"{p_rated/10**3:.0f} MW reference turbine"
                         + f" (Latitude: {self.lat_point},"
                         + f" longitude: {self.lon_point},"
                         + f" and height: {self.h_point} m)",
                         pad=0)
            ax.legend(loc='best', fontsize=12,
                      bbox_to_anchor=(0.97, 0.75))
            plt.tight_layout()
            plt.show()

        return AEPs, CFs
