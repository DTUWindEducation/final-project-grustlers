# pylint: disable=C0103  # Disable warnings about snake case variable naming
# pylint: disable=C0116 # tests dont need docstring
from pathlib import Path
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from scipy.integrate import quad

from WRA import classes
from WRA import functions


@pytest.mark.parametrize(
    "v_x, v_y, expected_velocity",
    [
        (3, 4, 5.0),
        (0, 0, 0.0),
        (5, 12, 13.0),
        (-3, -4, 5.0)
    ]
)
def test_velocity_calculation(v_x, v_y, expected_velocity):
    wind = classes.WindCalculation(v_x, v_y, 50, [10, 100])
    assert wind.get_velocity_site() == pytest.approx(expected_velocity)


@pytest.mark.parametrize(
    "v_x, v_y, expected_rad",
    [
        (0, 1, np.pi),                # wind from South
        (1, 0, 1.5*np.pi),          # wind from West
        (0, -1, 0.0),            # wind from North
        (-1, 0, 0.5*np.pi),        # wind from East
        (1, 1, (1.5*np.pi - np.arctan2(1, 1)) % (2*np.pi)),  # NW
        (-1, -1, (1.5*np.pi - np.arctan2(-1, -1)) % (2*np.pi)),  # SE
    ]
)
def test_get_angle_rad(v_x, v_y, expected_rad):
    wind = classes.WindCalculation(v_x, v_y, 50, [10, 100])
    result_rad = wind.get_angle_rad()
    assert result_rad == pytest.approx(expected_rad, rel=1e-5)


@pytest.mark.parametrize(
    "v_x, v_y, expected_angle_deg",
    [
        (0, 1, 180),           # wind from South
        (1, 0, 270.0),         # wind from West
        (0, -1, 0),            # wind from North
        (-1, 0, 90.0),         # wind from East
        (1, 1, 225.0),         # wind from South-West
    ]
)
def test_get_angle_deg(v_x, v_y, expected_angle_deg):
    wind = classes.WindCalculation(v_x, v_y, 50, [10, 100])
    assert wind.get_angle_deg() == pytest.approx(expected_angle_deg)


@pytest.mark.parametrize(
    "height, error",
    [
        (5, "Height '5 m' is not between the two reference heights"),
        (111, "Height '111 m' is not between the two reference heights")
    ]
)
def test_invalid_height_raises_value_error(height, error):
    with pytest.raises(ValueError, match=error):
        classes.WindCalculation(1, 1, height, [10, 100])


def test_get_alpha():
    # Let (vx, vy) = (3, 4) → v = 5
    # Let (vx_ref, vy_ref) = (6, 8) → v_ref = 10
    # So expected alpha = log(5 / 10) / log(10 / 100) ~ 0.3010

    v_x = np.array([[3], [6]])
    v_y = np.array([[4], [8]])

    wind = classes.WindCalculation(v_x, v_y, 50, [10, 100])

    v, v_ref = wind.get_velocity_site() 

    expected_alpha = np.log(v / v_ref) / np.log(10 / 100)

    alpha = wind.get_alpha()
    assert alpha == pytest.approx(expected_alpha)


def test_calculate_wind_at_height():
    # u_ref = 5.0 (from get_velocity_site)
    # h_point = 20, h_ref[0] = 10
    # alpha = 0.2
    # Expected: u_z = 5.0 * (20 / 10)^0.2
    
    v_x = np.array([[3], [6]])
    v_y = np.array([[4], [8]])

    wind = classes.WindCalculation(v_x, v_y, 50, [10, 100])

    v_ref = wind.get_velocity_site()[0]

    test_alpha = wind.get_alpha()

    expected_u_z = v_ref * (wind.h_point / wind.h_ref[0]) ** test_alpha
    u_z = wind.calculate_wind_at_height()
    assert u_z == pytest.approx(expected_u_z)


# Dataset intialization, dataset loading tested in func_test.py
# As this is set, the tests will expect values from the provided dataset

path_main = Path(__file__)
path_dir = path_main.parent
path_package = path_dir.parent 

folder_name = 'inputs'
folder_path = path_package / folder_name
dataset = functions.load_nc_folder_to_dataset(folder_path)


@pytest.mark.parametrize(
    "data, index, u10_expec, u100_expec",
    [
        (dataset, 5, -3.24005126953125, -3.5185699462890625),
        (dataset, 500, -0.17340087890625, 0.0379486083984375)
    ]
)
def test_load_components_vx(dataset, index, u10_expec, u100_expec):
    ds1 = classes.DataSite(dataset, latitude=55.5, longitude=8,
                           ref_heights=[10, 100], name="P1")

    vx, _ = ds1.load_components()

    assert vx[0][index] == pytest.approx(u10_expec)
    assert vx[1][index] == pytest.approx(u100_expec)


@pytest.mark.parametrize(
    "data, index, v10_expec, v100_expec",
    [
        (dataset, 5, -2.6804351806640625, -2.96392822265625),
        (dataset, 500, 6.5696258544921875, 8.38238525390625)
    ]
)
def test_load_components_vy(dataset, index, v10_expec, v100_expec):
    ds1 = classes.DataSite(dataset, latitude=55.5, longitude=8,
                           ref_heights=[10, 100], name="P1")

    _, vy = ds1.load_components()

    assert vy[0][index] == pytest.approx(v10_expec)
    assert vy[1][index] == pytest.approx(v100_expec)


@patch("matplotlib.pyplot.show")
def test_plot_velocity(mock_show):
    # wind speeds alreay tested, pure plot test
    ds1 = classes.DataSite(dataset, latitude=55.5, longitude=8,
                           ref_heights=[10, 100], name="P1")

    ds1.plot_velocity_site()
    mock_show.assert_called_once()


def test_plot_vel_except():
    ds1 = classes.DataSite(dataset, latitude=55.5, longitude=8,
                           ref_heights=[10, 100], name="P1")
    with pytest.raises(ValueError, match=r"Year '1776' is not included"):
        ds1.plot_velocity_site(year=1776)


def test_interp_load_compnents_vx():
    is1 = classes.InterpolatedSite(dataset, latitude_point=55.5,
                                   longitude_point=8, height_point=50,
                                   ref_heights=[10, 100], name="IS1")

    vx, _ = is1.load_components_at_sites()

    u10_expec = dataset['u10'].values[0][0, 1]
    u100_expec = dataset['u100'].values[0][0, 1]

    assert vx['u10'][0][0, 1] == pytest.approx(u10_expec)
    assert vx['u100'][0][0, 1] == pytest.approx(u100_expec)


def test_interp_load_compnents_vy():
    is1 = classes.InterpolatedSite(dataset, latitude_point=55.6,
                                   longitude_point=7.9, height_point=50,
                                   ref_heights=[10, 100], name="IS1")

    _, vy = is1.load_components_at_sites()

    v10_expec = dataset['v10'].values[0][0, 1]
    v100_expec = dataset['v100'].values[0][0, 1]

    assert vy['v10'][0][0, 1] == pytest.approx(v10_expec)
    assert vy['v100'][0][0, 1] == pytest.approx(v100_expec)


def test_interp_components():
    is1 = classes.InterpolatedSite(dataset, latitude_point=55.6,
                                   longitude_point=7.9, height_point=50,
                                   ref_heights=[10, 100], name="IS1")
    
    time_steps = np.arange(is1.data_length)
    grid = (time_steps, is1.lats, is1.lons)
    points = np.column_stack((
        time_steps,
        np.full_like(time_steps,
                     is1.lat_point,
                     dtype=float),
        np.full_like(time_steps,
                     is1.lon_point,
                     dtype=float)
    ))

    u10_expec = interpn(grid, is1.v_x_sites['u10'],
                        points, bounds_error=True)
    u100_expec = interpn(grid, is1.v_x_sites['u100'],
                         points, bounds_error=True)
    v10_expec = interpn(grid, is1.v_y_sites['v10'],
                        points, bounds_error=True)
    v100_expec = interpn(grid, is1.v_y_sites['v100'],
                         points, bounds_error=True)

    dict_test = is1.interpolate_wind_components()

    assert dict_test['u10'] == pytest.approx(u10_expec)
    assert dict_test['u100'] == pytest.approx(u100_expec)
    assert dict_test['v10'] == pytest.approx(v10_expec)
    assert dict_test['v100'] == pytest.approx(v100_expec)


def test_interp_angle():
    is1 = classes.InterpolatedSite(dataset, latitude_point=55.6,
                                   longitude_point=7.9, height_point=50,
                                   ref_heights=[10, 100], name="IS1")

    thetas_h_ref = np.unwrap(is1.get_angle_rad(), axis=0)
    theta_rad = (
        thetas_h_ref[0]
        + (is1.h_point - is1.h_ref[0])
        / (is1.h_ref[1] - is1.h_ref[0])
        * (thetas_h_ref[1] - thetas_h_ref[0])
    ) % (2 * np.pi)

    theta_deg = theta_rad * 180 / np.pi

    rad, deg = is1.interpolate_angle()

    assert rad == pytest.approx(theta_rad)
    assert deg == pytest.approx(theta_deg)


@patch("matplotlib.pyplot.show")
def test_weibull_dist(mock_show):
    # k, A already tested by pdf_weib func test, so this is a plot test
    is1 = classes.InterpolatedSite(dataset, latitude_point=55.6,
                                   longitude_point=7.9, height_point=50,
                                   ref_heights=[10, 100], name="IS1")

    k, A = is1.weibull_distribution(year=1999)
    mock_show.assert_called_once()


def test_weibull_except():
    is1 = classes.InterpolatedSite(dataset, latitude_point=55.6,
                                   longitude_point=7.9, height_point=50,
                                   ref_heights=[10, 100], name="IS1")
    with pytest.raises(ValueError, match=r"Year '1776' is not included"):
        k, A = is1.weibull_distribution(year=1776)


@patch("matplotlib.pyplot.show")
def test_wind_rose(mock_show):
    # speed, dir already tested earlier, so plot test
    is1 = classes.InterpolatedSite(dataset, latitude_point=55.6,
                                   longitude_point=7.9, height_point=50,
                                   ref_heights=[10, 100], name="IS1")
    
    speed, dir = is1.show_wind_rose()
    mock_show.assert_called_once()


power_curve_path = path_package / 'inputs/NREL_Reference_5MW_126.csv'
power_curve_data = pd.read_csv(power_curve_path)


@patch("matplotlib.pyplot.show")
def test_get_AEP(mock_show, power_curve=power_curve_data):
    latitude_point = 55.7
    longitude_point = 7.9
    height_point = 90
    year = 2000
    eta = 1.0  # turbine availability
    u_in = power_curve['Wind Speed [m/s]'][0]  # cut-in wind speed
    u_out = power_curve['Wind Speed [m/s]'][power_curve.index[-1]]

    is_test = classes.InterpolatedSite(dataset, latitude_point,
                                       longitude_point, height_point,
                                       ref_heights=[10, 100], name=None)
    k, A = is_test.weibull_distribution(year=year, show_plot=False)

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

    p_rated = max(power_curve['Power [kW]'])

    expected_AEP = eta * 8760 * hourly_avg / 10**3  # MWh
    expected_CF = hourly_avg / p_rated

    assert is_test.get_AEP(power_curve, year, show_power_curve=True) == (
                           pytest.approx(expected_AEP),
                           pytest.approx(expected_CF))
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_compare_AEPs_years(mock_show, power_curve=power_curve_data):
    latitude_point = 55.7
    longitude_point = 7.9
    height_point = 90
    year = 2000

    years_col = pd.to_datetime(dataset.time.values).year
    years = np.unique(years_col)
    expected_AEPs = {}
    expected_CFs = {}

    is_test = classes.InterpolatedSite(dataset, latitude_point,
                                       longitude_point, height_point,
                                       ref_heights=[10, 100], name=None)

    for year in years:
        (expected_AEPs[f"{year}"],
            expected_CFs[f"{year}"]) = is_test.get_AEP(
                power_curve, year=year)

    AEPs, CFs = is_test.compare_AEPs_years(power_curve_data)
    
    assert AEPs == expected_AEPs
    assert CFs == expected_CFs
    mock_show.assert_called_once()
