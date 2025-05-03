
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.integrate import quad
from WRA import classes
from WRA import functions


def test_load_nc_folder_to_dataset():
    folder_path = Path(__file__).parent.parent / 'inputs'
    assert isinstance(functions.load_nc_folder_to_dataset(folder_path),
                      xr.Dataset)


@pytest.mark.parametrize(
    "k, A, u, expected_pdf",
    [
        (0.5, 1, 8, 1.044852e-2),
        (0.5, 1, 12, 4.51793e-3),
        (1.5, 2, 8, 5.03194e-4),
        (1.5, 2, 12, 7.60918e-7)
    ]
)


def test_pdf_weib(k, A, u, expected_pdf):
    pdf_values = functions.pdf_weib(k, A, u)
    assert pdf_values == pytest.approx(expected_pdf)


# Dataset intialization, dataset loading tested in func_test.py
# As this is set, the tests will expect values from the provided dataset

path_main = Path(__file__)
path_dir = path_main.parent
path_package = path_dir.parent

folder_name = 'inputs'
folder_path = path_package / folder_name
dataset = functions.load_nc_folder_to_dataset(folder_path)
power_curve_path = path_package / 'inputs/NREL_Reference_5MW_126.csv'
power_curve_data = pd.read_csv(power_curve_path)


def test_get_AEP(power_curve=power_curve_data):
        latitude_point = 55.7
        longitude_point = 7.9
        height_point = 90
        year=2000
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

        assert is_test.get_AEP(power_curve, year) == (pytest.approx(expected_AEP),
                                                      pytest.approx(expected_CF))

def test_compare_AEPs_years(power_curve=power_curve_data, show_comparison=False):
        latitude_point = 55.7
        longitude_point = 7.9
        height_point = 90
        year=2000
        eta = 1.0  # turbine availability

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
    
        AEPs, CFs = is_test.compare_AEPs_years(power_curve_data,
                                               show_comparison=False)
        
        assert AEPs == expected_AEPs
        assert CFs == expected_CFs


