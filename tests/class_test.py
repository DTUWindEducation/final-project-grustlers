
from pathlib import Path
import pytest
import numpy as np
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
def test_load_components_vx(data, index, u10_expec, u100_expec):
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
def test_load_components_vy(data, index, v10_expec, v100_expec):
    ds1 = classes.DataSite(dataset, latitude=55.5, longitude=8,
                           ref_heights=[10, 100], name="P1")

    _, vy = ds1.load_components()

    assert vy[0][index] == pytest.approx(v10_expec)
    assert vy[1][index] == pytest.approx(v100_expec)


