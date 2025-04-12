import pytest
import numpy as np
from WRA import classes

@pytest.mark.parametrize(
    "v_x, v_y, heights, expected_velocity",
    [
        (3, 4, [10,100], 5.0),
        (0, 0, [10,100], 0.0),
        (5, 12, [10,100], 13.0),
        (-3, -4, [10,100], 5.0)
    ]
)
def test_velocity_calculation(v_x, v_y, heights, expected_velocity):
    wind = classes.WindCalculation(v_x, v_y, heights)
    assert wind.get_velocity_site() == pytest.approx(expected_velocity)
