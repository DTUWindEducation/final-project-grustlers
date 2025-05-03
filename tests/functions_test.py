# pylint: disable=C0103  # Disable warnings about snake case variable naming
# pylint: disable=C0116 # tests dont need docstring
from pathlib import Path
import pytest
import xarray as xr
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
