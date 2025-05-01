# pylint: disable=C0103  # Disable warnings about snake case variable naming
"""
This module contains general functions that do not belong to a specific class.

Functions:
- load_nc_folder_to_dataset
- pdf_weib
"""

import os
import xarray as xr
import numpy as np


def load_nc_folder_to_dataset(folder_path):
    """
    Load all NetCDF (.nc) files from a specified folder and compile them into
    a single xarray.Dataset.

    Parameters:
    ----------
    folder_path : pathlib.Path
        Path to the folder containing the NetCDF (.nc) files.

    Returns:
    -------
    ds_all : xarray.Dataset
        A combined dataset containing data from all .nc files in the
        specified folder.
    """
    files_in_folder = os.listdir(folder_path)  # List all files in the folder
    nc_files = [file for file in files_in_folder if file.endswith('.nc')]
    nc_files = sorted(nc_files, key=lambda x: x)  # Sort years ascending order
    # Combine folder path with file names to get the full path for each file:
    nc_files_paths = [folder_path / nc_file for nc_file in nc_files]

    ds_all = xr.open_mfdataset(nc_files_paths, decode_timedelta=False)
    return ds_all


def pdf_weib(k, A, u):
    """
    Compute the probability density function (PDF) of the Weibull distribution.

    Parameters:
    ----------
    k : float
        Shape parameter of the Weibull distribution.
    A : float
        Scale parameter of the Weibull distribution.
    u : numpy.ndarray or float
        Velocities in m/s at which to evaluate the PDF.

    Returns:
    -------
    numpy.ndarray or float
        The PDF value(s) evaluated at u.
    """
    return (k / A) * (u / A)**(k - 1) * np.exp(- (u / A)**k)
