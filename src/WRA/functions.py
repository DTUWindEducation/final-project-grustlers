import os
import xarray as xr
import numpy as np
# from pathlib import Path


def load_nc_folder_to_dataset(folder_path):
    '''takes a folder path and returns a compiled dataset of
      all .nc files in that folder'''
    files_in_folder = os.listdir(folder_path)
    nc_files = [file for file in files_in_folder if file.endswith('.nc')]
    nc_files = sorted(nc_files, key=lambda x: x)
    nc_files_paths = [folder_path / nc_file for nc_file in nc_files]

    ds_all = xr.open_mfdataset(nc_files_paths)
    return ds_all


def pdf_weib(k, A, u):
    return (k / A) * (u / A)**(k - 1) * np.exp(- (u / A)**k)
