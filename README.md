[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Great Package

Team: Grustlers

## Overview

'A brief overview of the package objective.'

This is a Python package intended for simple wind resource assessment using ERA5 reanalysis data. The package has specifically been designed to work with multiple NetCDF4 files (.nc files provided in the 'inputs' folder) but should with small modifications work for other file types as well.

The data provided is hourly wind data at 4 different locations (specified by latitude and longitude) and 2 different heights (for the data provided at 10 and 100m). This combined with power curves for selected turbines will then provide the user with key wind energy metrics such as wind speed distribution, wind roses, and AEP for the aforementioned turbines.

The package allows users for visualizations to easily interpret the data for the initial stage of wind energy project planning.

## Quick-start guide

creating an environment from scratch<br />
python version, packages

## Architecture

'A description of the package architecture, with at least one diagram.' <br />
'A description of the class(es) you have implemented in your package, with clear reference to the file name of the code inside src.'

The package has been split into 2 parts, with the main one being the 3 classes and the secondary one being the (standalone) functions.

The 3 classes provided consist of the superclass 'WindCalculation' and the 2 subclasses 'DataSite' and 'InterpolatedSite'. All 3 classes are provided in 'classes.py' in the 'src' folder.
* The 'WindCalculation' class is intended for basic wind calculations such as velocity based on vector components, direction, and interpolation using the power-law. This class is essential for the subclasses, however, using it as a standalone class would require some pre-handling of the data provided in the 'inputs' folder.
* The 'DataSite' class is intended purely for investigating the data provided at the sites and is much more simple than the two other classes as it only contains a method which does the pre-handling for the 'WindCalculation' class.
* The third and final class 'InterpolatedSite' contains most of the remaining methods intended in the package, those being interpolation of both position (regarding altitude and longitude) as well as height. Given the inputs for a desired point, the class then provides all of the data and visualizations mentioned in the overview of the package.

For a more in-depth description of each of the methods provided by the 3 classes please refer to the docstrings provided in the 'classes.py' file in the 'src' folder.

In addition to the 3 classes the package also provides 2 functions provided in 'functions.py' in the 'src' folder.
* The first function 'load_nc_folder_to_dataset' is a quality of life function intended for providing the user with a easy way of inputting different data files, however, is also made specifically with the NetCDF4 files (.nc files) provided in the 'inputs' folder in mind.
* The second function 'pdf_weib' is simply the probability density function (PDF) which is used when working with the wind speed distribution in the 'InterpolatedSite' class.

A graphical overview of the classes and functions can be seen below:

[indsæt billede :)))]

In addition to a general overview of the architecture of the package a flowchart of an intended use case has also been provided to show how the different methods within the classes work together. In the flow chart the most typical use case (an object from the 'InterpolatedSite' class) is being shown:

[indsæt billede 2 :)))]


## Peer review

'A description of what peer review (if any) you have implemented.'
her kan vi måske bruge noget af feedback vi fik fra forrige opgave hvis det passer