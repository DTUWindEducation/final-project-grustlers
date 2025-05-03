[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Wind Resource Assessment Package

Team: Grustlers

## Overview

This is a Python package intended for simple wind resource assessment (WRA) using ERA5 reanalysis data. The package has specifically been designed to work with multiple NetCDF4 files (.nc files provided in the `inputs` folder) but should with small modifications work for other file types as well.

The data provided is hourly wind data at 4 different locations (specified by latitude and longitude) and 2 different heights (10 and 100 m for the provided data) near the Horns Rev 1 offshore wind farm. This combined with power curves for selected turbines will then provide the user with key wind energy metrics such as wind speed distribution, wind roses, and AEP.

The package allows users to visualize and easily interpret the data for the initial stage of wind energy project planning.

## Quick-start guide

Firstly, make sure that the explorer is located with the `FINAL-PROJECT-GRUSTLERS` as the upper most layer so that the subsequent usage of paths and relative paths work as intended.

Secondly, to ensure that the code works as intended please set up a new environment with the Python version used during the code development and including the required packages. An explanation how to set up such an environment is given below:

1. Create a new environment using the correct Python version:<br />
`conda create -n <name> python=3.13 -y`<br />
Note that you might need to download this version, if you haven't downloaded the newer Python versions in a while.<br />
You may also need to use the Anaconda prompt to create the environment.

2. Ensure that you are in the environment you just created, by e.g.:<br />
Pressing `CTRL + SHIFT + P` to open the Command Palette in VS Code (`CMD + SHIFT + P` on macOS).<br />
Then select `Python: Select Interpreter` and choose the correct environment.
 
3. Install the package incl. dependencies:<br />
If you wish to run pytests, or edit the package, you can install it as editable:<br />
`pip install -e .`<br />
Otherwise, simply install the package:<br />
`pip install .`<br />
You may need to open a new terminal, to ensure that the new environment is actived correctly.

If you followed the steps correctly, you should now be able to run `main.py` to experience the package.

## Architecture

The package has been split into 2 parts, with the main one being the 3 classes and the secondary one being the (standalone) functions.

The 3 classes provided consist of the superclass 'WindCalculation' and the 2 subclasses 'DataSite' and 'InterpolatedSite'. All 3 classes are provided in `classes.py` in the `src` folder.
* The 'WindCalculation' class is intended for basic wind calculations such as velocity-based on vector components, direction, and interpolation using the power-law. This class is essential for the subclasses, however, using it as a standalone class would require some pre-handling of the data provided in the `inputs` folder. Therefore, it is not used directly in main.
* The 'DataSite' class is intended purely for investigating the data provided at the sites and is much more simple than the two other classes as it only contains two methods, one for the pre-handling of the data for the 'WindCalculation' class and one plotting the velocities at the given heights.
* The third and final class 'InterpolatedSite' contains most of the remaining methods intended in the package, those being interpolation of both position (regarding latitude and longitude) as well as height. Given the inputs for a desired point, the class then provides all of the data and visualizations mentioned in the overview of the package.

For a more in-depth description of each of the methods provided by the 3 classes please refer to the docstrings provided in the `classes.py` file in the `src` folder.

In addition to the 3 classes the package also provides 2 functions provided in `functions.py` in the `src` folder.
* The first function 'load_nc_folder_to_dataset' is a quality of life function that allows the user to load multiple NetCDF4 (.nc) files in a single line (given that the files are in the same folder).
* The second function 'pdf_weib' is simply the probability density function (PDF) which is used when working with the wind speed distribution in the 'InterpolatedSite' class.

A graphical overview of the classes and functions can be seen below:

<img src="diagrams/Overview.svg" alt="">

In addition to a general overview of the architecture of the package a flowchart of an intended use case has also been provided to show how the different methods within the classes work together. In the flow chart the most typical use case (an object from the 'InterpolatedSite' class) is being shown:

<img src="diagrams/Flowchart.svg" alt="">

## Peer review

Based on feedback from the previous project we have tried to adjust the quick-start guide, as it wasn't entirely clear last time. Additionally the doc-strings for the methods and functions should be more clear this time compared to last as well.

## Git workflow

To ensure a smooth development of the package, two approaches were taken:<br />
Firstly, as we didn't have a perfect overview of how to make the package in the most efficient way possible, the start of the project consisted of a lot of collaborative coding to ensure that all the team members agreed on the structure.<br />
When the very early parts of the work had been completed in a 'dev' branch (to avoid any commits to 'main'), several sub-branches from the 'dev' branch were then made: 'classing', 'cleaning', 'diagramming', and 'testing' (technically the last 3 were made as copies of 'classing' as seen in the diagram below). These all had their respective purposes for the project and were occasionally PR'ed to the 'dev' branch which was subsequently PR'd to 'main'. The distribution of tasks into different branches allowed for all the team members to work on different parts of the project without disturbing each other.

<img src="diagrams/GitWorkflow.svg" alt="">