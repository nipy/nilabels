# LABelsToolkit
Python 2.7

LABelsToolkit is a set of tools to automatise simple manipulations and measurements of medical images and images 
segmentations in nifti format.

Each tool provided can have as input 
* Nibabel instances, numpy arrays and others.
* Paths to the corresponding objects trough the facades called **agents**.


## How to "install" 

Being a work in progress at this stage of development, this software is not packaged with wheel. 
To install it in a local python distribution (ideally in a virtual environment) please execute the following instructions:
```
cd <folder where to clone the code>
git clone https://github.com/SebastianoF/LABelsToolkit.git
cd LABelsToolkit
pip install -e .
```
The code is installed in development mode and every modification to the code will directly affect the code.

## Documentations

+ [Wiki-pages documentation](https://github.com/SebastianoF/LabelsManager/wiki)


## Code testing

To run the test, type `nosetests` in the root directory of the project.

## Licence

Copyright (c) 2017, Sebastiano Ferraris. LabelsManager is available as free open-source software under 
[MIT License](https://github.com/SebastianoF/LabelsManager/blob/master/LICENCE.txt)

