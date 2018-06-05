# LABelsToolkit
Python 2.7

LABelsToolkit is a set of tools to automatise simple manipulations and measurements of medical images and images 
segmentations in nifti format.


### [Motivations](https://github.com/SebastianoF/LABelsToolkit/wiki/Motivations)

### [What you can do with LABelsToolkit](https://github.com/SebastianoF/LABelsToolkit/wiki/What-you-can-do-with-LABelsToolkit)

### [Design pattern](https://github.com/SebastianoF/LABelsToolkit/wiki/Design-Pattern)

## How to install (in development mode) 


+ Install [NiftySeg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_install)
+ Install [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)
+ Install [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)

+ Install python requirements in requirements.txt with
    `pip install -r requirements.txt`
in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).


+ Install LABelsToolkit: the current version is not (yet) pip installable. It can be installed in development mode.
To proceede, initialise a virtual environment and execute the following instructions:
```
cd <folder where to clone the code>
git clone https://github.com/SebastianoF/LABelsToolkit.git
cd LABelsToolkit
pip install -e .
```
In development mode every change made to your local code will be directly affecting the libray installed in the python distribution
without the need of reinstalling.

## Documentations

+ [Wiki-pages documentation](https://github.com/SebastianoF/LABelsToolkit/wiki)


## Code testing

Code testing is a work in progress. We are aiming at reaching the 80% of coverage for the methods acting on numpy arrays and nibabel images, below the facade.
To run the test, type `nosetests` in the root directory of the project.

## Licence

Copyright (c) 2017, Sebastiano Ferraris. LABelsToolkit (ex. LabelsManager) is available as free open-source software under 
[MIT License](https://github.com/SebastianoF/LABelsToolkit/blob/master/LICENCE.txt)

