# nilabel (ex LABelsToolkit)

Python 2.7 (upgrading in progress)

Nilabel is a set of tools to automatise simple manipulations and measurements of medical images and images 
segmentations in nifti format.

## Warm up example

Let's say you need to change the labels values from [1, 2, 3, 4, 5, 6] to [2, 3, 4, 5, 6, 7] in a list of 10 
segmentations `file{1..10}.nii.gz`. You can then apply the tool `relabel` under `tools.manipulations.relabel` as:

```python
from nilabel.main import Nilabel as NiL


nil = NiL(<input_folder>, <output_folder>)

for i in range(1, 11):
    input_file_name  = 'file{}.nii.gz'.format(i)
    output_file_name = 'file{}.nii.gz'.format(i)
    nil.manipulate.relabel(input_file_name, output_file_name,
                           [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
```

## Further infos

+ [Motivations](https://github.com/SebastianoF/nilabel/wiki/Motivations)

+ [What you can do with nilabel](https://github.com/SebastianoF/nilabel/wiki/What-you-can-do-with-LABelsToolkit)

+ [Design pattern](https://github.com/SebastianoF/nilabel/wiki/Design-Pattern)

## How to install (in development mode) 


+ Install [NiftySeg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_install)
+ Install [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)
+ Install [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)

+ Install python requirements in requirements.txt with
    `pip install -r requirements.txt`
in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).


+ Install nilabel: the current version is not (yet) pip installable. It can be installed in development mode.
To proceede, initialise a virtual environment and execute the following instructions:
```
cd <folder where to clone the code>
git clone https://github.com/SebastianoF/nilabel.git
cd nilabel
pip install -e .
```
In development mode every change made to your local code will be directly affecting the libray installed in the python distribution
without the need of reinstalling.

## Documentations

+ [Wiki-pages documentation](https://github.com/SebastianoF/nilabel/wiki)


## Code testing

Code testing is a work in progress. We are aiming at reaching the 80% of coverage for the methods acting on numpy arrays and nibabel images, below the facade.
To run the test, type `nosetests -s` in the root directory of the project. Note that at the first run
a testing dataset is generated. This may take some minutes.

## Licence

Copyright (c) 2017, Sebastiano Ferraris. nilabel  (ex. LABelsToolkit ex. LabelsManager) is provided as it is and 
it is available as free open-source software under 
[MIT License](https://github.com/SebastianoF/nilabel/blob/master/LICENCE.txt)
