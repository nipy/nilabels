# nilabel 

(ex. [LABelsToolkit](https://github.com/SebastianoF/LABelsToolkit))

Python 2.7 (upgrading to Python 3 in progress)

Nilabel is a set of tools to automatise simple manipulations and measurements of medical images and images 
segmentations in nifti format.

## Introductory example

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


+ Install [NiftySeg](https://github.com/KCL-BMEIS/NiftySeg)
+ Install [NiftyReg](https://github.com/KCL-BMEIS/niftyreg)
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


## Acknowledgements
+ This repository is developed within the [gift-SURG research project](http://www.gift-surg.ac.uk).
+ This work was supported by Wellcome / Engineering and Physical Sciences Research Council (EPSRC) [WT101957; NS/A000027/1; 203145Z/16/Z]. 
Sebastiano Ferraris is supported by the EPSRC-funded UCL Centre for Doctoral Training in Medical Imaging (EP/L016478/1) and Doctoral Training Grant (EP/M506448/1). 
