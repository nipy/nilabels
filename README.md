# NiLabels 

(ex. [LABelsToolkit](https://github.com/SebastianoF/LABelsToolkit))

NiLabels is a set of tools to automatise simple manipulations and measurements of medical images 
segmentations in nifti format.

### Features

+ Written in Python 2.7

+ Simplify the access to a range of tools and algorithms for the manipulation of medical image segmentations (see more under [motivations](https://github.com/SebastianoF/nilabels/wiki/Motivations))

+ After importing NiLabels, you can quickly: 
    + check if there are missing labels in a segmentation. 
    + counts the number of connected components.
    + get the corresponding 4D RGB image (with colors provided in ITK-snap or fsl labels descriptor format) 
    + apply rotations and translations to the image header
    + change the segmentation numbering, erase or merge labels 
    + compute Dice's score, covariance distance, Hausdorff distance and normalised symmetric contour distance between segmentations 
    + get the array of values at the voxel below a given label 
    + symmetrise a segmentation 
    + [...and more](https://github.com/SebastianoF/nilabels/wiki/What-you-can-do-with-nilabels)

+ Facade design pattern to make it easily extendible (see the [docs](https://github.com/SebastianoF/nilabels/wiki/Design-Pattern))


### Non-features (work in progress)

+ Not yet Python 3, back compatible python 2.7
+ Not yet 80% coverage
+ Not yet pip-installable
+ [... and more](https://github.com/SebastianoF/nilabels/wiki/Work-in-Progress)


### Introductory example

Given a segmentation `my_input_data/my_segm.nii.gz` you want to change the labels values from [1, 2, 3, 4, 5, 6] to [2, 12, 4, 7, 5, 6]. Then:

```python
import nilabels as nis


nis_app = nis.App('my_input_data')
nis_app.manipulate.relabel('my_segm.nii.gz', 'my_new_segm.nii.gz', [1, 2, 3, 4, 5, 6], [2, 12, 4, 7, 5, 6])
```

### How to install (in development mode) 

+ Install [NiftySeg](https://github.com/KCL-BMEIS/NiftySeg)
+ Install [NiftyReg](https://github.com/KCL-BMEIS/niftyreg)
+ Install [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)

**Note:** NiftySeg, NiftySeg and ITK-Snap are required only for advanced functions.

+ Install python requirements in requirements.txt with
    `pip install -r requirements.txt`
in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).


+ Install NiLabels: the current version is not (yet) pip installable. It can be installed in development mode.
To proceede, initialise a virtual environment and execute the following instructions:
```
cd <folder where to clone the code>
git clone https://github.com/SebastianoF/nilabels.git
cd nilabel
pip install -e .
```
In development mode every change made to your local code will be directly affecting the libray installed in the python distribution
without the need of reinstalling.


### Documentations

+ [Wiki-pages documentation](https://github.com/SebastianoF/nilabels/wiki)


### Code testing

Code testing is a work in progress. We are aiming at reaching the 80% of coverage for the methods acting on numpy arrays and nibabel images, below the facade.
To run the test, `pip install -U pytest-cov` followed by:
```bash
py.test --cov-report html --cov nilabel --verbose
open htmlcov/index.html
```
The first time the test a run a dummy dataset is created. This may take some minutes of computations and 150MB of space.

### Licencing and Copyright

Copyright (c) 2017, Sebastiano Ferraris. NiLabels  (ex. LABelsToolkit) is provided as it is and 
it is available as free open-source software under 
[MIT License](https://github.com/SebastianoF/nilabels/blob/master/LICENCE.txt)


### Acknowledgements

+ This repository is developed within the [GIFT-surg research project](http://www.gift-surg.ac.uk).
+ This work was supported by Wellcome / Engineering and Physical Sciences Research Council (EPSRC) [WT101957; NS/A000027/1; 203145Z/16/Z]. 
Sebastiano Ferraris is supported by the EPSRC-funded UCL Centre for Doctoral Training in Medical Imaging (EP/L016478/1) and Doctoral Training Grant (EP/M506448/1). 
