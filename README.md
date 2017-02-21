# Label manager

Quick manipulation of medical images segmentations.
Main image format considered for images is nifti (.nii or .nii.gz).

## Examples

What can you do with the package labels_manager:



## Instructions
The code can be installed as a python package on a virtualenv (strongly recommended as it is recommended to avoid installing this package on system python interpreter).
As usual I do not take responsibility for any damage this package may cause to everything of anyone.
It is based on NiftyReg, NiftySeg, ITK-snap and some python (2.7) standard libraries

+ Install [NiftySeg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_install)
+ Install [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)
+ Install [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)

+ Install python requirements in requirements.txt with

    pip install -r requirements.txt

in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

### Detailed instructions after installing NiftyReg, NiftySeg and requirement

1) activate the virtualenvironment in the root folder of the repository.

To install as a library (option 1):
2a) python setup.py sdist
2b) cd ../
2c) pip install label_manager/dist/LabelsManager-XX.tar.gz
where XX is the chosen version
To install as a library (option 2):
2a) python setup.py install

To install in [development mode](http://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode) (modality 1) 
2a) python setup.py develop
To install in development mode (option 2)
2a) pip install -e .

To verify that it works:
3) python
4) from labels_manager.main import LabelsManager as LM
5) lm = LM('/some/folder/containing/images')

To uninstall:
 pip uninstall LabelsManager
 
To uninstall in case something really wrong happen:
  sudo rm -rf /path_to_site_package_in_virtualenv/site-packages/LabelsManager*