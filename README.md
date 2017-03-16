# Labels Manager (labels_manager)
Python 2.7

An essential package for simple and generic manipulations of medical images segmentations in nifti format (.nii or .nii.gz).

## What can you do with label manager

What can you do with the package labels_manager: access all the functions stored in tools, programmed to work with numpy array, using the facade manager that works with paths to nifti images. 
Tools consists of simple manipulations of labels, mainly based on [NiftySeg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_install) and [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install).
The visualisation software preferred is [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3).

Examples of tools are divided in:
* Cutter: cut each timepoint of a 4d volumes with the same 3d mask
* Merger: to stack images an to merge labels stored in a 4d volume to a 3d one.
* Splitter: opposite direction as merger.
* Relabeller: change the values of the labels according to given permutation or labels lists.
* Symmetriser: symmetrise a segmentation from one side to the anatomical image to the other, by flipping or with rigid or affine registration to compensate for asymmetries.
* Measurements: compute distances between patches (generated with morphological tools). Compute the centroid of each label of a segmentation, compute linear measurements.

The dev branch is devoted to prototyping algorithms and for benchmarking respect to the state of the art segmentation algorithms (NiftySeg).
It currently contains some early developments of label fusion algorithms.

## Instructions
The tools and manager in the code can be installed as a python package on a virtualenv 
(strongly recommended as it is recommended to avoid installing this package on system python interpreter).
It is based on NiftyReg, NiftySeg, ITK-snap and some python (2.7) standard libraries

+ Install [NiftySeg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftySeg_install)
+ Install [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)
+ Install [ITK-snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)

+ Install python requirements in requirements.txt with

    pip install -r requirements.txt

in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

### To use as a package in a virtualenv.

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
5) lm = LM('/some/folder/containing/nifti/images')

To uninstall:
 pip uninstall LabelsManager
 
To delete the library in the virtualenv in case something really wrong happen and pip uninstall will not work:
  sudo rm -rf /path_to_site_package_in_virtualenv/site-packages/LabelsManager*
 
## Tests - Nosetest
To check that everything works run nosetests in the root directory of the project.
  
## Working examples
To see some toy examples of what can be done with LabelsManager, go to LabelsManager/examples and run simple_relabelling_examples.py after running
run generate_images_examples.py and after selecting the example you want to run in the module itself. Explore other examples as manipulator_example.py.

