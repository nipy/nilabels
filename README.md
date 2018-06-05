# LABelsToolkit
Python 2.7

LABelsToolkit is a set of tools to automatise simple manipulations and measurements of medical images and images 
segmentations in nifti format.

## Motivation

An neurological or anatomical image is usually a file saved in nifti format. 
The algorithm written to manipulate it can act on the image header or on the image data (or the corresponding numpy array) where is it tested.

Apply the algorithm to the image usually involves the repetitive task of loading the image with nibabel, extract
the header or the data, apply the algorithm, create a new instance of the modified nifti image, and finally save it.

LABelsToolkit is aimed at reducing this at one step, allowing to have as input and output directly the path to the input image and the output image.
 

## Intended design pattern:

The code is designed around the command design pattern.

Each core function is designed to act on numpy arrays or nibabel instances, and it is tested on dummy examples. 

The facade methods, acting on paths to the images in nifti format, access the core method and return another image 
as output.

## What you can do with LABelsToolkit

Main features of the code are:

1) Check if there are missing labels and returns the number of connected components per label.

2) Get the stacks of 4d images so that they are predisposed for label fusion algorithms (with niftySeg and others)
    
3) Quickly manipulate the components of the header, applying rotations respect to the center of the image or the origin, change the datatype or the translational part.
    
4) Normalise the intensities of an image given the values below a list of labels of a segmentation, getting the contour of a segmentation, and produce graftings between segmentations.

5) Relabel a segmentation given the list of old and the corresponding new labels, permute labels according to a given permutation, keep only one label and assign all the others the same value,
get the probabilistic prior from a stack of segmentations.

6) Clean a segmentation leaving only a pre-defined number of connected components for each label and merging the extra components with the closest label.

7) Compute volumes and distances between segmentations, such as the dice score, covariance distance, Hausdorff distance and normalised symmetric contour distance.

8) Manipulate the shape of the volume of an image, extending a slice an new dimension, splitting the labels in a 4d image and merge it back, and cut each slice
of a 4d volume for a given 3d binary compatible segmentation.

9) Symmetrize a segmentation according to an axis (with and without registration based on an underpinning anatomy).


## How to install (in development mode) 

The current version is not (yet) pip installable. It can be installed in development mode.
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

