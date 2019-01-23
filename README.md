
[![coverage](https://github.com/SebastianoF/nilabels/blob/master/coverage.svg)](https://github.com/SebastianoF/nilabels/blob/master/coverage.svg)
[![CircleCI](https://circleci.com/gh/SebastianoF/nilabels.svg?style=svg)](https://circleci.com/gh/SebastianoF/nilabels)

<p align="center">
<img src="https://github.com/SebastianoF/nilabels/blob/master/logo_low.png" width="300">
</p>

# NiLabels

NiLabels is a cacophony of tools to automatise simple manipulations and measurements of medical image
segmentations in nifti format. It is heavily based on and influenced by the library [NiBabel](http://nipy.org/nibabel/)

+ Written in [Python 3.6](https://docs.python-guide.org/) back compatible with 2.7
+ [Motivations](https://github.com/SebastianoF/nilabels/wiki/Motivations)
+ [Features](https://github.com/SebastianoF/nilabels/wiki/What-you-can-do-with-nilabels)
+ [Design pattern](https://github.com/SebastianoF/nilabels/wiki/Design-Pattern)
+ [Work in progress](https://github.com/SebastianoF/nilabels/wiki/Work-in-Progress)

### Introductory example

Given a segmentation `my_segm.nii.gz` imagine you want to change the labels values from [1, 2, 3, 4, 5, 6] to [2, 12, 4, 7, 5, 6]
and save the result in `my_new_segm.nii.gz`. Then:

```python
import nilabels as nil


nil_app = nil.App()
nil_app.manipulate_labels.relabel('my_segm.nii.gz', 'my_new_segm.nii.gz',  [1, 2, 3, 4, 5, 6], [2, 12, 4, 7, 5, 6])

```

### Instructions

+ [Documentation](https://github.com/SebastianoF/nilabels/wiki)
+ [How to install](https://github.com/SebastianoF/nilabels/wiki/Instructions)
+ [How to run the tests](https://github.com/SebastianoF/nilabels/wiki/Testing)


### Licencing and Copyright

Copyright (c) 2017, Sebastiano Ferraris. NiLabels  (ex. [LABelsToolkit](https://github.com/SebastianoF/LABelsToolkit))
is provided as it is and it is available as free open-source software under
[MIT License](https://github.com/SebastianoF/nilabels/blob/master/LICENCE.txt)


### Acknowledgements

+ This repository had begun within the [GIFT-surg research project](http://www.gift-surg.ac.uk).
+ This work was supported by Wellcome / Engineering and Physical Sciences Research Council (EPSRC) [WT101957; NS/A000027/1; 203145Z/16/Z]. 
Sebastiano Ferraris is supported by the EPSRC-funded UCL Centre for Doctoral Training in Medical Imaging (EP/L016478/1) and Doctoral Training Grant (EP/M506448/1). 
