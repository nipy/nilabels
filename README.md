
[![coverage](https://github.com/nipy/nilabels/blob/master/coverage.svg)](https://github.com/SebastianoF/nilabels/blob/master/coverage.svg)
[![CircleCI](https://circleci.com/gh/nipy/nilabels.svg?style=svg)](https://circleci.com/gh/nipy/nilabels)
[![PyPI version](https://badge.fury.io/py/nilabels.svg)](https://badge.fury.io/py/nilabels)


<p align="center">
<img src="https://github.com/SebastianoF/nilabels/blob/master/logo_low.png" width="300">
</p>

# NiLabels

NiLabels is a cacophony of tools to automate simple manipulations and measurements of medical image
segmentations in nifti format. It is strongly based on and influenced by the library [NiBabel](http://nipy.org/nibabel/)

+ Written in [Python 3.6](https://docs.python-guide.org/) back compatible with 2.7
+ [Motivations](https://github.com/SebastianoF/nilabels/wiki/Motivations)
+ [Features](https://github.com/SebastianoF/nilabels/wiki/What-you-can-do-with-nilabels)
+ [Design pattern](https://github.com/SebastianoF/nilabels/wiki/Design-Pattern)
+ [Work in progress](https://github.com/SebastianoF/nilabels/wiki/Work-in-Progress)

### Introductory examples

#### 1 Manipulate labels: relabel

Given a segmentation, imagine you want to change the labels values from [1, 2, 3, 4, 5, 6] to [2, 12, 4, 7, 5, 6]
and save the result in `my_new_segm.nii.gz`. Then:

```python
import nilabels as nil


nil_app = nil.App()
nil_app.manipulate_labels.relabel('my_segm.nii.gz', 'my_new_segm.nii.gz',  [1, 2, 3, 4, 5, 6], [2, 12, 4, 7, 5, 6])

```

#### 2 Manipulate labels: clean a segmentation

Given a parcellation for which we expect a single connected component per label, we want to have it cleaned from all the
extra components, merging them with the closest labels.

```python
import nilabels as nil


nil_app = nil.App()

nil_app.check.number_connected_components_per_label('noisy_segm.nii.gz', where_to_save_the_log_file='before_cleaning.txt')
nil_app.manipulate_labels.clean_segmentation('noisy_segm.nii.gz', 'cleaned_segm.nii.gz', force_overwriting=True)
nil_app.check.number_connected_components_per_label('cleaned_segm.nii.gz', where_to_save_the_log_file='after_cleaning.txt')

```
<p align="center">
<img src="https://github.com/SebastianoF/nilabels/blob/master/examples/cleaning_before_after.png" width="600">
</p>


Before cleaning `check.number_connected_components_per_label` would return:
```

Label 0 has 1 connected components
Label 1 has 13761 connected components
Label 2 has 14175 connected components
Label 3 has 14373 connected components
Label 4 has 1016 connected components
Label 5 has 806 connected components
Label 6 has 816 connected components
Label 7 has 1281 connected components
Label 8 has 977 connected components
Label 9 has 746 connected components
```

The same command after cleaning:
```
Label 0 has 1 connected components
Label 1 has 1 connected components
Label 2 has 1 connected components
Label 3 has 1 connected components
Label 4 has 1 connected components
Label 5 has 1 connected components
Label 6 has 1 connected components
Label 7 has 1 connected components
Label 8 has 1 connected components
Label 9 has 1 connected components
```

More tools are introduced in the [documentation](https://github.com/SebastianoF/nilabels/wiki/What-you-can-do-with-nilabels).

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
