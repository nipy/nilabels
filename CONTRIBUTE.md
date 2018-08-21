# Contributing to nilabel

Thank you for being there!

Nilabel (ex LABelsToolkit) started as a python package containing a range of heterogeneous imaging tools to perform
quick manipulations and measurement on segmentations from ipython or jupyter notebook. 
Initially planned to support several projects undertook by the initial author, after some development and refactoring 
 it is not intended to be part of the Nipy ecosystem, to provide the community with another hopefully helpful tool 
  in neuroimaging research.

## Code of Conduct

This project adopts the [Covenant Code of Conduct](https://contributor-covenant.org/). 
By participating, you are expected to uphold this code. 
 
## Before starting 

Please familiarise with the design pattern and the nomenclature employed.
 + **tools:** core methods are all there, divivded by final intended aim. A tool acts on the numpy arrays or on 
 instances of nibabel images.
 + **agents** are facades collecting all the tools, and make them act directly on the paths to the nifti images.  
 + **main:** is facade of the facades under agents folder package. This collects all the methods under 
     the agents facades, therefore accessing to all the tools.
     
Typical usage in an ipython session involves importing the main facade, and then some tab completition to browse
 the provided methods.
 
## Contributions: Questions, bugs, issues and new features 

+ For any issue bugs or question related to the code, please raise an issue in the 
[nilabel issue page](https://github.com/SebastianoF/nilabel/issues).

+ Propose here as well improvements suggestions and new features.

+ **Please use a new issue for each thread:** make your issue re-usable and reachable by other users that may have 
encountered a similar problem.

+ If you forked the repository and made some contributions that you would like to integrate in the git master branch, 
you can do a [git pull request](https://yangsu.github.io/pull-request-tutorial/). Please **check tests are all passed** 
before this (type nosetests in the code root folder).

  
## Styleguides

+ The code follows the [PEP-8](https://www.python.org/dev/peps/pep-0008/) style convention. 
+ Please follow the [ITK standard prefix commit message convention](https://itk.org/Wiki/ITK/Git/Develop) for commit messages. 
+ Please use the prefix `pfi_` and `pfo_` for the variable names containing path to files and path to folders respectively

## To-Do list and work in progress

Please see under [todo wiki-page](https://github.com/SebastianoF/nilabel/wiki/Work-in-Progress) 
for the future intended future work and directions.
