#!/usr/bin/env python

import os
from setuptools import setup, find_packages


def requirements2list(pfi_txt='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    f = open(os.path.join(here, pfi_txt), 'r')
    l = []
    for line in f.readlines():
        l.append(line.replace('\n', ''))
    return l


setup(name='nilabels',
      version='v0.0.8',  # update also in definitions.py
      description='Toolkit to manipulate and measure image segmentations in nifti format.',
      author='sebastiano ferraris',
      author_email='sebastiano.ferraris@gmail.com',
      license='MIT',
      url='https://github.com/nipy/nilabels',
      packages=find_packages(),
      install_requires=requirements2list()
     )

