#!/usr/bin/env python

from nilabels.definitions import __version__
from setuptools import setup, find_packages
# from distutils.core import setup

setup(name='nilabels',
      version='v0.0.6',
      description='Toolkit to manipulate and measure image segmentations in nifti format.',
      author='sebastiano ferraris',
      author_email='sebastiano.ferraris@gmail.com',
      license='MIT',
      url='https://github.com/SebastianoF/nilabels',
      packages=find_packages(),
     )

