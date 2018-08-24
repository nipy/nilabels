#!/usr/bin/env python

from nilabels.__init__ import __version__
from setuptools import setup, find_packages
# from distutils.core import setup

setup(name='nilabels',
      version=__version__,
      description='Toolkit to manipulate and measure image segmentations in nifti format.',
      author='sebastiano ferraris',
      author_email='sebastiano.ferraris@gmail.com',
      license='MIT',
      url='https://github.com/SebastianoF/nilabels',
      packages=find_packages(),
     )

