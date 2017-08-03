#!/usr/bin/env python

from setuptools import setup, find_packages
# from distutils.core import setup

setup(name='LabelsManager',
      version='1.0.dev',
      description='Manage the labels of the segmentations of nifti images.',
      author='sebastiano ferraris',
      author_email='sebastiano.ferraris@gmail.com',
      license='MIT',
      url='https://github.com/SebastianoF/LabelsManager',
      packages=find_packages(),
     )

