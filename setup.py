#!/usr/bin/env python

from LABelsToolkit.__init__ import __version__
from setuptools import setup, find_packages
# from distutils.core import setup

setup(name='LABelsToolkit',
      version=__version__,
      description='Simple semgentation manager in nifti format.',
      author='sebastiano ferraris',
      author_email='sebastiano.ferraris@gmail.com',
      license='MIT',
      url='https://github.com/SebastianoF/LABelsToolkit',
      packages=find_packages(),
     )

