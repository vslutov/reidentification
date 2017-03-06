#!/bin/env python
# -*- coding: utf-8 -*-

"""Reidentification experiment."""

from setuptools import setup, find_packages

VERSION = "0.0.1"

setup(name='reidentification',
      version=VERSION,
      description=__doc__,
      maintainer='vslutov',
      maintainer_email='vladimir.lutov@graphics.cs.msu.ru',
      # url='',
      license='WTFPL',
      platforms=['any'],
      classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Environment :: Console"],
      install_requires=['numpy==1.12.0',
                        'h5py==2.6.0',
                        'pillow==4.0.0',
                        'scipy==0.18.1',
                        'scikit-image==0.12.3',
                        'scikit-learn==0.18.1',
                        'tensorflow==0.12.1',
                        'Keras==1.2.1',
                        'tabulate==0.7.7',
                        'pydot-ng==1.0.0',
                       ],
      packages=find_packages(),
      include_package_data=True,
      entry_points={'console_scripts': ['reidentification = reidentification.__main__:main']})
