#!/bin/env python
# -*- coding: utf-8 -*-

"""Reidentification experiment."""

from setuptools import setup, find_packages

VERSION = "0.1.1"

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
      install_requires=['numpy>=1.12.1',

                        'Keras>=2.0,<3',
                        'theano>=0.9',
                        'tensorflow-gpu>=1.1,<2',

                        'scipy>=0.19',
                        'scikit-image>=0.13',
                        'scikit-learn>=0.18',

                        'tabulate>=0.7',
                        'pydot-ng>=1.0,<2',
                        'pillow>=4.1',

                        'h5py>=2.7,<3',
                        'hdf5storage>=0.1',
                       ],
      packages=find_packages(),
      include_package_data=True,
      entry_points={'console_scripts': ['reidentification = reidentification.__main__:main']})
