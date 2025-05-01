#!/usr/bin/env python

# setup script for SemiSim

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get long description from README if available
try:
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A deep learning model for predicting liquid-liquid phase separation (LLPS) using sequence and structural features.'

setup(
    name='semisim',
    version='0.1.0',
    description='SemiSim: A deep learning framework for LLPS prediction using sequence and NetSurfP features',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SemiSim/SemiSim-MU-',
    author='Moriah Miles',
    author_email='your.email@example.com',  # replace with your actual email
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit-learn',
        'matplotlib',  # optional
    ],
    extras_require={
        'dev': ['jupyter', 'pytest']
    },
    tests_require=['pytest'],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
)
