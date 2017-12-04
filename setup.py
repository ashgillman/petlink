#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
from setuptools.extension import Extension

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext


cython_compile_args = [
    '-O2',
    '-Wno-cpp', # deprecated NumPy API
    '-Wno-unused-but-set-variable',
    '-Wno-unused-function',
]

unlisting = Extension(
    'petlink.listmode.unlisting',
    [os.path.join('petlink', 'listmode', 'unlisting.pyx')],
    include_dirs=[numpy.get_include()],
    extra_compile_args=cython_compile_args,
)

data_files = [
    os.path.join('listmode', 'templates', '*.hs'),
]


setup(
    name='petlink',
    description='PETLINK and Interfile tools',
    author='Ashely Gillman',
    version='0.0.1',
    license='nil',
    packages=find_packages(exclude=['tests']),
    ext_modules=cythonize([unlisting]),
    package_data=dict(petlink=data_files),
    setup_requires=['pytest-runner', 'cython'],
    install_requires=['numpy', 'pyparsing', 'pydicom'],
    tests_require=['pytest'],
)
