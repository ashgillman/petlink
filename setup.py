#!/usr/bin/env python3

import os
from glob import glob
from setuptools import setup, find_packages

setup(
    name='petlink',
    description='PETLINK and Interfile tools',
    author='Ashely Gillman',
    version='0.0.0',
    license='nil',
    packages=find_packages(exclude=['*.tests']),
)
