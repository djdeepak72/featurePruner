#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from pathlib import Path


exec(open("featurePruner/version.py").read())

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

LICENSE = (this_directory / "LICENSE.txt").read_text()

install_requires = [
 
    "pandas>=0.24.0",
    "numpy",
    "scipy",
    "optuna>=1.1.0",
    "lightgbm==2.3.1",
    "fastprogress>=0.2.2",
    "pytest-runner",
    "seaborn"
    ]

setup(
    name="featurePruner",
    description="A python package that uses permutation importance logic & custom clustering heuristics to find the most important variables in your modelling tasks quickly & effectively",
	long_description=long_description,
	long_description_content_type='text/markdown',
    author="DJ",
    author_email="willofdeepak@gmail.com",
    install_requires=install_requires,
    license=LICENSE,
    packages=find_packages(include=["featurePruner"]),
    python_requires=">=3.7",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    license_files=('LICENSE.txt'),
    version=__version__,
)