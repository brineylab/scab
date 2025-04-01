# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import os

from setuptools import find_packages, setup

# the actual __version__ is read from version.py, not assigned directly
# this causes the linter to freak out when we mention __version__ in setup()
# to fix that, we fake assign it here
__version__ = None

# read version
version_file = os.path.join(os.path.dirname(__file__), "scab", "version.py")
with open(version_file) as f:
    exec(f.read())

# read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# read long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scab",
    version=__version__,
    author="Bryan Briney",
    author_email="briney@scripps.edu",
    description="Single cell analysis of B cells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brineylab/scab",
    packages=find_packages(),
    scripts=[
        "bin/batch_cellranger",
        "bin/scabranger",
    ],
    entry_points={
        "console_scripts": [
            "scab=scab.scripts.scab:cli",
        ]
    },
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
)
