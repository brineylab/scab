import os
import sys

try:
    from setuptools import setup
except ImportError:
    sys.exit("ERROR: setuptools is required.\n")


try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
# try:
#     from pip.req import parse_requirements
# except ImportError:
#     sys.exit('ERROR: pip is required.\n')


if os.environ.get("READTHEDOCS", None):
    # Set empty install_requires to get install to work on readthedocs
    install_requires = []
else:
    req_file = "requirements.txt"
    try:
        reqs = parse_requirements(req_file, session=False)
    except TypeError:
        reqs = parse_requirements(req_file)
    try:
        install_requires = [str(r.req) for r in reqs]
    except AttributeError:
        install_requires = [str(r.requirement) for r in reqs]

# read version
exec(open("scab/version.py").read())

config = {
    "description": "Single cell analysis of B cells",
    "author": "Bryan Briney",
    "url": "https://www.github.com/briney/scab",
    "author_email": "briney@scripps.edu",
    "version": __version__,
    "install_requires": install_requires,
    "packages": ["scab"],
    "scripts": ['bin/batch_cellranger', 'bin/scabranger'],
    "name": "scab",
    "include_package_data": True,
    "classifiers": [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
}

setup(**config)
