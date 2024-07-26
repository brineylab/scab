# import os
# import sys

# try:
#     from setuptools import setup
# except ImportError:
#     sys.exit("ERROR: setuptools is required.\n")


# try:  # for pip >= 10
#     from pip._internal.req import parse_requirements
# except ImportError:  # for pip <= 9.0.3
#     from pip.req import parse_requirements
# # try:
# #     from pip.req import parse_requirements
# # except ImportError:
# #     sys.exit('ERROR: pip is required.\n')


# if os.environ.get("READTHEDOCS", None):
#     # Set empty install_requires to get install to work on readthedocs
#     install_requires = []
# else:
#     req_file = "requirements.txt"
#     try:
#         reqs = parse_requirements(req_file, session=False)
#     except TypeError:
#         reqs = parse_requirements(req_file)
#     try:
#         install_requires = [str(r.req) for r in reqs]
#     except AttributeError:
#         install_requires = [str(r.requirement) for r in reqs]

# # read version
# exec(open("scab/version.py").read())

# config = {
#     "description": "Single cell analysis of B cells",
#     "author": "Bryan Briney",
#     "url": "https://www.github.com/briney/scab",
#     "author_email": "briney@scripps.edu",
#     "version": __version__,
#     "install_requires": install_requires,
#     "packages": ["scab"],
#     "scripts": ['bin/batch_cellranger', 'bin/scabranger'],
#     "name": "scab",
#     "include_package_data": True,
#     "classifiers": [
#         "License :: OSI Approved :: MIT License",
#         "Programming Language :: Python :: 3.6",
#         "Programming Language :: Python :: 3.7",
#         "Programming Language :: Python :: 3.8",
#         "Topic :: Scientific/Engineering :: Bio-Informatics",
#     ],
# }

# setup(**config)


import os

from setuptools import find_packages, setup

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
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
)
