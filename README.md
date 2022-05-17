![](https://img.shields.io/pypi/v/scab.svg?colorB=blue)
[![Build Status](https://travis-ci.com/briney/scab.svg?branch=master)](https://travis-ci.com/briney/scab)
[![Documentation Status](https://readthedocs.org/projects/scab/badge/?version=latest)](https://scab.readthedocs.io/en/latest/?badge=latest)
![](https://img.shields.io/pypi/pyversions/scab.svg)
![](https://img.shields.io/badge/license-MIT-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# scab

Single cell analysis of B cells.
scab is a core component of the ab\[x\] toolkit for antibody sequence analysis.
  
  - Source code: [github.com/briney/scab](https://github.com/briney/scab)  
  - Documentation: [scab.readthedocs.org](https://scab.readthedocs.io)  
  - Download: [pypi.python.org/pypi/scab](https://pypi.python.org/pypi/scab)  
  
### install  
`pip install scab`  


### api  
The intended use of scab is through the public API, enabling incorporation of scab's methods and utilities into integrated analysis pipelines or for interative analysis of antibody repertoires in notebook-style computing environments like Jupyter. See the scab [documentation](http://scab.readthedocs.io) for more detail about the API.  


### testing  
To run the test suite, clone or download the repository and run `pytest ./` from the top-level directory. The same tests are run after every commit using TravisCI.  
  

### requirements  
Python 3.6+  
abstar
abutils>=0.2.5
anndata
dnachisel
fastcluster
harmonypy
leidenalg
matplotlib
mnemonic
natsort
numpy
pandas
prettytable
pytest
python-Levenshtein
scanpy
scanorama
scipy
scrublet
scvelo
seaborn>=0.11
umap-learn
  
All of the above dependencies can be installed with pip, and will be installed automatically when installing scab with pip.  
If you're new to Python, a great way to get started is to install the [Anaconda Python distribution](https://www.continuum.io/downloads), which includes pip as well as a ton of useful scientific Python packages.
  
