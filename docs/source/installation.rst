installation
============

The easiest way to install scab is to use pip::

    $ pip install scab

If you don't have pip, the Anaconda_ Python distribution contains pip along 
with a ton of useful scientific Python packages and is a great way to get 
started with Python.  

Stable_ and development_ versions of scab can also be downloaded from Github. 
You can manually install the latest development version of scab with::

    $ git clone https://github.com/briney/scab
    $ cd scab/
    $ git checkout development
    $ python setup.py install

.. tip::  
    If installing manually via setup.py and you don't already have scikit-bio installed, 
    you may get an error when setuptools attempts to install scikit-bio. This can be fixed 
    by first installing scikit-bio with pip::

        $ pip install scikit-bio

    and then retrying the manual install of scab.  


Requirements
------------

* Python 3.6+
* abutils_
* abstar_
* anndata
* dnachisel
* fastcluster
* harmonypy
* leidenalg
* matplotlib
* mnemonic
* natsort
* numpy
* pandas
* prettytable
* pytest
* python-Levenshtein
* scanpy
* scanorama
* scipy
* scrublet
* scvelo
* seaborn
* umap-learn



.. _Anaconda: https://www.continuum.io/downloads
.. _stable: https://github.com/briney/scab/releases
.. _development: https://github.com/briney/scab
.. _abutils: https://github.com/briney/abutils
.. _abstar: https://github.com/briney/abstar
