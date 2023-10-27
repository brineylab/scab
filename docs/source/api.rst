.. _api:


api reference
=============

read and write: ``io``
-----------------------

.. toctree::
    :hidden:

    modules/io

Read and integrate the ouput data produced by 10x Genomics' CellRanger, 
load and save integrated ``AnnData`` objects to ``h5ad``-formatted files, and concatenate 
multiple ``AnnData`` objects.  

.. tip:: 
    All of the methods in ``io`` are also available directly from ``scab`` itself, 
    so ``scab.read_10x_mtx()`` is identical to ``scab.io.read_10x_mtx()``. This was 
    done to make the API more consistent with scanpy. 

Each function is designed to replicate the behavior of their scanpy or anndata equivalent, but 
to add necessary fuctionality to accomodate BCR and/or TCR sequence data. Additionally, we have added two 
additional functions -- ``load`` and ``save`` which are equivalent to ``read`` and ``write`` in Scanpy but 
which we belive are named more descriptively. You can use ``scab.read()``// ``scab.load()`` and ``scab.write()``// 
``scab.save()`` interchangably. Re-implementing these functions is necessary, because BCR/TCR 
annotations cannot be writted to ``h5ad``-formatted files directly, so ``scab.save()`` serializes them prior 
to saving, and de-serializes when loading with ``scab.load()``. This means that while ``scab.load()`` can be
used to read files saved with ``scanpy.write()``, ``scanpy.read()`` will not properly load files containing 
BCR/TCR annotations that were saved using ``scab.save()``.

.. currentmodule:: scab.io

.. autosummary::
    :nosignatures:

    read_10x_mtx
    load 
    save
    concat



preprocessing: ``pp``
---------------------

.. toctree::
    :hidden:

    modules/pp

Filtering and normalization of GEX data, doublet detection and removal. 

.. currentmodule:: scab.pp

.. autosummary::
   :nosignatures:

   filter_and_normalize
   remove_doublets
   scrublet
   doubletdetection


tools: ``tl``
---------------------

.. toctree::
    :hidden:
    :maxdepth: 2

    modules/tl

Filtering and normalization of GEX data, doublet detection and removal. 


batch correction
~~~~~~~~~~~~~~~~

.. currentmodule:: scab.tools.batch_correction

.. autosummary::
   :nosignatures:

   combat
   harmony
   mnn
   scanorama



cell hashes
~~~~~~~~~~~~~~~~

.. currentmodule:: scab.tools.cellhashes

.. autosummary::
   :nosignatures:

   demultiplex


clonality
~~~~~~~~~~~~~~~~

.. currentmodule:: scab.tools.clonify

.. autosummary::
   :nosignatures:

   clonify



embeddings
~~~~~~~~~~~~~~~~

.. currentmodule:: scab.tools.embeddings

.. autosummary::
   :nosignatures:

   pca
   umap



specificity
~~~~~~~~~~~~~~~~

.. currentmodule:: scab.tools.specificity

.. autosummary::
   :nosignatures:

   classify_specificity




.. toctree::

    modules/vdj








.. toctree::

    modules/pl






.. _scrublet: https://github.com/swolock/scrublet
.. _doubletdetection: https://github.com/JonathanShor/DoubletDetection
.. _ComBat: https://github.com/brentp/combat.py
.. _mutual nearest neighbors: https://github.com/chriscainx/mnnpy
.. _Scanorama: https://github.com/brianhie/scanorama
.. _clonify: https://github.com/briney/clonify  