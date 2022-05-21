
scab: single cell analysis of B cells
===================================================================
 
scab provides a computational framework for integrating, analyzing and 
visualizing single B cell multi-omics data. We have developed a simple, 
straightforward API which should feel quite familiar to users of scanpy_) 
and  allows sophisticated analyses with just a few lines of code:  


.. image:: images/scab_console-workflow.jpg
  :width: 750
  :alt: example workflow in scab
  :align: center
  :class: only-light
  

.. image:: images/scab_console-workflow_inverted.jpg
  :width: 750
  :alt: example workflow in scab
  :align: center
  :class: only-dark

  
|   
  
scab is a standards-based toolkit built on the ``AnnData`` objects used 
by scanpy_. We integrate several tools specifically designed to facilitate 
analysis of adaptive immune cells and receptor repertoires.     


|  

.. toctree::
   :maxdepth: 1
   :caption: getting started

   overview
   installation


|

.. toctree::
   :maxdepth: 1
   :caption: usage

   examples
   api
   

|

.. toctree::
   :maxdepth: 1
   :caption: about

   license
   news


|

.. toctree::
   :maxdepth: 1
   :caption: related projects

   abstar <https://github.com/briney/abstar>
   abutils <https://github.com/briney/abutils>
   abcloud <https://github.com/briney/abcloud>
   clonify <https://github.com/briney/clonify-python>





.. _scanpy: https://github.com/scverse/scanpy
