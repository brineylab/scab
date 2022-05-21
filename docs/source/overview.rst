overview
========

As single cell omics tools become increasingly important for 
characterizing adaptive immunity, we noted the need for open, 
easy-to-use software designed specifically for analysis of 
adaptive immune cells.  

We built **scab** to fill this need. It was engineered to use an 
API that should be familiar to users of scanpy_, which is the 
most widely used Python package for general single cell omics 
analysis. Beyond the API similarities, scab builds directly 
on the models and functions introduced by scanpy to create 
specialized tools that address issues related specifically 
to the analysis of adaptive immune single cell omics data.  


tools
---------

scab provides a range of utilities for all stages of single cell 
adaptive immune analysis, including data I/O, immune receptor 
annotation, sample demultiplexing, antigen specificity classification, 
and clonal lineage assignment. scab also includes 
a variety of visualization tools designed to facilitate exploratory 
analyses and generate publication-quality figures.  


file and data standards
------------------------

From the start, scab was designed to be compatible with file and 
data formats that have been accepted as standards in the immunology 
community. 10x Genomics' Chromium is the most widely used platform 
for single cell omics analysis, and scab's data IO functions are 
designed to work directly with ``CellRanger`` output files without 
needing any intermediate processing. BCR and TCR annotations 
conform to the standards of the Adaptive Immune Receptor Repertoire 
(AIRR) community. Output data files are in the widely used `.h5ad` 
format, ensuring interoperability with a wide range of existing 
and future software.


.. _scanpy: https://github.com/scverse/scanpy

