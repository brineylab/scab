Overview
========

As single cell omics tools become increasingly important for 
characterizing adaptive immunity, we noted the need for open, 
easy-to-use software designed specifically for analysis of 
adaptive immune cells.  

We build scab to fill this need. It was engineered to use an 
API that should be familiar to users of scanpy_, which is the 
most widely used Python package for general single cell omics 
analysis. Beyond API similarities, scab also builds directly 
on the models and functions introduced by scanpy to create 
specialized tools that address specific issues that arise during 
analysis of adaptive immune single cell omics data.  


Workflows
---------

scab provides a range of utilities for all stages of single cell 
adaptive immune analysis, including data IO, immune receptor 
annotation, sample demultiplexing, specificity classification, 
and clonal lineage assignment. scab also includes a variety of 
visualization tools designed for rapid exploratory analyses as 
well as generating publication-quality figures.  


File and data standards
------------------------

From the start, scab was designed to be compatible with file, data and 
schema standards of the immunology community. 10x Genomics' Chromium 
platform is the most widely used platform for single cell omics analysis, 
and scab's data ingestion functions are designed to work directly with 
``CellRanger`` output files, requiring no intermediate processing. BCR 
and TCR annotations conform to the standards of the Adaptive Immune 
Receptor Repertoire (AIRR) community. Output data files are in the 
widely used `.h5ad` format, ensuring interoperability with a wide 
range of existing tools.


