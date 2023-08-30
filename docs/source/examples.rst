.. _examples:

examples
============

We have designed scab to be used primarily in interactive notebook-like 
programming environments like Jupyter. Although it may have a steeper learning 
curve than a GUI-based tool, we believe that the gains in flexibility and 
customizability are more than worth the tradeoff.  

The scab API is quite similar to that of scanpy_ [Wolf18]_. This is by design, as 
we are big fans of the scanpy API and are also striving minimize the learning curve 
for users already familiar with scanpy. Additoinally, scab builds on the ``AnnData`` 
object at the core of scanpy to integrate BCR/TCR and antigen specificity data.  

Below are a few hypothetical use cases, with functional code examples. The 
`scab Github repository`_` includes interactive examples with sample datasets so that 
users can take scab for a more comprehensive test drive. 


example #1
------------
Our first example is relatively simple. We're starting with two single cell libraries 
(cell hashes and B cell VDJ) generated from a set of multiplexed samples of 
enriched B cells on a single 10x Genomics Chromium Controller reaction. We have two 
primary outputs from CellRanger: 1) a counts matrix, which includes only 
cell hashes; and 2) assembled BCR contigs with associated summary 
annotations. With this dataset, we'd like to do the following:  

  - read, annotate and integrate the input data (cell hashes and BCR sequences)  
  - demultiplex the samples using cell hashes and rename the samples  
  - filter out any cells without paired heavy/light chains  
  - assign BCR clonal lineages  
  - make a lineage donut plot for each sample, colored by VH gene use  



.. code-block:: python

    import scab

    # read, integrate and annotate the input data
    adata = scab.read_10x_mtx(
        mtx_path  = '/path/to/filtered_bc_matrix',
        bcr_file  = '/path/to/filtered_contigs.fasta',
        bcr_annot = '/path/to/filtered_summary.csv'
    )

    # demultiplex the samples using cell hashes and rename the samples
    sample_names = {
        'control1': 'CellHash1',
        'control2': 'CellHash2',
        'test1': 'CellHash3',
        'test2': 'CellHash4'
    }
    adata = scab.tl.demultiplex(adata, rename=sample_names)

    # filter out any cells that don't contain a single BCR pair
    adata = adata[adata.obs.bcr_pairing == "single pair"]

    # assign BCR clonal lineages
    adata = scab.vdj.clonifuy(adata)

    # make a lineage donut plot for each sample, colored by VH gene use
    for sample in adata.obs.sample.unique():
        a = adata[adata.obs.sample == sample]
        scab.pl.lineage_donut(a, hue='v_gene', chain='heavy')

|
|

example #2
------------
Next, we have a more complex set of libraries, generated from multiplexed 
peripheral blood mononuclear cell (PBMC) samples. The PBMCs were labeled with 
a panel of CITE-seq antibodies and we recovered BCR and TCR sequences, to produce 
the following CellRanger outputs: 1) a counts matrix, including GEX, cell hash and 
CITE-seq (feature barcode) UMI counts; 2) assembled BCR contigs with associated summary 
annotations; and 3) assembled TCR contigs with associated summary annotations. With 
this dataset, we'd like to:

  - read, annotate and integrate all of the input data 
  - demultiplex the samples using cell hashes, and rename the samples using a dictionary mapping 
    sample names to cell hash names
  - preprocess the GEX data, including leiden clustering and UMAP embedding 
  - for each CITE-seq antibody, make a pair of plots comparing transcription and cell surface abundance 
  - group TCR sequences into clonotypes 
  - select cells expressing a clonally expanded TCR 


.. code-block:: python

    import scab

    # read, integrate and annotate the input data
    adata = scab.read_10x_mtx(
        mtx_path  = '/path/to/filtered_bc_matrix',
        bcr_file  = '/path/to/BCR/filtered_contigs.fasta',
        bcr_annot = '/path/to/BCR/filtered_summary.csv',
        tcr_file  = '/path/to/TCR/filtered_contigs.fasta',
        tcr_annot = '/path/to/TCR/filtered_summary.csv'
    )

    # demultiplex the samples using cell hashes and rename the samples
    sample_names = {
        'donor123': 'CellHash1',
        'donor456': 'CellHash2',
        'donor789': 'CellHash3'
    }
    adata = adata.tl.demultiplex(adata, rename=sample_names)

    # preprocess the GEX data and compute the UMAP embedding
    adata = scab.pp.filter_and_normalize(adata)
    adata = scab.tl.umap(adata)

    # for each CITE-seq antibody, make a pair of plots comparing transcription and expression
    gene2citeseq = {
        'gene_name1': 'citeseq_name1',
        ...
        'gene_nameN': 'citeseq_nameN'
    }
    for gene, citeseq in gene2citeseq.items():
        scab.pl.umap(adata, colors=[gene, citeseq])

    # group TCR sequences into clonotypes 
    adata = scab.vdj.group_clonotypes(adata)

    # select cells expressing a clonally expanded TCR
    expanded = adata[adata.obs.clonotype_size > 1]



.. _scanpy: https://github.com/scverse/scanpy
.. _abutils: https://github.com/briney/abutils
.. _scab Github repository: htts://github.com/briney/scab

