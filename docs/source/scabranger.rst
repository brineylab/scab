.. _scabranger:

scabranger
============

CellRanger_ is a great way to process 10x Genomics data, but running it 
can often be cumbersome. This is especially true if you are running 
multiple samples, or if your data is spread across multiple sequencing
runs. A typical workflow might look like this:

.. code-block:: console

    $ cellranger mkfastq --run path/to/myrun --csv samplesheet.csv --output-dir fastq
    $ cellranger multi --id sample1 --csv sample1_config.csv
    $ cellranger multi --id sample2 --csv sample2_config.csv
    ...
    $ cellranger multi --id sampleN --csv sampleN_config.csv

Each of these steps can take quite awhile to run, and if you have a lot of
samples, you need to keep coming back to the terminal to start the next
command. This is where ``scabranger`` comes in. It is a simple wrapper around 
CellRanger_ that allows you to automate the processing of multiple samples 
across one or more sequencing runs. Running ``scabranger`` is as simple as:

.. code-block:: console

    $ scabranger -p path/to/myproject -c config.yaml

A ``scabranger`` run is defined by a single YAML file that specifies the
location of your sequencing run data, the samples to process, and the
libraries to use for each sample. Here is an example:

.. code-block:: yaml

    # SEQUENCING RUNS
    sequencing_runs:
        MyRun:
            path: path/to/my_run
            is_compressed: False
            simple_csv: my_run.csv
            copy_to_project: False
        MyOtherRun:
            url: https://s3.amazonaws.com/brineylab/runs/my_other_run.tar.gz
            samplesheet: my_other_run.csv

    # SAMPLES
    samples:
        sample1:
            Gene Expression: sample1_gex
            Antibody Capture: sample1_features
            VDJ-B: sample1_ig
        sample2:
            Gene Expression: sample2_gex
            VDJ-T: sample1_tcr

    # REFERENCES
    gex_reference:
        default: /path/to/default_gex_reference
    vdj_reference:
        default: /path/to/default_vdj_reference
        sample2: /path/to/alternate_vdj_reference
    feature_reference:
        default: /path/to/default_feature_reference

    # CELLRANGER OPTIONS
    cli_options:
        mkfastq:
            default: ""
        multi:
            default: ""
            sample2: "--expect-cells=5000 --no-bam=true"

    # MISCELLENEOUS OPTIONS
    uiport: 40909 # port for the cellranger UI
    cellranger: cellranger # cellranger invocation



scabranger configuration
------------------------
The only thing you need to run ``scabranger`` (aside from data, of course) is a
YAML-formatted configuration file and the name of a project directory into which 
the results will be deposited. The configuration file specifies the location of your
sequencing run data, the samples to process, the libraries that have been generated 
for each sample, and any other options you want to pass to CellRanger. Let's walk 
through each of these sections in a little more detail.

  
sequencing runs
~~~~~~~~~~~~~~~
This section is required, and you must provide information for at least one sequencing
run. Each run is identified by a unique name -- you can pick whatever name you want, but 
names with spaces and/or special characters may cause unexpected problems. The following 
options are available:  

    - **``path``**: The path to the sequencing run data. This should be a local path to a 
      directory containing the sequencing run data or to a compressed file containing
      the sequencing run data. This option is mutually exclusive with ``url``.
    - **``url``**: A URL to a compressed file containing the sequencing run data. This option 
      is mutually exclusive with ``path``.  

    .. note:: 
        at least one of ``path`` or ``url`` must be provided.
  

    - ``simple_csv``: A simple CSV file containing the sample name and the index sequences
      for each sample. This option is mutually exclusive with ``samplesheet``.
    - ``samplesheet``: A CSV file containing the sample name and the index sequences for 
      each sample. This option is mutually exclusive with ``simple_csv``.

    .. note:: 
        at least one of ``simple_csv`` or ``samplesheet`` must be provided.
  

    - ``is_compressed``: A boolean indicating whether the sequencing run data is compressed.
    - ``copy_to_project``: A boolean indicating whether the sequencing run data should be 
      copied to the project directory. If ``True``, the data will be copied to the project 
      directory. Only really applicable when paired with ``path``, because if ``url`` is 
      provided, the linked data is downloaded into the project directory regardless of the 
      value of ``copy_to_project``. The default value is ``True``.

The library names in the ``simple_csv`` or ``samplesheet`` files must match the library 
names in the ``samples`` configuration block. 

.. tip:: 
    If libraries are present in more than one sequencing run (for example, the libraries 
    were re-sequenced to increase the total amount of data generated), the matched libraries 
    should be given identical names in the ``samplesheet`` or ``simple_csv`` files for each 
    run. If named in this way, ``scabranger`` can automatically combine the data from all
    applicable runs when running CellRanger.


samples
~~~~~~~
This section is required, and you must provide information for at least one sample. Each 
sample is identified by a unique name -- you can pick whatever name you want, but names 
with spaces and/or special characters may cause unexpected problems. For each sample, you
You must specify the libraries that have been generated using a key/value pair in which the 
key is the name of the library type and the value is the name of the library. 

.. warning:: 
    While `samples` can be given arbitrary names, library names must match the name of a 
    library present in the ``samplesheet`` or ``simple_csv`` files provided in the 
    `sequencing runs` configuration block.

The following library types are available: 

    - ``Gene Expression``: The name of the library containing the gene expression data for 
      this sample.
    - ``VDJ-B``: The name of the library containing the B-cell VDJ data for this sample. 
    - ``VDJ-T``: The name of the library containing the T-cell VDJ data for this sample.
    - ``VDJ-T-GD``: The name of the library containing the T-cell VDJ data (gamma-delta chains) 
      for this sample.
    - ``Antibody Capture``: The name of the library containing the antibody capture data for 
      this sample.
    - ``Antigen Capture``: The name of the library containing Barcode Enabled Antigen Mapping
      (BEAM) data for this sample.
    - ``CRISPR Guide Capture``: The name of the library containing the CRISPR guide capture 
      data for this sample.
    - ``Custom``: The name of the library containing custom feature barcode data for this sample.

At least one library must be provided for each sample. If you do not have data for a particular 
library type, you can omit it from the sample definition. For example, if you only have gene
expression data for a sample, you can define the sample like this:

.. code-block:: yaml

    samples:
        sample1:
            Gene Expression: sample1_gex


references
~~~~~~~~~~
This section is required, and you must provide at least one reference for each library type you 
are using. Each reference type (GEX, VDJ, and Feature) has a default reference that will be used 
for all samples unless a sample-specific reference is provided. The default references are
specified using the ``default`` key. Sample-specific references are specified using the sample
name as the key. For example, if you have a sample named ``sample2`` that uses a different VDJ
reference than the default, you would specify it like this:

.. code-block:: yaml

    vdj_reference:
        default: /path/to/default_vdj_reference
        sample2: /path/to/alternate_vdj_reference


cli options
~~~~~~~~~~~
This section is optional, and you can provide options for any or all of the CellRanger commands 
you want to run. Each command has a ``default`` option that will be used for all samples unless
a sample-specific option is provided. Sample-specific options are specified using the sample name
as the key. For example, if you have a sample named ``sample2`` that uses a different number of
expected cells than the default and for which you would prefer that BAMs not be generated, you 
would specify the additional options (which will be passed diretly to ``cellranger multi``) 
like this:

.. code-block:: yaml

    cli_options:
        multi:
            default: ""
            sample2: "--expect-cells=5000 --no-bam=true"


miscellaneous options
~~~~~~~~~~~~~~~~~~~~~



.. _CellRanger: https://support.10xgenomics.com/single-cell-vdj/software/pipelines/latest/what-is-cell-ranger