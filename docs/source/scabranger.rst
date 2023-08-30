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
The only thing you need to run `scabranger` (aside from data, of course) is a
YAML-formatted configuration file and the name of a project directory into which 
the results will be deposited. The configuration file specifies the location of your
sequencing run data, the samples to process, the libraries that have been generated 
for each sample, and any other options you want to pass to CellRanger. Let's walk 
through each of these sections in a little more detail.

  
sequencing runs
~~~~~~~~~~~~~~~
This section is required, and you must provide information for at least one sequencing
run. Each run is identified by a unique name, and the following options are available:  

    - ``path``: The path to the sequencing run data. This should be a local path to a 
      directory containing the sequencing run data or to a compressed file containing
      the sequencing run data. This option is mutually exclusive with ``url``.
    - ``url``: A URL to a compressed file containing the sequencing run data. This option 
      is mutually exclusive with ``path``.  

.. note:: at least one of ``path`` or ``url`` must be provided.

    - ``simple_csv``: A simple CSV file containing the sample name and the index sequences
      for each sample. This option is mutually exclusive with ``samplesheet``.
    - ``samplesheet``: A CSV file containing the sample name and the index sequences for 
      each sample. This option is mutually exclusive with ``simple_csv``.

.. note:: at least one of ``simple_csv`` or ``samplesheet`` must be provided.

    - ``is_compressed``: A boolean indicating whether the sequencing run data is compressed.
    - ``copy_to_project``: A boolean indicating whether the sequencing run data should be 
      copied to the project directory. If ``True``, the data will be copied to the project 
      directory. Only really applicable when paired with ``path``, because if ``url`` is 
      provided, the linked data is downloaded into the project directory regardless of the 
      value of ``copy_to_project``. The default value is ``True``.




.. _CellRanger: https://support.10xgenomics.com/single-cell-vdj/software/pipelines/latest/what-is-cell-ranger