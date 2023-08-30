.. _scabranger:

scabranger
============

CellRanger_ is a great way to process 10x Genomics data, but running it 
can often be cumbersome. This is especially true if you are running 
multiple samples, or if your data is spread across multiple sequencing
runs. A typical workflow might look like this:

.. code-block:: console

    $ cellranger mkfastq --run=path/to/myrun --csv=samplesheet.csv --output-dir=fastq
    $ cellranger multi --id=sample1 --csv=sample1_config.csv
    $ cellranger multi --id=sample2 --csv=sample2_config.csv
    ...
    $ cellranger multi --id=sampleN --csv=sampleN_config.csv

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






.. _CellRanger: https://support.10xgenomics.com/single-cell-vdj/software/pipelines/latest/what-is-cell-ranger