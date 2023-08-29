# creating `scabranger` config files

`scabranger` is a wrapper around 10x Genomics' [CellRanger](https://support.10xgenomics.com/single-cell-vdj/software/overview/welcome) software. CellRanger is great, but fully processing raw sequencing data requires several steps, and the analysis for each sample needs to be run separately. That means manually running CellRanger commands every hour or two until all of your samples are processed. That's annoying. 

`scabranger` is designed to fix that problem by iteratively running the entire CellRanger pipeline on each of your samples. This way, you just run `scabranger` once for your entire experiment. We also try to make it easy to process experiments that include data from multiple sequencing runs -- regardless of whether your samples were split across runs, or you performed an additional sequencing run on the same set of samples to get more data.

To do all this, `scabranger` uses a YAML-formatted configuration file, which allows you to tell `scabranger` about your sequencing runs, samples, and libraries. There are several main *configuration blocks* in our YAML file:  
  - **sequencing runs**: contains information about each of the sequencing runs used in the experiment  
  - **samples**: contains information about the samples (for example, `donor_1`) and libraries (for example, GEX or VDJ) used in the experiment  
  - **references**: contains information about the references to be used, which are typically species-specific  
  - **CLI options**: provides a way to use sample-specific options to any of the major CellRanger tools  
  
<br>

## sequencing runs
Here's an example of a *sequencing runs* configuration block:
```yaml
sequencing_runs:
  NS001:
    path: ./ns001_run-data/
    is_compressed: False
    simple_csv: ./ns001.csv
    copy_to_project: False

  NS002:
    url: s3://brineylab/run_data/ns002.tar.gz
    samplesheet: /data/samplesheets/ns002.csv
```

Each configuration block ***must*** start with the name of the block (in this case, `sequencing_runs:` -- make sure the colon is there too!) and the text must be fully left justified (all the way to the left, no leading spaces or tabs). This is part of the YAML specificiation, and if it's not properly formatted, the Python YAML parser won't be able to read your config file.

Following the block title, you need to include at least one sequencing run. Each sequencing run makes up its own sub-block, which starts with the name of the sequencing run, like so:  
```yaml
  NS001:
    path: ./ns001_run-data/
    is_compressed: False
    simple_csv: ./ns001.csv
    copy_to_project: False
```
 The run name can be whatever you want, but spaces and special characters may cause issues so don't go too crazy. Each run must have a unique name. The run name must be indented two spaces, and should include a colon just like the block title. Below the run name (and indented an additional two spaces), you can provide additional information about the run. The following options are accepted:

   * ***path***: path to the run folder, if the run is either present on your local server or on an attached volume
   * ***url***: download link for the sequencing run, typically on S3 or similar
   * ***simple_csv***: path to a [CellRanger-formatted CSV file](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/using/mkfastq) for use with ``cellranger mkfastq``
   * ***samplesheet***: path to an [Illumina-formatted samplesheet](https://support-docs.illumina.com/SHARE/SampleSheetv2/Content/SHARE/FrontPages/SampleSheetv2.htm)
   * ***is_compressed***: set to `True` if *path* or *url* points to a single compressed file or `False` if *path* is a run data folder. Automatically set to `True` if *url* is provided.
   * ***copy_to_project***: if `True`, the run data folder will be copied into the project directory. The default is `True`.

You must provide either ***path*** or ***url***. Also, you must provide either ***simple_csv*** or ***samplesheet***. All other parameters are optional.
  
<br>  

## samples
Here's an example of a *samples* configuration block:
```yaml
samples:
  sample1:
    Gene Expression: sample1_gex
    Antibody Capture: sample1_features
    VDJ-B: sample1_ig
    VDJ-T: sample1_tcr
    VDJ-T-GD: sample1_tcr-gd

  sample2:
    Gene Expression: sample2_gex
    CRISPR Guide Capture: sample2_crispr
```

Like the *sequencing runs* configuration block, this block must start with the title: `samples:`. Again, the title must include the colon and should be fully left justified.  

After the title, you must include at least one sample (although you can provide as many as you want). Each sample makes up its own sub-block, which contains all of the libraries that should be processed for each sample, like so:
```yaml
  sample1:
    Gene Expression: sample1_gex
    Antibody Capture: sample1_features
    VDJ-B: sample1_ig
    VDJ-T: sample1_tcr
    VDJ-T-GD: sample1_tcr-gd
```

Like the *sequencing runs* block, you can name the samples essentially whatever you want but each sample name must be unique. Sample names must also be indented two spaces and include a colon at the end. Within each sample sub-block, you can provide key-value pairs, with the keys corresponding to a [10x Genomics library type](https://support.10xgenomics.com/single-cell-vdj/software/pipelines/latest/using/multi) and values being the name of a corresponding library found in either the `simple_csv` or `samplesheet` of at least one of the runs listed in the *sequencing runs* block. Currently acceptable library types are:
  - `Gene Expression`: single cell RNAseq libraries
  - `VDJ-B`: B cell VDJ libraries
  - `VDJ-T`" T cell VDJ libraries (alpha/beta chains only)
  - `VDJ-T-GD`: T cell VDJ libraries (gamma/delta chains only)
  - `Antibody Capture`: feature barcode libraries that involve labeling cell-surface proteins with barcoded antibodies (CITE-seq, cellhashes, etc)
  - `Antigen Capture`: Barcode Enabled Antigen Mapping (BEAM) libraries
  - `CRISPR Guide Capture`: libraries used in CRISPR purturbation ([Perturb-Seq](https://www.sciencedirect.com/science/article/pii/S0092867416316105?via%3Dihub)) assays
  - `Custom`: any other feature barcode-style library

<br>  
  
## references
Here's an example of a *references* configuration block:
```yaml
gex_reference:
  default: /path/to/default_transcriptome
  sample1: /path/to/alternate_transcriptome

vdj_reference:
  default: /path/to/default_vdj_reference
  sample1: /path/to/alternate_vdj_reference

feature_reference:
  default: /path/to/default_feature_reference
```
This block is a bit different than the first two we've looked at. Instead of having a single title for the whole block, we have a separate, fully left justified title for each reference type: `gex_reference`, `vdj_reference`, and `feature_reference`. Within each reference sub-block, you can supply paths to the appropriate reference. The `default` reference will be used on all samples unless otherwise requested. Alternate references can be supplied in a sample-specific manner. For the `gex_reference` example above, `/path/to/default_transcriptome` would be used for all samples ***except*** for `sample1`, which would use `/path/to/alternate_transcriptome`. 

***NOTE:** sample names supplied to any of the reference sub-blocks must match the sample name given in the **samples** sub-block*

Supplying multiple references is uncommon, and is typically only done when samples in a single collection of 10x Genomics reactions are derived from different species.

<br>  
  
## CLI options
Here's an example of a *CLI options* configuration block:
```yaml
cli_options:
  mkfastq:
    default: ""
    NS002: "--rc-i2-override=true"

  multi:
    default: "--localcores=32 --localmem=256"
    sample1: "--expect-cells=5000 --no-bam=true"
```

In this block, we're back to our usual layout with a single, left-justified block title (`cli_options:`). The title of each sub-block should be the name of a CellRanger command, of which there are currently only two choices: `mkfastq` or `multi`. Like the *references* block, we can supply a `default` set of CLI options as well as run- or sample-specific CLI options, with the `default` options used for all runs/samples for which specific CLI options are not provided. The CLI options should be provided as a single string, and will be appended to the command at runtime. Non-default options for `mkfastq` should match sequencing run names (as supplied in the *sequencing runs* configuration block), and non-default options for `multi` should match sample names (as supplied in the *samples* configuration block).


