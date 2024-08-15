# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from typing import Iterable, Optional, Union

import click
from abutils.utils.click import HiddenClickOption, parse_dict_from_string

from ..cellranger.scabranger import cellranger_pipeline as run_cellranger_pipeline
from ..ont.vdj import ont_vdj as run_ont_vdj
from ..version import __version__


@click.group()
def cli():
    pass


# --------------------------------
#           VERSION
# --------------------------------
@cli.command("version")
def version():
    """
    Print the version of scab.
    """
    print(f"scab v{__version__}")


# --------------------------------
#          CELLRANGER
# --------------------------------


@cli.command("cellranger")
@click.option(
    "-p",
    "--project-path",
    type=str,
    required=True,
    help="Path to a directory in which tmp, log and output files will be deposited.",
)
@click.option(
    "-c",
    "--config-file",
    type=str,
    required=True,
    help="Path to a configuration file. in YAML format.",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="If set, run in debug mode which results in temporary files being retained and additional logging.",
)
def cellranger(project_path: str, config_file: str, debug: bool = False):
    """
    Run the cellranger pipeline.
    """
    run_cellranger_pipeline(
        project_path=project_path, config_file=config_file, debug=debug
    )


# --------------------------------
#           ONT-VDJ
# --------------------------------


@cli.command("ont-vdj")
@click.argument(
    "input_path",
    type=str,
)
@click.argument(
    "project_path",
    type=str,
)
@click.option(
    "--barcode-description",
    type=click.Choice(["TXG_v2"], case_sensitive=False),
    show_default=True,
    default="TXG_v2",
    help="Name of the built-in barcode description, which contains information about flanking adapters, whitelisted barcodes, and presence of a UMI.",
)
@click.option(
    "--clustering-downsample",
    type=int,
    show_default=True,
    default=5000,
    help="Number of sequences to downsample to prior to clustering",
)
@click.option(
    "--clustering-algo",
    type=click.Choice(["vsearch", "mmseqs"], case_sensitive=False),
    show_default=True,
    default="vsearch",
    help="Algorithm to use for clustering",
)
@click.option(
    "--clustering-threshold",
    type=float,
    show_default=True,
    default=0.6,
    help="Identity threshold for clustering prior to consensus sequence generation",
)
@click.option(
    "--min-cluster-size",
    type=int,
    show_default=True,
    default=100,
    help="Minimum number of sequences in a cluster to be considered for consensus sequence generation",
)
@click.option(
    "--consensus-downsample",
    type=int,
    show_default=True,
    default=200,
    help="Number of sequences to downsample to prior to building consensus sequences",
)
@click.option(
    "--alignment-algo",
    type=click.Choice(["famsa", "muscle", "mafft"], case_sensitive=False),
    show_default=True,
    default="famsa",
    help="Algorithm to use for multiple sequence alignment",
)
@click.option(
    "--alignment-kwargs",
    type=str,
    callback=parse_dict_from_string,
    default=None,
    help="Keyword arguments to pass to the alignment function. Format must be 'key1=val1,key2=val2'",
)
@click.option(
    "--n-processes",
    type=int,
    default=None,
    help="Number of processes to use for clustering, alignment, and consensus sequence generation. By default, the number of processes is set to the number of available CPU cores.",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    help="Whether to print verbose output",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Whether to run in debug mode, which results in temporary files being retained and additional logging.",
)
@click.option(
    "--copy-inputs-to-project",
    cls=HiddenClickOption,
    is_flag=True,
    default=True,
)
def ont_vdj(
    input_path: Union[str, Iterable[str]],
    project_path: str,
    barcode_description: str = "txg_v2",
    copy_inputs_to_project: bool = False,
    clustering_downsample: int = 5000,
    consensus_downsample: int = 200,
    min_cluster_size: int = 100,
    clustering_threshold: float = 0.6,
    clustering_algo: str = "vsearch",
    alignment_algo: str = "famsa",
    alignment_kwargs: Optional[dict] = None,
    n_processes: Optional[int] = None,
    verbose: bool = True,
    debug: bool = False,
):
    """
    Run the ONT-VDJ pipeline, which builds consensus VDJ sequences from Oxford Nanopore data.

    \b
    INPUT_PATH can be a FASTQ file or a directory containing FASTQ files. Gzipped files are supported.
    PROJECT_PATH is the path to a directory in which tmp, log and output files will be deposited.
    """
    run_ont_vdj(
        fastq_files=input_path,
        project_path=project_path,
        barcode_description=barcode_description,
        copy_inputs_to_project=copy_inputs_to_project,
        clustering_downsample=clustering_downsample,
        consensus_downsample=consensus_downsample,
        min_cluster_size=min_cluster_size,
        clustering_threshold=clustering_threshold,
        clustering_algo=clustering_algo,
        alignment_algo=alignment_algo,
        alignment_kwargs=alignment_kwargs,
        n_processes=n_processes,
        verbose=verbose,
        debug=debug,
    )
