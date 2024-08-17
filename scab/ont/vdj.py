# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import logging
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Optional, Union

import abutils
import polars as pl
from natsort import natsorted
from tqdm.auto import tqdm

from .barcodes import get_barcode_definition, parse_barcodes
from .consensus import cluster_and_consensus


def ont_vdj(
    fastq_files: Union[str, Iterable[str]],
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
    chunksize: int = 1000,
    verbose: bool = True,
    debug: bool = False,
):
    """ """
    # setup project directory
    project_path = os.path.abspath(project_path)
    abutils.io.make_dir(project_path)
    log_directory = os.path.join(project_path, "logs")
    temp_directory = os.path.join(project_path, "temp")
    abutils.io.make_dir(log_directory)
    abutils.io.make_dir(temp_directory)

    # setup logging
    abutils.log.setup_logging(
        logfile=os.path.join(log_directory, "scab_ont-vdj.log"),
        add_stream_handler=verbose,
        print_log_location=False,
    )
    logger = abutils.log.get_logger(log_directory)
    _log_run_parameters(
        logger,
        project_path,
        barcode_description,
        n_processes,
        chunksize,
        copy_inputs_to_project,
        clustering_downsample,
        consensus_downsample,
        min_cluster_size,
        clustering_threshold,
        clustering_algo,
        alignment_algo,
        alignment_kwargs,
        debug,
    )

    # processes
    if n_processes is None:
        n_processes = mp.cpu_count()

    # process inputs:
    input_files = []
    if isinstance(fastq_files, str):
        fastq_files = [fastq_files]
    for fastq_file in fastq_files:
        if os.path.isfile(fastq_file):
            input_files.append(fastq_file)
        elif os.path.isdir(fastq_file):
            input_files.extend(
                abutils.io.list_files(
                    fastq_file,
                    extension=["fastq.gz", "fastq", "fq.gz", "fq"],
                    recursive=True,
                )
            )
        else:
            raise FileNotFoundError(f"Input path {fastq_file} does not exist")
    input_files = natsorted(input_files)
    _log_inputfile_info(logger, input_files)

    # copy inputs to project directory
    if copy_inputs_to_project:
        logger.info("copying input files to project directory")
        copied_inputs = []
        inputs_directory = os.path.join(project_path, "inputs")
        abutils.io.make_dir(inputs_directory)
        if verbose:
            input_files = tqdm(
                input_files, bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:25}{r_bar}"
            )
        for input_file in input_files:
            shutil.copy(input_file, inputs_directory)
            copied_inputs.append(
                os.path.join(inputs_directory, os.path.basename(input_file))
            )
        input_files = copied_inputs
    logger.info("")
    logger.info("")

    # set up temp directory for chunked inputs
    chunked_inputs_directory = os.path.join(temp_directory, "chunked_inputs")
    abutils.io.make_dir(chunked_inputs_directory)

    # TODO: chunk the input files
    logger.info("splitting input files into job-sized chunks")
    chunked_inputs = []
    if verbose:
        progress_bar = tqdm(
            total=len(input_files),
            bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:25}{r_bar}",
        )
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        chunking_kwargs = {
            "output_directory": chunked_inputs_directory,
            "chunksize": chunksize,
        }
        futures = [
            executor.submit(abutils.io.split_fastx, input_file, **chunking_kwargs)
            for input_file in input_files
        ]
        for future in as_completed(futures):
            chunked_inputs.extend(future.result())
            if verbose:
                progress_bar.update(1)
    if verbose:
        progress_bar.close()

    # # single threaded version of the above chunking operation
    # if verbose:
    #     input_files = tqdm(
    #         input_files, bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:25}{r_bar}"
    #     )
    # for input_file in input_files:
    #     chunked_inputs.extend(
    #         abutils.io.split_fastx(
    #             fastx_file=input_file,
    #             output_directory=chunked_inputs_directory,
    #             chunksize=chunksize,
    #         )
    #     )

    input_files = chunked_inputs
    logger.info("")
    logger.info("")

    # set up temp and log directories for barcode parsing
    barcode_temp_directory = os.path.join(temp_directory, "barcodes")
    barcode_log_directory = os.path.join(log_directory, "barcodes")
    abutils.io.make_dir(barcode_temp_directory)
    abutils.io.make_dir(barcode_log_directory)

    # get barcode segment descriptions
    barcode_segments = get_barcode_definition(barcode_description)

    # parse and correct barcodes/UMIs
    parquet_files = []
    logger.info("parsing/correcting barcodes")
    if verbose:
        progress_bar = tqdm(
            total=len(input_files),
            bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:25}{r_bar}",
        )
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        parsing_kwargs = {
            "barcode_segments": barcode_segments,
            "output_directory": barcode_temp_directory,
            "log_directory": barcode_log_directory,
            "show_progress": False,
        }
        futures = [
            executor.submit(parse_barcodes, input_file, **parsing_kwargs)
            for input_file in input_files
        ]
        for future in as_completed(futures):
            parquet_file = future.result()
            if parquet_file is not None:
                parquet_files.append(parquet_file)
            if verbose:
                progress_bar.update(1)
    if verbose:
        progress_bar.close()
    logger.info("")

    # concat the output Parquet files
    concat_parquet = os.path.join(project_path, "parsed_barcodes.parquet")
    dfs = [pl.scan_parquet(pq_file) for pq_file in parquet_files]
    concat_df = pl.concat(dfs, rechunk=False)
    concat_df.sink_parquet(concat_parquet)
    # remove temp parquet files only if not in debug mode
    if not debug:
        for pf in parquet_files:
            os.remove(pf)

    # setup temp and log directories for clustering/consensus
    consensus_temp_directory = os.path.join(temp_directory, "consensus")
    consensus_log_directory = os.path.join(log_directory, "consensus")
    abutils.io.make_dir(consensus_temp_directory)
    abutils.io.make_dir(consensus_log_directory)

    if alignment_kwargs is None:
        alignment_kwargs = {}
        if alignment_algo == "famsa":
            alignment_kwargs["guide_tree"] = "upgma"

    # filter barcodes and make a separate parquet file for each
    barcode_parquet_files = []
    logger.info("splitting into barcode-specific Parquet files")
    df = pl.read_parquet(concat_parquet)
    passed_barcodes = list(
        df["barcode"]
        .value_counts()
        .filter(pl.col("count") >= min_cluster_size)["barcode"]
    )
    if verbose:
        passed_barcodes = tqdm(
            passed_barcodes, bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:25}{r_bar}"
        )
    for barcode in passed_barcodes:
        barcode_parquet_file = os.path.join(consensus_temp_directory, f"{barcode}")
        _df = df.filter(pl.col("barcode") == barcode)
        _df.write_parquet(barcode_parquet_file)
        barcode_parquet_files.append(barcode_parquet_file)

    # cluster and build consensus sequences
    logger.info("clustering and consensus building")
    all_consensus = []
    if verbose:
        progress_bar = tqdm(
            total=len(barcode_parquet_files),
            bar_format="{desc:<2.5}{percentage:3.0f}%|{bar:25}{r_bar}",
        )
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        consensus_kwargs = {
            "log_directory": consensus_log_directory,
            "clustering_algo": clustering_algo,
            "clustering_threshold": clustering_threshold,
            "clustering_downsample": clustering_downsample,
            "consensus_downsample": consensus_downsample,
            "min_cluster_size": min_cluster_size,
            "alignment_algo": alignment_algo,
            "alignment_kwargs": alignment_kwargs,
        }
        # submit
        logger.info("submitting jobs to executor")
        futures = []
        for i, f in enumerate(barcode_parquet_files):
            futures.append(
                executor.submit(cluster_and_consensus, f, **consensus_kwargs)
            )
            if i == 0:
                logger.info("first job submitted")
            if i == 1:
                logger.info("second job submitted")

        # futures = [
        #     executor.submit(cluster_and_consensus, f, **consensus_kwargs)
        #     for f in barcode_parquet_files
        # ]
        # wait
        for future in as_completed(futures):
            consensus = future.result()
            all_consensus.append(consensus)
            if verbose:
                progress_bar.update(1)
    if verbose:
        progress_bar.close()
    logger.info("")

    # remove temp barcode parquet files only if not in debug mode
    if not debug:
        for pf in barcode_parquet_files:
            os.remove(pf)

    # write the consensus sequences to a file
    logger.info("writing consensus sequences to file")
    consensus_file = os.path.join(project_path, "consensus.fasta")
    with open(consensus_file, "w") as f:
        f.write("\n".join([c.fasta for c in all_consensus]))


# ===============================
#
#       PRINTING/LOGGING
#
# ===============================


def _log_run_parameters(
    logger: logging.Logger,
    project_path: str,
    barcode_description: str,
    n_processes: Optional[int],
    chunksize: int,
    copy_inputs_to_project: bool,
    clustering_downsample: int,
    consensus_downsample: int,
    min_cluster_size: int,
    clustering_threshold: float,
    clustering_algo: str,
    alignment_algo: str,
    alignment_kwargs: Optional[dict],
    debug: bool,
) -> None:
    # printing the splash line-by-line makes the log files look nicer
    for line in ONT_VDJ_SPLASH.split("\n"):
        logger.info(line)
    logger.info("")
    logger.info("RUN PARAMETERS")
    logger.info("==============")
    logger.info(f"PROJECT PATH: {project_path}")
    logger.info(f"BARCODE DESCRIPTION: {barcode_description}")
    logger.info(f"MIN CLUSTER SIZE: {min_cluster_size}")
    logger.info(f"CLUSTERING ALGO: {clustering_algo}")
    logger.info(f"CLUSTERING THRESHOLD: {clustering_threshold}")
    logger.info(f"CLUSTERING DOWNSAMPLE: {clustering_downsample}")
    logger.info(f"CONSENSUS DOWNSAMPLE: {consensus_downsample}")
    logger.info(f"ALIGNMENT ALGO: {alignment_algo}")
    logger.info(f"ALIGNMENT KWARGS: {alignment_kwargs}")
    logger.info(f"NUM PROCESSES: {n_processes if n_processes is not None else 'auto'}")
    logger.info(f"CHUNKSIZE: {chunksize}")
    logger.info(f"COPY INPUTS TO PROJECT: {copy_inputs_to_project}")
    logger.info(f"DEBUG: {debug}")
    logger.info("")


def _log_inputfile_info(logger: logging.Logger, input_files: Iterable[str]) -> None:
    num_files = len(input_files)
    plural = "files" if num_files > 1 else "file"
    logger.info("")
    logger.info("INPUT FILES")
    logger.info("===========")
    logger.info(f"found {num_files} input {plural}:")
    if num_files < 6:
        for f in input_files:
            logger.info(f"  {os.path.basename(f)}")
    else:
        for f in input_files[:5]:
            logger.info(f"  {os.path.basename(f)}")
        logger.info(f"  ... and {num_files - 5} more")
    logger.info("")


ONT_VDJ_SPLASH = """
                    __                   __                 __  _ 
   ______________ _/ /_     ____  ____  / /_     _   ______/ / (_)
  / ___/ ___/ __ `/ __ \   / __ \/ __ \/ __/____| | / / __  / / / 
 (__  ) /__/ /_/ / /_/ /  / /_/ / / / / /_/_____/ |/ / /_/ / / /  
/____/\___/\__,_/_.___/   \____/_/ /_/\__/      |___/\__,_/_/ /   
                                                         /___/   
"""
