# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import os
import random
import traceback
from typing import Iterable, Optional

import abutils
import pandas as pd

# import polars as pl
from abutils import Sequence
from abutils.tools.log import SimpleLogger


def cluster_and_consensus(
    parquet_file: str,
    log_directory: Optional[str] = None,
    temp_directory: Optional[str] = None,
    clustering_threshold: float = 0.6,
    min_cluster_size: int = 100,
    clustering_downsample: int = 5000,
    clustering_algo: str = "vsearch",
    consensus_downsample: int = 200,
    alignment_algo: str = "famsa",
    alignment_kwargs: dict = {"guide_tree": "upgma"},
) -> Iterable[Sequence]:
    """
    Cluster and make consensus sequences starting from a Parquet file of barcode/umi-extracted sequence data.

    Parameters
    ----------
    parquet_file : str
        Path to a Parquet file containing barcode/umi-extracted sequence data.

    log_directory : Optional[str], optional
        Path to a directory to log the clustering and consensus sequence generation process, by default None.

    clustering_threshold : float, optional
        The clustering threshold for the clustering algorithm, by default 0.6.

    min_cluster_size : int, optional
        The minimum size of a cluster to be considered valid, by default 100.

    clustering_downsample : int, optional
        The number of sequences to downsample to for clustering, by default 5000.

    clustering_algo : str, optional
        The clustering algorithm to use, by default "vsearch".

    consensus_downsample : int, optional
        The number of sequences to downsample to for consensus sequence generation, by default 200.

    alignment_algo : str, optional
        The alignment algorithm to use, by default "famsa".

    alignment_kwargs : dict, optional
        Keyword arguments to pass to the alignment algorithm, by default {"guide_tree": "upgma"}.

    Returns
    -------
    consensuses : List[Sequence]
        A list of consensus sequences.

    """
    barcode = os.path.basename(parquet_file)

    # setup logging
    logger = SimpleLogger(os.path.join(log_directory, barcode))
    bc_string = f"  BARCDODE: {barcode}  "
    logger.log("=" * len(bc_string))
    logger.log(bc_string)
    logger.log("=" * len(bc_string))
    logger.log("")

    # get sequences
    # we originally used polars, but it's incompatible with multiprocessing
    # using fork on Unix, so we need to use pandas instead
    # see: https://docs.pola.rs/user-guide/misc/multiprocessing/
    # it's possible that we can use polars if we use "spawn" instead of "fork",
    # so we can look into that in the future.
    df = pd.read_parquet(parquet_file)
    # df = pl.read_parquet(parquet_file)

    seqs = [
        Sequence(row["sequence"], id=row["sequence_id"]) for _, row in df.iterrows()
    ]
    # seqs = abutils.io.from_polars(df)
    logger.log("TOTAL SEQUENCES:", len(seqs))
    if len(seqs) > clustering_downsample:
        logger.log(
            f"CLUSTERING DOWNSAMPLING: from {len(seqs)} to {clustering_downsample} sequences"
        )
        seqs = random.sample(seqs, clustering_downsample)
    logger.log("")

    try:
        # clustering
        clusters = abutils.tl.cluster(
            seqs,
            threshold=clustering_threshold,
            algo=clustering_algo,
            threads=1,
            temp_dir=temp_directory,
        )
        logger.log("NUM CLUSTERS:", len(clusters))
        logger.log("CLUSTER_SIZES:", ", ".join([str(c.size) for c in clusters]))
        # filter clusters by size
        clusters = [c for c in clusters if c.size >= min_cluster_size]
        logger.log(
            "CLUSTER SIZE FILTERING:",
            f"{len(clusters)} clusters met the minimum size threshold ({min_cluster_size})",
        )

        # make consensus sequences
        consensuses = []
        # alignment for consensus generation should only use 1 thread
        alignment_kwargs["threads"] = 1
        for i, cluster in enumerate(clusters, 1):
            try:
                cluster_name = f"{barcode}_{i}"
                logger.log("")
                logger.log(cluster_name)
                logger.log("-" * len(cluster_name))

                n_umis = df[df["sequence_id"].isin(cluster.seq_ids)]["umi"].nunique()
                # n_umis = df.filter(
                #     pl.col("sequence_id").is_in(cluster.seq_ids)
                # ).n_unique("umi")
                n_reads = len(cluster.sequences)
                consensus = abutils.tl.make_consensus(
                    cluster.sequences,
                    name=f"{barcode}_{i}",
                    downsample_to=consensus_downsample,
                    algo=alignment_algo,
                    alignment_kwargs=alignment_kwargs,
                )
                consensus["n_umis"] = n_umis
                consensus["n_reads"] = n_reads
                consensuses.append(consensus)
                logger.log("NUM UMIS:", n_umis)
                logger.log("NUM READS:", n_reads)
                logger.log("CONSENSUS SEQUENCE:")
                logger.log("")
            except Exception as e:
                logger.exception("CONSENSUS EXCEPTION:", traceback.format_exc())
            finally:
                logger.checkpoint()

        # log consensus sequences
        logger.log("")
        for cons in consensuses:
            logger.log(cons.fasta)

    # handle and log exceptions
    except Exception as e:
        logger.log("CLUSTER AND CONSENSUS EXCEPTION:", traceback.format_exc())

    # write log file
    finally:
        logger.log("\n\n")
        if logger.log_file is not None:
            logger.write()

    return consensuses
