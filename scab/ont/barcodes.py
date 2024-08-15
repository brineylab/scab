# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import gzip
import os
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import abutils
import polars as pl
import rapidfuzz
from abutils import Sequence
from rapidfuzz.process import extract
from tqdm.auto import tqdm

WHITELIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whitelists")


# ----------------------------------
#         BARCODE PARSING
# ----------------------------------


def parse_barcodes(
    fastq_file: str,
    barcode_segments: Union["BarcodeSegment", Iterable["BarcodeSegment"]],
    output_directory: str,
    log_directory: str,
    show_progress: bool = False,
) -> str:
    """
    Parses barcode segments from a FASTQ file.

    Parameters
    ----------
    fastq_file : str
        The path to the FASTQ file to parse. Gzip-compressed files are supported.

    barcode_segments : Union[BarcodeSegment, Iterable[BarcodeSegment]]
        The barcode segments to parse.

    output_directory : str
        The directory to write the output to.

    log_directory : str
        The directory to write the log to.

    show_progress : bool, optional
        Whether to show a progress bar, by default False

    Returns
    -------
    str
        The path to the output Parquet file.

    """
    sample_name = os.path.basename(fastq_file).split(".fastq")[0]
    # setup logging
    logger = abutils.log.SimpleLogger(log_file=os.path.join(log_directory, sample_name))
    logger.log("SAMPLE:", sample_name)

    # read input FASTQ file
    sequences = abutils.io.read_fastx(fastq_file)
    if not sequences:
        logger.log(f"INPUT ERROR: input file {fastq_file} has no sequences")
        return 0

    # parse barcode segments
    processed = []
    failed = []
    if isinstance(barcode_segments, BarcodeSegment):
        barcode_segments = [barcode_segments]
    if show_progress:
        sequences = tqdm(sequences)
    for seq in sequences:
        try:
            # log the sequence
            seq_string = f"  SEQUENCE: {seq.id}  "
            logger.log("\n\n")
            logger.log("=" * len(seq_string))
            logger.log(seq_string)
            logger.log("=" * len(seq_string))
            logger.log(seq.fasta)

            # parse barcode segments
            parsed_segments = []
            failed_segments = []
            for seg in barcode_segments:
                logger.log("BARCODE SEGMENT:", seg)
                try:
                    parsed_segment = parse_barcode_segment(seq, seg)
                    if parsed_segment.is_good:
                        parsed_segments.append(parsed_segment)
                        logger.log("  --> PARSED!")
                    else:
                        failed_segments.append(parsed_segment)
                        logger.log("  --> FAILED!")
                except Exception as e:
                    logger.exception(
                        "BARCODE PARSING EXCEPTION:", seg, traceback.format_exc()
                    )
            # annotate sequences with the correct number of properly parsed segments
            if len(parsed_segments) == len(barcode_segments):
                bc = "+".join([seg.corrected for seg in parsed_segments])
                raw_bc = "+".join([seg.raw for seg in parsed_segments])
                umi = "+".join(
                    [seg.umi for seg in parsed_segments if seg.umi is not None]
                )
                seq.barcode = bc
                seq.raw_barcode = raw_bc
                seq.umi = umi

                # correct the read orientation
                if all([seg.is_rc for seg in parsed_segments]):
                    seq.was_rc = True
                    seq.sequence = seq.reverse_complement
                else:
                    seq.was_rc = False
                processed.append(seq)

                # log the results
                logger.log("RAW BARCODE:      ", raw_bc)
                logger.log("CORRECTED BARCODE:", bc)
                logger.log("UMI:", umi)
                logger.log("REV COMP:", seq.was_rc)
            else:
                failed.append(seq)
        except Exception as e:
            logger.exception("SEQUENCE PROCESSING EXCEPTION:", traceback.format_exc())
        finally:
            logger.checkpoint()

    # write successfully processed sequences to file
    output_fname = os.path.basename(fastq_file).split(".fastq")[0]
    output_file = os.path.join(output_directory, output_fname)
    output_data = []
    for p in processed:
        d = {
            "barcode": p.barcode,
            "raw_barcode": p.raw_barcode,
            "umi": p.umi,
            "sequence_id": p.id,
            "sequence": p.sequence,
            "was_rc": p.was_rc,
        }
        output_data.append(d)

    # log a basic summary of barcode parsing
    logger.log("OUTPUT FILENAME:", output_fname)
    logger.log("NUM PROCESSED SEQUENCES:", len(processed))
    logger.log("NUM FAILED SEQUENCES:", len(failed))

    # write output and log files
    output_df = pl.DataFrame(output_data)
    output_df.write_parquet(output_file)
    logger.write()

    # return the number of successfully processed sequences
    return output_file


def parse_barcode_segment(
    sequence: Sequence,
    segment: "BarcodeSegment",
) -> "ParsedBarcodeSegment":
    """
    Parses a single barcode segment from a sequence. Identfies the barcode region
    by aligning the sequence to the barcode segment's adapters, then extracts the
    barcode sequence (and optionally the UMI) from the identified region.

    Parameters
    ----------
    sequence : Sequence
        The sequence to parse.

    segment : BarcodeSegment
        The barcode segment to parse.

    Returns
    -------
    ParsedBarcodeSegment
        The parsed barcode segment.
    """
    is_rc = False
    s = sequence.sequence

    # find the first adapter
    adapter = segment.adapters[0]
    threshold = segment.adapter_score_threshold(adapter)
    aln1 = abutils.tl.semiglobal_alignment(adapter, s)
    if aln1.score < threshold:
        # if the initial alignment fails, check the reverse complement
        rc = sequence.reverse_complement
        aln1 = abutils.tl.semiglobal_alignment(adapter, rc)
        if aln1.score < threshold:
            return ParsedBarcodeSegment(is_good=False)
        else:
            is_rc = True
            s = sequence.reverse_complement

    # find the second adapter
    adapter = segment.adapters[1]
    threshold = segment.adapter_score_threshold(adapter)
    aln2 = abutils.tl.semiglobal_alignment(adapter, s)
    if aln2.score < threshold:
        return ParsedBarcodeSegment(is_good=False)

    # extract the barcode and UMI
    expected_interval_length = segment.length
    if segment.has_umi:
        expected_interval_length += segment.umi_length
    interval = s[aln1.target_end + 1 : aln2.target_begin]

    # check the barcode/UMI interval length
    if len(interval) != expected_interval_length:
        return ParsedBarcodeSegment(is_good=False)
    else:
        raw = interval[: segment.length]
        corrected = correct_barcode(raw, segment.whitelist)
        if corrected is not None:
            if segment.has_umi:
                umi = interval[segment.length :]
            else:
                umi = None
            return ParsedBarcodeSegment(
                raw=raw,
                corrected=corrected,
                umi=umi,
                is_good=True,
                is_rc=is_rc,
            )
    return ParsedBarcodeSegment(is_good=False)


# ----------------------------------
#        BARCODE CORRECTION
# ----------------------------------


def correct_barcode(
    barcode: str,
    whitelist: Iterable,
    score_cutoff: int = 6,
    scorer: Callable = rapidfuzz.distance.Levenshtein.distance,
) -> Optional[str]:
    """
    Corrects a raw barcode using a whitelist.

    Parameters
    ----------
    barcode : str
        The raw barcode to correct.

    whitelist : Iterable
        The whitelist to use for correction.

    score_cutoff : int, optional
        The score cutoff to use for correction, by default 6

    scorer : Callable, optional
        The scorer to use for correction, by default rapidfuzz.distance.Levenshtein.distance

    Returns
    -------
    Optional[str]
        The corrected barcode, or None if no suitable correct barcode is found.

    """
    result = extract(barcode, whitelist, scorer=scorer, score_cutoff=score_cutoff)

    # check for a "best" match
    if result:
        top_hit = result[0]
        top_bc = top_hit[0]
        top_bc_distance = top_hit[1]
    else:
        return None

    # check for a "second best" match
    if len(result) > 1:
        next_match_diff = result[1][1] - top_bc_distance
    else:
        next_match_diff = len(barcode)
    if next_match_diff >= 1:
        return top_bc
    return None


# ----------------------------------
#        BARCODE SEGMENTS
# ----------------------------------


@dataclass
class BarcodeSegment:
    """
    Defines the layout of a barcode segment, including the adapters that
    flank the barcode, barcode/UMI lengths, and an optional barcode whitelist

    Attributes
    ----------
    length : int
        The length of the barcode segment

    adapters : Iterable[str]
        The adapters that flank the barcode

    whitelist_file : Optional[str]
        The path to a whitelist file. Gzip-compressed files are supported.

    umi_length : int
        The length of the UMI. If not provided, the UMI is assumed to be absent.

    """

    length: int
    adapters: Iterable[str]
    whitelist_file: Optional[str] = None
    umi_length: int = 0
    _whitelist = None

    @property
    def has_umi(self):
        """
        Whether the barcode segment has a UMI.
        """
        return self.umi_length > 0

    @property
    def whitelist(self):
        """
        The barcode whitelist, parsed from the whitelist file.
        """
        if self._whitelist is None:
            whitelist = []
            # if we have a whitelist_file, parse it
            if self.whitelist_file is not None:
                if not os.path.exists(self.whitelist_file):
                    raise FileNotFoundError(
                        f"Whitelist file {self.whitelist_file} not found"
                    )
                # read the whitelist file
                open_func = gzip.open if self.whitelist_file.endswith(".gz") else open
                with open_func(self.whitelist_file) as f:
                    for line in f:
                        if bc := line.strip():
                            whitelist.append(bc)
            # if not, the whitelist is empty
            self._whitelist = whitelist
        return self._whitelist

    def adapter_score_threshold(self, adapter):
        """
        Get the adapter score threshold for a given adapter.
        """
        return int(len(adapter) * 1.5)


@dataclass
class ParsedBarcodeSegment:
    is_good: bool
    is_rc: Optional[bool] = None
    raw: Optional[str] = None
    corrected: Optional[str] = None
    umi: Optional[str] = None


# ----------------------------------
#       BARCODE DEFINITIONS
# ----------------------------------


def get_barcode_definition(barcode_name: str):
    """
    Get a barcode definition by name. Supported definitions are:
    - TXG_v2

    Parameters
    ----------
    barcode_name : str
        The name of the barcode definition

    Returns
    -------
    BarcodeSegment
        The barcode definition

    """
    if barcode_name.lower() == "txg_v2":
        return BarcodeSegment(**TXG_v2)
    else:
        raise ValueError(f"Barcode {barcode_name} not supported")


TXG_v2 = {
    "length": 16,
    "adapters": ["GATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT", "TTTCTTATATGGG"],
    "umi_length": 10,
    "whitelist_file": os.path.join(WHITELIST_DIR, "737K-august-2016.txt"),
}
