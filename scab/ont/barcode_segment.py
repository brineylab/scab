# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import gzip
import os
from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = [
    "BarcodeSegment",
    "ParsedBarcodeSegment",
    "get_barcode_definition",
    "WHITELIST_DIR",
]

WHITELIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whitelists")


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

TXG_v2 = {
    "length": 16,
    "adapters": ["GATCTACACTCTTTCCCTACACGACGCTCTTCCGATCT", "TTTCTTATATGGG"],
    "umi_length": 10,
    "whitelist_file": os.path.join(WHITELIST_DIR, "737K-august-2016.txt"),
}
