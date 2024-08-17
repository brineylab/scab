# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import os

import pytest

from scab.ont.barcodes import BarcodeSegment, get_barcode_definition


def test_get_barcode_definition():
    barcode_definition = get_barcode_definition("txg_v2")
    assert barcode_definition is not None
    assert isinstance(barcode_definition, BarcodeSegment)
    assert len(barcode_definition.whitelist) > 0
