#!/usr/bin/env python
# filename: specificity.py


#
# Copyright (c) 2021 Bryan Briney
# License: The MIT license (http://opensource.org/licenses/MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import os
import re
import sys
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame

import anndata
from anndata import AnnData

from ..io import read_10x_mtx


__all__ = ["classify_specificity"]


def classify_specificity(
    adata: anndata.AnnData,
    raw: Union[AnnData, str],
    agbcs: Optional[Iterable] = None,
    groups: Optional[dict] = None,
    rename: Optional[dict] = None,
    percentile: float = 0.997,
    percentile_dict: Optional[dict] = None,
    threshold_dict: Optional[dict] = None,
    agbc_regex: str = "agbc",
    update: bool = True,
    uns_batch: Optional[str] = None,
    verbose: bool = True,
) -> Union[AnnData, DataFrame]:
    """
    Classifies BCR specificity using antigen barcodes (**AgBCs**). Thresholds are computed by 
    analyzing background AgBC UMI counts in empty droplets.

    .. caution:: 
       In order to set accurate thresholds, we must remove all cell-containing droplets 
       from the ``raw`` counts matrix. Because ``adata`` comprises only cell-containing 
       droplets, we simply remove all of the droplet barcodes in ``adata`` from ``raw``. 
       Thus, it is **very important** that ``adata`` and ``raw`` are well matched.  
       
       For example, if processing a single Chromium reaction containing several multiplexed samples, 
       ``adata`` should contain all of the multiplexed samples, since the raw matrix produced 
       by CellRanger will also include all droplets in the reaction. If ``adata`` was missing 
       one or more samples, cell-containing droplets cannot accurately be removed from ``raw`` 
       and classification accuracy will be adversely affected.
    
    Parameters
    ----------
    adata : anndata.AnnData  
        Input ``AnnData`` object. Log2-normalized AgBC UMI counts should be found in 
        ``adata.obs``. If data was read using ``scab.read_10x_mtx()``, the resulting 
        ``AnnData`` object will already be correctly formatted.  
            
    raw : anndata.AnnData or str  
        Raw matrix data. Either a path to a directory containing the raw ``.mtx`` file
        produced by CellRanger, or an ``anndata.AnnData`` object containing the raw 
        matrix data. As with `adata`, log2-normalized AgBC UMIs should be found at 
        ``raw.obs``.  

        .. tip::
            If reading the raw counts matrix with ``scab.read_10x_mtx()``, it can be 
            helpful to include ``ignore_zero_quantile_agbcs=False``. In some cases with
            very little AgBC background, AgBCs can be incorrectly removed from the raw
            counts matrix.  
            
    agbcs : iterable object, optional
        A list of AgBCs to be classified. Either `agbcs`` or `groups`` is required. 
        If both are provided, both will be used.
        
    groups : dict, optional  
        A ``dict`` mapping specificity names to a list of one or more AgBCs. This 
        is particularly useful when multiple AgBCs correspond to the same antigen 
        (either because dual-labeled AgBCs were used, or because several AgBCs are 
        closely-related molecules that would be expected to compete for BCR binding). 
        Either `agbcs` or `groups` is required. If both are provided, both will be used.
            
    rename : dict, optional  
        A ``dict`` mapping AgBC or group names to a new name. Keys should be present in 
        either ``agbcs`` or ``groups.keys()``. If only a subset of AgBCs or groups are 
        provided in ``rename``, then only those AgBCs or groups will be renamed.
            
    percentile : float, default=0.997  
        Percentile used to compute the AgBC classification threshold using `raw` data. Default 
        is ``0.997``, which corresponds to three standard deviations.
            
    percentile_dict : dict, optional  
        A ``dict`` mapping AgBC or group names to the desired `percentile`. If only a subset 
        of AgBCs or groups are provided in `percentile_dict`, all others will use `percentile`.
            
    update : bool, default=True  
        If ``True``, update `adata` with grouped UMI counts and classifications. If ``False``, 
        a Pandas ``DataFrame`` containg classifications will be returned and `adata` will 
        not be modified. 

    uns_batch: str, default=None
        If provided, `uns_batch` will add batch information to the percentile and threshold
        data stored in ``adata.uns``. This results in an additional layer of nesting, which 
        allows concatenating multiple ``AnnData`` objects represeting different batches for 
        which classification is performed separately. If not provided, the data stored in ``uns`` 
        would be formatted like::

            adata.uns['agbc_percentiles'] = {agbc1: percentile1, ...}
            adata.uns['agbc_thresholds'] = {agbc1: threshold1, ...}  

        If `uns_batch` is provided, ``uns`` will be formatted like::

            adata.ubs['agbc_percentiles'] = {uns_batch: {agbc1: percentile1, ...}}
            adata.ubs['agbc_thresholds'] = {uns_batch: {agbc1: threshold1, ...}}

    verbose : bool, default=True  
        If ``True``, calculated threshold values are printed.  
            
    
    Returns
    -------
    output : ``anndata.AnnData`` or ``pandas.DataFrame``
        If `update` is ``True``, an updated `adata` object containing specificity classifications \
        is returned. Otherwise, a Pandas ``DataFrame`` containing specificity classifications \
        is returned.  
    
    """
    adata_groups = {}
    classifications = {}
    percentiles = percentile_dict if percentile_dict is not None else {}
    thresholds = threshold_dict if threshold_dict is not None else {}
    rename = rename if rename is not None else {}

    # process AgBCs and specificity groups
    if groups is None:
        groups = {}
        # if neither agbcs nor groups are provided,
        # find agbcs in obs columns using regex
        if agbcs is None:
            agbc_pattern = re.compile(agbc_regex, flags=re.IGNORECASE)
            agbcs = [re.search(agbc_pattern, i) is not None for i in adata.obs.columns]
    if agbcs is not None:
        for a in agbcs:
            groups[a] = [a]

    # load raw data, if necessary
    if isinstance(raw, str):
        if os.path.isdir(raw):
            raw = read_10x_mtx(raw, ignore_zero_quantile_agbcs=False)
        else:
            err = "\nERROR: raw must be either an AnnData object or a path "
            err += "to the raw matrix output folder from CellRanger.\n"
            print(err)
            sys.exit()

    # remove cell-containing droplets from raw
    no_cell = [o not in adata.obs_names for o in raw.obs_names]
    empty = raw[no_cell]

    # classify AgBC specificities
    if verbose:
        print("")
        print("  THRESHOLDS  ")
        print("--------------")
    uns_thresholds = {}
    uns_percentiles = {}
    for group, barcodes in groups.items():
        # remove missing AgBCs
        in_adata = [b for b in barcodes if b in adata.obs]
        in_empty = [b for b in barcodes if b in empty.obs]
        in_both = list(set(in_adata) & set(in_empty))
        if any([not in_adata, not in_empty]):
            err = f"\nERROR: group {group} cannot be processed because all AgBCs "
            err += "are missing from input or raw datasets.\n"
            if not in_adata:
                err += f"input is missing {', '.join([b for b in barcodes if b not in in_adata])}\n"
            if not in_empty:
                err += f"raw is missing {', '.join([b for b in barcodes if b not in in_empty])}\n"
            print(err)
            del groups[group]
            continue
        if len(in_adata) != len(in_empty):
            warn = f"\nWARNING: not all AgBCs for group {group} can be found in data and raw.\n"
            warn += f"input contains {', '.join(in_adata)}\n"
            warn += f"raw contains {', '.join(in_empty)}\n"
            print(warn)
            groups[group] = [bc for bc in barcodes if bc in in_both]
        group_name = rename.get(group, group)
        # compute thresholds
        if group_name in thresholds:
            pctile = "NA"
            threshold = thresholds[group_name]
        else:
            pctile = percentiles.get(group_name, percentile)
            # thresholds for each barcode
            _empty = np.array([np.exp2(empty.obs[bc]) - 1 for bc in in_empty])
            raw_bc_thresholds = {
                bc: np.quantile(_e, pctile) for _e, bc in zip(_empty, in_empty)
            }
            # UMI counts for the entire group
            _data = np.sum([np.exp2(adata.obs[bc]) - 1 for bc in in_adata], axis=0)
            adata_groups[group_name] = np.log2(_data + 1)
            # threshold for the entire group (sum of the individual barcode thresholds)
            raw_threshold = np.sum(list(raw_bc_thresholds.values()))
            threshold = np.log2(raw_threshold + 1)
        classifications[group_name] = adata_groups[group_name] > threshold
        # update uns dicts
        uns_thresholds[group] = threshold
        uns_percentiles[group] = pctile
        if verbose:
            print(group_name)
            print(f"percentile: {pctile}")
            print(f"threshold: {threshold}")
            for bc, rt in raw_bc_thresholds.items():
                print(f"  - {bc}: {np.log2(rt + 1)}")
            print("")
    if update:
        for g, group_data in adata_groups.items():
            adata.obs[g] = group_data
            adata.obs[f"is_{g}"] = classifications[g]
            if uns_batch is not None:
                adata.uns["agbc_thresholds"] = {uns_batch: uns_thresholds}
                adata.uns["agbc_percentiles"] = {uns_batch: uns_percentiles}
            else:
                adata.uns["agbc_thresholds"] = uns_thresholds
                adata.uns["agbc_percentiles"] = uns_percentiles
        return adata
    else:
        return pd.DataFrame(classifications, index=adata.obs_names)
