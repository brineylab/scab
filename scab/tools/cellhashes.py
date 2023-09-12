#!/usr/bin/env python
# filename: cellhashes.py


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

from natsort import natsorted

import numpy as np
import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt

import scanpy as sc

from anndata import AnnData

from sklearn.neighbors import KernelDensity

from scipy.signal import argrelextrema


__all__ = ["demultiplex", "assign_cellhashes"]


def demultiplex(
    adata: AnnData,
    hash_names: Optional[Iterable] = None,
    cellhash_regex: str = "cell ?hash",
    ignore_cellhash_case: bool = True,
    rename: Optional[dict] = None,
    assignment_key: str = "cellhash_assignment",
    threshold_minimum: float = 4.0,
    threshold_maximum: float = 10.0,
    kde_minimum: float = 0.0,
    kde_maximum: float = 15.0,
    assignments_only: bool = False,
    debug: bool = False,
) -> Union[AnnData, Series]:
    """
    Demultiplexes cells using cell hashes.

    Parameters
    ----------

    adata : anndata.Anndata  
        ``AnnData`` object containing cellhash UMI counts in ``adata.obs``.
        
    hash_names : iterable object, optional  
        List of hashnames, which correspond to column names in ``adata.obs``. 
        Overrides cellhash name matching using `cellhash_regex`. If not provided, 
        all columns in ``adata.obs`` that match `cellhash_regex` will be assumed 
        to be hashnames and processed. 
        
    cellhash_regex : str, default='cell ?hash'  
        A regular expression (regex) string used to identify cell hashes. The regex 
        must be found in all cellhash names. The default is ``'cell ?hash'``, which 
        combined with the default setting for `ignore_cellhash_regex_case`, will 
        match ``'cellhash'`` or ``'cell hash'`` anywhere in the cell hash name and 
        in any combination of upper or lower case letters.  

    ignore_cellhash_regex_case : bool, default=True  
        If ``True``, matching to `cellhash_regex` will ignore case.  
        
    rename : dict, optional  
        A ``dict`` linking cell hash names (column names in ``adata.obs``) to the 
        preferred batch name. For example, if the cell hash name ``'Cellhash1'`` 
        corresponded to the sample ``'Sample1'``, an example `rename` argument 
        would be::

                {'Cellhash1': 'Sample1'}

        This would result in all cells classified as positive for ``'Cellhash1'`` being
        labeled as ``'Sample1'`` in the resulting assignment column (``adata.obs.sample`` 
        by default, adjustable using `assignment_key`).

    assignment_key : str, default='cellhash_assignment'  
        Column name (in ``adata.obs``) into which cellhash assignments will be stored.  

    threshold_minimum : float, default=4.0  
        Minimum acceptable log2-normalized UMI count threshold. Potential thresholds 
        below this cutoff value will be ignored.

    threshold_maximum : float, default=10.0  
        Maximum acceptable log2-normalized UMI count threshold. Potential thresholds 
        above this cutoff value will be ignored.  

    kde_maximum : float, default=15.0  
        Upper limit of the KDE plot (in log2-normalized UMI counts). This should 
        be less than `threshold_maximum`, or you may obtain strange results.  

    assignments_only : bool, default=False  
        If ``True``, return a pandas ``Series`` object containing only the group 
        assignment. Suitable for appending to an existing dataframe. If ``False``,
        an updated `adata` object is returned, containing cell hash group assignemnts
        at ``adata.obs.assignment_key``

    debug : bool, default=False  
        If ``True``, saves cell hash KDE plots and prints intermediate information 
        for debugging.  


    Returns
    -------
    output : ``anndata.AnnData`` or ``pandas.Series``  
        By default, an updated `adata` is returned with cell hash assignment groups \
        stored in the `assignment_key` column of ``adata.obs``. If `assignments_only` \
        is ``True``, a ``pandas.Series`` of lineage assignments is returned.

    """
    # parse hash names
    if hash_names is None:
        if ignore_cellhash_case:
            cellhash_pattern = re.compile(cellhash_regex, flags=re.IGNORECASE)
        else:
            cellhash_pattern = re.compile(cellhash_regex)
        hash_names = [
            o for o in adata.obs.columns if re.search(cellhash_pattern, o) is not None
        ]
    hash_names = [h for h in hash_names if h != assignment_key]
    if rename is None:
        rename = {}
    # compute thresholds
    thresholds = {}
    for hash_name in hash_names:
        if debug:
            print(hash_name)
        try:
            thresh = _get_feature_cutoff(
                adata.obs[hash_name].dropna(),
                threshold_minimum=threshold_minimum,
                threshold_maximum=threshold_maximum,
                kde_minimum=kde_minimum,
                kde_maximum=kde_maximum,
                debug=debug,
            )
        except Exception as e:
            thresh = None
            if debug:
                print("")
                print(f"ERROR: could not calculate a threshold for {hash_name}")
                print(e)
                print("")
        if thresh is not None:
            thresholds[hash_name] = thresh
    hash_names = [h for h in hash_names if h in thresholds]
    if debug:
        print("THRESHOLDS")
        print("----------")
        for hash_name in hash_names:
            print(f"{hash_name}: {thresholds[hash_name]}")
    assignments = []
    for _, row in adata.obs[hash_names].iterrows():
        a = [h for h in hash_names if row[h] >= thresholds[h]]
        if len(a) == 1:
            assignment = rename.get(a[0], a[0])
        elif len(a) > 1:
            assignment = "doublet"
        else:
            assignment = "unassigned"
        assignments.append(assignment)
    if assignments_only:
        return pd.Series(assignments, index=adata.obs_names)
    else:
        adata.obs[assignment_key] = assignments
        return adata


def _get_feature_cutoff(
    vals,
    threshold_maximum: float = 10.0,
    threshold_minimum: float = 4.0,
    kde_minimum: float = 0.0,
    kde_maximum: float = 15.0,
    debug: bool = False,
    show_cutoff_value: bool = False,
    cutoff_text: str = "cutoff",
    debug_figfile: Optional[str] = None,
) -> float:
    a = np.array(vals)
    k = _bw_silverman(a)
    kde = KernelDensity(kernel="gaussian", bandwidth=k).fit(a.reshape(-1, 1))
    s = np.linspace(kde_minimum, kde_maximum, num=int(kde_maximum * 100))
    e = kde.score_samples(s.reshape(-1, 1))

    all_min, all_max = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    if len(all_min) > 1:
        _all_min = np.array(
            [
                m
                for m in all_min
                if s[m] <= threshold_maximum and s[m] >= threshold_minimum
            ]
        )
        if _all_min.shape[0]:
            min_vals = zip(_all_min, e[_all_min])
            mi = sorted(min_vals, key=lambda x: x[1])[0][0]
            cutoff = s[mi]
        else:
            cutoff = None
    elif len(all_min) == 1:
        mi = all_min[0]
        cutoff = s[mi]
    else:
        cutoff = None
    if debug:
        if cutoff is not None:
            # plot
            plt.plot(s, e)
            plt.fill_between(s, e, y2=[min(e)] * len(s), alpha=0.1)
            plt.vlines(
                cutoff,
                min(e),
                max(e),
                colors="k",
                alpha=0.5,
                linestyles=":",
                linewidths=2,
            )
            # text
            text_xadj = 0.025 * (max(s) - min(s))
            cutoff_string = (
                f"{cutoff_text}: {round(cutoff, 3)}"
                if show_cutoff_value
                else cutoff_text
            )
            plt.text(
                cutoff - text_xadj,
                max(e),
                cutoff_string,
                ha="right",
                va="top",
                fontsize=14,
            )
            # style
            ax = plt.gca()
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="both", labelsize=12)
            ax.set_xlabel("$\mathregular{log_2}$ UMI counts", fontsize=14)
            ax.set_ylabel("kernel density", fontsize=14)
            # save or show
            if debug_figfile is not None:
                plt.tight_layout()
                plt.savefig(debug_figfile)
            else:
                plt.show()
        print("bandwidth: {}".format(k))
        print("local minima: {}".format(s[all_min]))
        print("local maxima: {}".format(s[all_max]))
        if cutoff is not None:
            print("cutoff: {}".format(cutoff))
        else:
            print(
                "WARNING: no local minima were found, so the threshold could not be calculated."
            )
        print("\n\n")
    return cutoff


def _bw_silverman(x):
    normalize = 1.349
    IQR = (np.percentile(x, 75) - np.percentile(x, 25)) / normalize
    std_dev = np.std(x, axis=0, ddof=1)
    if IQR > 0:
        A = np.minimum(std_dev, IQR)
    else:
        A = std_dev
    n = len(x)
    return 0.9 * A * n ** (-0.2)


# for backwards compatibility
def assign_cellhashes(
    adata: AnnData,
    hash_names: Optional[Iterable] = None,
    cellhash_regex: str = "cell ?hash",
    ignore_cellhash_case: bool = True,
    rename: Optional[dict] = None,
    assignment_key: str = "cellhash_assignment",
    threshold_minimum: float = 4.0,
    threshold_maximum: float = 10.0,
    kde_minimum: float = 0.0,
    kde_maximum: float = 15.0,
    assignments_only: bool = False,
    debug: bool = False,
) -> Union[AnnData, Series]:
    """
    Demultiplexes cells using cell hashes.

    Parameters
    ----------

    adata : anndata.Anndata  
        ``AnnData`` object containing cellhash UMI counts in ``adata.obs``.
        
    hash_names : iterable object, optional  
        List of hashnames, which correspond to column names in ``adata.obs``. 
        Overrides cellhash name matching using `cellhash_regex`. If not provided, 
        all columns in ``adata.obs`` that match `cellhash_regex` will be assumed 
        to be hashnames and processed. 
        
    cellhash_regex : str, default='cell ?hash'  
        A regular expression (regex) string used to identify cell hashes. The regex 
        must be found in all cellhash names. The default is ``'cell ?hash'``, which 
        combined with the default setting for `ignore_cellhash_regex_case`, will 
        match ``'cellhash'`` or ``'cell hash'`` anywhere in the cell hash name and 
        in any combination of upper or lower case letters.  

    ignore_cellhash_regex_case : bool, default=True  
        If ``True``, matching to `cellhash_regex` will ignore case.  
        
    rename : dict, optional  
        A ``dict`` linking cell hash names (column names in ``adata.obs``) to the 
        preferred batch name. For example, if the cell hash name ``'Cellhash1'`` 
        corresponded to the sample ``'Sample1'``, an example `rename` argument 
        would be::

                {'Cellhash1': 'Sample1'}

        This would result in all cells classified as positive for ``'Cellhash1'`` being
        labeled as ``'Sample1'`` in the resulting assignment column (``adata.obs.sample`` 
        by default, adjustable using `assignment_key`).

    assignment_key : str, default='cellhash_assignment'  
        Column name (in ``adata.obs``) into which cellhash assignments will be stored.  

    threshold_minimum : float, default=4.0  
        Minimum acceptable log2-normalized UMI count threshold. Potential thresholds 
        below this cutoff value will be ignored.

    threshold_maximum : float, default=10.0  
        Maximum acceptable log2-normalized UMI count threshold. Potential thresholds 
        above this cutoff value will be ignored.  

    kde_maximum : float, default=15.0  
        Upper limit of the KDE plot (in log2-normalized UMI counts). This should 
        be less than `threshold_maximum`, or you may obtain strange results.  

    assignments_only : bool, default=False  
        If ``True``, return a pandas ``Series`` object containing only the group 
        assignment. Suitable for appending to an existing dataframe. If ``False``,
        an updated `adata` object is returned, containing cell hash group assignemnts
        at ``adata.obs.assignment_key``

    debug : bool, default=False  
        If ``True``, saves cell hash KDE plots and prints intermediate information 
        for debugging.  


    Returns
    -------
    output : ``anndata.AnnData`` or ``pandas.Series``  
        By default, an updated `adata` is returned with cell hash assignment groups \
        stored in the `assignment_key` column of ``adata.obs``. If `assignments_only` \
        is ``True``, a ``pandas.Series`` of lineage assignments is returned.

    """
    return demultiplex(
        adata=adata,
        hash_names=hash_names,
        cellhash_regex=cellhash_regex,
        ignore_cellhash_case=ignore_cellhash_case,
        rename=rename,
        assignment_key=assignment_key,
        threshold_maximum=threshold_maximum,
        threshold_minimum=threshold_minimum,
        kde_maximum=kde_maximum,
        assignments_only=assignments_only,
        debug=debug,
    )
