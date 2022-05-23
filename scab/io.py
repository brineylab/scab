#!/usr/bin/env python
# filename: io.py


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


import codecs
import pickle
import re

import numpy as np

import scanpy as sc

import abstar
from abutils.core.pair import Pair, assign_pairs
from abutils.core.sequence import read_csv, read_fasta, read_json


def read_10x_mtx(
    mtx_path,
    bcr_file=None,
    bcr_annot=None,
    bcr_format="fasta",
    bcr_delimiter="\t",
    bcr_id_key="sequence_id",
    bcr_sequence_key="sequence",
    bcr_id_delimiter="_",
    bcr_id_delimiter_num=1,
    tcr_file=None,
    tcr_annot=None,
    tcr_format="fasta",
    tcr_delimiter="\t",
    tcr_id_key="sequence_id",
    tcr_sequence_key="sequence",
    tcr_id_delimiter="_",
    tcr_id_delimiter_num=1,
    chain_selection_func=None,
    abstar_output_format="airr",
    abstar_germ_db="human",
    gex_only=False,
    hashes=None,
    cellhash_regex="cell ?hash",
    ignore_cellhash_case=True,
    agbcs=None,
    agbc_regex="agbc",
    ignore_agbc_case=True,
    log_transform_cellhashes=True,
    ignore_zero_quantile_cellhashes=True,
    rename_cellhashes=None,
    log_transform_agbcs=True,
    ignore_zero_quantile_agbcs=True,
    rename_agbcs=None,
    log_transform_features=True,
    ignore_zero_quantile_features=True,
    rename_features=None,
    feature_suffix="_FBC",
    cellhash_quantile=0.95,
    agbc_quantile=0.95,
    feature_quantile=0.95,
    verbose=True,
):

    """
    Reads and integrates output files from 10x Genomics' CellRanger into a single ``AnnData`` object.  

    Datasets can include gene expression (GEX), cell hashes, antigen barcodes (AgBCs), feature
    barcodes, and assembled BCR or TCR contig sequences.

    Parameters  
    ----------  
    mtx_path : str  
        Path to a CellRanger counts matrix folder, typically either ``'sample_feature_bc_matrix'`` 
        or ``'raw_feature_bc_matrix'``.  

    [bcr|tcr]_file : str, optional  
        Path to a file containing BCR/TCR data. The file can be in one of several formats:  

                - FASTA-formatted file, as output by CellRanger  

                - delimited text file, containing annotated BCR/TCR sequences  

                - JSON-formatted file, containing annotated BCR/TCR sequences  

    [bcr|tcr]_annot : str, optional  
        Path to the CSV-formatted BCR/TCR annotations file produced by CellRanger. Matching the 
        annotation file to `[bcr|tcr]_file` is preferred -- if ``'all_contig.fasta'`` is the supplied 
        `[bcr|tcr]_file`, then ``'all_contig_annotations.csv'`` is the appropriate annotation file.  

    [bcr|tcr]_format : str, default='fasta'  
        Format of the input `[bcr|tcr]_file`. Options are: ``'fasta'``, ``'delimited'``, and 
        ``'json'``. If `[bcr|tcr]_format` is ``'fasta'``, `abstar`_ 
        will be run on the input data to obtain annotated BCR/TCR data. By default, abstar will 
        produce `AIRR-formatted`_  (tab-delimited) annotations.  

    [bcr|tcr]_delimiter : str, default='\t'  
        Delimiter used in `[bcr|tcr]_file`. Only used if `[bcr|tcr]_format` is ``'delimited'``.
        Default is ``'\t'``, which conforms to AIRR-C data standards.  

    [bcr|tcr]_id_key : str, default='sequence_id'  
        Name of the column or field in `[bcr|tcr]_file` that corresponds to the sequence ID.   

    [bcr|tcr]_sequence_key : str, default='sequence'  
        Name of the column or field in `[bcr|tcr]_file` that corresponds to the VDJ sequence.  

    [bcr|tcr]_id_delimiter : str, default='_'  
        The delimiter used to separate the droplet and contig components of the sequence ID.
        For example, default CellRanger names are formatted as: ``'AAACCTGAGAACTGTA-1_contig_1'``, where 
        ``'AAACCTGAGAACTGTA-1'`` is the droplet identifier and ``'contig_1'`` is the contig identifier.  

    [bcr|tcr]_id_delimiter_num : str, default=1  
        The occurance (1-based numbering) of the `[bcr|tcr]_id_delimiter`.  

    abstar_output_format : str, default='airr'  
        Format for abstar annotations. Only used if `[bcr|tcr]_format` is ``'fasta'``. 
        Options are ``'airr'``, ``'json'`` and ``'tabular'``.  

    abstar_germ_db : str, default='human'  
        Germline database to be used for annotation of BCR/TCR data. Built-in abstar options 
        include: ``'human'``, ``'macaque'``, ``'mouse'`` and ``'humouse'``. Only used if 
        one or both of `[bcr|tcr]_format` is ``'fasta'``.

    gex_only : bool, default=False  
        If ``True``, return only gene expression data and ignore features and hashes. Note that
        VDJ data will still be included in the returned ``AnnData`` object if `[bcr|tcr]_file` 
        is provided.  

    cellhash_regex : str, default='cell ?hash'  
        A regular expression (regex) string used to identify cell hashes. The regex 
        must be found in all hash names. The default, combined with the
        default setting for `ignore_hash_regex_case`, will match ``'cellhash'`` or ``'cell hash'``
        in any combination of upper and lower case letters.  

    ignore_cellhash_regex_case : bool, default=True  
        If ``True``, searching for `hash_regex` will ignore case.  

    agbc_regex : str, default='agbc'  
        A regular expression (regex) string used to identify AgBCs. The regex 
        must be found in all AgBC names. The default, combined with the
        default setting for `ignore_hash_regex_case`, will match ``'agbc'``
        in any combination of upper and lower case letters.  

    ignore_agbc_regex_case : bool, default=True  
        If ``True``, searching for `agbc_regex` will ignore case.  

    log_transform_cellhashes : bool, default=True  
        If ``True``, cell hash UMI counts will be log2-plus-1 transformed.  

    log_transform_agbcs : bool, default=True  
        If ``True``, AgBC UMI counts will be log2-plus-1 transformed.  
        
    log_transform_features : bool, default=True  
        If ``True``, feature UMI counts will be log2-plus-1 transformed.  

    ignore_zero_quantile_cellhashes : bool, default=True  
        If ``True``, any hashes for which the `cellhash_quantile`
        percentile have a count of zero are ignored. Default is ``True`` and the default 
        `cellhash_quantile` is ``0.95``, resulting in cellhashes with zero counts for the 95th
        percentile being ignored.  
        
    ignore_zero_median_agbcs : bool, default=True  
        If ``True``, any AgBCs for which the `agbc_quantile`
        percentile have a count of zero are ignored. Default is ``True`` and the default 
        `agbc_quantile` is ``0.95``, resulting in AgBCs with zero counts for the 95th
        percentile being ignored.  

    ignore_zero_median_features : bool, default=True  
        If ``True``, any features for which the `feature_quantile`
        percentile have a count of zero are ignored. Default is ``True`` and the default 
        `feature_quantile` is ``0.95``, resulting in features with zero counts for the 95th
        percentile being ignored.  

    rename_cellhashes : dict, optional  
        A dictionary with keys and values corresponding to the existing and 
        new cellhash names, respectively. For example, ``{'CellHash1': 'donor123}`` would result in the 
        renaming of ``'CellHash1'`` to ``'donor123'``. Cellhashes not found in the `rename_cellhashes` 
        dictionary will not be renamed.  

    rename_agbcs : dict, optional  
        A dictionary with keys and values corresponding to the existing and 
        new AgBC names, respectively. For example, ``{'AgBC1': 'Influenza H1'}`` would result in the 
        renaming of ``'AgBC1'`` to ``'Influenza H1'``. AgBCs not found in the `rename_agbcs` 
        dictionary will not be renamed.  

    rename_features : dict, optional  
        A dictionary with keys and values corresponding to the existing and 
        new feature names, respectively. For example, ``{'FeatureBC1': 'CD19}`` would result in the 
        renaming of ``'FeatureBC1'`` to ``'CD19'``. Features not found in the `rename_features` 
        dictionary will not be renamed.  

    feature_suffix : str, default='_FBC'  
        Suffix to add to the end of each feature name. Useful because feature 
        names may overlap with gene names. The default value will result in the feature 
        ``'CD19'`` being renamed to ``'CD19_FBC'``. The suffix is added after feature renaming. 
        To skip the addition of a feature suffix, simply supply an empty string (``''``) as the argument.  

    cellhash_quantile : float, default=0.95  
        Percentile for which cellhashes with zero counts will be ignored if
        `ignore_zero_quantile_cellhashes` is ``True``. Default is ``0.95``, which is equivalent to the
        95th percentile.  

    agbc_quantile : float, default=0.95  
        Percentile for which AgBCs with zero counts will be ignored if
        `ignore_zero_quantile_agbcs` is ``True``. Default is ``0.95``, which is equivalent to the
        95th percentile.  

    feature_quantile : float, default=0.95  
        Percentile for which features with zero counts will be ignored if
        `ignore_zero_quantile_features` is ``True``. Default is ``0.95``, which is equivalent to the
        95th percentile.  

    verbose : bool, default=True  
        Print progress updates.  
    

    Returns
    -------
    adata : ``anndata.AnnData``
        An ``AnnData`` object containing gene expression data, with VDJ information located
        at ``adata.obs.bcr`` and/or ``adata.obs.tcr``, and cellhash and feature barcode data 
        found in ``adata.obs``. If ``gex_only`` is ``True``, cellhash and feature barcode data 
        are not returned.  


    .. _abstar:
        https://github.com/briney/abstar

    .. _AIRR-formatted:
        https://docs.airr-community.org/en/stable/datarep/rearrangements.html
    """
    # read 10x matrix file
    if verbose:
        print("reading 10x Genomics matrix file...")
    adata = sc.read_10x_mtx(mtx_path, gex_only=False)
    gex = adata[:, adata.var.feature_types == "Gene Expression"]

    # process BCR data:
    if bcr_file is not None:
        if bcr_format == "delimited":
            delim_renames = {'\t': 'tab', ',': 'comma'}
            if verbose:
                d = delim_renames.get(bcr_delimiter, f"'{bcr_delimiter}'")
                print(f"reading {d}-delimited BCR data...")
            sequences = read_csv(
                bcr_file,
                delimiter=bcr_delimiter,
                id_key=bcr_id_key,
                sequence_key=bcr_sequence_key,
            )
        elif bcr_format == "json":
            if verbose:
                print("reading JSON-formatted BCR data...")
            sequences = read_json(
                bcr_file, id_key=bcr_id_key, sequence_key=bcr_sequence_key
            )
        elif bcr_format == "fasta":
            if verbose:
                print("reading FASTA-formatted BCR data...")
            raw_seqs = read_fasta(bcr_file)
            if verbose:
                print("annotating BCR sequences with abstar...")
            sequences = abstar.run(
                raw_seqs, output_type=abstar_output_format, germ_db=abstar_germ_db
            )
        pairs = assign_pairs(
            sequences,
            id_key=bcr_id_key,
            delim=bcr_id_delimiter,
            delim_occurance=bcr_id_delimiter_num,
            chain_selection_func=chain_selection_func,
            tenx_annot_file=bcr_annot,
        )
        pdict = {p.name: p for p in pairs}
        gex.obs["bcr"] = [pdict.get(o, Pair([])) for o in gex.obs_names]

    # process TCR data:
    if tcr_file is not None:
        if tcr_format == "delimited":
            delim_renames = {'\t': 'tab', ',': 'comma'}
            if verbose:
                d = delim_renames.get(tcr_delimiter, f"'{tcr_delimiter}'")
                print(f"reading {d}-delimited BCR data...")
            sequences = read_csv(
                tcr_file,
                delimiter=tcr_delimiter,
                id_key=tcr_id_key,
                sequence_key=tcr_sequence_key,
            )
        elif tcr_format == "json":
            if verbose:
                print("reading JSON-formatted TCR data...")
            sequences = read_json(
                tcr_file, id_key=tcr_id_key, sequence_key=tcr_sequence_key
            )
        elif tcr_format == "fasta":
            if verbose:
                print("reading FASTA-formatted TCR data...")
            raw_seqs = read_fasta(tcr_file)
            if verbose:
                print("annotating TCR sequences with abstar...")
            sequences = abstar.run(
                raw_seqs,
                receptor="tcr",
                output_type=abstar_output_format,
                germ_db=abstar_germ_db,
            )
        pairs = assign_pairs(
            sequences,
            id_key=tcr_id_key,
            delim=tcr_id_delimiter,
            delim_occurance=tcr_id_delimiter_num,
            chain_selection_func=chain_selection_func,
            tenx_annot_file=tcr_annot,
        )
        pdict = {p.name: p for p in pairs}
        gex.obs["tcr"] = [pdict.get(o, Pair([])) for o in gex.obs_names]
    if gex_only:
        return gex

    # parse out features and cellhashes
    non_gex = adata[:, adata.var.feature_types != "Gene Expression"]
    if ignore_cellhash_case:
        cellhash_pattern = re.compile(cellhash_regex, flags=re.IGNORECASE)
    else:
        cellhash_pattern = re.compile(cellhash_regex)
    if ignore_agbc_case:
        agbc_pattern = re.compile(agbc_regex, flags=re.IGNORECASE)
    else:
        agbc_pattern = re.compile(agbc_regex)
    hashes = non_gex[
        :, [re.search(cellhash_pattern, i) is not None for i in non_gex.var.gene_ids]
    ]
    agbcs = non_gex[
        :, [re.search(agbc_pattern, i) is not None for i in non_gex.var.gene_ids]
    ]
    features = non_gex[
        :,
        [
            all(
                [
                    re.search(cellhash_pattern, i) is None,
                    re.search(agbc_pattern, i) is None,
                ]
            )
            for i in non_gex.var.gene_ids
        ],
    ]

    # process cellhash data
    if verbose:
        print("processing cellhash data...")
    hash_df = hashes.to_df()[hashes.var_names]
    if ignore_zero_quantile_cellhashes:
        hash_df = hash_df[
            [
                h
                for h in hash_df.columns.values
                if hash_df[h].quantile(q=cellhash_quantile) > 0
            ]
        ]
    if log_transform_cellhashes:
        hash_df += 1
        hash_df = hash_df.apply(np.log2)
    if rename_cellhashes is None:
        rename_cellhashes = {}
    for h in hash_df:
        gex.obs[rename_cellhashes.get(h, h)] = hash_df[h]

    # process AgBC data
    if verbose:
        print("processing AgBC data...")
    agbc_df = agbcs.to_df()[agbcs.var_names]
    if ignore_zero_quantile_agbcs:
        agbc_df = agbc_df[
            [
                a
                for a in agbc_df.columns.values
                if agbc_df[a].quantile(q=agbc_quantile) > 0
            ]
        ]
    if log_transform_agbcs:
        agbc_df += 1
        agbc_df = agbc_df.apply(np.log2)
    if rename_agbcs is None:
        rename_agbcs = {}
    for a in agbc_df:
        gex.obs[rename_agbcs.get(a, a)] = agbc_df[a]

    # process feature barcode data
    if verbose:
        print("processing feature barcode data...")
    feature_df = features.to_df()[features.var_names]
    if ignore_zero_quantile_features:
        feature_df = feature_df[
            [
                h
                for h in feature_df.columns.values
                if feature_df[h].quantile(q=feature_quantile) > 0
            ]
        ]
    if log_transform_features:
        feature_df += 1
        feature_df = feature_df.apply(np.log2)
    if rename_features is None:
        rename_features = {f: f"{f}{feature_suffix}" for f in feature_df}
    for f in feature_df:
        gex.obs[rename_features.get(f, f)] = feature_df[f]
    return gex


def read(h5ad_file):
    """
    Reads a serialized ``AnnData`` object. Similar to ``scanpy.read()``, except that ``scanpy`` 
    does not support serialized BCR/TCR data. If BCR/TCR data is included in the serialized ``AnnData``
    file, it will be separately deserialized into the original ``abutils.Pair`` objects.

    Parameters
    ----------
    h5ad_file : str  
        Path to the output file. The output will be written in ``h5ad`` format and must
        include ``'.h5ad'`` as the file extension. If it is not included, the extension will automatically
        be added. Required.   


    Returns
    -------
    adata : ``anndata.AnnData``

    """
    adata = sc.read(h5ad_file)
    if "bcr" in adata.obs:
        # unpickle BCR data
        adata.obs["bcr"] = [
            pickle.loads(codecs.decode(b.encode(), "base64")) for b in adata.obs.bcr
        ]
    if "tcr" in adata.obs:
        # unpickle TCR data
        adata.obs["tcr"] = [
            pickle.loads(codecs.decode(t.encode(), "base64")) for t in adata.obs.tcr
        ]
    return adata


def write(adata, h5ad_file):
    """
    Serialized and writes an ``AnnData`` object to disk in ``h5ad`` format. Similar to 
    ``scanpy.write()``, except that ``scanpy`` does not support serializing BCR/TCR data. This
    function serializes ``abutils.Pair`` objects stored in either ``adata.obs.bcr`` or 
    ``adata.obs.tcr`` using ``pickle`` prior to writing the ``AnnData`` object to disk.

    Parameters
    ----------

    adata : anndata.AnnData  
        An ``AnnData`` object containing gene expression, feature barcode and 
        VDJ data. ``scab.read_10x_mtx()`` can be used to construct a multi-omics ``AnnData`` object
        from raw CellRanger outputs. Required.

    h5ad_file : str  
        Path to the output file. The output will be written in ``h5ad`` format and must
        include ``'.h5ad'`` as the file extension. If it is not included, the extension will automatically
        be added. Required.    
    """
    if not h5ad_file.endswith("h5ad"):
        h5ad_file += ".h5ad"
    _adata = adata.copy()
    if "bcr" in _adata.obs:
        # pickle BCR data
        _adata.obs["bcr"] = [
            codecs.encode(pickle.dumps(b), "base64").decode() for b in _adata.obs.bcr
        ]
    if "tcr" in adata.obs:
        # pickle TCR data
        _adata.obs["tcr"] = [
            codecs.encode(pickle.dumps(t), "base64").decode() for t in _adata.obs.tcr
        ]
    _adata.write(h5ad_file)


def save(adata, h5ad_file):
    """
    Serialized and writes an ``AnnData`` object to disk in ``h5ad`` format. Similar to 
    ``scanpy.write()``, except that ``scanpy`` does not support serializing BCR/TCR data. This
    function serializes ``abutils.Pair`` objects stored in either ``adata.obs.bcr`` or 
    ``adata.obs.tcr`` using ``pickle`` prior to writing the ``AnnData`` object to disk.

    Parameters
    ----------

    adata : anndata.AnnData  
        An ``AnnData`` object containing gene expression, feature barcode and 
        VDJ data. ``scab.read_10x_mtx()`` can be used to construct a multi-omics ``AnnData`` object
        from raw CellRanger outputs. Required.

    h5ad_file : str  
        Path to the output file. The output will be written in ``h5ad`` format and must
        include ``'.h5ad'`` as the file extension. If it is not included, the extension will automatically
        be added. Required.    
    """
    write(adata, h5ad_file)
