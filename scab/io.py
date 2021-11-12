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


import re

import numpy as np

import scanpy as sc

import abstar
from abutils.core.pair import Pair, assign_pairs
from abutils.core.sequence import read_csv, read_fasta, read_json


def read_10x_mtx(mtx_path, bcr_file=None, bcr_annotations=None, bcr_format='csv', bcr_delimiter='\t',
                 bcr_id_key='sequence_id', bcr_sequence_key='sequence', bcr_id_delimiter='_', bcr_id_delimiter_num=1,
                 tcr_file=None, tcr_annotations=None, tcr_format='csv', tcr_delimiter='\t',
                 tcr_id_key='sequence_id', tcr_sequence_key='sequence', tcr_id_delimiter='_', tcr_id_delimiter_num=1,
                 h_selection_func=None, l_selection_func=None, abstar_output_format='airr',
                 gex_only=False, cellhash_regex='cell ?hash', ignore_cellhash_case=True,
                 agbc_regex='agbc', ignore_agbc_case=True,
                 log_transform_cellhashes=True, ignore_zero_median_cellhashes=True, rename_cellhashes=None,
                 log_transform_agbcs=True, ignore_zero_median_agbcs=True, rename_agbcs=None,
                 log_transform_features=True, ignore_zero_median_features=True, rename_features=None, feature_suffix='_FBC',
                 verbose=True):

    '''
    Reads 10x Genomics matrix and VDJ files and outputs GEX, cellhash, feature barcode and VDJ data in a single AnnData object.

    Args:
    -----

        mtx_path (str): Path to the 10x Genomics matrix folder (as accepted by ``scanpy.read_10x_mtx()``)

        [bcr|tcr]_file (str): Path to a file containing VDJ data. The file can be in of the following formats:
                1) a FASTA-formatted file, as output by CellRanger
                2) a delimited text file, containing annotated VDJ sequences
                3) a JSON file, containing annotated VDJ sequences

        [bcr|tcr]_annotations (str): Path to the CSV-formatted VDJ annotations file produced by CellRanger. 
            Matching the annotation file to ``vdj_file`` is preferred -- if ``all_contig.fasta`` is the supplied 
            ``vdj_file``, then ``all_contig_annotations.csv`` is the appropriate annotation file.

        [bcr|tcr]_format (str): Format of the input ``[bcr|tcr]_file``. Options are: ``'fasta'``, ``'csv'``, and ``'json'``.
            Default is ``'csv'``. If ``[bcr|tcr]_format`` is ``'fasta'``, ``abstar`` will be run on  the input data to 
            obtain annotated VDJ data. By default, ``abstar`` will produce AIRR-formatted (tab-delimited) annotations.

        [bcr|tcr]_delimiter (str): Delimiter used in ``[bcr|tcr]_file``. Only used if ``[bcr|tcr]_format`` is ``'csv'``.
            Default is ``'\t'``, which conforms to AIRR-C data standards.

        [bcr|tcr]_id_key (str): Name of the column or field in ``[bcr|tcr]_file`` that corresponds to the sequence ID. 
            Default is ``'sequence_id'``, which is compatible with standardized AIRR-C data formatting.

        [bcr|tcr]_id_key (str): Name of the column or field in ``[bcr|tcr]_file`` that corresponds to the VDJ sequence. 
            Default is ``'sequence'``, which is compatible with standardized AIRR-C data formatting.

        [bcr|tcr]_id_delimiter (str): The delimiter used to separate the droplet and contig components of the sequence ID.
            For example, default CellRanger names are formatted as: ``'AAACCTGAGAACTGTA-1_contig_1'``, where 
            ``'AAACCTGAGAACTGTA-1'`` is the droplet identifier and ``'contig_1'`` is the contig identifier. 
            Default is '_', which matches the format used by CellRanger.

        [bcr|tcr]_id_delimiter_num (str): The occurance (1-based numbering) of the ``[bcr|tcr]_id_delimiter``. Default is ``1``,
            which matches the format used by CellRanger.

        abstar_output_format (str): Format for abstar annotations. Only used if ``[bcr|tcr]_format`` is ``'fasta'``. 
            Options are ``'airr'``, ``'json'`` and ``'tabular'``. Default is ``'airr'``.

        gex_only (bool): If ``True``, return only gene expression data and ignore features and hashes. Note that
            VDJ data will still be included in the returned ``AnnData`` object if ``[bcr|tcr]_file`` is provided.
            Default is ``False``.

        cellhash_regex (str): A regular expression (regex) string used to identify cell hashes. The regex 
            must be found in all hash names. The default is ``'cell ?hash'``, which combined with the
            default setting for ``ignore_hash_regex_case``, will match ``'cellhash'`` or ``'cell hash'``
            in any combination of upper and lower case letters.

        ignore_cellhash_regex_case (bool): If ``True``, searching for ``hash_regex`` will ignore case.
            Default is ``True``.

        agbc_regex (str): A regular expression (regex) string used to identify AgBCs. The regex 
            must be found in all AgBC names. The default is ``'agbc'``, which combined with the
            default setting for ``ignore_hash_regex_case``, will match ``'agbc'``
            in any combination of upper and lower case letters.

        ignore_agbc_regex_case (bool): If ``True``, searching for ``agbcregex`` will ignore case.
            Default is ``True``.

        log_transform_cellhashes (bool): If ``True``, cellhash UMI counts will be log2 transformed 
            (after adding 1 to the raw count). Default is ``True``.

        log_transform_agbcs (bool): If ``True``, AgBC UMI counts will be log2 transformed 
            (after adding 1 to the raw count). Default is ``True``.
        
        log_transform_features (bool): If ``True``, feature UMI counts will be log2 transformed 
            (after adding 1 to the raw count). Default is ``True``.

        ignore_zero_median_cellhashes (bool): If ``True``, any hashes containing a meadian
            count of ``0`` will be ignored and not returned in the hash dataframe. Default
            is ``True``.
        
        ignore_zero_median_agbcs (bool): If ``True``, any AgBCs containing a meadian
            count of ``0`` will be ignored and not returned in the AgBC dataframe. Default
            is ``True``.

        ignore_zero_median_features (bool): If ``True``, any features containing a meadian
            count of ``0`` will be ignored and not returned in the feature dataframe. Default
            is ``True``.

        rename_cellhashes (dict): A dictionary with keys and values corresponding to the existing and 
            new cellhash names, respectively. For example, ``{'CellHash1': 'donorABC}`` would result in the 
            renaming of ``'CellHash1'`` to ``'donorABC'``. Cellhashes not found in the ``rename_cellhashes`` 
            dictionary will not be renamed.

        rename_agbcs (dict): A dictionary with keys and values corresponding to the existing and 
            new AgBC names, respectively. For example, ``{'AgBC1': 'Lassa_GPC}`` would result in the 
            renaming of ``'AgBC1'`` to ``'LassaGPC'``. AgBCs not found in the ``rename_agbcs`` 
            dictionary will not be renamed.

        rename_features (dict): A dictionary with keys and values corresponding to the existing and 
            new feature names, respectively. For example, ``{'FeatureBC1': 'CD19}`` would result in the 
            renaming of ``'FeatureBC1'`` to ``'CD19'``. Features not found in the ``rename_features`` 
            dictionary will not be renamed.

        feature_suffix (str): Suffix to add to the end of each feature name. Useful because feature 
            names may overlap with gene names. Default is ``'_FBC'`` which would result in the feature 
            ``'CD19'`` being renamed to ``'CD19_FBC'``. The suffix is added after feature renaming. 
            To skip the addition of a feature suffix, simply supply an empty string (``''``) as the argument.

        verbose (bool): Print progress updates. Default is ``True``.
    
    Returns:
    --------

        anndata.AnnData: an ``AnnData`` object containing gene expression data, with VDJ information located
            at ``adata.obs.vdj``, and cellhash and feature barcode data found in ``adata.obs``. If ``gex_only`` 
            is ``True``, cellhash and feature barcode data are not returned. If ``vdj_file`` is ``None``, 
            VDJ information is not returned.
    '''
    # read 10x matrix file
    if verbose:
                print('reading 10x Genomics matrix file...')
    adata = sc.read_10x_mtx(mtx_path, gex_only=False)
    gex = adata[:,adata.var.feature_types == 'Gene Expression']

    # process BCR data:
    if bcr_file is not None:
        if bcr_format == 'csv':
            if verbose:
                print('reading CSV-formatted BCR data...')
            sequences = read_csv(bcr_file, delimiter=bcr_delimiter,
                                 id_key=bcr_id_key, sequence_key=bcr_sequence_key)
        elif bcr_format == 'json':
            if verbose:
                print('reading JSON-formatted BCR data...')
            sequences = read_json(bcr_file, id_key=bcr_id_key, sequence_key=bcr_sequence_key)
        elif bcr_format == 'fasta':
            if verbose:
                print('reading FASTA-formatted BCR data...')
            raw_seqs = read_fasta(bcr_file)
            if verbose:
                print('annotating BCR sequences with abstar...')
            sequences = abstar.run(raw_seqs, output_type=abstar_output_format)
        pairs = assign_pairs(sequences, name=bcr_id_key,
                             delim=bcr_id_delimiter, delim_occurance=bcr_id_delimiter_num,
                             h_selection_func=h_selection_func, l_selection_func=l_selection_func,
                             tenx_annot_file=bcr_annotations)
        pdict = {p.name: p for p in pairs}
        gex.obs['bcr'] = [pdict.get(o, Pair([])) for o in gex.obs_names]

    # process TCR data:
    if tcr_file is not None:
        if tcr_format == 'csv':
            if verbose:
                print('reading CSV-formatted TCR data...')
            sequences = read_csv(tcr_file, delimiter=tcr_delimiter,
                                 id_key=tcr_id_key, sequence_key=tcr_sequence_key)
        elif tcr_format == 'json':
            if verbose:
                print('reading JSON-formatted TCR data...')
            sequences = read_json(tcr_file, id_key=tcr_id_key, sequence_key=tcr_sequence_key)
        elif tcr_format == 'fasta':
            if verbose:
                print('reading FASTA-formatted TCR data...')
            raw_seqs = read_fasta(tcr_file)
            if verbose:
                print('annotating TCR sequences with abstar...')
            sequences = abstar.run(raw_seqs, output_type=abstar_output_format)
        pairs = assign_pairs(sequences, name=tcr_id_key,
                             delim=tcr_id_delimiter, delim_occurance=tcr_id_delimiter_num,
                             h_selection_func=h_selection_func, l_selection_func=l_selection_func,
                             tenx_annot_file=tcr_annotations)
        pdict = {p.name: p for p in pairs}
        gex.obs['tcr'] = [pdict.get(o, Pair([])) for o in gex.obs_names]
    if gex_only:
        return gex
    
    # parse out features and cellhashes
    non_gex = adata[:, adata.var.feature_types != 'Gene Expression']
    if ignore_cellhash_case:
        cellhash_pattern = re.compile(cellhash_regex, flags=re.IGNORECASE)
    else:
        cellhash_pattern = re.compile(cellhash_regex)
    if ignore_agbc_case:
        agbc_pattern = re.compile(agbc_regex, flags=re.IGNORECASE)
    else:
        agbc_pattern = re.compile(agbc_regex)
    hashes = non_gex[:, [re.search(cellhash_pattern, i) is not None for i in non_gex.var.gene_ids]]
    agbcs = non_gex[:, [re.search(agbc_pattern, i) is not None for i in non_gex.var.gene_ids]]
    features = non_gex[:, [all([re.search(cellhash_pattern, i) is None,
                                re.search(agbc_pattern, i) is None]) for i in non_gex.var.gene_ids]]
    
    # process cellhash data
    if verbose:
        print('processing cellhash data...')
    hash_df = hashes.to_df()[hashes.var_names]
    if ignore_zero_median_cellhashes:
        hash_df = hash_df[[h for h in hash_df.columns.values if hash_df[h].median() > 0]]
    if log_transform_cellhashes:
        hash_df += 1
        hash_df = hash_df.apply(np.log2)
    if rename_cellhashes is None:
        rename_cellhashes = {}
    for h in hash_df:
        gex.obs[rename_cellhashes.get(h, h)] = hash_df[h]

    # process cellhash data
    if verbose:
        print('processing AgBC data...')
    agbc_df = agbcs.to_df()[agbcs.var_names]
    if ignore_zero_median_agbcs:
        agbc_df = agbc_df[[a for a in agbc_df.columns.values if agbc_df[a].median() > 0]]
    if log_transform_agbcs:
        agbc_df += 1
        agbc_df = agbc_df.apply(np.log2)
    if rename_agbcs is None:
        rename_agbcs = {}
    for a in agbc_df:
        gex.obs[rename_agbcs.get(a, a)] = agbc_df[a]
    
    # make feature dataframe
    if verbose:
        print('processing feature barcode data...')
    feature_df = features.to_df()[features.var_names]
    if ignore_zero_median_features:
        feature_df = feature_df[[h for h in feature_df.columns.values if feature_df[h].median() > 0]]
    if log_transform_features:
        feature_df += 1
        feature_df = feature_df.apply(np.log2)
    if rename_features is None:
        rename_features = {f: f'{f}{feature_suffix}' for f in feature_df}
    for f in feature_df:
        gex.obs[rename_features.get(f, f)] = feature_df[f]
    return gex












