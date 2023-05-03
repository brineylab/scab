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
from collections.abc import Iterable
import pathlib
import pickle
import re
import typing
from typing import Any, Callable, Collection, Dict, Literal, Optional, Union

import numpy as np

import scanpy as sc

import anndata
from anndata import AnnData

# from anndata.compat import Literal
from anndata._core.merge import StrategiesLiteral

import abstar
from abutils.core.pair import Pair, assign_pairs
from abutils.core.sequence import read_csv, read_fasta, read_json

from .vdj import merge_bcr, merge_tcr


def read_10x_mtx(
    mtx_path: str,
    *,
    bcr_file: Optional[str] = None,
    bcr_annot: Optional[str] = None,
    bcr_format: Literal["fasta", "delimited", "json"] = "fasta",
    bcr_delimiter: str = "\t",
    bcr_id_key: str = "sequence_id",
    bcr_sequence_key: str = "sequence",
    bcr_id_delimiter: str = "_",
    bcr_id_delimiter_num: int = 1,
    tcr_file: Optional[str] = None,
    tcr_annot: Optional[str] = None,
    tcr_format: Literal["fasta", "delimited", "json"] = "fasta",
    tcr_delimiter: str = "\t",
    tcr_id_key: str = "sequence_id",
    tcr_sequence_key: str = "sequence",
    tcr_id_delimiter: str = "_",
    tcr_id_delimiter_num: int = 1,
    chain_selection_func: Optional[Callable] = None,
    abstar_output_format: Literal["airr", "json"] = "airr",
    abstar_germ_db: str = "human",
    gex_only: bool = False,
    hashes: Optional[Iterable] = None,
    cellhash_regex: str = "cell ?hash",
    ignore_cellhash_case: bool = True,
    agbcs: Optional[Iterable] = None,
    agbc_regex: str = "agbc",
    ignore_agbc_case: bool = True,
    log_transform_cellhashes: bool = True,
    ignore_zero_quantile_cellhashes: bool = True,
    rename_cellhashes: Optional[Dict[str, str]] = None,
    log_transform_agbcs: bool = True,
    ignore_zero_quantile_agbcs: bool = True,
    rename_agbcs: Optional[Dict[str, str]] = None,
    log_transform_features: bool = True,
    ignore_zero_quantile_features: bool = True,
    rename_features: Optional[Dict[str, str]] = None,
    feature_suffix: str = "_FBC",
    cellhash_quantile: Union[float, int] = 0.95,
    agbc_quantile: Union[float, int] = 0.95,
    feature_quantile: Union[float, int] = 0.95,
    cache: bool = True,
    verbose: bool = True,
) -> AnnData:
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

    ignore_zero_quantile_agbcs : bool, default=True
        If ``True``, any AgBCs for which the `agbc_quantile`
        percentile have a count of zero are ignored. Default is ``True`` and the default
        `agbc_quantile` is ``0.95``, resulting in AgBCs with zero counts for the 95th
        percentile being ignored.

    ignore_zero_quantile_features : bool, default=True
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
    adata = sc.read_10x_mtx(mtx_path, gex_only=False, cache=cache)
    gex = adata[:, adata.var.feature_types == "Gene Expression"]

    # process BCR/TCR data
    if bcr_file is not None:
        gex = merge_bcr(
            adata=gex,
            bcr_file=bcr_file,
            bcr_annot=bcr_annot,
            bcr_format=bcr_format,
            bcr_delimiter=bcr_delimiter,
            bcr_id_key=bcr_id_key,
            bcr_sequence_key=bcr_sequence_key,
            bcr_id_delimiter=bcr_id_delimiter,
            bcr_id_delimiter_num=bcr_id_delimiter_num,
            chain_selection_func=chain_selection_func,
            abstar_output_format=abstar_output_format,
            abstar_germ_db=abstar_germ_db,
            verbose=verbose,
        )
    if tcr_file is not None:
        gex = merge_tcr(
            adata=gex,
            tcr_file=tcr_file,
            tcr_annot=tcr_annot,
            tcr_format=tcr_format,
            tcr_delimiter=tcr_delimiter,
            tcr_id_key=tcr_id_key,
            tcr_sequence_key=tcr_sequence_key,
            tcr_id_delimiter=tcr_id_delimiter,
            tcr_id_delimiter_num=tcr_id_delimiter_num,
            chain_selection_func=chain_selection_func,
            abstar_output_format=abstar_output_format,
            abstar_germ_db=abstar_germ_db,
            verbose=verbose,
        )

    if gex_only:
        return gex
    non_gex = adata[:, adata.var.feature_types != "Gene Expression"]
    if non_gex.shape[1] == 0:
        return gex

    # parse out features and cellhashes
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


def read(h5ad_file: Union[str, pathlib.Path]) -> AnnData:
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


def write(adata: AnnData, h5ad_file: Union[str, pathlib.Path]):
    """
    Serialized and writes an ``AnnData`` object to disk in ``h5ad`` format. Similar to
    ``scanpy.write()``, except that ``scanpy`` does not support serializing BCR/TCR data. This
    function serializes ``abutils.Pair`` objects stored in either ``adata.obs.bcr`` or
    ``adata.obs.tcr`` using ``pickle`` prior to writing the ``AnnData`` object to disk.

    Parameters
    ----------

    adata
        An ``AnnData`` object containing gene expression, feature barcode and
        VDJ data. ``scab.read_10x_mtx()`` can be used to construct a multi-omics ``AnnData`` object
        from raw CellRanger outputs.

    h5ad_file
        Path to the output file. The output will be written in ``h5ad`` format and must
        include ``'.h5ad'`` as the file extension. If it is not included, the extension will automatically
        be added.
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


def save(adata: AnnData, h5ad_file: Union[str, pathlib.Path]):
    """
    Serialized and writes an ``AnnData`` object to disk in ``h5ad`` format. Similar to
    ``scanpy.write()``, except that ``scanpy`` does not support serializing BCR/TCR data. This
    function serializes ``abutils.Pair`` objects stored in either ``adata.obs.bcr`` or
    ``adata.obs.tcr`` using ``pickle`` prior to writing the ``AnnData`` object to disk.

    Parameters
    ----------

    adata
        An ``AnnData`` object containing gene expression, feature barcode and
        VDJ data. ``scab.read_10x_mtx()`` can be used to construct a multi-omics ``AnnData`` object
        from raw CellRanger outputs.

    h5ad_file
        Path to the output file. The output will be written in ``h5ad`` format and must
        include ``'.h5ad'`` as the file extension. If it is not included, the extension will automatically
        be added.
    """
    write(adata, h5ad_file)


def concat(
    adatas: Union[Collection[AnnData], "typing.Mapping[str, AnnData]"],
    *,
    axis: Literal[0, 1] = 0,
    join: Literal["inner", "outer"] = "inner",
    merge: Union[StrategiesLiteral, Callable, None] = None,
    uns_merge: Union[StrategiesLiteral, Callable, None] = "unique",
    label: Optional[str] = None,
    keys: Optional[Collection] = None,
    index_unique: Optional[str] = None,
    fill_value: Optional[Any] = None,
    pairwise: bool = False,
    obs_names_make_unique: bool = True,
) -> AnnData:
    """Concatenates AnnData objects using ``anndata.concat()``.
    Documentation was copied almost verbatim from the ``anndata.concat()`` `docstring`_.

    The only major difference is that the default for `uns_merge` has been changed from
    ``None`` (which doesn't merge any of the data in ``adata.uns``) to ``'unique'``, which
    only merges ``adata.uns`` elements for which there is only one possible value.

    Parameters
    ----------

    adatas
        The objects to be concatenated. If a Mapping is passed, keys are used for the `keys`
        argument and values are concatenated.

    axis
        Which axis to concatenate along. ``0`` is row-wise, ``1`` is column-wise.

    join
        How to align values when concatenating. If ``"outer"``, the union of the other axis
        is taken. If ``"inner"``, the intersection is taken. For example::

    merge
        How elements not aligned to the axis being concatenated along are selected.
        Currently implemented strategies include:
        * ``None``: No elements are kept.
        * ``"same"``: Elements that are the same in each of the objects.
        * ``"unique"``: Elements for which there is only one possible value.
        * ``"first"``: The first element seen at each from each position.
        * ``"only"``: Elements that show up in only one of the objects.

    uns_merge
        How the elements of ``.uns`` are selected. Uses the same set of strategies as
        the `merge` argument, except applied recursively.

    label
        Column in axis annotation (i.e. ``.obs`` or ``.var``) to place batch information in.
        If it's None, no column is added.

    keys
        Names for each object being added. These values are used for column values for
        `label` or appended to the index if `index_unique` is not ``None``. Defaults to
        incrementing integer labels.

    index_unique
        Whether to make the index unique by using the keys. If provided, this
        is the delimeter between "{orig_idx}{index_unique}{key}". When ``None``,
        the original indices are kept.

    fill_value
        When ``join="outer"``, this is the value that will be used to fill the introduced
        indices. By default, sparse arrays are padded with zeros, while dense arrays and
        DataFrames are padded with missing values.

    pairwise
        Whether pairwise elements along the concatenated dimension should be included.
        This is False by default, since the resulting arrays are often not meaningful.

    obs_names_make_unique
        If ``True``, will call ``obs_names_make_unique()`` on the concatenated ``AnnData``
        object prior to returning. Default is ``True``.


    Notes
    -----
    .. warning::
        If you use ``join='outer'`` this fills 0s for sparse data when
        variables are absent in a batch. Use this with care. Dense data is
        filled with ``NaN``.


    Examples
    --------
    Preparing example objects
    >>> import anndata as ad, pandas as pd, numpy as np
    >>> from scipy import sparse
    >>> a = ad.AnnData(
    ...     X=sparse.csr_matrix(np.array([[0, 1], [2, 3]])),
    ...     obs=pd.DataFrame({"group": ["a", "b"]}, index=["s1", "s2"]),
    ...     var=pd.DataFrame(index=["var1", "var2"]),
    ...     varm={"ones": np.ones((2, 5)), "rand": np.random.randn(2, 3), "zeros": np.zeros((2, 5))},
    ...     uns={"a": 1, "b": 2, "c": {"c.a": 3, "c.b": 4}},
    ... )
    >>> b = ad.AnnData(
    ...     X=sparse.csr_matrix(np.array([[4, 5, 6], [7, 8, 9]])),
    ...     obs=pd.DataFrame({"group": ["b", "c"], "measure": [1.2, 4.3]}, index=["s3", "s4"]),
    ...     var=pd.DataFrame(index=["var1", "var2", "var3"]),
    ...     varm={"ones": np.ones((3, 5)), "rand": np.random.randn(3, 5)},
    ...     uns={"a": 1, "b": 3, "c": {"c.b": 4}},
    ... )
    >>> c = ad.AnnData(
    ...     X=sparse.csr_matrix(np.array([[10, 11], [12, 13]])),
    ...     obs=pd.DataFrame({"group": ["a", "b"]}, index=["s1", "s2"]),
    ...     var=pd.DataFrame(index=["var3", "var4"]),
    ...     uns={"a": 1, "b": 4, "c": {"c.a": 3, "c.b": 4, "c.c": 5}},
    ... )

    Concatenating along different axes

    >>> ad.concat([a, b]).to_df()
        var1  var2
    s1   0.0   1.0
    s2   2.0   3.0
    s3   4.0   5.0
    s4   7.0   8.0
    >>> ad.concat([a, c], axis=1).to_df()
        var1  var2  var3  var4
    s1   0.0   1.0  10.0  11.0
    s2   2.0   3.0  12.0  13.0

    Inner and outer joins

    >>> inner = ad.concat([a, b])  # Joining on intersection of variables
    >>> inner
    AnnData object with n_obs × n_vars = 4 × 2
        obs: 'group'
    >>> (inner.obs_names, inner.var_names)
    (Index(['s1', 's2', 's3', 's4'], dtype='object'),
    Index(['var1', 'var2'], dtype='object'))
    >>> outer = ad.concat([a, b], join="outer") # Joining on union of variables
    >>> outer
    AnnData object with n_obs × n_vars = 4 × 3
        obs: 'group', 'measure'
    >>> outer.var_names
    Index(['var1', 'var2', 'var3'], dtype='object')
    >>> outer.to_df()  # Sparse arrays are padded with zeroes by default
        var1  var2  var3
    s1   0.0   1.0   0.0
    s2   2.0   3.0   0.0
    s3   4.0   5.0   6.0
    s4   7.0   8.0   9.0

    Keeping track of source objects

    >>> ad.concat({"a": a, "b": b}, label="batch").obs
       group batch
    s1     a     a
    s2     b     a
    s3     b     b
    s4     c     b
    >>> ad.concat([a, b], label="batch", keys=["a", "b"]).obs  # Equivalent to previous
       group batch
    s1     a     a
    s2     b     a
    s3     b     b
    s4     c     b
    >>> ad.concat({"a": a, "b": b}, index_unique="-").obs
         group
    s1-a     a
    s2-a     b
    s3-b     b
    s4-b     c

    Combining values not aligned to axis of concatenation

    >>> ad.concat([a, b], merge="same")
    AnnData object with n_obs × n_vars = 4 × 2
        obs: 'group'
        varm: 'ones'
    >>> ad.concat([a, b], merge="unique")
    AnnData object with n_obs × n_vars = 4 × 2
        obs: 'group'
        varm: 'ones', 'zeros'
    >>> ad.concat([a, b], merge="first")
    AnnData object with n_obs × n_vars = 4 × 2
        obs: 'group'
        varm: 'ones', 'rand', 'zeros'
    >>> ad.concat([a, b], merge="only")
    AnnData object with n_obs × n_vars = 4 × 2
        obs: 'group'
        varm: 'zeros'

    The same merge strategies can be used for elements in `.uns`

    >>> dict(ad.concat([a, b, c], uns_merge="same").uns)
    {'a': 1, 'c': {'c.b': 4}}
    >>> dict(ad.concat([a, b, c], uns_merge="unique").uns)
    {'a': 1, 'c': {'c.a': 3, 'c.b': 4, 'c.c': 5}}
    >>> dict(ad.concat([a, b, c], uns_merge="only").uns)
    {'c': {'c.c': 5}}
    >>> dict(ad.concat([a, b, c], uns_merge="first").uns)
    {'a': 1, 'b': 2, 'c': {'c.a': 3, 'c.b': 4, 'c.c': 5}}


    .. _docstring
        https://github.com/scverse/anndata/blob/master/anndata/_core/merge.py#L628
    """
    adata = anndata.concat(
        adatas,
        axis=axis,
        join=join,
        merge=merge,
        uns_merge=uns_merge,
        label=label,
        keys=keys,
        index_unique=index_unique,
        fill_value=fill_value,
        pairwise=pairwise,
    )

    if obs_names_make_unique:
        adata.obs_names_make_unique()
    return adata
