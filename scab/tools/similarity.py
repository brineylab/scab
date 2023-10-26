#!/usr/bin/env python
# filename: similarity.py


#
# Copyright (c) 2023 Bryan Briney
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


from typing import Iterable, Optional, Union

from natsort import natsorted

import abutils
from abutils.tools.similarity import RepertoireSimilarity, RepertoireSimilarities

import scanpy as sc

import anndata


def repertoire_similarity(
    adata: anndata.AnnData,
    batch_key: str = "batch",
    batches: Optional[Iterable[str]] = None,
    minimum_batch_size: Optional[int] = None,
    method: str = "morisita-horn",
    features: Union[str, Iterable[str], None] = None,
    n_iters: int = 1,
    subsample_size: Optional[int] = None,
    sample_with_replacement: bool = False,
    pairs_only: bool = False,
    chain: Optional[str] = None,
    force_self_comparisons: bool = False,
    seed: Union[int, float, str, None] = None,
) -> Union[float, RepertoireSimilarity, RepertoireSimilarities]:
    """
    Compute the pairwise similarity between two or more repertoires.

    Parameters
    ----------
    adata : anndata.AnnData
        ``AnnData`` object containing BCR sequence data.

    batch_key : str, optional
        The key in ``adata.obs`` containing batch information. Default is ``"batch"``.

    batches : list, optional
        A list of batches to use for similarity calculation. If ``None``, all batches
        in ``adata.obs[batch_key]`` will be used. Default is ``None``.

    minimum_batch_size : int, optional
        The minimum number of sequences required in a batch to be included in the
        similarity calculation. Default is ``None``, which will include all batches
        regardless of size.

    method : str, optional
        The similarity method to use. Default is ``"morisita-horn"``.

    features : str, list, optional
        The features to use for similarity calculation. Default is ``["v_gene", "j_gene", "cdr3_length"]``.

    n_iters : int, optional
        The number of iterations to perform. Default is ``1``.

    subsample_size : int, optional
        The number of sequences to subsample from each repertoire. If ``None``, the smallest repertoire
        will be used as the subsample size. Default is ``None``.

    sample_with_replacement : bool, optional
        Whether to subsample with replacement. Default is ``False``.

    pairs_only : bool, optional
        Whether to use only paired sequences. Default is ``False``.

    chain : str, optional
        The chain to use for similarity calculation. Default is ``None``, which will use all chains.

    force_self_comparisons : bool, optional
        Whether to force self-comparisons. Only used if exactly two repertoires are provided.
        Default is ``False``, which performs only a single pairwise comparison when exactly
        two repertoires are provided. If more than two repertoires are provided, all pairwise
        comparisons will be performed, including self-comparisons.

    seed : int, float, str, optional
        The seed to use for random number generation. Default is ``None``.

    Returns
    -------
    float or RepertoireSimilarity or RepertoireSimilarities
        If only two batches are provided and `n_iters` is ``1``, the similarity value will be returned as a float.
        If only two batches are provided and `n_iters` is greater than ``1``, a ``RepertoireSimilarity`` object will be
        returned. If more than two batches are provided, a ``RepertoireSimilarities`` object will be
        returned.

    """
    adata = adata.copy()
    # batches
    if batches is None:
        batches = natsorted(adata.obs[batch_key].unique())
    if len(batches) < 2:
        raise ValueError(f'Only one batch found in "adata.obs[{batch_key}]"')
    # features
    if features is None:
        features = ["v_gene", "j_gene", "cdr3_length"]
    elif isinstance(features, str):
        features = [features]
    if len(features) == 0:
        raise ValueError(
            "No features provided for similarity calculation.",
            "Please provide at least one feature.",
        )
    # minimum reperoire size
    if minimum_batch_size is None:
        minimum_batch_size = 0
    # parse repertoires
    repertoires = []
    for batch in batches:
        repertoire = adata[adata.obs[batch_key] == batch].obs.bcr.to_list()
        if len(repertoire) < minimum_batch_size:
            continue
        repertoires.append(repertoire)
    if len(repertoires) < 2:
        raise ValueError(
            f"Only {len(repertoires)} repertoires found with at least {minimum_batch_size} sequences"
        )
    # similarity
    similarity = abutils.tl.repertoire_similarity(
        repertoires,
        method=method,
        features=features,
        n_iters=n_iters,
        subsample_size=subsample_size,
        sample_with_replacement=sample_with_replacement,
        pairs_only=pairs_only,
        chain=chain,
        force_self_comparisons=force_self_comparisons,
        seed=seed,
    )
    return similarity
