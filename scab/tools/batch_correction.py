#!/usr/bin/env python
# filename: batch_correction.py


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


import os
import re
import sys
from typing import Optional, Union, Iterable

from natsort import natsorted

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scanpy as sc

import anndata

import statsmodels.api as sm

from .embeddings import pca, umap


__all__ = ["combat", "harmony", "mnn", "scanorama"]


def combat(
    adata: anndata.AnnData,
    batch_key: str = "batch",
    covariates: Optional[Iterable] = None,
    post_correction_umap: bool = True,
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Batch effect correction using ComBat_ [Johnson07]_.

    .. seealso::
        | W. Evan Johnson, Cheng Li, Ariel Rabinovic
        | Adjusting batch effects in microarray expression data using empirical Bayes methods
        | *Biostatistics* 2007, doi: 10.1093/biostatistics/kxj037


    Parameters
    ----------

    adata : anndata.AnnData
        ``AnnData`` object containing gene counts data.

    batch_key : str, default='batch'
        Name of the column in adata.obs that corresponds to the batch.

    covariates : iterable object, optional
        List of additional covariates besides the batch variable such as adjustment variables
        or biological condition. Not including covariates may lead to the removal of real
        biological signal.

    post_correction_umap : bool, default=True
        If ``True``, UMAP will be computed on the post-integration data using
        ``scab.tl.umap()``.

    verbose : bool, default=True
        If ``True``, print progress.


    Returns
    -------
    adata : ``anndata.AnnData``


    .. _ComBat:
        https://github.com/brentp/combat.py

    """
    if verbose:
        print("")
        print("------")
        print("COMBAT")
        print("------")
    adata_combat = sc.AnnData(X=adata.raw.X, var=adata.raw.var, obs=adata.obs)
    adata_combat.layers = adata.layers
    adata_combat.raw = adata_combat
    # run combat
    sc.pp.combat(adata_combat, key=batch_key, covariates=covariates)
    sc.pp.highly_variable_genes(adata_combat)
    # UMAP, if desired
    if post_correction_umap:
        if verbose:
            print("")
        adata = umap(adata, verbose=verbose)
    return adata_combat


def harmony(
    adata: anndata.AnnData,
    batch_key: str = "batch",
    adjusted_basis: str = "X_pca_harmony",
    n_dim: int = 50,
    force_pca: bool = False,
    post_correction_umap: bool = True,
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Data integration and batch correction using `mutual nearest neighbors`_ [Haghverdi19]_. Uses the
    ``scanpy.external.pp.mnn_correct()`` function.

    .. seealso::
        | Laleh Haghverdi, Aaron T L Lun, Michael D Morgan & John C Marioni
        | Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors
        | *Nature Biotechnology* 2019, doi: 10.1038/nbt.4091

    Parameters
    ----------
    adata : anndata.AnnData
        ``AnnData`` object containing gene counts data.

    batch_key : str, default='batch'
        Name of the column in adata.obs that corresponds to the batch.

    adjusted_basis : str, default='X_pca_harmony'
        Name of the basis in ``adata.obsm`` that will be added by `harmony`.

    n_dim : int, default=50
        Number of dimensions to use for PCA.

    force_pca : bool, default=False
        If ``True``, PCA will be run even if ``adata.obsm['X_pca']`` already exists.

    post_correction_umap : bool, default=True
        If ``True``, UMAP will be computed on the batch corrected data using ``scab.tl.umap()``.

    verbose : bool, default=True
        If ``True``, print progress.


    Returns
    -------
    adata : ``anndata.AnnData``


    .. _mutual nearest neighbors:
        https://github.com/chriscainx/mnnpy

    """
    adata = adata.copy()
    if verbose:
        print("")
        print("--------")
        print("HARMONY")
        print("--------")
    # PCA must be run first
    if "X_pca" not in adata.obsm or force_pca:
        adata = pca(adata, n_pcs=n_dim)
    sc.external.pp.harmony_integrate(
        adata, key=batch_key, basis="X_pca", adjusted_basis=adjusted_basis
    )
    # UMAP, if desired
    if post_correction_umap:
        if verbose:
            print("")
        adata = umap(adata, use_rep=adjusted_basis, verbose=verbose)
    return adata


def mnn(
    adata: anndata.AnnData,
    batch_key: str = "batch",
    min_hvg_batches: int = 1,
    post_correction_umap: bool = True,
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Data integration and batch correction using `mutual nearest neighbors`_ [Haghverdi19]_. Uses the
    ``scanpy.external.pp.mnn_correct()`` function.

    .. seealso::
        | Laleh Haghverdi, Aaron T L Lun, Michael D Morgan & John C Marioni
        | Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors
        | *Nature Biotechnology* 2019, doi: 10.1038/nbt.4091

    Parameters
    ----------
    adata : anndata.AnnData
        ``AnnData`` object containing gene counts data.

    batch_key : str, default='batch'
        Name of the column in adata.obs that corresponds to the batch.

    min_hvg_batches : int, default=1
        Minimum number of batches in which highly variable genes are found in order to be included
        in the list of genes used for batch correction. Default is ``1``, which results in the use
        of all HVGs found in any batch.

    post_correction_umap : bool, default=True
        If ``True``, UMAP will be computed on the batch corrected data using ``scab.tl.umap()``.

    verbose : bool, default=True
        If ``True``, print progress.


    Returns
    -------
    adata : ``anndata.AnnData``


    .. _mutual nearest neighbors:
        https://github.com/chriscainx/mnnpy

    """
    if verbose:
        print("")
        print("---")
        print("MNN")
        print("---")
    adata_mnn = adata.raw.to_adata()
    adata_mnn.layers = adata.layers
    # variable genes
    sc.pp.highly_variable_genes(
        adata_mnn, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key=batch_key
    )
    var_select = adata_mnn.var.highly_variable_nbatches >= min_hvg_batches
    var_genes = var_select.index[var_select]
    # split per batch into new objects.
    batch_names = adata_mnn.obs[batch_key].cat.categories.tolist()
    batches = [adata_mnn[adata_mnn.obs[batch_key] == b] for b in batch_names]
    # run MNN correction
    cdata = sc.external.pp.mnn_correct(
        *batches,
        svd_dim=50,
        batch_key=batch_key,
        save_raw=True,
        var_subset=var_genes,
    )
    corr_data = cdata[0][:, var_genes]
    # UMAP, if desired
    if post_correction_umap:
        if verbose:
            print("")
        corr_data = umap(corr_data, verbose=verbose)
    return corr_data


def scanorama(
    adata: anndata.AnnData,
    batch_key: str = "batch",
    scanorama_key: str = "X_Scanorama",
    n_dim: int = 50,
    post_correction_umap: bool = True,
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Batch correction using Scanorama_ [Hie19]_.

    .. seealso::
        | Brian Hie, Bryan Bryson, and Bonnie Berger
        | Efficient integration of heterogeneous single-cell transcriptomes using Scanorama
        | *Nature Biotechnology* 2019, doi: 10.1038/s41587-019-0113-3

    Parameters
    ----------

    adata : anndata.AnnData
        ``AnnData`` object containing gene counts data.

    batch_key : str, default='batch'
        Name of the column in ``adata.obs`` that corresponds to the batch.

    post_correction_umap : bool, default=True
        If ``True``, UMAP will be computed on the batch corrected data using ``scab.tl.umap()``.

    verbose : bool, default=True
        If ``True``, print progress.


    Returns
    -------
    adata : ``anndata.AnnData``


    .. _Scanorama:
        https://github.com/brianhie/scanorama

    """
    import scanorama

    if verbose:
        print("")
        print("SCANORAMA")
        print("---------")
    # make sure obs names are unique, since we'll need them to incorporate
    # Scanorama integrations into the original adata object
    adata.obs_names_make_unique()
    # PCA must be run first
    if "X_pca" not in adata.obsm.keys():
        adata = pca(adata, n_pcs=n_dim)
    # Scanorama needs raw gene counts, not normalized counts
    adata_scanorama = adata.raw.to_adata()
    adata_scanorama.layers = adata.layers
    # Scanorama needs the datasets to be dividied into individual batches
    # rather than just passing a batch key
    batch_names = adata_scanorama.obs[batch_key].cat.categories.tolist()
    adatas = [adata_scanorama[adata_scanorama.obs[batch_key] == b] for b in batch_names]
    # run Scanorama
    scanorama.integrate_scanpy(adatas, dimred=n_dim)
    # add the Scanorama to the "complete" adata object
    obs_names = [ad.obs_names for ad in adatas]
    integrations = [ad.obsm["X_scanorama"] for ad in adatas]
    integrate_dict = {}
    for names, integration in zip(obs_names, integrations):
        for n, i in zip(names, integration):
            integrate_dict[n] = i
    all_s = np.array([integrate_dict[o] for o in adata.obs_names])
    adata.obsm[scanorama_key] = all_s
    if post_correction_umap:
        if verbose:
            print("")
        adata = umap(adata, use_rep=scanorama_key, verbose=verbose)
    return adata
