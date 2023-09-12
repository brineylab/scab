#!/usr/bin/env python
# filename: tl.py


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


from natsort import natsorted

import numpy as np
import pandas as pd

from scipy import stats
from scipy.signal import argrelextrema

import statsmodels.api as sm

from .tools.batch_correction import *
from .tools.cellhashes import *
from .tools.embeddings import *
from .tools.specificity import *


# def pca(
#     adata: anndata.AnnData,
#     solver: str = "arpack",
#     n_pcs: int = 40,
#     ignore_ig: bool = True,
#     verbose: bool = True,
# ) -> anndata.AnnData:
#     """
#     Performs PCA, neighborhood graph construction and UMAP embedding.
#     PAGA is optional, but is performed by default.

#     Parameters
#     ----------

#     adata : anndata.AnnData
#         ``AnnData`` object containing gene counts data.

#     solver : str, default='arpack'
#         Solver to use for the PCA.

#     n_pcs : int, default=40
#         Number of principal components to use when computing the neighbor graph.
#         Although the default value is generally appropriate, it is sometimes useful
#         to empirically determine the optimal value for `n_pcs`.

#     ignore_ig : bool, default=True
#         Ignores immunoglobulin V, D and J genes when computing the PCA.


#     Returns
#     -------
#     adata : ``anndata.AnnData``


#     .. _PAGA:
#         https://github.com/theislab/paga
#     .. _computing neighbors:
#         https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html

#     """
#     if verbose:
#         print("performing PCA...")
#     if ignore_ig:
#         _adata = adata.copy()
#         _adata.var["highly_variable"] = _adata.var["highly_variable"] & ~(
#             _adata.var["ig"]
#         )
#         sc.tl.pca(_adata, svd_solver=solver, n_comps=n_pcs, use_highly_variable=True)
#         adata.obsm["X_pca"] = _adata.obsm["X_pca"]
#         adata.varm["PCs"] = _adata.varm["PCs"]
#         adata.uns["pca"] = {
#             "variance_ratio": _adata.uns["pca"]["variance_ratio"],
#             "variance": _adata.uns["pca"]["variance"],
#         }
#     else:
#         sc.tl.pca(adata, svd_solver=solver, n_comps=n_pcs, use_highly_variable=True)
#     return adata


# def umap(
#     adata: anndata.AnnData,
#     solver: str = "arpack",
#     n_neighbors: int = 20,
#     n_pcs: int = 40,
#     force_pca: bool = False,
#     ignore_ig: bool = True,
#     paga: bool = True,
#     use_rna_velocity: bool = False,
#     use_rep: Optional[str] = None,
#     random_state: Union[int, float, str] = 42,
#     resolution: float = 1.0,
#     verbose: bool = True,
# ) -> anndata.AnnData:
#     """
#     Performs PCA, neighborhood graph construction and UMAP embedding.
#     PAGA is optional, but is performed by default.

#     Parameters
#     ----------

#     adata : anndata.AnnData
#         ``AnnData`` object containing gene counts data.

#     solver : str, default='arpack'
#         Solver to use for the PCA.

#     n_neighbors : int, default=10
#         Number of neighbors to calculate for the neighbor graph.

#     n_pcs : int, default=40
#         Number of principal components to use when computing the neighbor graph.
#         Although the default value is generally appropriate, it is sometimes useful
#         to empirically determine the optimal value for `n_pcs`.

#     force_pca : bool, default=False
#         Construct the PCA even if it has already been constructed (``"X_pcs"`` exists
#         in ``adata.obsm``). Default is ``False``, which will use an existing PCA.

#     ignore_ig : bool, default=True
#         Ignores immunoglobulin V, D and J genes when computing the PCA.

#     paga : bool, default=True
#         If ``True``, performs partition-based graph abstraction (PAGA_) prior to
#         UMAP embedding.

#     use_rna_velocity : bool, default=False
#         If ``True``, uses RNA velocity information to compute PAGA. If ``False``,
#         this option is ignored.

#     use_rep : str, optional
#         Representation to use when `computing neighbors`_. For example, if data have
#         been batch normalized with ``scanorama``, the representation
#         should be ``'Scanorama'``. If not provided, ``scanpy``'s default
#         representation is used.

#     random_state : int, optional
#         Seed for the random state used by ``sc.tl.umap``.

#     resolution : float, default=1.0
#         Resolution for Leiden clustering.


#     Returns
#     -------
#     adata : ``anndata.AnnData``


#     .. _PAGA:
#         https://github.com/theislab/paga
#     .. _computing neighbors:
#         https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html

#     """
#     if verbose:
#         print("")
#         print("UMAP EMBEDDING")
#         print("--------------")
#     # PCA
#     if any([force_pca, "X_pca" not in adata.obsm_keys()]):
#         if verbose:
#             print("  - computing PCA")
#         adata = pca(
#             adata, solver=solver, n_pcs=n_pcs, ignore_ig=ignore_ig, verbose=False
#         )
#     # neighbors
#     if verbose:
#         print("  - calculating neighbors")
#     sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
#     # leiden clustering
#     if verbose:
#         print("  - leiden clustering")
#     sc.tl.leiden(adata, resolution=resolution)
#     # umap (with or without PAGA first)
#     if paga:
#         if verbose:
#             print("  - paga")
#         sc.tl.paga(adata, use_rna_velocity=use_rna_velocity)
#         if use_rna_velocity:
#             adata.uns["paga"]["connectivities"] = adata.uns["paga"][
#                 "transitions_confidence"
#             ]
#         sc.pl.paga(adata, plot=False)
#         init_pos = "paga"
#     else:
#         init_pos = "spectral"
#     if verbose:
#         print("  - umap")
#     sc.tl.umap(adata, init_pos=init_pos, random_state=random_state)
#     return adata


# def dimensionality_reduction(
#     adata: anndata.AnnData,
#     solver: str = "arpack",
#     n_neighbors=20,
#     n_pcs=40,
#     ignore_ig=True,
#     paga=True,
#     use_rna_velocity=False,
#     use_rep=None,
#     random_state=42,
#     resolution=1.0,
#     verbose=True,
# ):
#     """
#     Performs PCA, neighborhood graph construction and UMAP embedding.
#     PAGA is optional, but is performed by default.

#     Parameters
#     ----------

#     adata : anndata.AnnData)
#         ``AnnData`` object containing gene counts data.

#     solver : str, default='arpack'
#         Solver to use for the PCA.

#     n_neighbors : int, default=10
#         Number of neighbors to calculate for the neighbor graph.

#     n_pcs : int, default=40
#         Number of principal components to use when computing the neighbor graph.
#         Although the default value is generally appropriate, it is sometimes useful
#         to empirically determine the optimal value for `n_pcs`.

#     paga : bool, default=True
#         If ``True``, performs partition-based graph abstraction (PAGA_) prior to
#         UMAP embedding.

#     use_rna_velocity : bool, default=False
#         If ``True``, uses RNA velocity information to compute PAGA. If ``False``,
#         this option is ignored.

#     use_rep : str, optional
#         Representation to use when `computing neighbors`_. For example, if data have
#         been batch normalized with ``scanorama``, the representation
#         should be ``'Scanorama'``. If not provided, ``scanpy``'s default
#         representation is used.

#     random_state : int, optional
#         Seed for the random state used by ``sc.tl.umap``.

#     resolution : float, default=1.0
#         Resolution for Leiden clustering.


#     Returns
#     -------
#     adata : ``anndata.AnnData``


#     .. _PAGA:
#         https://github.com/theislab/paga
#     .. _computing neighbors:
#         https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html

#     """
#     if verbose:
#         print("performing PCA...")
#     if ignore_ig:
#         _adata = adata.copy()
#         _adata.var["highly_variable"] = _adata.var["highly_variable"] & ~(
#             _adata.var["ig"]
#         )
#         sc.tl.pca(_adata, svd_solver=solver, n_comps=n_pcs, use_highly_variable=True)
#         adata.obsm["X_pca"] = _adata.obsm["X_pca"]
#         adata.varm["PCs"] = _adata.varm["PCs"]
#         adata.uns["pca"] = {
#             "variance_ratio": _adata.uns["pca"]["variance_ratio"],
#             "variance": _adata.uns["pca"]["variance"],
#         }
#     else:
#         sc.tl.pca(adata, svd_solver=solver, n_comps=n_pcs, use_highly_variable=True)
#     if verbose:
#         print("calculating neighbors...")
#     sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
#     if verbose:
#         print("leiden clustering...")
#     sc.tl.leiden(adata, resolution=resolution)
#     if paga:
#         if verbose:
#             print("paga...")
#         #         solid_edges = 'transitions_confidence' if use_rna_velocity else 'connectivities'
#         sc.tl.paga(adata, use_rna_velocity=use_rna_velocity)
#         if use_rna_velocity:
#             adata.uns["paga"]["connectivities"] = adata.uns["paga"][
#                 "transitions_confidence"
#             ]
#         sc.pl.paga(adata, plot=False)
#         if verbose:
#             print("umap...")
#         sc.tl.umap(
#             adata,
#             init_pos="paga",
#             random_state=random_state,
#         )
#     else:
#         if verbose:
#             print("umap...")
#         sc.tl.umap(adata, random_state=random_state)
#     return adata


# def combat(adata, batch_key="batch", covariates=None, dim_red=True):
#     """
#     Batch effect correction using ComBat_ [Johnson07]_.

#     .. seealso::
#         | W. Evan Johnson, Cheng Li, Ariel Rabinovic
#         | Adjusting batch effects in microarray expression data using empirical Bayes methods
#         | *Biostatistics* 2007, doi: 10.1093/biostatistics/kxj037


#     Parameters
#     ----------

#     adata : anndata.AnnData
#         ``AnnData`` object containing gene counts data.

#     batch_key : str, default='batch'
#         Name of the column in adata.obs that corresponds to the batch.

#     covariates : iterable object, optional
#         List of additional covariates besides the batch variable such as adjustment variables
#         or biological condition. Not including covariates may lead to the removal of real
#         biological signal.

#     dim_red : bool, default=True
#         If ``True``, dimentionality reduction will be performed on the post-integration data using
#         ``scab.tl.dimensionality_reduction()``.

#     Returns
#     -------
#     adata : ``anndata.AnnData``


#     .. _ComBat:
#         https://github.com/brentp/combat.py

#     """
#     adata_combat = sc.AnnData(X=adata.raw.X, var=adata.raw.var, obs=adata.obs)
#     adata_combat.layers = adata.layers
#     adata_combat.raw = adata_combat
#     # run combat
#     sc.pp.combat(adata_combat, key=batch_key, covariates=covariates)
#     sc.pp.highly_variable_genes(adata_combat)
#     if dim_red:
#         adata_combat = dimensionality_reduction(adata_combat)
#     return adata_combat


# def mnn(adata, batch_key="batch", min_hvg_batches=1, dim_red=True):
#     """
#     Data integration and batch correction using `mutual nearest neighbors`_ [Haghverdi19]_. Uses the
#     ``scanpy.external.pp.mnn_correct()`` function.

#     .. seealso::
#         | Laleh Haghverdi, Aaron T L Lun, Michael D Morgan & John C Marioni
#         | Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors
#         | *Nature Biotechnology* 2019, doi: 10.1038/nbt.4091

#     Parameters
#     ----------
#     adata : anndata.AnnData
#         ``AnnData`` object containing gene counts data.

#     batch_key : str, default='batch'
#         Name of the column in adata.obs that corresponds to the batch.

#     min_hvg_batches : int, default=1
#         Minimum number of batches in which highly variable genes are found in order to be included
#         in the list of genes used for batch correction. Default is ``1``, which results in the use
#         of all HVGs found in any batch.

#     dim_red : bool, default=True
#         If ``True``, dimentionality reduction will be performed on the post-integration data using
#         ``scab.tl.dimensionality_reduction()``.


#     Returns
#     -------
#     adata : ``anndata.AnnData``


#     .. _mutual nearest neighbors:
#         https://github.com/chriscainx/mnnpy

#     """
#     adata_mnn = adata.raw.to_adata()
#     adata_mnn.layers = adata.layers
#     # variable genes
#     sc.pp.highly_variable_genes(
#         adata_mnn, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key=batch_key
#     )
#     var_select = adata_mnn.var.highly_variable_nbatches >= min_hvg_batches
#     var_genes = var_select.index[var_select]
#     # split per batch into new objects.
#     batches = adata_mnn.obs[batch_key].cat.categories.tolist()
#     alldata = {}
#     for batch in batches:
#         alldata[batch] = adata_mnn[adata_mnn.obs[batch_key] == batch,]
#     # run MNN correction
#     cdata = sc.external.pp.mnn_correct(
#         *[alldata[b] for b in batches],
#         svd_dim=50,
#         batch_key=batch_key,
#         save_raw=True,
#         var_subset=var_genes,
#     )
#     corr_data = cdata[0][:, var_genes]
#     if dim_red:
#         corr_data = dimensionality_reduction(corr_data)
#     return corr_data


# def scanorama(
#     adata, batch_key="batch", scanorama_key="X_Scanorama", n_dim=50, dim_red=True
# ):
#     """
#     Batch correction using Scanorama_ [Hie19]_.

#     .. seealso::
#         | Brian Hie, Bryan Bryson, and Bonnie Berger
#         | Efficient integration of heterogeneous single-cell transcriptomes using Scanorama
#         | *Nature Biotechnology* 2019, doi: 10.1038/s41587-019-0113-3

#     Parameters
#     ----------

#     adata : anndata.AnnData
#         ``AnnData`` object containing gene counts data.

#     batch_key : str, default='batch'
#         Name of the column in ``adata.obs`` that corresponds to the batch.

#     dim_red : bool, default=True
#         If ``True``, dimentionality reduction will be performed on the post-integration data using
#         ``scab.tl.dimensionality_reduction``.


#     Returns
#     -------
#     adata : ``anndata.AnnData``


#     .. _Scanorama:
#         https://github.com/brianhie/scanorama

#     """
#     import scanorama

#     # make sure obs names are unique, since we'll need them to incorporate
#     # Scanorama integrations into the original adata object
#     # Also, Scanorama needs the raw gene counts, not normalized
#     adata_scanorama = adata.raw.to_adata()
#     adata_scanorama.layers = adata.layers
#     # Scanorama needs the datasets to be dividied into individual batches
#     # rather than just passing a batch key
#     batch_names = adata_scanorama.obs[batch_key].cat.categories.tolist()
#     adatas = [adata_scanorama[adata_scanorama.obs[batch_key] == b] for b in batch_names]
#     # run Scanorama
#     scanorama.integrate_scanpy(adatas, dimred=n_dim)
#     # add the Scanorama to the "complete" adata object
#     obs_names = [ad.obs_names for ad in adatas]
#     integrations = [ad.obsm["X_Scanorama"] for ad in adatas]
#     integrate_dict = {}
#     for obs_name, integration in zip(obs_names, integrations):
#         for o, i in zip(obs_names, integration):
#             integrate_dict[o] = i
#     all_s = np.array([integrate_dict[o] for o in adata.obs_names])
#     adata.obsm["X_scanorama"] = all_s
#     return adata


# def classify_specificity(
#     adata,
#     raw,
#     agbcs=None,
#     groups=None,
#     rename=None,
#     percentile=0.997,
#     percentile_dict=None,
#     update=True,
#     uns_batch=None,
#     verbose=True,
# ):
#     """
#     Classifies BCR specificity using antigen barcodes (**AgBCs**). Thresholds are computed by
#     analyzing background AgBC UMI counts in empty droplets.

#     .. note::
#        In order to set accurate thresholds, we must remove all cell-containing droplets
#        from the ``raw`` counts matrix. Because ``adata`` comprises only cell-containing
#        droplets, we simply remove all of the droplet barcodes in ``adata`` from ``raw``.
#        Thus, it is **very important** that ``adata`` and ``raw`` are well matched.

#        For example, if processing a single Chromium reaction containing several multiplexed samples,
#        ``adata`` should contain all of the multiplexed samples, since the raw matrix produced
#        by CellRanger will also include all droplets in the reaction. If ``adata`` was missing
#        one or more samples, cell-containing droplets cannot accurately be removed from ``raw``
#        and classification accuracy will be adversely affected.

#     Parameters
#     ----------
#     adata : anndata.AnnData
#         Input ``AnnData`` object. Log2-normalized AgBC UMI counts should be found in
#         ``adata.obs``. If data was read using ``scab.read_10x_mtx()``, the resulting
#         ``AnnData`` object will already be correctly formatted.

#     raw : anndata.AnnData or str
#         Raw matrix data. Either a path to a directory containing the raw ``.mtx`` file
#         produced by CellRanger, or an ``anndata.AnnData`` object containing the raw
#         matrix data. As with `adata`, log2-normalized AgBC UMIs should be found at
#         ``raw.obs``.

#         .. tip::
#             If reading the raw counts matrix with ``scab.read_10x_mtx()``, it can be
#             helpful to include ``ignore_zero_quantile_agbcs=False``. In some cases with
#             very little AgBC background, AgBCs can be incorrectly removed from the raw
#             counts matrix.

#     agbcs : iterable object, optional
#         A list of AgBCs to be classified. Either `agbcs`` or `groups`` is required.
#         If both are provided, both will be used.

#     groups : dict, optional
#         A ``dict`` mapping specificity names to a list of one or more AgBCs. This
#         is particularly useful when multiple AgBCs correspond to the same antigen
#         (either because dual-labeled AgBCs were used, or because several AgBCs are
#         closely-related molecules that would be expected to compete for BCR binding).
#         Either `agbcs` or `groups` is required. If both are provided, both will be used.

#     rename : dict, optional
#         A ``dict`` mapping AgBC or group names to a new name. Keys should be present in
#         either ``agbcs`` or ``groups.keys()``. If only a subset of AgBCs or groups are
#         provided in ``rename``, then only those AgBCs or groups will be renamed.

#     percentile : float, default=0.997
#         Percentile used to compute the AgBC classification threshold using `raw` data. Default
#         is ``0.997``, which corresponds to three standard deviations.

#     percentile_dict : dict, optional
#         A ``dict`` mapping AgBC or group names to the desired `percentile`. If only a subset
#         of AgBCs or groups are provided in `percentile_dict`, all others will use `percentile`.

#     update : bool, default=True
#         If ``True``, update `adata` with grouped UMI counts and classifications. If ``False``,
#         a Pandas ``DataFrame`` containg classifications will be returned and `adata` will
#         not be modified.

#     uns_batch: str, default=None
#         If provided, `uns_batch` will add batch information to the percentile and threshold
#         data stored in ``adata.uns``. This results in an additional layer of nesting, which
#         allows concateenating multiple ``AnnData`` objects represeting different batches for
#         which classification is performed separately. If not provided, the data stored in ``uns``
#         would be formatted like::

#             adata.uns['agbc_percentiles'] = {agbc1: percentile1, ...}
#             adata.uns['agbc_thresholds'] = {agbc1: threshold1, ...}

#         If `uns_batch` is provided, ``uns`` will be formatted like::

#             adata.ubs['agbc_percentiles'] = {uns_batch: {agbc1: percentile1, ...}}
#             adata.ubs['agbc_thresholds'] = {uns_batch: {agbc1: threshold1, ...}}

#     verbose : bool, default=True
#         If ``True``, calculated threshold values are printed.


#     Returns
#     -------
#     output : ``anndata.AnnData`` or ``pandas.DataFrame``
#         If `update` is ``True``, an updated `adata` object containing specificity classifications \
#         is returned. Otherwise, a Pandas ``DataFrame`` containing specificity classifications \
#         is returned.

#     """
#     adata_groups = {}
#     classifications = {}
#     percentiles = percentile_dict if percentile_dict is not None else {}
#     rename = rename if rename is not None else {}

#     # process AgBCs and specificity groups
#     if all([agbcs is None, groups is None]):
#         err = "ERROR: either agbcs or groups must be provided."
#         print("\n" + err + "\n")
#         sys.exit()
#     if groups is None:
#         groups = {}
#     if agbcs is not None:
#         for a in agbcs:
#             groups[a] = [
#                 a,
#             ]

#     # load raw data, if necessary
#     if isinstance(raw, str):
#         if os.path.isdir(raw):
#             raw = read_10x_mtx(raw, ignore_zero_quantile_agbcs=False)
#         else:
#             err = "\nERROR: raw must be either an AnnData object or a path to the raw matrix output folder from CellRanger.\n"
#             print(err)
#             sys.exit()

#     # remove cell-containing droplets from raw
#     no_cell = [o not in adata.obs_names for o in raw.obs_names]
#     empty = raw[no_cell]

#     # classify AgBC specificities
#     if verbose:
#         print("")
#         print("  THRESHOLDS  ")
#         print("--------------")
#     uns_thresholds = {}
#     uns_percentiles = {}
#     for group, barcodes in groups.items():
#         # remove missing AgBCs
#         in_adata = [b for b in barcodes if b in adata.obs]
#         in_empty = [b for b in barcodes if b in empty.obs]
#         in_both = list(set(in_adata) & set(in_empty))
#         if any([not in_adata, not in_empty]):
#             err = f"\nERROR: group {group} cannot be processed because all AgBCs are missing from input or raw datasets.\n"
#             if not in_adata:
#                 err += f"input is missing {', '.join([b for b in barcodes if b not in in_adata])}\n"
#             if not in_empty:
#                 err += f"raw is missing {', '.join([b for b in barcodes if b not in in_empty])}\n"
#             print(err)
#             del groups[group]
#             continue
#         if len(in_adata) != len(in_empty):
#             warn = f"\nWARNING: not all AgBCs for group {group} can be found in data and raw.\n"
#             warn += f"input contains {', '.join(in_adata)}\n"
#             warn += f"raw contains {', '.join(in_empty)}\n"
#             print(warn)
#             groups[group] = [bc for bc in barcodes if bc in in_both]
#         group_name = rename.get(group, group)
#         pctile = percentiles.get(group_name, percentile)
#         # thresholds for each barcode
#         _empty = np.array([np.exp2(empty.obs[bc]) - 1 for bc in in_empty])
#         raw_bc_thresholds = {
#             bc: np.quantile(_e, pctile) for _e, bc in zip(_empty, in_empty)
#         }
#         # UMI counts for the entire group
#         _data = np.sum([np.exp2(adata.obs[bc]) - 1 for bc in in_adata], axis=0)
#         adata_groups[group_name] = np.log2(_data + 1)
#         # threshold for the entire group (sum of the individual barcode thresholds)
#         raw_threshold = np.sum(list(raw_bc_thresholds.values()))
#         threshold = np.log2(raw_threshold + 1)
#         classifications[group_name] = adata_groups[group_name] > threshold
#         # update uns dicts
#         uns_thresholds[group] = threshold
#         uns_percentiles[group] = pctile
#         if verbose:
#             print(group_name)
#             print(f"percentile: {pctile}")
#             print(f"threshold: {threshold}")
#             for bc, rt in raw_bc_thresholds.items():
#                 print(f"  - {bc}: {np.log2(rt + 1)}")
#             print("")
#     if update:
#         for g, group_data in adata_groups.items():
#             adata.obs[g] = group_data
#             adata.obs[f"is_{g}"] = classifications[g]
#             if uns_batch is not None:
#                 adata.uns["agbc_thresholds"] = {uns_batch: uns_thresholds}
#                 adata.uns["agbc_percentiles"] = {uns_batch: uns_percentiles}
#             else:
#                 adata.uns["agbc_thresholds"] = uns_thresholds
#                 adata.uns["agbc_percentiles"] = uns_percentiles
#         return adata
#     else:
#         return pd.DataFrame(classifications, index=adata.obs_names)


def calculate_agbc_confidence(
    adata,
    control_adata,
    agbcs,
    update=True,
    batch_key=None,
    batch_control_data=True,
    verbose=True,
):
    """
    Computes AgBC confidence using a control dataset.

    Args:
    -----

        adata (anndata.AnnData): ``AnnData`` object with AgBC count data (log2 transformed) located in ``adata.obs``.

        control_adata (anndata.AnnData): ``AnnData`` object with AgBC count data (log2 transformed) located in
            ``control_adata.obs``. Should contain data from control cells (not antigen sorted) which will be used to
            compute the confidence values

        agbcs (list): List of AgBC names. Each AgBC name must be present in both ``adata.obs`` and ``control_adata.obs``.

        update (bool): If ``True``, ``adata.obs`` is updated with confidence values (column names are ``f'{agbc}_confidence'``).
            If ``False``, a ``DataFrame`` is returned containing the confidence values. Default is ``True``.


    Returns:
    --------
    Updated ``adata`` if ``update`` is ``True`` or a ``pandas.DataFrame`` if ``update`` is ``False``.

    """
    conf_data = {}
    # check to make sure AgBCs are in both datasets
    if any([a not in control_adata.obs for a in agbcs]):
        missing = [a for a in agbcs if a not in control_adata.obs]
        if verbose:
            print(
                "Ignoring the following AgBCs, as they were not found in the control data: ",
                end="",
            )
            print(", ".join(missing))
        agbcs = [a for a in agbcs if a not in missing]
    if any([a not in adata.obs for a in agbcs]):
        missing = [a for a in agbcs if a not in adata.obs]
        if verbose:
            print(
                "Ignoring the following AgBCs, as they were not found in the data: ",
                end="",
            )
            print(", ".join(missing))
        agbcs = [a for a in agbcs if a not in missing]
    # split data into batches
    if batch_key is not None:
        if batch_key not in adata.obs:
            print(
                f"ERROR: the supplied batch key ({batch_key}) was not found in the input data."
            )
            return
        batch_names = natsorted(adata.obs[batch_key].unique())
        if verbose:
            print(f'Found {len(batch_names)} batches: {", ".join(batch_names)}')
        batches = [adata[adata.obs[batch_key] == b] for b in batch_names]
        if batch_control_data:
            if batch_key not in control_adata.obs:
                print(
                    f"ERROR: the supplied batch key ({batch_key}) was not found in the control data."
                )
                return
            control_batches = [
                control_adata[control_adata.obs[batch_key] == b] for b in batch_names
            ]
        else:
            control_batches = [control_adata] * len(batch_names)
    else:
        batch_names = [None]
        batches = [adata]
        control_batches = [control_adata]
    # calculate confidence
    for barcode in agbcs:
        bc_conf = {}
        for name, data, control in zip(batch_names, batches, control_batches):
            # get the fit parameters
            y = np.exp2(control.obs[barcode]) - 1
            x = np.ones(y.shape)
            res = sm.NegativeBinomial(y, x).fit(start_params=[0.1, 0.1], disp=False)
            mu = np.exp(res.params[0])
            alpha = res.params[1]
            size = 1.0 / alpha
            prob = size / (size + mu)
            # estimate the distribution
            dist = stats.nbinom(size, prob)
            # calculate confidences
            conf = [dist.cdf(np.exp2(v) - 1) for v in data.obs[barcode]]
            bc_conf.update({k: v for k, v in zip(data.obs_names, conf)})
        confidence = [bc_conf[o] for o in adata.obs_names]
        if update:
            adata.obs[f"{barcode}_confidence"] = confidence
        else:
            conf_data[f"{barcode}_confidence"] = confidence
    if update:
        return adata
    else:
        return pd.DataFrame(conf_data, index=adata.obs_names)


# # for backwards compatibility
# def assign_cellhashes(adata, **kwargs):
#     return demultiplex(adata, **kwargs)


# def demultiplex(
#     adata,
#     hash_names=None,
#     cellhash_regex="cell ?hash",
#     ignore_cellhash_case=True,
#     rename=None,
#     assignment_key="cellhash_assignment",
#     threshold_minimum=4.0,
#     threshold_maximum=10.0,
#     kde_maximum=15.0,
#     assignments_only=False,
#     debug=False,
# ):
#     """
#     Demultiplexes cells using cell hashes.

#     Parameters
#     ----------

#     adata : anndata.Anndata
#         ``AnnData`` object containing cellhash UMI counts in ``adata.obs``.

#     hash_names : iterable object, optional
#         List of hashnames, which correspond to column names in ``adata.obs``.
#         Overrides cellhash name matching using `cellhash_regex`. If not provided,
#         all columns in ``adata.obs`` that match `cellhash_regex` will be assumed
#         to be hashnames and processed.

#     cellhash_regex : str, default='cell ?hash'
#         A regular expression (regex) string used to identify cell hashes. The regex
#         must be found in all cellhash names. The default is ``'cell ?hash'``, which
#         combined with the default setting for `ignore_cellhash_regex_case`, will
#         match ``'cellhash'`` or ``'cell hash'`` anywhere in the cell hash name and
#         in any combination of upper or lower case letters.

#     ignore_cellhash_regex_case : bool, default=True
#         If ``True``, matching to `cellhash_regex` will ignore case.

#     rename : dict, optional
#         A ``dict`` linking cell hash names (column names in ``adata.obs``) to the
#         preferred batch name. For example, if the cell hash name ``'Cellhash1'``
#         corresponded to the sample ``'Sample1'``, an example `rename` argument
#         would be::

#                 {'Cellhash1': 'Sample1'}

#         This would result in all cells classified as positive for ``'Cellhash1'`` being
#         labeled as ``'Sample1'`` in the resulting assignment column (``adata.obs.sample``
#         by default, adjustable using `assignment_key`).

#     assignment_key : str, default='cellhash_assignment'
#         Column name (in ``adata.obs``) into which cellhash assignments will be stored.

#     threshold_minimum : float, default=4.0
#         Minimum acceptable log2-normalized UMI count threshold. Potential thresholds
#         below this cutoff value will be ignored.

#     threshold_maximum : float, default=10.0
#         Maximum acceptable log2-normalized UMI count threshold. Potential thresholds
#         above this cutoff value will be ignored.

#     kde_maximum : float, default=15.0
#         Upper limit of the KDE plot (in log2-normalized UMI counts). This should
#         be less than `threshold_maximum`, or you may obtain strange results.

#     assignments_only : bool, default=False
#         If ``True``, return a pandas ``Series`` object containing only the group
#         assignment. Suitable for appending to an existing dataframe. If ``False``,
#         an updated `adata` object is returned, containing cell hash group assignemnts
#         at ``adata.obs.assignment_key``

#     debug : bool, default=False
#         If ``True``, saves cell hash KDE plots and prints intermediate information
#         for debugging.


#     Returns
#     -------
#     output : ``anndata.AnnData`` or ``pandas.Series``
#         By default, an updated `adata` is returned with cell hash assignment groups \
#         stored in the `assignment_key` column of ``adata.obs``. If `assignments_only` \
#         is ``True``, a ``pandas.Series`` of lineage assignments is returned.

#     """
#     # parse hash names
#     if hash_names is None:
#         if ignore_cellhash_case:
#             cellhash_pattern = re.compile(cellhash_regex, flags=re.IGNORECASE)
#         else:
#             cellhash_pattern = re.compile(cellhash_regex)
#         hash_names = [
#             o for o in adata.obs.columns if re.search(cellhash_pattern, o) is not None
#         ]
#     hash_names = [h for h in hash_names if h != assignment_key]
#     if rename is None:
#         rename = {}
#     # compute thresholds
#     thresholds = {}
#     for hash_name in hash_names:
#         if debug:
#             print(hash_name)
#         thresholds[hash_name] = positive_feature_cutoff(
#             adata.obs[hash_name].dropna(),
#             threshold_minimum=threshold_minimum,
#             threshold_maximum=threshold_maximum,
#             kde_maximum=kde_maximum,
#             debug=debug,
#         )
#     if debug:
#         print("THRESHOLDS")
#         print("----------")
#         for hash_name in hash_names:
#             print(f"{hash_name}: {thresholds[hash_name]}")
#     assignments = []
#     for _, row in adata.obs[hash_names].iterrows():
#         a = [h for h in hash_names if row[h] >= thresholds[h]]
#         if len(a) == 1:
#             assignment = rename.get(a[0], a[0])
#         elif len(a) > 1:
#             assignment = "doublet"
#         else:
#             assignment = "unassigned"
#         assignments.append(assignment)
#     if assignments_only:
#         return pd.Series(assignments, index=adata.obs_names)
#     else:
#         adata.obs[assignment_key] = assignments
#         return adata


# def positive_feature_cutoff(
#     vals,
#     threshold_maximum=10.0,
#     threshold_minimum=4.0,
#     kde_maximum=15.0,
#     debug=False,
#     show_cutoff_value=False,
#     cutoff_text="cutoff",
#     debug_figfile=None,
# ):
#     a = np.array(vals)
#     k = _bw_silverman(a)
#     kde = KernelDensity(kernel="gaussian", bandwidth=k).fit(a.reshape(-1, 1))
#     s = np.linspace(0, kde_maximum, num=int(kde_maximum * 100))
#     e = kde.score_samples(s.reshape(-1, 1))

#     all_min, all_max = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
#     if len(all_min) > 1:
#         _all_min = np.array(
#             [
#                 m
#                 for m in all_min
#                 if s[m] <= threshold_maximum and s[m] >= threshold_minimum
#             ]
#         )
#         min_vals = zip(_all_min, e[_all_min])
#         mi = sorted(min_vals, key=lambda x: x[1])[0][0]
#         cutoff = s[mi]
#     elif len(all_min) == 1:
#         mi = all_min[0]
#         cutoff = s[mi]
#     else:
#         cutoff = None
#     if debug:
#         if cutoff is not None:
#             # plot
#             plt.plot(s, e)
#             plt.fill_between(s, e, y2=[min(e)] * len(s), alpha=0.1)
#             plt.vlines(
#                 cutoff,
#                 min(e),
#                 max(e),
#                 colors="k",
#                 alpha=0.5,
#                 linestyles=":",
#                 linewidths=2,
#             )
#             # text
#             text_xadj = 0.025 * (max(s) - min(s))
#             cutoff_string = (
#                 f"{cutoff_text}: {round(cutoff, 3)}"
#                 if show_cutoff_value
#                 else cutoff_text
#             )
#             plt.text(
#                 cutoff - text_xadj,
#                 max(e),
#                 cutoff_string,
#                 ha="right",
#                 va="top",
#                 fontsize=14,
#             )
#             # style
#             ax = plt.gca()
#             for spine in ["right", "top"]:
#                 ax.spines[spine].set_visible(False)
#             ax.tick_params(axis="both", labelsize=12)
#             ax.set_xlabel("$\mathregular{log_2}$ UMI counts", fontsize=14)
#             ax.set_ylabel("kernel density", fontsize=14)
#             # save or show
#             if debug_figfile is not None:
#                 plt.tight_layout()
#                 plt.savefig(debug_figfile)
#             else:
#                 plt.show()
#         print("bandwidth: {}".format(k))
#         print("local minima: {}".format(s[all_min]))
#         print("local maxima: {}".format(s[all_max]))
#         if cutoff is not None:
#             print("cutoff: {}".format(cutoff))
#         else:
#             print(
#                 "WARNING: no local minima were found, so the threshold could not be calculated."
#             )
#         print("\n\n")
#     return cutoff


# def negative_feature_cutoff(
#     vals,
#     threshold_maximum=10.0,
#     threshold_minimum=4.0,
#     kde_maximum=15.0,
#     denominator=2.0,
#     debug=False,
#     show_cutoff_value=False,
#     cutoff_text="cutoff",
#     debug_figfile=None,
# ):
#     a = np.array(vals)
#     k = _bw_silverman(a)
#     kde = KernelDensity(kernel="gaussian", bandwidth=k).fit(a.reshape(-1, 1))
#     s = np.linspace(0, kde_maximum, num=int(kde_maximum * 100))
#     e = kde.score_samples(s.reshape(-1, 1))

#     all_min, all_max = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
#     if len(all_min) > 1:
#         _all_min = np.array(
#             [
#                 m
#                 for m in all_min
#                 if s[m] <= threshold_maximum and s[m] >= threshold_minimum
#             ]
#         )
#         min_vals = zip(_all_min, e[_all_min])
#         mi = sorted(min_vals, key=lambda x: x[1])[0][0]
#         ma = [m for m in all_max if s[m] < s[mi]][-1]
#         cutoff = s[int((mi + ma) / denominator)]
#     elif len(all_min) == 1:
#         mi = all_min[0]
#         ma = all_max[0]
#         cutoff = s[int((mi + ma) / denominator)]
#     else:
#         cutoff = None
#     if debug:
#         if cutoff is not None:
#             # plot
#             plt.plot(s, e)
#             plt.fill_between(s, e, y2=[min(e)] * len(s), alpha=0.1)
#             plt.vlines(s[mi], min(e), e[mi], colors="k", alpha=0.5, linestyles=":")
#             plt.vlines(s[ma], min(e), e[ma], colors="k", alpha=0.5, linestyles=":")
#             plt.vlines(
#                 cutoff,
#                 min(e),
#                 max(e),
#                 colors="k",
#                 alpha=0.5,
#                 linestyles=":",
#                 linewidths=2,
#             )
#             # text
#             text_ymin = min(e) + (0.025 * (max(e) - min(e)))
#             text_xadj = 0.025 * (max(s) - min(s))
#             plt.text(
#                 s[mi] + text_xadj,
#                 text_ymin,
#                 "local\nmin",
#                 ha="left",
#                 va="bottom",
#                 fontsize=12,
#             )
#             plt.text(
#                 s[ma] - text_xadj,
#                 text_ymin,
#                 "local\nmax",
#                 ha="right",
#                 va="bottom",
#                 fontsize=12,
#             )
#             cutoff_string = (
#                 f"{cutoff_text}: {round(cutoff, 3)}"
#                 if show_cutoff_value
#                 else cutoff_text
#             )
#             plt.text(
#                 cutoff + text_xadj,
#                 max(e),
#                 cutoff_string,
#                 ha="left",
#                 va="top",
#                 fontsize=14,
#             )
#             # style
#             ax = plt.gca()
#             for spine in ["right", "top"]:
#                 ax.spines[spine].set_visible(False)
#             ax.tick_params(axis="both", labelsize=12)
#             ax.set_xlabel("$\mathregular{log_2}$ UMI counts", fontsize=14)
#             ax.set_ylabel("kernel density", fontsize=14)
#             # save or show
#             if debug_figfile is not None:
#                 plt.tight_layout()
#                 plt.savefig(debug_figfile)
#             else:
#                 plt.show()
#         print("bandwidth: {}".format(k))
#         print("local minima: {}".format(s[all_min]))
#         print("local maxima: {}".format(s[all_max]))
#         if cutoff is not None:
#             print("cutoff: {}".format(cutoff))
#         else:
#             print(
#                 "WARNING: no local minima were found, so the threshold could not be calculated."
#             )
#         print("\n\n")
#     return cutoff


# def _bw_silverman(x):
#     normalize = 1.349
#     IQR = (np.percentile(x, 75) - np.percentile(x, 25)) / normalize
#     std_dev = np.std(x, axis=0, ddof=1)
#     if IQR > 0:
#         A = np.minimum(std_dev, IQR)
#     else:
#         A = std_dev
#     n = len(x)
#     return 0.9 * A * n ** (-0.2)
