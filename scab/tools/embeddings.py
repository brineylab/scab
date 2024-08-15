#!/usr/bin/env python
# filename: embeddings.py


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


import warnings
from typing import Optional, Union

import anndata
import scanpy as sc

__all__ = ["pca", "umap", "dimensionality_reduction"]


def pca(
    adata: anndata.AnnData,
    solver: str = "arpack",
    n_pcs: int = 50,
    ignore_ig: bool = True,
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Performs PCA, neighborhood graph construction and UMAP embedding.
    PAGA is optional, but is performed by default.

    Parameters
    ----------

    adata : anndata.AnnData
        ``AnnData`` object containing gene counts data.

    solver : str, default='arpack'
        Solver to use for the PCA.

    n_pcs : int, default=50
        Number of principal components to use when computing the neighbor graph.
        Although the default value is generally appropriate, it is sometimes useful
        to empirically determine the optimal value for `n_pcs`.

    ignore_ig : bool, default=True
        Ignores immunoglobulin V, D and J genes when computing the PCA.


    Returns
    -------
    adata : ``anndata.AnnData``


    .. _PAGA:
        https://github.com/theislab/paga
    .. _computing neighbors:
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html

    """
    if verbose:
        print("performing PCA...")
    if ignore_ig:
        _adata = adata.copy()
        _adata.var["highly_variable"] = (
            _adata.var["highly_variable"] & ~(_adata.var["ig"])
        )
        sc.tl.pca(_adata, svd_solver=solver, n_comps=n_pcs, use_highly_variable=True)
        adata.obsm["X_pca"] = _adata.obsm["X_pca"]
        adata.varm["PCs"] = _adata.varm["PCs"]
        adata.uns["pca"] = {
            "variance_ratio": _adata.uns["pca"]["variance_ratio"],
            "variance": _adata.uns["pca"]["variance"],
        }
    else:
        sc.tl.pca(adata, svd_solver=solver, n_comps=n_pcs, use_highly_variable=True)
    return adata


def umap(
    adata: anndata.AnnData,
    solver: str = "arpack",
    n_neighbors: Optional[int] = None,
    n_pcs: Optional[int] = None,
    force_pca: bool = False,
    ignore_ig: bool = True,
    paga: bool = True,
    batch_key: Optional[str] = None,
    use_rna_velocity: bool = False,
    use_rep: Optional[str] = None,
    random_state: Union[int, float, str] = 42,
    resolution: float = 1.0,
    verbose: bool = True,
) -> anndata.AnnData:
    """
    Performs PCA, neighborhood graph construction and UMAP embedding.
    PAGA is optional, but is performed by default.

    Parameters
    ----------

    adata : anndata.AnnData
        ``AnnData`` object containing gene counts data.

    solver : str, default='arpack'
        Solver to use for the PCA.

    n_neighbors : int, default=10
        Number of neighbors to calculate for the neighbor graph.

    n_pcs : int, default=40
        Number of principal components to use when computing the neighbor graph.
        Although the default value is generally appropriate, it is sometimes useful
        to empirically determine the optimal value for `n_pcs`.

    force_pca : bool, default=False
        Construct the PCA even if it has already been constructed (``"X_pcs"`` exists
        in ``adata.obsm``). Default is ``False``, which will use an existing PCA.

    ignore_ig : bool, default=True
        Ignores immunoglobulin V, D and J genes when computing the PCA.

    paga : bool, default=True
        If ``True``, performs partition-based graph abstraction (PAGA_) prior to
        UMAP embedding.

    batch_key : str, optional
        If ``adata`` contains batch information, this is the key in ``adata.obs``
        that contains the batch information. If provided, neighbors will be computed
        using `batch-balanced KNN`_ (``scanpy.external.pp.bbknn``) rather than
        ``scanpy.pp.neighbors``.

    use_rna_velocity : bool, default=False
        If ``True``, uses RNA velocity information to compute PAGA. If ``False``,
        this option is ignored.

    use_rep : str, optional
        Representation to use when `computing neighbors`_. For example, if data have
        been batch normalized with ``scanorama``, the representation
        should be ``'Scanorama'``. If not provided, ``scanpy``'s default
        representation is used.

    random_state : int, optional
        Seed for the random state used by ``sc.tl.umap``.

    resolution : float, default=1.0
        Resolution for Leiden clustering.


    Returns
    -------
    adata : ``anndata.AnnData``


    .. _PAGA:
        https://github.com/theislab/paga
    .. _computing neighbors:
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html
    .. _batch-balanced KNN:
        https://github.com/Teichlab/bbknn

    """
    if verbose:
        print("")
        print("UMAP EMBEDDING")
        print("--------------")
    if batch_key is not None:
        n_neighbors = n_neighbors if n_neighbors is not None else 3
        n_pcs = n_pcs if n_pcs is not None else 50
    else:
        n_neighbors = n_neighbors if n_neighbors is not None else 15
        n_pcs = n_pcs if n_pcs is not None else 40
    # PCA
    if any([force_pca, "X_pca" not in adata.obsm_keys()]):
        if verbose:
            print("  - computing PCA")
        adata = pca(
            adata, solver=solver, n_pcs=n_pcs, ignore_ig=ignore_ig, verbose=False
        )
    # neighbors
    if batch_key is not None:
        if verbose:
            print("  - using BBKNN to calculate neighbors")
        sc.external.pp.bbknn(
            adata,
            batch_key=batch_key,
            n_pcs=n_pcs,
            neighbors_within_batch=n_neighbors,
            use_rep=use_rep,
        )
    else:
        if verbose:
            print("  - calculating neighbors")
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
        )
    # leiden clustering
    if verbose:
        print("  - leiden clustering")
    sc.tl.leiden(adata, resolution=resolution)
    # umap (with or without PAGA first)
    if paga:
        if verbose:
            print("  - paga")
        sc.tl.paga(adata, use_rna_velocity=use_rna_velocity)
        if use_rna_velocity:
            adata.uns["paga"]["connectivities"] = adata.uns["paga"][
                "transitions_confidence"
            ]
        sc.pl.paga(adata, plot=False)
        init_pos = "paga"
    else:
        init_pos = "spectral"
    if verbose:
        print("  - umap")
    sc.tl.umap(adata, init_pos=init_pos, random_state=random_state)
    return adata


def dimensionality_reduction(*args, **kwargs) -> anndata.AnnData:
    """
    Deprecated, but retained for backwards compatibility. Use ``scab.tl.umap()`` instead.
    """
    warnings.warn(
        "scab.tl.dimensionality_reduction() is deprecated, use scab.tl.umap() instead",
        DeprecationWarning,
    )
    return umap(*args, **kwargs)
