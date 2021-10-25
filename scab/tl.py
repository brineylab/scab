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


import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
import statsmodels.api as sm


def dimensionality_reduction(adata, solver='arpack', n_neighbors=20, n_pcs=40,
                             paga=True, use_rna_velocity=False, rep=None,
                             random_state=None, resolution=1.0, verbose=True):
    '''
    performs PCA and UMAP embedding

    Args:
    -----

        solver (str): Solver to use for the PCA. Default is ``'arpack'``.

        n_neighbors (int): Number of neighbors to calculate for the neighbor graph.
                           Default is ``10``.

        n_pcs (int): Number of principal components to use when calculating the
                     neighbor graph. Default is ``40``. Although the default value
                     is generally appropriate, it is advisable to use 
                     ``GEX.plot_pca_variance`` to empiracally determine the optimal
                     value for ``n_pcs``.

        paga (bool): If ``True``, performs partition-based graph abstraction prior to
                     UMAP embedding. Default is ``True``.

        use_rna_velocity (bool): If ``True``, uses RNA velocity information to compute PAGA.
                                 If ``paga`` is ``False``, this option is ignored. Default is ``False``.

        rep (str): Representation to use when computing neighbors with ``sc.pp.neignbors``. For 
                   example, if data have been batch normalized with ``scanorama``, the representation
                   should be ``'Scanorama'``. Default is ``None``, which uses ``scanpy``'s default 
                   representation.

        random_state (int): Seed for the random state used by ``sc.tl.umap``. Default is ``None``.

        resolution (float): Resolution for Leiden clustering. Default is ``1.0``.
    '''    
    if verbose:
        print('performing PCA...')
    sc.tl.pca(adata, svd_solver=solver, n_comps=n_pcs, use_highly_variable=True)
    if verbose:
        print('calculating neighbors...')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=rep)
    if verbose:
        print('leiden clustering...')
    sc.tl.leiden(adata, resolution=resolution)
    if paga:
        if verbose:
            print('paga...')
#         solid_edges = 'transitions_confidence' if use_rna_velocity else 'connectivities'
        sc.tl.paga(adata, use_rna_velocity=use_rna_velocity)
        if use_rna_velocity:
            adata.uns['paga']['connectivities'] = adata.uns['paga']['transitions_confidence']
        sc.pl.paga(adata, plot=False)
        if verbose:
            print('umap...')
        sc.tl.umap(adata, init_pos='paga', random_state=random_state, )
    else:
        if verbose:
            print('umap...')
        sc.tl.umap(adata, random_state=random_state)
    return adata


def combat(adata, batch_key='sample', covariates=None, dim_red=True):
    '''
    Data integration and batch correction using mutual nearest neighbors. Uses the 
    ``scanpy.external.pp.mnn_correct()`` function.

    Args:
    -----

        adata (anndata.AnnData): AnnData object containing gene expression and/or feature barcode count data.

        batch_key (str): Name of the column in adata.obs that corresponds to the batch. Default is ``'sample'``.

        covariates (iterable): List of additional covariates besides the batch variable such as adjustment variables 
                               or biological condition. Not including covariates may lead to the removal of real
                               biological signal. Default is ``None``, which corresponds to no covariates.

        dim_red (bool): If ``True``, dimentionality reduction will be performed on the post-integration data using 
                        ``scab.tl.data_reduction``. Default is ``True``.

    Returns:
    --------

        anndata.AnnData
    '''
    adata_combat = sc.AnnData(X=adata.raw.X, var=adata.raw.var, obs=adata.obs)
    adata_combat.layers = adata.layers
    adata_combat.raw = adata_combat
    # run combat
    sc.pp.combat(adata_combat, key=batch_key, covariates=covariates)
    sc.pp.highly_variable_genes(adata_combat)
    if dim_red:
        adata_combat = dimensionality_reduction(adata_combat)
    return adata_combat


def mnn(adata, batch_key='sample', min_hvg_batches=1, dim_red=True):
    '''
    Data integration and batch correction using mutual nearest neighbors. Uses the 
    ``scanpy.external.pp.mnn_correct()`` function.

    Args:
    -----

        adata (anndata.AnnData): AnnData object containing gene expression and/or feature barcode count data.

        batch_key (str): Name of the column in adata.obs that corresponds to the batch. Default is ``'sample'``.

        min_hvg_batches (int): Minimum number of batches in which highly variable genes are found in order to be included
                               in the list of genes used for batch correction. Default is ``1``, which results in the use 
                               of all HVGs found in any batch.
        
        dim_red (bool): If ``True``, dimentionality reduction will be performed on the post-integration data using 
                        ``scab.tl.data_reduction``. Default is ``True``.

    Returns:
    --------

        anndata.AnnData
    '''
    adata_mnn = adata.raw.to_adata()
    adata_mnn.layers = adata.layers
    # variable genes
    sc.pp.highly_variable_genes(adata_mnn, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key=batch_key)
    var_select = adata_mnn.var.highly_variable_nbatches >= min_hvg_batches
    var_genes = var_select.index[var_select]
    # split per batch into new objects.
    batches = adata_mnn.obs[batch_key].cat.categories.tolist()
    alldata = {}
    for batch in batches:
        alldata[batch] = adata_mnn[adata_mnn.obs[batch_key] == batch,]
    # run MNN correction
    cdata = sc.external.pp.mnn_correct(*[alldata[b] for b in batches], 
                                       svd_dim=50,
                                       batch_key=batch_key,
                                       save_raw=True,
                                       var_subset=var_genes)
    corr_data = cdata[0][:,var_genes]
    if dim_red:
        corr_data = dimensionality_reduction(corr_data)
    return corr_data


def scanorama(adata, batch_key='sample', dim_red=True):
    '''
    Data integration and batch correction using Scanorama. Uses the ``scanorama.integrate_scanpy()`` function.

    Args:
    -----

        adata (anndata.AnnData): AnnData object containing gene expression and/or feature barcode count data.

        batch_key (str): Name of the column in adata.obs that corresponds to the batch. Default is ``'sample'``.

        dim_red (bool): If ``True``, dimentionality reduction will be performed on the post-integration data using 
                        ``scab.tl.data_reduction``. Default is ``True``.

    Returns:
    --------

        anndata.AnnData
    '''
    import scanorama
    adata_scanorama = adata.raw.to_adata()
    adata_scanorama.layers = adata.layers
    # split per batch into new objects.
    batches = adata_scanorama.obs[batch_key].cat.categories.tolist()
    alldata = {}
    for batch in batches:
        alldata[batch] = adata_scanorama[adata_scanorama.obs[batch_key] == batch,]
    adatas = [alldata[s] for s in alldata.keys()]
    # run scanorama
    scanorama.integrate_scanpy(adatas,  dimred=50)
    scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]
    all_s = np.concatenate(scanorama_int)
    adata_scanorama.obsm["Scanorama"] = all_s
    if dim_red:
        adata_scanorama = dimensionality_reduction(adata_scanorama, rep="Scanorama")
    return adata_scanorama



def calculate_agbc_confidence(adata, control_adata, agbcs, update=True):
    '''
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
    
    '''
    conf_data = {}
    for barcode in agbcs:
        # get the fit parameters
        y = np.exp2(control_adata.obs[barcode]) - 1
        x = np.ones(y.shape)
        res = sm.NegativeBinomial(y, x).fit(start_params=[0.1, 0.1], disp=False)
        mu = np.exp(res.params[0])
        alpha = res.params[1]
        size = 1. / alpha
        prob = size / (size + mu)
        # estimate the distribution
        dist = stats.nbinom(size, prob)
        # calculate confidences
        confidence = [dist.cdf(np.exp2(v) - 1) for v in adata.obs[barcode]]
        if update:
            adata.obs[f'{barcode}_confidence'] = confidence
        else:
            conf_data[f'{barcode}_confidence'] = confidence
    if update:
        return adata
    else:
        return pd.DataFrame(conf_data, index=adata.obs_names)






