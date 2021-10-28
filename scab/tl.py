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

import re

from natsort import natsorted

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scanpy as sc

from sklearn.neighbors import KernelDensity

from scipy import stats
from scipy.signal import argrelextrema
from scipy.stats import scoreatpercentile

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



def calculate_agbc_confidence(adata, control_adata, agbcs, update=True,
                              batch_key=None, batch_control_data=True, verbose=True):
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
    # check to make sure AgBCs are in both datasets
    if any([a not in control_adata.obs for a in agbcs]):
        missing = [a for a in agbcs if a not in control_adata.obs]
        if verbose:
            print('Ignoring the following AgBCs, as they were not found in the control data: ', end='')
            print(', '.join(missing))
        agbcs = [a for a in agbcs if a not in missing]
    if any([a not in adata.obs for a in agbcs]):
        missing = [a for a in agbcs if a not in adata.obs]
        if verbose:
            print('Ignoring the following AgBCs, as they were not found in the data: ', end='')
            print(', '.join(missing))
        agbcs = [a for a in agbcs if a not in missing]
    # split data into batches
    if batch_key is not None:
        if batch_key not in adata.obs:
            print(f'ERROR: the supplied batch key ({batch_key}) was not found in the input data.')
            return
        batch_names = natsorted(adata.obs[batch_key].unique())
        if verbose:
            print(f'Found {len(batch_names)} batches: {", ".join(batch_names)}')
        batches = [adata[adata.obs[batch_key] == b] for b in batch_names]
        if batch_control_data:
            if batch_key not in control_adata.obs:
                print(f'ERROR: the supplied batch key ({batch_key}) was not found in the control data.')
                return
            control_batches = [control_adata[control_adata.obs[batch_key] == b] for b in batch_names]
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
            size = 1. / alpha
            prob = size / (size + mu)
            # estimate the distribution
            dist = stats.nbinom(size, prob)
            # calculate confidences
            conf = [dist.cdf(np.exp2(v) - 1) for v in data.obs[barcode]]
            bc_conf.update({k: v for k, v in zip(data.obs_names, conf)})
        confidence = [bc_conf[o] for o in adata.obs_names]
        if update:
            adata.obs[f'{barcode}_confidence'] = confidence
        else:
            conf_data[f'{barcode}_confidence'] = confidence
    if update:
        return adata
    else:
        return pd.DataFrame(conf_data, index=adata.obs_names)



def assign_cellhashes(adata, hash_names=None, cellhash_regex='cell ?hash', ignore_cellhash_case=True,
                      batch_names=None, batch_key='batch',
                      threshold_minimum=4.0, threshold_maximum=10.0, kde_maximum=15.0, 
                      assignments_only=False, debug=False):
    '''
    Assigns cells to hash groups based on cell hashing data.

    Args:
    -----

        df (pd.DataFrame): Squareform input dataframe, containing cellhash UMI counts. Indexes
            should be cells, columns should be cell hashes.
        
        hash_names (iterable): List of hashnames, which correspond to column names in ``adata.obs``. 
            Overrides cellhash name matching using ``cellhash_regex``. If not provided, all columns 
            in ``adata.obs`` that are matched using ``cellhash_regex`` will be assumed to be hashnames. 
        
        cellhash_regex (str): A regular expression (regex) string used to identify cell hashes. The regex 
            must be found in all cellhash names. The default is ``'cell ?hash'``, which combined with the
            default setting for ``ignore_cellhash_regex_case``, will match ``'cellhash'`` or ``'cell hash'``
            in any combination of upper and lower case letters.

        ignore_cellhash_regex_case (bool): If ``True``, searching for ``cellhash_regex`` will ignore case.
            Default is ``True``.
        
        batch_names (dict): Dictionary relating hasnhames (column names in ``adata.obs``) to the preferred
            batch name. For example, if the hashname ``'Cellhash1'`` corresponded to the sample 
            ``'Sample1'``, an example ``batch_names`` argument would be::

                {'Cellhash1': 'Sample1'}

        batch_key (str): Column name (in ``adata.obs``) into which cellhash classifications will be 
            stored. Default is ``'batch'``.

        threshold_minimum (float): Minimum acceptable kig2-normalized UMI threshold. Potential 
            thresholds below this value will be ignored. Default is ``4.0``.

        threshold_maximum (float): Maximum acceptable kig2-normalized UMI threshold. Potential 
            thresholds above this value will be ignored. Default is ``10.0``.

        kde_maximum (float): Upper limit of the KDE (in log2-normalized UMI counts). This should 
            be below the maximum number of UMI counts, or else strange results may occur. Default 
            is ``15.0``.

        assignments_only (bool): If ``True``, return a pandas ``Series`` object containing only the 
            group assignment. Suitable for appending to an existing dataframe.

        debug (bool): produces plots and prints intermediate information for debugging. Default is 
            ``False``.
    '''
    # parse hash names
    if hash_names is None:
        if ignore_cellhash_case:
            cellhash_pattern = re.compile(cellhash_regex, flags=re.IGNORECASE)
        else:
            cellhash_pattern = re.compile(cellhash_regex)
        hash_names = [re.search(cellhash_pattern, o) is not None for o in adata.obs.columns]
    if batch_names is None:
        batch_names = {}
    # compute thresholds
    thresholds = {}
    for hash_name in hash_names:
        if debug:
            print(hash_name)
        thresholds[hash_name] = positive_feature_cutoff(adata.obs[hash_name],
                                                        threshold_minimum=threshold_minimum,
                                                        threshold_maximum=threshold_maximum,
                                                        kde_maximum=kde_maximum,
                                                        debug=debug)
    if debug:
        print('THRESHOLDS')
        print('----------')
        for hash_name in hash_names:
            print(f'{hash_name}: {thresholds[hash_name]}')
    assignments = []
    for _, row in adata.obs[hash_names].iterrows():
        a = [h for h in hash_names if row[h] >= thresholds[h]]
        if len(a) == 1:
            assignment = batch_names.get(a[0], a[0])
        elif len(a) > 1:
            assignment = 'doublet'
        else:
            assignment = 'unassigned'
        assignments.append(assignment)
    if assignments_only:
        return pd.Series(assignments, index=adata.obs_names)
    else:
        adata.obs[batch_key] = assignments
        return adata


def positive_feature_cutoff(vals, threshold_maximum=10.0, threshold_minimum=4.0, kde_maximum=15.0,
                            debug=False, show_cutoff_value=False, cutoff_text='cutoff', debug_figfile=None):
    a = np.array(vals)
    k = _bw_silverman(a)
    kde = KernelDensity(kernel='gaussian', bandwidth=k).fit(a.reshape(-1, 1))
    s = np.linspace(0, kde_maximum, num=int(kde_maximum * 100))
    e = kde.score_samples(s.reshape(-1,1))
    
    all_min, all_max = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    if len(all_min) > 1:
        _all_min = np.array([m for m in all_min if s[m] <= threshold_maximum and s[m] >= threshold_minimum])
        min_vals = zip(_all_min, e[_all_min])
        mi = sorted(min_vals, key=lambda x: x[1])[0][0]
        cutoff = s[mi]
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
            plt.vlines(cutoff, min(e), max(e),
                       colors='k', alpha=0.5, linestyles=':', linewidths=2)
            # text
            text_xadj = 0.025 * (max(s) - min(s))
            cutoff_string = f'{cutoff_text}: {round(cutoff, 3)}' if show_cutoff_value else cutoff_text
            plt.text(cutoff - text_xadj, max(e), cutoff_string, ha='right', va='top', fontsize=14)
            # style
            ax = plt.gca()
            for spine in ['right', 'top']:
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_xlabel('$\mathregular{log_2}$ UMI counts', fontsize=14)
            ax.set_ylabel('kernel density', fontsize=14)
            # save or show
            if debug_figfile is not None:
                plt.tight_layout()
                plt.savefig(debug_figfile)
            else:
                plt.show()
        print('bandwidth: {}'.format(k))
        print('local minima: {}'.format(s[all_min]))
        print('local maxima: {}'.format(s[all_max]))
        if cutoff is not None:
            print('cutoff: {}'.format(cutoff))
        else:
            print('WARNING: no local minima were found, so the threshold could not be calculated.')
        print('\n\n')
    return cutoff


def negative_feature_cutoff(vals, threshold_maximum=10.0, threshold_minimum=4.0, kde_maximum=15.0, denominator=2.0,
                            debug=False, show_cutoff_value=False, cutoff_text='cutoff', debug_figfile=None):
    a = np.array(vals)
    k = _bw_silverman(a)
    kde = KernelDensity(kernel='gaussian', bandwidth=k).fit(a.reshape(-1, 1))
    s = np.linspace(0, kde_maximum, num=int(kde_maximum * 100))
    e = kde.score_samples(s.reshape(-1,1))
    
    all_min, all_max = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    if len(all_min) > 1:
        _all_min = np.array([m for m in all_min if s[m] <= threshold_maximum and s[m] >= threshold_minimum])
        min_vals = zip(_all_min, e[_all_min])
        mi = sorted(min_vals, key=lambda x: x[1])[0][0]
        ma = [m for m in all_max if s[m] < s[mi]][-1]
        cutoff = s[int((mi + ma) / denominator)]
    elif len(all_min) == 1:
        mi = all_min[0]
        ma = all_max[0]
        cutoff = s[int((mi + ma) / denominator)]
    else:
        cutoff = None
    if debug:
        if cutoff is not None:
            # plot
            plt.plot(s, e)
            plt.fill_between(s, e, y2=[min(e)] * len(s), alpha=0.1)
            plt.vlines(s[mi], min(e), e[mi], colors='k', alpha=0.5, linestyles=':')
            plt.vlines(s[ma], min(e), e[ma], colors='k', alpha=0.5, linestyles=':')
            plt.vlines(cutoff, min(e), max(e),
                       colors='k', alpha=0.5, linestyles=':', linewidths=2)
            # text
            text_ymin = min(e) + (0.025 * (max(e) - min(e)))
            text_xadj = 0.025 * (max(s) - min(s))
            plt.text(s[mi] + text_xadj, text_ymin, 'local\nmin', ha='left', va='bottom', fontsize=12)
            plt.text(s[ma] - text_xadj, text_ymin, 'local\nmax', ha='right', va='bottom', fontsize=12)
            cutoff_string = f'{cutoff_text}: {round(cutoff, 3)}' if show_cutoff_value else cutoff_text
            plt.text(cutoff + text_xadj, max(e), cutoff_string, ha='left', va='top', fontsize=14)
            # style
            ax = plt.gca()
            for spine in ['right', 'top']:
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_xlabel('$\mathregular{log_2}$ UMI counts', fontsize=14)
            ax.set_ylabel('kernel density', fontsize=14)
            # save or show
            if debug_figfile is not None:
                plt.tight_layout()
                plt.savefig(debug_figfile)
            else:
                plt.show()
        print('bandwidth: {}'.format(k))
        print('local minima: {}'.format(s[all_min]))
        print('local maxima: {}'.format(s[all_max]))
        if cutoff is not None:
            print('cutoff: {}'.format(cutoff))
        else:
            print('WARNING: no local minima were found, so the threshold could not be calculated.')
        print('\n\n')
    return cutoff


def _bw_silverman(x):
    normalize = 1.349
    IQR = (scoreatpercentile(x, 75) - scoreatpercentile(x, 25)) / normalize
    std_dev = np.std(x, axis=0, ddof=1)
    if IQR > 0:
        A = np.minimum(std_dev, IQR)
    else:
        A = std_dev
    n = len(x)
    return .9 * A * n ** (-0.2)



