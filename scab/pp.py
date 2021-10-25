#!/usr/bin/env python
# filename: pp.py


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

import scanpy as sc
import scrublet



def filter_and_normalize(adata, make_var_names_unique=True, min_genes=200, min_cells=None,
                         n_genes_by_counts=2500, percent_mito=10, hvg_batch_key=None,
                         remove_ig=True, ig_removal_pattern='IG[HKL][VDJ][1-9].+|TR[ABDG][VDJ][1-9]',
                         target_sum=None, n_top_genes=None, normalization_flavor='cell_ranger', log=True,
                         scale_max_value=None, save_raw=True, verbose=True):
    '''
    performs quality filtering and normalization of 10x Genomics count data

    Args:
    -----

        adata (anndata.AnnData): AnnData object containing gene count data.

        make_var_names_unique (bool): If ``True``, ``adata.var_names_make_unique()`` will be called
                           before filtering and normalization. Default is ``True``.

        min_genes (int): Minimum number of identified genes for a droplet to be considered a valid cell.
                         Passed to ``sc.pp.filter_cells()`` as the ``min_genes`` kwarg. Default is ``200``.

        min_cells (int): Minimum number of cells in which a gene has been identified. Genes below this
                         threshold will be filtered. Default is ``None``, which uses a dynamic 
                         threshold equal to 0.1% of the total number of cells in the dataset. 

        n_genes_by_counts (int): Threshold for filtering cells based on the number of genes by counts.
                                 Default is ``2500``, which results in cells with more than 2500 genes by counts
                                 being filtered from the dataset.

        percent_mito (float): Threshold for filtering cells based on the fraction of mitochondrial genes.
                              Default is ``10``, which results in cells with more than 10% mitochondrial
                              genes being filtered from the dataset.

        hvg_batch_key (str): When processing an ``AnnData`` object containing multiple samples that may
                             require integration and batch correction, ``hvg_batch_key`` will be passed to
                             ``sc.pp.highly_variable_genes()`` to force separate identification of highly
                             variable genes for each batch. Default is ``None``, which results in highly
                             variable genes being computed on the entire dataset.

        remove_ig (bool): If ``True``, immunoglobulin genes will be filtered from the dataset. Default is ``True``.

        ig_removal_pattern (str): Regular expression pattern used to identify immunoglobulin genes. Default is
                                  ``'IG[HKL][VDJ][1-9].+|TR[ABDG][VDJ][1-9]"``, which filteres all immunoglobulin
                                  germline gene segments (V, D and J). Constant region genes are not removed.

        target_sum (int): Target read count for normalization, passed to ``sc.pp.normalize_total()``. Default
                          is ``None``, which uses the median count of all cells (pre-normalization).

        n_top_genes (int): The number of top highly variable genes to retain. Default is ``None``, which results
                           in the default number of genes for the selected normalization flavor.

        normalization_flavor (str): Options are ``'cell_ranger'``, ``'seurat'`` or ``'seurat_v3'``. Default
                                    is ``'cell_ranger'``.

        log (bool): If ``True``, counts will be log2 transformed. Default is ``True``.

        scale_max_value (float): Value at which normalized count values will be clipped. Default is ``None``,
                                 which results in no clipping.

        doublet_removal (bool): If ``True``, doublets will be identified using ``scrublet.scrub_doublets()``
                                and removed.

        save_raw (bool): If ``True``, normalized and filtered data will be saved to ``adata.raw`` prior to
                         scaling, regressing out mitochondrial genes, and doublet removal. Default is ``True``.

        verbose (bool): If ``True``, progress updates will be printed. Default is ``True``.
    '''

    if make_var_names_unique:
        adata.var_names_make_unique()
    if verbose:
        print(f'filtering cells with fewer than {min_genes} genes...')
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells is None:
        min_cells = int(0.001 * adata.shape[0])
    if verbose:
        print(f'filtering genes found in fewer than {min_cells} cells...')
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if remove_ig:
        if verbose:
            print('removing immunoglobulin genes...')
        pattern = re.compile(ig_removal_pattern)
        not_ig = [g for g in adata.var.index if not re.match(pattern, g)]
        adata = adata[:, not_ig]
    # filter by mito and n_genes_by_counts
    if verbose:
        print('filtering mitochondrial genes...')
    if 'mt' not in adata.var:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],
                                   percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < percent_mito, :]
    adata = adata[adata.obs.n_genes_by_counts < n_genes_by_counts, :]
    # normalize and log transform
    if verbose:
        print('normalizing...')
    if normalization_flavor == 'seurat_v3':
        if n_top_genes is None:
            n_top_genes = 3500
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=hvg_batch_key, 
                                    flavor=normalization_flavor)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        if log:
            sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        if log:
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=hvg_batch_key,
                                    flavor=normalization_flavor)
    if save_raw:
        adata.raw = adata
    if verbose:
        print('regressing out mitochondrial genes...')
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    if verbose:
        print('scaling...')
    sc.pp.scale(adata, max_value=scale_max_value)
    return adata


def predict_doublets(adata, verbose=True):
    '''
    Predicts doublets using scrublet.

    Args:
    -----

        adata (anndata.AnnData): AnnData object containing gene count data.

        verbose (bool): If ``True``, progress updates will be printed. Default is ``True``.

    Returns:
    --------

        Returns an anndata.AnnData object with doublet predictions found at ``adata.obs.predicted_doublets``..
    '''
    scrub = scrublet.Scrublet(adata.raw.X)
    adata.obs['doublet_scores'], adata.obs['predicted_doublets'] = scrub.scrub_doublets(verbose=verbose)
    if verbose:
        scrub.plot_histogram()
        print('Identified {} potential doublets'.format(sum(adata.obs['predicted_doublets'])))
    return adata


def remove_doublets(adata, verbose=True):
    '''
    Removes doublets (doublet identification performed using scrublet).

    Args:
    -----

        adata (anndata.AnnData): AnnData object containing gene count data.

        verbose (bool): If ``True``, progress updates will be printed. Default is ``True``.

    Returns:
    --------

        Returns an anndata.AnnData object without droplets identified as doublets.
    '''
    if "predicted_doublets" not in adata.obs.columns:
        adata = predict_doublets(adata, verbose=verbose)
    singlets = [not o for o in adata.obs['predicted_doublets']]
    adata = adata[singlets,:]
    return adata