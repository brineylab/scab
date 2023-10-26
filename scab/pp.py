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
from typing import Optional, Union

import scanpy as sc

from anndata import AnnData


def filter_and_normalize(
    adata: AnnData,
    make_var_names_unique: bool = True,
    min_genes: int = 200,
    min_cells: Optional[float] = None,
    n_genes_by_counts: int = 2500,
    percent_mito: Union[int, float] = 10,
    percent_ig: Union[int, float] = 100,
    hvg_batch_key: Optional[str] = None,
    ig_regex_pattern: str = "IG[HKL][VDJ][1-9].+|TR[ABDG][VDJ][1-9]",
    regress_out_mt: bool = False,
    regress_out_ig: bool = False,
    target_sum: Optional[int] = None,
    n_top_genes: Optional[int] = None,
    normalization_flavor: str = "cell_ranger",
    log: bool = True,
    scale_max_value: Optional[float] = None,
    save_raw: bool = True,
    verbose: bool = True,
) -> AnnData:
    """
    performs quality filtering and normalization of 10x Genomics count data

    Parameters
    ----------

    adata : anndata.AnnData
        ``AnnData`` object containing gene count data.

    make_var_names_unique : bool, default=True
        If ``True``, ``adata.var_names_make_unique()`` will be called before filtering and
        normalization.

    min_genes : int, default=200
        Minimum number of identified genes for a droplet to be considered a valid cell.
        Passed to ``sc.pp.filter_cells()`` as the ``min_genes`` parameter.

    min_cells : int, optional
        Minimum number of cells in which a gene has been identified. Genes below this
        threshold will be filtered. If not provided, a dynamic threshold equal to
        0.1% of the total number of cells in the dataset will be used.

    n_genes_by_counts : int, default=2500
        Threshold for filtering cells based on the number of genes by counts.

    percent_mito : int or float, default=10
        Threshold for filtering cells based on the percentage of mitochondrial genes.

    hvg_batch_key : str, optional
        When processing an ``AnnData`` object containing multiple samples that may
        require integration and batch correction, `hvg_batch_key` will be passed to
        ``sc.pp.highly_variable_genes()`` to force separate identification of highly
        variable genes for each batch. If not provided, variable genes will be computed
        on the entire dataset.

    ig_regex_pattern : str, default='IG[HKL][VDJ][1-9].+|TR[ABDG][VDJ][1-9]'
        Regular expression pattern used to identify immunoglobulin genes. The default
        is designed to match all immunoglobulin germline gene segments (V, D and J).
        Constant region genes are not matched.

    target_sum : int, optional
        Target read count for normalization, passed to ``sc.pp.normalize_total()``. If not
        provided, the median count of all cells (pre-normalization) is used.

    n_top_genes : int, optional
        The number of top highly variable genes to retain. If not provided, the default
        number of genes for the selected normalization flavor is used.

    normalization_flavor : str, default='cell_ranger'
        Options are ``'cell_ranger'``, ``'seurat'`` or ``'seurat_v3'``.

    log : bool, default=True
        If ``True``, counts will be log-plus-1 transformed.

    scale_max_value : float, optional
        Value at which normalized count values will be clipped. Default is no clipping.

    save_raw : bool, default=True
        If ``True``, normalized and filtered data will be saved to ``adata.raw`` prior to
        scaling and regressing out mitochondrial/immmunoglobulin genes.

    verbose : bool, default=True
        If ``True``, progress updates will be printed.

    """

    if make_var_names_unique:
        adata.var_names_make_unique()
    if verbose:
        print(f"filtering cells with fewer than {min_genes} genes...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells is None:
        min_cells = int(0.001 * adata.shape[0])
    if verbose:
        print(f"filtering genes found in fewer than {min_cells} cells...")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if verbose:
        print("QC...")
    ig_pattern = re.compile(ig_regex_pattern)
    adata.var["ig"] = [re.match(ig_pattern, g) is not None for g in adata.var.index]
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ig"], percent_top=None, log1p=False, inplace=True
    )
    if verbose:
        print("filtering based on percent Ig and percent mito...")
    adata = adata[adata.obs.pct_counts_ig < percent_ig, :]
    adata = adata[adata.obs.pct_counts_mt < percent_mito, :]
    adata = adata[adata.obs.n_genes_by_counts < n_genes_by_counts, :]
    # normalize and log transform
    if verbose:
        print("normalizing...")
    if normalization_flavor == "seurat_v3":
        if n_top_genes is None:
            n_top_genes = 3500
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            batch_key=hvg_batch_key,
            flavor=normalization_flavor,
        )
        sc.pp.normalize_total(adata, target_sum=target_sum)
        if log:
            sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        if log:
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            batch_key=hvg_batch_key,
            flavor=normalization_flavor,
        )
    if save_raw:
        adata.raw = adata
    if regress_out_mt:
        if verbose:
            print("regressing out mitochondrial genes...")
        sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    if regress_out_ig:
        if verbose:
            print("regressing out immunoglobulin genes...")
        sc.pp.regress_out(adata, ["total_counts", "pct_counts_ig"])
    if verbose:
        print("scaling...")
    sc.pp.scale(adata, max_value=scale_max_value)
    return adata


def scrublet(adata: AnnData, verbose: bool = True) -> AnnData:
    """
    Predicts doublets using scrublet_ [Wolock19]_.

    .. seealso::
       | Samuel L. Wolock, Romain Lopez, Allon M. Klein  
       | Scrublet: Computational Identification of Cell Doublets in Single-Cell Transcriptomic Data  
       | *Cell Systems* 2019. https://doi.org/10.1016/j.cels.2018.11.005  


    Parameters
    ----------

    adata : anndata.AnnData  
        ``AnnData`` object containing gene count data.

    verbose : bool, default=True  
        If ``True``, progress updates will be printed.  

    Returns
    -------
    Returns an updated `adata` object with doublet predictions found at \
    ``adata.obs.is_doublet`` and doublet scores at ``adata.obs.doublet_score``.


    .. _scrublet: https://github.com/swolock/scrublet
    """
    import scrublet

    scrub = scrublet.Scrublet(adata.raw.X)
    adata.obs["doublet_score"], adata.obs["is_doublet"] = scrub.scrub_doublets(
        verbose=verbose
    )
    if verbose:
        scrub.plot_histogram()
        print("Identified {} potential doublets".format(sum(adata.obs["is_doublet"])))
    return adata


def doubletdetection(
    adata: AnnData,
    n_iters: int = 25,
    use_phenograph: bool = False,
    standard_scaling: bool = True,
    p_thresh: float = 1e-16,
    voter_thresh: float = 0.5,
    verbose: bool = False,
) -> AnnData:
    """
    Predicts doublets using doubletdetection_ [Gayoso20]_.  

    .. seealso::
       | Adam Gayoso, Jonathan Shor, Ambrose J Carr, Roshan Sharma, Dana Pe'er  
       | DoubletDetection (Version v3.0)  
       | *Zenodo* 2020. http://doi.org/10.5281/zenodo.2678041  


    Parameters
    ----------

    adata : anndata.AnnData  
        ``AnnData`` object containing gene counts data.

    n_iters : int, default=25  
        Iterations of doubletdetection to perform.  

    use_phenograph : bool, default=False  
        Passed directly to ``doubletdection.BoostClassifier()``.  

    standard_scaling : bool, default=True  
        Passed directly to ``doubletdection.BoostClassifier()``.  

    p_thresh : float, default=1e-16  
        P-value threshold for doublet classification.  

    voter_thresh : float, default=0.5  
        Voter threshold, as a fraction of all voters.  

    verbose : bool, default=True  
        If ``True``, progress updates will be printed.  

    Returns
    -------
    Returns an updated `adata` object with doublet predictions found at ``adata.obs.is_doublet`` \
    and doublet scores at ``adata.obs.doublet_score``. Note that ``adata.obs.is_doublet`` values are \
    ``0.0`` and ``1.0``, not ``True`` and ``False``. This is the default output of ``doubletdetection`` \
    and is useful for plotting doublets using ``scanpy.pl.umap()``, which cannot handle boolean \
    color values.  


    .. _doubletdetection: https://github.com/JonathanShor/DoubletDetection

    """
    import doubletdetection

    clf = doubletdetection.BoostClassifier(
        n_iters=n_iters,
        use_phenograph=use_phenograph,
        verbose=verbose,
        standard_scaling=standard_scaling,
    )
    fit = clf.fit(adata.raw.X)
    doublets = fit.predict(p_thresh=p_thresh, voter_thresh=voter_thresh)
    adata.obs["is_doublet"] = doublets
    adata.obs["doublet_score"] = clf.doublet_score()
    return adata


def remove_doublets(
    adata: AnnData,
    doublet_identification_method: Optional[str] = None,
    verbose: bool = True,
) -> AnnData:
    """
    Removes doublets. If not already performed, doublet identification is performed 
    using either doubletdetection or scrublet.

    Parameters
    ----------

    adata : anndata.AnnData): 
        ``AnnData`` object containing gene count data.  

    doublet_identification_method : str, default='doubletdetection'  
        Method for identifying doublets. Only used if ``adata.obs.is_doublet`` does not
        already exist. Options are ``'doubletdetection'`` and ``'scrublet'``.

    verbose : bool, default=True  
        If ``True``, progress updates will be printed.  

    Returns
    -------
    An updated ``adata`` object that does not contain observations that were \
    identified as doublets.

    """
    if "is_doublet" not in adata.obs.columns:
        if doublet_identification_method.lower() == "scrublet":
            adata = scrublet(adata, verbose=verbose)
        else:
            adata = doubletdetection(adata, verbose=verbose)
    singlets = [not o for o in adata.obs["is_doublet"]]
    adata = adata[singlets, :]
    return adata
