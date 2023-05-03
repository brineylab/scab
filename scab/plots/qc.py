#!/usr/bin/env python
# filename: qc.py


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


from collections import Counter
import os
import re
from typing import Iterable, Optional

import matplotlib.pyplot as plt

import scanpy as sc

import seaborn as sns

from anndata import AnnData


def qc_metrics(
    adata: AnnData,
    ngenes_cutoff: int = 2500,
    mito_cutoff: float = 10.0,
    ig_cutoff: float = 100.0,
    read_count_bounds: Iterable = [0, 5000],
    gene_count_bounds: Iterable = [0, 500],
    fig_dir: Optional[str] = None,
    fig_prefix: Optional[str] = None,
):
    """
    
    """
    if "ig" not in adata.var:
        pattern = re.compile("IG[HKL][VDJ][1-9].+|TR[ABDG][VDJ][1-9]")
        adata.var["ig"] = [
            False if re.match(pattern, a) is None else True for a in adata.var.index
        ]
    if "mt" not in adata.var:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["ig", "mt"], percent_top=None, log1p=False, inplace=True
    )

    palette = {"include": "#202020", "exclude": "#C0C0C0"}
    hue_order = ["include", "exclude"]

    # plot Ig
    ig_hue = [
        "include" if i < ig_cutoff else "exclude" for i in adata.obs.pct_counts_ig
    ]
    ig_counter = Counter(ig_hue)
    g = sns.JointGrid(data=adata.obs, x="total_counts", y="pct_counts_ig")
    g.plot_joint(
        sns.scatterplot,
        s=10,
        linewidth=0,
        hue=ig_hue,
        hue_order=hue_order,
        palette=palette,
    )
    g.plot_marginals(sns.kdeplot, shade=True, color="#404040")
    g.ax_joint.set_xlabel("total counts", fontsize=16)
    g.ax_joint.set_ylabel("immunoglobulin counts (%)", fontsize=16)
    g.ax_joint.tick_params(axis="both", labelsize=13)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    labels = [f"{l} ({ig_counter[l]})" for l in labels]
    g.ax_joint.legend(
        handles, labels, title="ig filter", title_fontsize=14, fontsize=13
    )
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f"{fig_prefix}_pct-counts-ig.pdf"
        else:
            fig_name = "pct_counts_ig.pdf"
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()

    # plot mito
    mito_hue = [
        "include" if i < mito_cutoff else "exclude" for i in adata.obs.pct_counts_mt
    ]
    mito_counter = Counter(mito_hue)
    g = sns.JointGrid(data=adata.obs, x="total_counts", y="pct_counts_mt")
    g.plot_joint(
        sns.scatterplot,
        s=10,
        linewidth=0,
        hue=mito_hue,
        hue_order=hue_order,
        palette=palette,
    )
    g.plot_marginals(sns.kdeplot, shade=True, color="#404040")
    g.ax_joint.set_xlabel("total counts", fontsize=16)
    g.ax_joint.set_ylabel("mitochondrial counts (%)", fontsize=16)
    g.ax_joint.tick_params(axis="both", labelsize=13)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    labels = [f"{l} ({mito_counter[l]})" for l in labels]
    g.ax_joint.legend(
        handles, labels, title="mito filter", title_fontsize=14, fontsize=13
    )
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f"{fig_prefix}_pct-counts-mt.pdf"
        else:
            fig_name = "pct_counts_mt.pdf"
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()

    # plot N genes by counts
    ngenes_hue = [
        "include" if i < ngenes_cutoff else "exclude"
        for i in adata.obs.n_genes_by_counts
    ]
    ngenes_counter = Counter(ngenes_hue)
    g = sns.JointGrid(data=adata.obs, x="total_counts", y="n_genes_by_counts")
    g.plot_joint(
        sns.scatterplot,
        s=10,
        linewidth=0,
        hue=ngenes_hue,
        hue_order=hue_order,
        palette=palette,
    )
    g.plot_marginals(sns.kdeplot, shade=True, color="#404040")
    g.ax_joint.set_xlabel("total counts", fontsize=16)
    g.ax_joint.set_ylabel("number of genes", fontsize=16)
    g.ax_joint.tick_params(axis="both", labelsize=13)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    labels = [f"{l} ({ngenes_counter[l]})" for l in labels]
    g.ax_joint.legend(
        handles, labels, title="genes filter", title_fontsize=14, fontsize=13
    )
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f"{fig_prefix}_n-genes-by-counts.pdf"
        else:
            fig_name = "n_genes_by_counts.pdf"
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()

    # histogram of read counts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 4])
    sns.histplot(data=adata.obs, x="total_counts", binwidth=500, ax=ax1)
    sns.histplot(data=adata.obs, x="total_counts", binwidth=100, ax=ax2)
    ax2.set_xlim(read_count_bounds)

    for ax in [ax1, ax2]:
        ax.set_xlabel("read count", fontsize=16)
        ax.set_ylabel("# of cells", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        for s in ["left", "right", "top"]:
            ax.spines[s].set_visible(False)
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f"{fig_prefix}_read-counts.pdf"
        else:
            fig_name = "read-counts.pdf"
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()

    # histogram of gene counts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 4])
    sns.histplot(data=adata.obs, x="n_genes_by_counts", binwidth=100, ax=ax1)
    sns.histplot(data=adata.obs, x="n_genes_by_counts", binwidth=10, ax=ax2)
    ax2.set_xlim(gene_count_bounds)

    for ax in [ax1, ax2]:
        ax.set_xlabel("gene count", fontsize=16)
        ax.set_ylabel("# of cells", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        for s in ["left", "right", "top"]:
            ax.spines[s].set_visible(False)
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f"{fig_prefix}_gene-counts.pdf"
        else:
            fig_name = "gene-counts.pdf"
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()
