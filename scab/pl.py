#!/usr/bin/env python
# filename: pl.py


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
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scanpy as sc

import seaborn as sns

from anndata import AnnData

from natsort import natsorted

from abstar.core.germline import get_germlines

from abutils.utils.utilities import nested_dict_lookup



# ===========================

#      QUALITY CONTROL

# ===========================


def qc_metrics(adata, ngenes_cutoff=2500, mito_cutoff=10, ig_cutoff=50,
               fig_dir=None, fig_prefix=None):
    if 'ig' not in adata.var:
        pattern = re.compile('IG[HKL][VDJ][1-9].+|TR[ABDG][VDJ][1-9]')
        adata.var['ig'] = [False if re.match(pattern, a) is None else True for a in adata.var.index]
    if 'mt' not in adata.var:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['ig', 'mt'],
                               percent_top=None, log1p=False, inplace=True)

    palette = {'include': '#202020', 'exclude': '#C0C0C0'}
    hue_order = ['include', 'exclude']

    # plot Ig
    ig_hue = ['include' if i < ig_cutoff else 'exclude' for i in adata.obs.pct_counts_ig]
    ig_counter = Counter(ig_hue)
    g = sns.JointGrid(data=adata.obs, x='total_counts', y='pct_counts_ig')
    g.plot_joint(sns.scatterplot, s=10, linewidth=0, 
                 hue=ig_hue, hue_order=hue_order, palette=palette)
    g.plot_marginals(sns.kdeplot, shade=True, color='#404040')
    g.ax_joint.set_xlabel('total counts', fontsize=16)
    g.ax_joint.set_ylabel('immunoglobulin counts (%)', fontsize=16)
    g.ax_joint.tick_params(axis='both', labelsize=13)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    labels = [f'{l} ({ig_counter[l]})' for l in labels]
    g.ax_joint.legend(handles, labels, title='ig filter', title_fontsize=14, fontsize=13)
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f'{fig_prefix}_pct-counts-ig.pdf'
        else:
            fig_name = 'pct_counts_ig.pdf'
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()

    # plot mito
    mito_hue = ['include' if i < mito_cutoff else 'exclude' for i in adata.obs.pct_counts_mt]
    mito_counter = Counter(mito_hue)
    g = sns.JointGrid(data=adata.obs, x='total_counts', y='pct_counts_mt')
    g.plot_joint(sns.scatterplot, s=10, linewidth=0, 
                 hue=mito_hue, hue_order=hue_order, palette=palette)
    g.plot_marginals(sns.kdeplot, shade=True, color='#404040')
    g.ax_joint.set_xlabel('total counts', fontsize=16)
    g.ax_joint.set_ylabel('mitochondrial counts (%)', fontsize=16)
    g.ax_joint.tick_params(axis='both', labelsize=13)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    labels = [f'{l} ({mito_counter[l]})' for l in labels]
    g.ax_joint.legend(handles, labels, title='mito filter', title_fontsize=14, fontsize=13)
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f'{fig_prefix}_pct-counts-mt.pdf'
        else:
            fig_name = 'pct_counts_mt.pdf'
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()

    # plot N genes by counts
    ngenes_hue = ['include' if i < ngenes_cutoff else 'exclude' for i in adata.obs.n_genes_by_counts]
    ngenes_counter = Counter(ngenes_hue)
    g = sns.JointGrid(data=adata.obs, x='total_counts', y='n_genes_by_counts')
    g.plot_joint(sns.scatterplot, s=10, linewidth=0, 
                 hue=ngenes_hue, hue_order=hue_order, palette=palette)
    g.plot_marginals(sns.kdeplot, shade=True, color='#404040')
    g.ax_joint.set_xlabel('total counts', fontsize=16)
    g.ax_joint.set_ylabel('number of genes', fontsize=16)
    g.ax_joint.tick_params(axis='both', labelsize=13)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    labels = [f'{l} ({ngenes_counter[l]})' for l in labels]
    g.ax_joint.legend(handles, labels, title='genes filter', title_fontsize=14, fontsize=13)
    if fig_dir is not None:
        plt.tight_layout()
        if fig_prefix is not None:
            fig_name = f'{fig_prefix}_n-genes-by-counts.pdf'
        else:
            fig_name = 'n_genes_by_counts.pdf'
        plt.savefig(os.path.join(fig_dir, fig_name))
    else:
        plt.show()




# ===========================

#         FEATURES

# ===========================


def feature_kde(data, x, y, hue=None, hue_order=None, colors=None, thresh=0.1,
                show_scatter=True, scatter_size=5, scatter_alpha=0.2,
                fill=False, kde_fill_alpha=0.7, kde_line_alpha=1.0,
                highlight_index=None, highlight_x=None, highlight_y=None, highlight_marker='x',
                highlight_size=90, highlight_color='k', highlight_name=None, highlight_alpha=0.8,
                xlabel=None, ylabel=None, equal_axes=True,
                legend_loc='best', legend_title=None, show_legend_title=True, legend_fontsize=12,
                return_ax=False, figsize=[6, 6], figfile=None, **kwargs):
    '''
    Produces a 2-dimensional KDE plot of two features.

    Args:

        data (anndata.AnnData or pd.DataFramne): An ``AnnData`` object or a ``Pandas`` dataframe
                                                 containing the input data. Required.

        x (str): Name of the column in ``data`` containing the feature to be plotted on the x-axis. Required.

        y (str): Name of the column in ``data`` containing the feature to be plotted on the y-axis. Required.

        hue (str): Name of the column in ``data`` containing categories for hue values. For scatter plots, 
                   the categories in ``hue`` will be plotted as different colored points. For KDE plots,
                   ``hue```` categories will each be plotted as differently colored KDE plots
                   on the same plot. 

        hue_order (iterable): Iterable of hue categories, in the order they should be plotted and listed
                              in the legend. If ```hue_order``` contains only a subset of the categories
                              present in ```data[hue]```, only the categories supplied in ```hue_order```
                              will be plotted.

        colors (iterable): List of colors to be used for ```'hue'``` categories. If ```'colors'``` is
                           shorter than the list of hue categories, colors will be reused.

        thresh (float): Threshold for the KDE. Default is ```0.1```.
        
        show_scatter (bool): Show the scatterplot beneath the transparent KDE plot. Default is ```True```.

        scatter_size (int, float): Size of the scatter points. Default is ```5```.

        scatter_alpha (float): Alpha of the scatter points. Default is ```0.2```.

        fill (bool): Fill the KDE plot. Default is ```True```.

        kde_fill_alpha (float): Alpha for the filled KDE plot. If ```fill``` is ```False```,
                                this option is ignored. Default is ```0.7```.
        
        kde_line_alpha (float): Alpha for the KDE plot lines. Default is ```1.0```.

        highlight_index (iterable): An iterabile of index names (present in ```data```) of points
                                    to be highlighted on the KDE plot. If provided, ```highlight_x```
                                    and ```highlight_y``` are ignored.

        highlight_x (iterable): An iterable of x-values for highlighted points. Also requires
                                ```highlight_y```.
        
        highlight_y (iterable): An iterable of y-values for highlighted points. Also requires
                                ```highlight_x```.
        
        highlight_marker (str): The marker style to be used for highlight points. Accepts 
                                standard matplotlib marker styles. Default is ```'x'```. 

        highlight_size (int): Size of the highlight marker. Default is ```90```.

        highlight_color (string or RGB list): Color of the highlight points. Default is black.

        highlight_name (str): Name of the highlights, to be used in the legend. If not supplied,
                              highlight points will not be included in the legend.
        
        highlight_alpha (float): Alpha for the highlight points. Default is ```0.8```.

        xlabel (str): Label for the x-axis. By default, the value for ```x``` is used.

        ylabel (str): Label for the y-axis. By default, the value for ```y``` is used.

        equal_axes (bool): If ```True```, the the limits of the x- and y-axis will be equal.
                           Default is ```True```.
        
        legend_loc (str): Location for the legend. Uses standard matplotlib locations. Default
                          is ```'best'```.

        legend title (str): Title for the legend. By default, ```hue``` is used.

        show_legend_title (bool): Whether or not to show the legend title. Default is ```True```.

        return_ax (bool): If ```True```, return the plot's ```ax``` object. Will not show or save
                          the plot. Default is ```False```.

        figsize (list): A list containg the dimensions of the plot. Default is ```[6, 6]```.

        figfile (str): Path to which the figure will be saved. If not provided, the figure will be
                       shown but not saved to file.

        kwargs: All other keyword arguments are passed to ``seaborn.kdeplot()``.
    '''

    # input data
    if isinstance(data, AnnData):
        _data = {}
        for var in [x, y, hue]:
            if var is not None:
                if any([var in data.obs.columns.values, var in data.var_names]):
                    _data[var] = data.obs_vector(var)
                else:
                    print('"{}" was not found in the supplied AnnData object.'.format(var))
                    return
        df = pd.DataFrame(_data, index=data.obs_names)
    else:
        _data = {}
        for var in [x, y, hue]:
            if var is not None:
                if var in data.columns.values:
                    _data[var] = data[var]
                else:
                    print('"{}" is not a column in the supplied dataframe'.format(x))
                    return
        df = pd.DataFrame(_data, index=data.index.values)

    # hue
    if hue is not None:
        if hue_order is None:
            hue_order = natsorted(list(set(df[hue])))
        df = df[df[hue].isin(hue_order)]
    else:
        hue_order = []

    # colors
    n_colors = max(1, len(hue_order))
    if colors is None:
        colors = sns.hls_palette(n_colors=n_colors)
        
    plt.figure(figsize=figsize)

    # scatterplots
    if show_scatter:
        if hue_order:
            for h, c in zip(hue_order, colors):
                d = df[df[hue] == h]
                plt.scatter(d[x], d[y], c=[c], s=scatter_size,
                            alpha=scatter_alpha, linewidths=0)
        else:
            plt.scatter(df[x], df[y], c=[colors[0]], s=scatter_size,
                            alpha=scatter_alpha, linewidths=0)

    # kdeplot
    if fill:
        if hue_order:
            sns.kdeplot(data=df, x=x, y=y, hue=hue, fill=True, alpha=kde_fill_alpha,
                        hue_order=hue_order, palette=colors, thresh=thresh, **kwargs)
        else:
            sns.kdeplot(data=df, x=x, y=y, fill=True, alpha=kde_fill_alpha, 
                        color=colors[0], thresh=thresh, **kwargs)
    if hue_order:
        ax = sns.kdeplot(data=df, x=x, y=y, hue=hue, alpha=kde_line_alpha,
                        hue_order=hue_order, palette=colors, thresh=thresh, **kwargs)
    else:
        ax = sns.kdeplot(data=df, x=x, y=y, alpha=kde_line_alpha,
                        color=colors[0], thresh=thresh, **kwargs)
    
    # highlighted points
    highlight = any([highlight_index is not None, all([highlight_x is not None, highlight_y is not None])])
    if highlight:
        if highlight_index is not None:
            hidata = df.loc[highlight_index]
            highlight_x = hidata[x]
            highlight_y = hidata[y]
        plt.scatter(highlight_x, highlight_y, zorder=10,
                    s=highlight_size,
                    c=highlight_color,
                    alpha=highlight_alpha,
                    marker=highlight_marker)

    # legend
    legend_labels = hue_order
    if fill:
        handles = []
        for c in colors:
            f = Patch(fc=c, alpha=kde_fill_alpha / 3)
            e = Patch(ec=c, fill=False, lw=1.5)
            handles.append((f, e))
#         handles = [Patch(fc=c, ec=c, alpha=kde_fill_alpha / 3, label=h) for c, h in zip(colors, hue_order)]
    else:
        handles = [Line2D([0], [0], color=c) for c in colors]
    if highlight_name is not None:
        legend_labels.append(highlight_name)
        handles.append(Line2D([0], [0], marker=highlight_marker, color='w',
                              mec=highlight_color,
                              mfc=highlight_color,
                              ms=highlight_size / 10))
    if show_legend_title:
        legend_title = legend_title if legend_title is not None else hue
    else:
        legend_title = None
    ax.legend(handles, legend_labels, loc=legend_loc, fontsize=legend_fontsize, title=legend_title, frameon=False)
    
    # style the plot
    ax.set_xlabel(xlabel if xlabel is not None else x, fontsize=16)
    ax.set_ylabel(ylabel if ylabel is not None else y, fontsize=16)
    ax.tick_params(axis='both', labelsize=13)

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position(('outward', 10))

    if equal_axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        axlim = [min([xlim[0], ylim[0]]), max([xlim[1], ylim[1]])]
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)
    
    if return_ax:
        return ax
    elif figfile is not None:
        plt.tight_layout()
        plt.savefig(figfile)
    else:
        plt.show()


    
def feature_scatter(data, x, y, hue=None, hue_order=None, color=None, cmap=None, marker='o', size=20, alpha=0.6,
                    highlight_index=None, highlight_x=None, highlight_y=None, highlight_marker='x',
                    highlight_size=90, highlight_color='k', highlight_name=None, highlight_alpha=0.9,
                    xlabel=None, ylabel=None, equal_axes=True, force_categorical_hue=False,
                    legend_loc='best', legend_title=None, legend_fontsize=13, legend_frameon=True,
                    cbar_width=35, cbar_height=5, cbar_loc='lower right', cbar_orientation='horizontal', 
                    cbar_bbox_to_anchor=None, cbar_flip_ticks=False,
                    cbar_title=None, cbar_title_loc=None, cbar_title_fontsize=12, 
                    return_ax=False, figsize=[6, 6], figfile=None):
    '''
    Produces a scatter plot of two features, optionally colored by a third feature.

    Args:

        data (anndata.AnnData or pd.DataFramne): An ``AnnData`` object or a ``Pandas`` dataframe
                                                 containing the input data. Required.

        x (str): Name of the column in ``data`` containing the feature to be plotted on the x-axis. Required.

        y (str): Name of the column in ``data`` containing the feature to be plotted on the y-axis. Required.

        hue (str): Name of the column in ``data`` containing categories for hue values. If ``hue`` is categorical,
                   each category will be plotted in a different color (using the ``color`` for the colors). If 
                   ``hue`` is continuous, points will be colored using a colormap (using ``cmap`` if supplied). 

        hue_order (iterable): Iterable of hue categories, in the order they should be plotted and listed
                              in the legend. If ```hue_order``` contains only a subset of the categories
                              present in ```data[hue]```, only the categories supplied in ```hue_order```
                              will be plotted.

        force_categorical_hue (bool): If ``True``, ``hue`` data will be treated as categorical, even if
                                      the data appear to be continuous. This results in ``color`` being used
                                      to color the points rather than ``cmap``. Default is ``False``.

        color (iterable): List of colors to be used for ``hue`` categories. If ``colors`` is
                          shorter than the list of hue categories, colors will be reused. Only used
                          if ``hue`` contains categorical data (``cmap`` is used for continuous data). 
                          Default is to use ``sns.color_palette()``.
        
        camp (str or matplotlib.color.Colormap): Colormap to be used for continuous ``hue`` data. Default 
                                                 is to use ``'flare'``.
        
        marker (str): Marker for the scatter plot. Accepts standard matplotlib marker styles.
                      Default is ``'o'``.

        size (int, float): Size of the scatter points. Default is ``20``.

        alpha (float): Alpha of the scatter points. Default is ``0.6``.

        highlight_index (iterable): An iterabile of index names (present in ```data```) of points
                                    to be highlighted on the scatter plot. If provided, ```highlight_x```
                                    and ```highlight_y``` are ignored.

        highlight_x (iterable): An iterable of x-values for highlighted points. Also requires
                                ```highlight_y```.
        
        highlight_y (iterable): An iterable of y-values for highlighted points. Also requires
                                ```highlight_x```.
        
        highlight_marker (str): The marker style to be used for highlight points. Accepts 
                                standard matplotlib marker styles. Default is ``'x'``. 

        highlight_size (int): Size of the highlight marker. Default is ``90``.

        highlight_color (string or RGB list): Color of the highlight points. Default is ``'k'`` (black).

        highlight_name (str): Name of the highlights, to be used in the legend. If not supplied,
                              highlight points will not be included in the legend.
        
        highlight_alpha (float): Alpha for the highlight points. Default is ``0.9``.

        xlabel (str): Label for the x-axis. By default, the value for ``x`` is used.

        ylabel (str): Label for the y-axis. By default, the value for ``y`` is used.

        equal_axes (bool): If ``True``, the the limits of the x- and y-axis will be equal.
                           Default is ``True``.
        
        legend_loc (str): Location for the legend. Uses standard matplotlib locations. Default
                          is ``'best'``.

        legend title (str): Title for the legend. By default, ``hue`` is used.

        show_legend_title (bool): Whether or not to show the legend title. Default is ``True``.

        return_ax (bool): If ``True``, return the plot's ``ax`` object. Will not show or save
                          the plot. Default is ``False``.

        figsize (list): A list containg the dimensions of the plot. Default is ``[6, 6]``.

        figfile (str): Path to which the figure will be saved. If not provided, the figure will be
                       shown but not saved to file.
    '''
    # input data
    if isinstance(data, AnnData):
        _data = {}
        for var in [x, y, hue]:
            if var is not None:
                if any([var in data.obs.columns.values, var in data.var_names]):
                    _data[var] = data.obs_vector(var)
                else:
                    print('"{}" was not found in the supplied AnnData object.'.format(var))
                    return
        df = pd.DataFrame(_data, index=data.obs_names)
    else:
        _data = {}
        for var in [x, y, hue]:
            if var is not None:
                if var in data.columns.values:
                    _data[var] = data[var]
                else:
                    print('"{}" is not a column in the supplied dataframe'.format(x))
                    return
        df = pd.DataFrame(_data, index=data.index.values)
    
    # hue and color
    continuous_hue = False
    if hue is not None:
        if all([isinstance(h, float) for h in df[hue]]) and not force_categorical_hue:
            continuous_hue = True
            hue_order = []
            if cmap is None:
                cmap = sns.color_palette("flare", as_cmap=True)
            else:
                cmap = plt.get_cmap(cmap)
            max_hue = df[hue].max()
            min_hue = df[hue].min()
            df['color'] = [cmap((h - min_hue) / (max_hue - min_hue)) for h in df[hue]]
        else:
            if hue_order is None:
                hue_order = natsorted(list(set(df[hue])))
            n_colors = max(1, len(hue_order))
            if color is None:
                color = sns.color_palette(n_colors=n_colors)
            if len(color) < n_colors:
                color = itertools.cycle(color)
            hue_dict = {h: c for h, c in zip(hue_order, color)}
            df['color'] = [hue_dict[h] for h in df[hue]]
    else:
        hue_order = []
        if color is not None:
            df['color'] = [color] * df.shape[0]
        else:
            df['color'] = [sns.color_palette()[0]] * df.shape[0]

    # scatterplot
    plt.figure(figsize=figsize)
    ax = plt.gca()
    if hue_order:
        for h in hue_order[::-1]:
            d = df[df[hue] == h]
            plt.scatter(d[x], d[y], c=d['color'], s=size, marker=marker,
                        alpha=alpha, linewidths=0, label=h)
    else:
        plt.scatter(df[x], df[y], c=df['color'], s=size, marker=marker,
                        alpha=alpha, linewidths=0)

    # highlighted points
    highlight = any([highlight_index is not None, all([highlight_x is not None, highlight_y is not None])])
    if highlight:
        if highlight_index is not None:
            hi_index = [h for h in highlight_index if h in df.index.values]
            hidata = df.loc[hi_index]
            highlight_x = hidata[x]
            highlight_y = hidata[y]
        plt.scatter(highlight_x, highlight_y, zorder=10,
                    s=highlight_size,
                    c=highlight_color,
                    alpha=highlight_alpha,
                    marker=highlight_marker,
                    label=highlight_name)
    # legend
    if not continuous_hue:
        if hue is not None:
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, title=legend_title, frameon=legend_frameon)
    # colorbar
    else:
        cbax = inset_axes(ax, width=f'{cbar_width}%', height=f'{cbar_height}%',
                          loc=cbar_loc, bbox_to_anchor=cbar_bbox_to_anchor,
                          bbox_transform=ax.transAxes) 
        fig = plt.gcf()
        norm = mpl.colors.Normalize(vmin=min_hue, vmax=max_hue)
        ticks = [round(t, 2) for t in np.linspace(min_hue, max_hue, num=4)]
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax,
                     orientation=cbar_orientation, ticks=ticks,)
        if cbar_orientation == 'horizontal':
            ticks_position = 'bottom' if cbar_flip_ticks else 'top'
            cbax.xaxis.set_ticks_position(ticks_position)
        else:
            ticks_position = 'left' if cbar_flip_ticks else 'right'
            cbax.yaxis.set_ticks_position(ticks_position)
        cbax.set_title(cbar_title, fontsize=cbar_title_fontsize, fontweight='medium')
            
    # style the plot
    ax.set_xlabel(xlabel if xlabel is not None else x, fontsize=16)
    ax.set_ylabel(ylabel if ylabel is not None else y, fontsize=16)
    ax.tick_params(axis='both', labelsize=13)

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_position(('outward', 10))

    if equal_axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        axlim = [min([xlim[0], ylim[0]]), max([xlim[1], ylim[1]])]
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)
    
    if return_ax:
        return ax
    elif figfile is not None:
        plt.tight_layout()
        plt.savefig(figfile)
    else:
        plt.show()


def cellhash_ridge(adata, hashname, category, colors=None, alpha=1.0,
                   categories=None, hide_extra_categories=False, rename=None, xmax=14,
                   ylabel_fontsize=11, xlabel=None, xlabel_fontsize=12, xtick_labelsize=11,
                   feature_label_xoffset=5, figfile=None):
    '''
    Docstring for feature_ridge.
    '''
    
    # input data
    data = adata.obs.copy()
    data = data[data[hashname] <= xmax]
    if category not in data.columns.values:
        print('"{}" is not a column in the supplied dataframe'.format(category))
        return
    
    # rename
    if rename is None:
        rename = {}
    else:
        if not any([k in data.columns.values for k in rename.keys()]):
            rename = {v: k for k, v in rename.items()}

    # categories
    category_set = data[category].unique()
    if categories is None:
        feature_cats = natsorted([c for c in category_set if rename.get(c, c) in data.columns.values])
        extra_cats = natsorted([c for c in category_set if rename.get(c, c) not in feature_cats])
        categories = feature_cats + extra_cats
    else:
        feature_cats = categories
        if hide_extra_categories:
            extra_cats = []
        else:
            extra_cats = [c for c in category_set if rename.get(c, c) not in feature_cats]
        categories = feature_cats + extra_cats

    # colors
    if colors is None:
        n_colors = len(feature_cats)
        colors = list(sns.color_palette(n_colors=n_colors))
        n_greys = len(extra_cats)
        greys = list(plt.get_cmap('Greys')(np.linspace(0, 1, n_greys + 2))[1:-1, :3])
        cdict = {h: c for h, c in zip(categories, colors + greys)}
    elif isinstance(colors, (list, tuple, np.ndarray, pd.core.series.Series)):
        colors = list(colors)
        if len(colors) < len(categories):
            n_greys = len(categories) - len(colors)
            greys = list(plt.get_cmap('Greys')(np.linspace(0, 1, n_greys + 2))[1:-1, :3])
        cdict = {h: c for h, c in zip(categories, colors + greys)}
    else:
        cdict = colors
        if len(cdict) < len(categories):
            missing = [k for k in categories if k not in cdict]
            n_greys = len(missing)
            greys = list(plt.get_cmap('Greys')(np.linspace(0, 1, n_greys + 2))[1:-1, :3])
            for m, g in zip(missing, greys):
                cdict[m] = g
    colors = [cdict[c] for c in categories]

    # plot
    g = sns.FacetGrid(data, row=category, hue=category,
                      aspect=7.5, height=0.75, palette=colors,
                      row_order=categories, hue_order=categories)
    g.map(sns.kdeplot, hashname, clip=[None, xmax], shade=True, alpha=alpha, lw=1.5)
    g.map(sns.kdeplot, hashname, clip=[None, xmax], color="w", lw=3)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                fontsize=ylabel_fontsize,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, hashname)

    # set the subplots to overlap
    g.fig.subplots_adjust(hspace=0.1)

    # remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(xticks=range(0, xmax + 1, 2))
    g.set(yticks=[])
    g.set(xlim=[-feature_label_xoffset, xmax + 0.25])
    g.despine(bottom=True, left=True)
    
    # xlabel
    if xlabel is not None:
        g.set(xlabel=xlabel)
    
    xlabel_position = ((xmax / 2) + feature_label_xoffset) / (xmax + feature_label_xoffset)
    for ax in g.axes.flat:
        ax.set_xlabel(ax.get_xlabel(),
                      x=xlabel_position,
                      fontsize=xlabel_fontsize)
        ax.tick_params(axis='x', labelsize=xtick_labelsize)
        
    # for ax in g.axes.flat:
    #     ax.set_xlabel(ax.get_xlabel(), fontsize=xlabel_fontsize)

    if figfile is not None:
        g.savefig(figfile)
    else:
        plt.show()


def feature_ridge(data, features, colors=None, rename=None,
                  xlabel='UMI count ($\mathregular{log_2}$)', 
                  ylabel_fontsize=11, xlabel_fontsize=12,
                  feature_label_xoffset=5, xmax=14, alpha=1.0,
                  figfile=None):
    '''
    Docstring for feature_ridge.
    '''
    
    # input data
    data = data.copy()
    features = [f for f in features if f in data.columns.values]
    melted = data.melt(value_vars=features, var_name='feature')
    
    # rename
    if rename is None:
        rename = {}
    else:
        if not any([k in data.columns.values for k in rename.keys()]):
            rename = {v: k for k, v in rename.items()}

    # colors
    if colors is None:
        n_colors = len(features)
        colors = list(sns.color_palette(n_colors=n_colors))
        cdict = {h: c for h, c in zip(features, colors)}
    elif isinstance(colors, (list, tuple, np.ndarray, pd.core.series.Series)):
        cdict = {h: c for h, c in zip(features, colors)}
    else:
        cdict = colors
    colors = [cdict[f] for f in features]

    # plot
    g = sns.FacetGrid(melted, row='feature', hue='feature',
                      aspect=7.5, height=0.75, palette=colors,
                      row_order=features, hue_order=features)
    g.map(sns.kdeplot, 'value', clip_on=False, shade=True, alpha=alpha, lw=1.5)
    g.map(sns.kdeplot, 'value', clip_on=False, color="w", lw=3)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                fontsize=ylabel_fontsize,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, 'feature')

    # set the subplots to overlap
    g.fig.subplots_adjust(hspace=0.1)

    # remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(xticks=range(0, xmax + 1, 2))
    g.set(yticks=[])
    g.set(xlim=[-feature_label_xoffset, xmax + 0.25])
    g.despine(bottom=True, left=True)
    
    # xlabel
    g.set(xlabel=xlabel)
    xlabel_position = ((xmax / 2) + feature_label_xoffset) / (xmax + feature_label_xoffset)
    for ax in g.axes.flat:
        ax.set_xlabel(ax.get_xlabel(),
                      x=xlabel_position,
                      fontsize=xlabel_fontsize)

    if figfile is not None:
        g.savefig(figfile)
    else:
        plt.show()




# ===========================

#           VDJ

# ===========================


def germline_use_barplot(adata, fig_file=None, gene_names=None, segment='v', chain='heavy',
                         germline_key='v_call', batch_key=None, batch_names=None,
                         palette=None, color=None, color_by_germline=False, germline_cmap=None,
                         show=False,  pairs_only=False, normalize=True, figfile=None):
    '''

    '''
    if batch_key is not None:
        batch_names = batch_names if batch_names is not None else natsorted(adata.obs.batch_key.unique())
        batches = [adata[adata.obs.batch_key == batch] for batch in batch_names]
    else:
        batch_names = [None, ]
        batches = [adata, ]

    batch_data = []
    all_gene_names = []
    for batch in batches:
        vdjs = batch.obs.vdj
        if pairs_only:
            vdjs = [v for v in vdjs if v.is_pair]
        # parse sequences
        if chain == 'heavy':
            seqs = [v.heavy for v in vdjs if v.heavy is not None]
        elif chain == 'light':
            seqs = [v.light for v in vdjs if v.light is not None]
        elif chain == 'kappa':
            lights = [v.light for v in vdjs if v.light is not None]
            seqs = [l for l in lights if l['chain'] == 'kappa']
        elif chain == 'lambda':
            lights = [v.light for v in vdjs if v.light is not None]
            seqs = [l for l in lights if l['chain'] == 'lambda']

        # retrieve germline genes
        klist = germline_key.split('.')
        germ_counts = Counter([nested_dict_lookup(s, klist) for s in seqs])
        if normalize:
            total = sum(germ_counts.values())
            germ_counts = {k: v / total for k, v in germ_counts.items()}
        batch_data.append(germ_counts)
        for gname in germ_counts.keys():
            if gname not in all_gene_names:
                all_gene_names.append(gname)

    gene_names = gene_names if gene_names is not None else natsorted(all_gene_names)

    # colors
    if palette is not None:
        colors = [[p] * len(gene_names) for _, p in itertools.zip_longest(batches, palette)]
    elif color_by_germline:
        color_dict

    for name, 


    germs = get_germlines(species, gene, chain=chain)
    keys = list(set([g.name.split('*')[0] for g in germs]))
    if segment.upper() == 'J':
        size = (6, 4)
    elif segment.upper() == 'D':
        size = (8, 4)
    else:
        size = (10, 4)
    genes = natsorted(keys)

    # parse germline data
    gtype = '{}_gene'.format(gene.lower())
    data = [s[gtype] for s in seqs]
    if level == 'gene':
        data = [d.split('*')[0].rstrip('D') for d in data]
    counts = Counter(data)
    xs = np.arange(len(genes))
    ys = [counts.get(g, 0) for g in genes]
    if normalize:
        ys = [y / sum(ys) for y in ys]
    short_chain = chain[0].upper()
    genes = [g.replace('IG{}{}'.format(short_chain, gene), '{}{}'.format(gene, short_chain)) for g in genes]

    # colors
    fams = [g.split('-')[0].replace('/OR', '') for g in genes]
    nr_fams = natsorted(set(fams))
    cdict = {f: c for f, c in zip(nr_fams, sns.hls_palette(len(nr_fams)))}
    colors = [cdict[f] for f in fams]

    # make the plot
    plt.bar(xs, ys, color=colors, tick_label=genes)
    ax = plt.gca()

    if normalize:
        ax.set_ylabel('Frequency (%)', fontsize=14)
    else:
        ax.set_ylabel('Sequence count', fontsize=14)
    ax.tick_params(axis='x', labelsize=14, labelrotatino=90)
    ax.tick_params(axis='y', labelsize=12)

    for s in ['left', 'right', 'top']:
        ax.spines[s].set_visible(False)

    if figfile is not None:
        plt.tight_layout()
        plt.savefit(fig_file)
    elif show:
        plt.show()
    else:
        return ax



def _get_germline_plot_colors(data):
    fams = [d.split('-')[0].replace('/OR', '') for d in data]
    nr_fams = natsorted(set(fams))
    cdict = {f: c for f, c in zip(nr_fams, sns.hls_palette(len(nr_fams)))}
    return [cdict[f] for f in fams]





