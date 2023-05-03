#!/usr/bin/env python
# filename: ridge.py


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


from typing import Optional, Iterable, Union

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from anndata import AnnData

import abutils

from ..ut import get_adata_values


def ridge(
    adata: AnnData,
    categories: Union[str, Iterable, None] = None,
    values: Union[str, Iterable, None] = None,
    order: Optional[Iterable] = None,
    palette: Union[dict, Iterable, None] = None,
    alt_color: Union[Iterable, str] = "lightgrey",
    alpha: float = 1.0,
    receptor: str = "bcr",
    chain: str = "heavy",
    linewidth: float = 0.0,
    outlinewidth: float = 1.5,
    xaxis_linewidth: float = 1.0,
    xaxis_linecolor: Union[Iterable, str, None] = None,
    xlabel: str = "UMI count ($\mathregular{log_2}$)",
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 11,
    bw: Union[str, float] = "scott",
    adjust_hspace: float = 0.1,
    category_label_fontweight: str = "bold",
    category_label_xoffset: float = 0.25,
    xmin: Union[int, float, None] = None,
    xmax: Union[int, float, None] = None,
    aspect: float = 15.0,
    height: float = 0.5,
    figfile: Optional[str] = None,
    show: bool = True,
):
    """
    Draws a ridge plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data to be plotted. Required.

    categories : Union[str, Iterable, None], optional
        Category information. Can be any of the following:
          - if `data` is provided and `values` is not provided, `categories`
            must be a list containing one or more column names in `data`
          - if both `data` and `values` are also provided, `categories` should be a
            column name in `data`
          - if `data` is not provided, `categories` must be a list of category names

    values : Union[str, Iterable, None], optional
        Value information. Can be any of the following:
          - if `data` and `categories` are provided, `values` should be a column name in `data`
          - if `data` is not provided, `values` must be a list of values
        Note that `values` cannot be provided without `categories`.

    order : Optional[Iterable], optional
        Order of categories in the plot. If not provided, categories will be sorted using ``natsort``.

    palette : Union[dict, Iterable, None], optional
        Color palette. Can be any of the following:
            - a dictionary mapping categories to colors
            - a list of colors
        If a ``dict`` is provided, any categories not included in the dictionary will be assigned
        `alt_color`. If a ``list`` is provided, the colors will be assigned to the categories in order,
        with colors reused if there are more categories than colors. If not provided, a default palette
        (created using ``sns.hls_palette()``) will be used.

    alt_color : Union[Iterable, str], optional
        Color to use for categories not included in the palette. If a ``list`` is provided, the color

    alpha : float, optional
        Alpha value for the density curves. Default is 1.0.

    receptor : str, default='bcr'
        Receptor for which data should be plotted. Options are ``'bcr'`` and ``'tcr'``.

    chain : str, default='heavy'
        If `categories`, and/or `values` are BCR/TCR annotation fields, chain for which annotation will 
        be retrieved. Options are ``'heavy'``, ``'light'``, ``'kappa'``, ``'lambda'``, ``'alpha'``,
        ``'beta'``, ``'delta'`` or ``'gamma'``.

    linewidth : float, optional
        Line width for the data line on the density curves. Default is 0.0.

    outlinewidth : float, optional
        Line width for the white outline of the density curves. The purpose is to provide visual
        separation if/when the density curves of adjacent ridge plots overlap. Default is 1.5.

    xaxis_linewidth : float, optional
        Line width for the x-axis line. Default is 1.0.

    xaxis_linecolor : Union[Iterable, str, None], optional
        Color for the x-axis line. If not provided, the color will be the same as the color
        used to plot the density curve.

    xlabel : str, optional
        Label for the x-axis. Default is "UMI count ($\mathregular{log_2}$)".

    xlabel_fontsize : int, optional
        Font size for the x-axis label. Default is 12.

    ylabel_fontsize : int, optional
        Font size for the y-axis labels. Default is 11.

    bw : Union[str, float], optional
        Bandwidth for the density curves. Can be any of the following:
            - a float
            - the name of a reference rule (see ``scipy.stats.gaussian_kde``)
        Default is "scott".

    adjust_hspace : float, optional
        Adjust the vertical space between subplots. Default is 0.1.

    category_label_fontweight : str, optional
        Font weight for the category labels. Default is "bold".

    category_label_xoffset : float, optional
        Horizontal offset for the category labels, as a fraction of the total plot width.
        Default is 0.25.

    xmin : Union[int, float, None], optional
        Minimum value for the x-axis. If not provided, the minimum value will be the data minimum.

    xmax : Union[int, float, None], optional
        Maximum value for the x-axis. If not provided, the maximum value will be the data maximum.

    aspect : float, optional
        Aspect ratio for the plot. Default is 15.0.

    height : float, optional
        Height of each individual ridge plot, in inches. Default is 0.5.

    figfile : Optional[str], optional
        If provided, the figure will be saved to this file. Default is None.

    show : bool, optional
        If True, the figure will be displayed. Default is True.

    Returns
    -------
    sns.FacetGrid
        If `show` is ``False`` and `figfile` is not provided, the ``FacetGrid`` object
        will be returned.

    """
    # process input data
    d = {}
    d[categories] = get_adata_values(adata, categories, receptor=receptor, chain=chain)
    d[values] = get_adata_values(adata, values, receptor=receptor, chain=chain)
    df = pd.DataFrame(d)
    # make the plot
    g = abutils.pl.ridge(
        data=df,
        categories=categories,
        values=values,
        order=order,
        palette=palette,
        alt_color=alt_color,
        alpha=alpha,
        linewidth=linewidth,
        outlinewidth=outlinewidth,
        xaxis_linewidth=xaxis_linewidth,
        xaxis_linecolor=xaxis_linecolor,
        xlabel=xlabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        bw=bw,
        adjust_hspace=adjust_hspace,
        category_label_fontweight=category_label_fontweight,
        category_label_xoffset=category_label_xoffset,
        xmin=xmin,
        xmax=xmax,
        aspect=aspect,
        height=height,
        figfile=None,
        show=False,
    )
    # save, show or return the ax
    if figfile:
        g.savefig(figfile)
    elif show:
        plt.show()
    return g
