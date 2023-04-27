#!/usr/bin/env python
# filename: bar.py


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

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from anndata import AnnData

import abutils

from ..ut import get_adata_values


def bar(
    adata: AnnData,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Union[str, Iterable, None] = None,
    order: Optional[Iterable] = None,
    hue_order: Optional[Iterable] = None,
    palette: Union[dict, Iterable, None] = None,
    color: Union[str, Iterable, None] = None,
    alt_color: Union[str, Iterable] = "#D3D3D3",
    receptor: str = "bcr",
    chain: str = "heavy",
    x_chain: Optional[str] = None,
    y_chain: Optional[str] = None,
    hue_chain: Optional[str] = None,
    normalize: bool = False,
    highlight: Union[str, Iterable, None] = None,
    highlight_color: Union[str, Iterable, None] = None,
    orientation: str = "vertical",
    plot_kwargs: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
    hide_legend: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_fontsize: Union[int, float] = 16,
    ylabel_fontsize: Union[int, float] = 16,
    xtick_labelsize: Union[int, float] = 14,
    ytick_labelsize: Union[int, float] = 14,
    xtick_labelrotation: Union[int, float] = 0,
    ytick_labelrotation: Union[int, float] = 0,
    ax: Optional[mpl.axes.Axes] = None,
    show: bool = False,
    figsize: Optional[Iterable] = None,
    figfile: Optional[str] = None,
):
    """
    Produces a bar plot of categorical data. For data with distinct batches, a stacked
    bar plot will be constructed.

    Parameters
    ----------
    adata : anndata.AnnData
        A ``anndata.AnnData`` object containing the input data.
    
    x : str
        Name of a column in `adata.obs` or a BCR/TCR annotation field to be plotted on the
        x-axis. BCR/TCR annotations can be further specified using `receptor` and `chain` or,
        if data from different chains is being analyzed, using `x_chain`. BCR/TCR annotation
        fields must contain numerical data. At least one of `x` and `y` is required.

    y : str
        Name of a column in `adata.obs` or a BCR/TCR annotation field to be plotted on the
        y-axis. BCR/TCR annotations can be further specified using `receptor` and `chain` or,
        if data from different chains is being analyzed, using `y_chain`. BCR/TCR annotation
        fields must contain numerical data. At least one of `x` and `y` is required.

    hue : str, optional
        Name of a column in `adata.obs` or an iterable of hue categories to be used to
        group data into stacked bars. If not provided, an un-stacked bar plot is created.

    data : pandas.DataFrame, optional
        A ``DataFrame`` object containing the input data. If provided, `x` and/or `y` should
        be column names in `data`.

    sequences : iterable of abutils.core.sequence.Sequence, optional
        An iterable of ``Sequence`` objects. If provided, `x`, `y` and/or `hue` should be annotations
        in the ``Sequence`` objects. Alternatively, `x`, `y` and/or `hue` can be an iterable of
        values to be plotted, but must be the same length as `sequences`.

    order : iterable object, optional
        List of `x` or `y` categories in the order they should be plotted. If `order` contains a
        subset of all categories found in `x` or `y`, only the supplied categories will be plotted.
        If not provided, categories will be plotted in ``natsort.natsorted()`` order.

    hue_order : iterable object, optional
        List of `hue` categories in the order they should be plotted. If `hue_order` contains a
        subset of all categories found in `hue`, only the supplied categories will be plotted.
        If not provided, `hue` categories will be plotted in ``natsort.natsorted()`` order.

    palette : dict, optional
        Dictionary mapping `hue`, `x` or `y` names to colors. If if keys in `palette` match
        more than one category, `hue` categories take priority. If `palette` is not provided,
        bars are colored using `color` (if `hue` is ``None``) or a palette is generated
        automatically using ``sns.hls_palette()``.

    color : str or iterable, optional
        Single color to be used for the bar plot. If not provided, the first color in the
        default ``Seaborn`` color palette will be used. If `highlight` is provided but
        `highlight_color` is not, `color` will be used to color highlighted bars.

    alt_color : str or iterable, default='#D3D3D3'
        Alternate color for the bar plot. Used to color categories not provided in `palette`
        or to color categories not present in `highlight`.

    receptor : str, default='bcr'
        Receptor for which data should be plotted. Options are ``'bcr'`` and ``'tcr'``.

    chain : str, default='heavy'
        If `x`, `y` and/or `hue` are BCR/TCR annotation fields, chain for which annotation will be
        retrieved. Options are ``'heavy'``, ``'light'``, ``'kappa'``, ``'lambda'``, ``'alpha'``,
        ``'beta'``, ``'delta'`` or ``'gamma'``.

    x_chain : str
        `chain` to be used for the x-axis. If not supplied, `chain` will be used. Only necessary
        when visualizing data from different chains on the same plot.

    y_chain : str
        `chain` to be used for the y-axis. If not supplied, `chain` will be used. Only necessary
        when visualizing data from different chains on the same plot.

    hue_chain : str
        `chain` to be used for the hue. If not supplied, `chain` will be used. Only necessary
        when visualizing data from different chains on the same plot.

    orientation : str, optional
        Orientation of the plot. Options are ``'vertical'`` or ``'horizontal'``. Default is
        ``'vertical'``.

    normalize : bool, default=False
        If ``True``, normalized frequencies are plotted instead of raw counts. If multiple `hue`
        categories are present, data will be normalized such that all
        bars extend from [0,1] and each stacked bar is sized according to the `hue`'s  fraction.
        If `hue` is not provided or there is only one `hue` category, the entire
        dataset is normalized.

    highlight : iterable, optional
        List of `x` or `hue` categories to be highlighted. If `highlight_color` is provided,
        categories in `highlight` will use `highlight_color` and all others will use `alt_color`.
        If `highlight_color` is not provided, `palette` will be used. If both `highlight_color`
        and `palette` are not provided, `color` will be used.

    highlight_color : str or iterable, optional
        Color to be used for categories in `highlight`. If

    plot_kwargs : dict, optional
        Dictionary containing keyword arguments that will be passed to ``pyplot.bar()``.

    legend_kwargs : dict, optional
        Dictionary containing keyword arguments that will be passed to ``ax.legend()``.

    hide_legend : bool, default=False
        By default, a plot legend will be shown if multiple batches are plotted. If ``True``,
        the legend will not be shown.

    xlabel : str, optional
        Text for the X-axis label.

    ylabel : str, optional
        Text for the Y-axis label.

    xlabel_fontsize : int or float, default=16
        Fontsize for the X-axis label text.

    ylabel_fontsize : int or float, default=16
        Fontsize for the Y-axis label text.

    xtick_labelsize : int or float, default=14
        Fontsize for the X-axis tick labels.

    ytick_labelsize : int or float, default=14
        Fontsize for the Y-axis tick labels.

    xtick_labelrotation : int or float, default=0
        Rotation of the X-axis tick labels.

    ytick_labelrotation : int or float, default=0
        Rotation of the Y-axis tick labels.

    show :bool, default=False
        If ``True``, plot is shown and the plot ``Axes`` object is not returned. Default
        is ``False``, which does not call ``pyplot.show()`` and returns the ``Axes`` object.

    figsize : iterable object, default=[6, 4]
        List containing the figure size (as ``[x-dimension, y-dimension]``) in inches.

    figfile : str, optional
        Path at which to save the figure file. If not provided, the figure is not saved
        and is either shown (if `show` is ``True``) or the ``Axes`` object is returned.
    """
    # get x and hue data
    d = {}
    if x is not None:
        d["x"] = get_adata_values(
            adata, x, receptor=receptor, chain=x_chain if x_chain is not None else chain
        )
        if xlabel is None:
            xlabel = x
        x = "x"
    if y is not None:
        d["y"] = get_adata_values(
            adata, y, receptor=receptor, chain=y_chain if y_chain is not None else chain
        )
        if ylabel is None:
            ylabel = y
        y = "y"
    if hue is not None:
        d[hue] = get_adata_values(
            adata,
            hue,
            receptor=receptor,
            chain=hue_chain if hue_chain is not None else chain,
        )
    df = pd.DataFrame(d, index=adata.obs.index)

    # drop all of the rows with missing `x` data
    df = df.dropna(subset=["x"])

    # make the plot
    ax = abutils.pl.bar(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        palette=palette,
        color=color,
        alt_color=alt_color,
        normalize=normalize,
        highlight=highlight,
        highlight_color=highlight_color,
        orientation=orientation,
        plot_kwargs=plot_kwargs,
        legend_kwargs=legend_kwargs,
        hide_legend=hide_legend,
        xlabel=xlabel,
        ylabel=ylabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        xtick_labelsize=xtick_labelsize,
        ytick_labelsize=ytick_labelsize,
        xtick_labelrotation=xtick_labelrotation,
        ytick_labelrotation=ytick_labelrotation,
        ax=ax,
        show=False,
        figsize=figsize,
        figfile=None,
    )

    # save, show or return the ax
    if figfile is not None:
        plt.tight_layout()
        plt.savefig(figfile)
    elif show:
        plt.show()
    else:
        return ax

