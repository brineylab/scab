#!/usr/bin/env python
# filename: umap.py


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


from typing import Optional, Iterable, Union

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from anndata import AnnData

import abutils

from ..tl import dimensionality_reduction
from ..ut import get_adata_values


def umap(
    adata: AnnData,
    hue: Optional[str] = None,
    marker: str = "o",
    hue_order: Optional[Iterable] = None,
    force_categorical_hue: bool = False,
    force_continuous_hue: bool = False,
    palette: Union[dict, Iterable, None] = None,
    color: Union[str, Iterable, None] = None,
    cmap: Union[str, mpl.colors.Colormap, None] = "magma_r",
    hue_min: Optional[float] = None,
    hue_max: Optional[float] = None,
    under_color: Union[str, Iterable, None] = [0.9, 0.9, 0.9],
    # zero_color: Union[str, Iterable, None] = None,
    size: Union[int, float] = 20,
    alpha: float = 0.6,
    n_col: int = 2,
    receptor: str = "bcr",
    chain: str = "heavy",
    hue_chain: Optional[str] = None,
    highlight_index: Optional[Iterable] = None,
    highlight_x: Optional[Iterable] = None,
    highlight_y: Optional[Iterable] = None,
    highlight_marker: str = "x",
    highlight_size: Union[int, float] = 90,
    highlight_color: Union[str, Iterable] = "k",
    highlight_name: Optional[str] = None,
    highlight_alpha: float = 0.9,
    plot_kwargs: Optional[dict] = None,
    legend_marker_alpha: Optional[float] = None,
    legend_on_data: bool = False,
    legend_fontsize: Union[int, float] = 12,
    legend_fontweight: str = "bold",
    legend_fontoutline: Union[int, float] = 3,
    legend_label_position_offsets: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
    hide_legend: bool = False,
    xlabel: str = "UMAP1",
    ylabel: str = "UMAP2",
    title: Optional[str] = None,
    title_fontsize: Union[int, float] = 20,
    title_fontweight: str = "normal",
    title_loc: str = "center",
    title_pad: Union[int, float, None] = None,
    show_title: bool = False,
    xlabel_fontsize: Union[int, float] = 16,
    ylabel_fontsize: Union[int, float] = 16,
    xtick_labelsize: Union[int, float] = 14,
    ytick_labelsize: Union[int, float] = 14,
    xtick_labelrotation: Union[int, float] = 0,
    ytick_labelrotation: Union[int, float] = 0,
    hide_ticks: bool = True,
    tiny_axis: bool = True,
    tiny_axis_xoffset: Union[int, float, None] = None,
    tiny_axis_yoffset: Union[int, float, None] = None,
    cbar_width: float = 0.35,
    cbar_height: float = 0.05,
    cbar_loc: str = "lower right",
    cbar_orientation: str = "horizontal",
    cbar_bbox_to_anchor: Optional[Iterable] = None,
    cbar_flip_ticks: bool = False,
    cbar_title: Optional[str] = None,
    cbar_title_fontsize: Union[int, float] = 12,
    cbar_title_labelpad: Union[int, float] = 8,
    hide_cbar: bool = False,
    equal_axes: bool = True,
    ax: Optional[mpl.axes.Axes] = None,
    show: bool = False,
    figsize: Optional[Iterable] = None,
    figfile: Optional[str] = None,
):
    """
    Produces a scatter plot.

    Parameters
    ----------

    adata : pandas.DataFrame
        A ``anndata.AnnData`` object containing the input data.

    hue : str, optional
        Name of a column in `adata.obs` or a BCR/TCR annotation field containing hue values.
        BCR/TCR annotations can be further specified using `receptor` and `chain` or, if data
        from different chains is being analyzed, using `hue_chain`. BCR/TCR annotation
        fields must contain numerical data.

    marker : str, dict or iterable object, optional
        Marker style for the scatter plot. Accepts any of the following:
          * a `matplotlib marker`_ string
          * a ``dict`` mapping `hue` categories to a `matplotlib marker`_ string
          * a ``list`` of `matplotlib marker`_ strings, which should be the same
              length as `x` and `y`.

    hue_order : iterable object, optional
        List of `hue` categories in the order they should be plotted. If `hue_order` contains a
        subset of all categories found in `hue`, only the supplied categories will be plotted.
        If not provided, `hue` categories will be plotted in ``natsort.natsorted()`` order.

    force_categorical_hue : bool, default=False
        If ``True``, `hue` data will be treated as categorical, even if the data appear to
        be continuous. This results in `color` being used to color the points rather than `cmap`.

    force_continuous_hue : bool, default=False
        If ``True``, `hue` data will be treated as continuous, even if the data appear to
        be categorical. This results in `cmap` being used to color the points rather than `color`.
        This may produce unexpected results and/or errors if used on non-numerical data.

    palette : dict, optional
        Dictionary mapping `hue`, `x` or `y` names to colors. If if keys in `palette` match
        more than one category, `hue` categories take priority. If `palette` is not provided,
        bars are colored using `color` (if `hue` is ``None``) or a palette is generated
        automatically using ``sns.hls_palette()``.

    color : str or iterable object, optional.
        Color for the plot markers. Can be any of the following:
          * a ``list`` or ``tuple`` containing RGB or RGBA values
          * a color string, either from `Matplotlib's set of named colors`_ or a hex color value
          * the name of a column in `data` that contains color values
          * a ``list`` of colors (either as strings or RGB tuples) to be used for `hue` categories.
            If `colors` is shorter than the list of hue categories, colors will be reused.

        .. tip::
            If a single RGB or RGBA ``list`` or ``tuple`` is provided and `hue` is also supplied,
            there may be unexpected results as ``scatter()`` will attempt to map each of the
            individual RGB(A) values to a hue category.

        Only used if `hue` contains categorical data (`cmap` is used for continuous data). If not
        provided, the `default Seaborn color palette`_ will be used.

    cmap : str or matplotlib.color.Colormap, default='flare'
        Colormap to be used for continuous `hue` data.

    zero_color : str or list of RGB(A) values
        Separate color for ``0`` values when `hue` is continuous. By default, `cmap` is
        used for all values. An example use would be GEx plots for which visualization is
        improved if ``0`` values are more obviously distinguished from low count values.

    size : str or float or iterable object, default=20
        Size of the scatter points. Either a

    alpha : float, default=0.6
        Alpha of the scatter points.

    n_col : int, default=2
        Number of plot columns if multiple `hue` values are provided.

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

    highlight_index : iterable object, optional
        An iterable of index names (present in `data.index`) of points to be highlighted on
        the plot. If provided, `highlight_x` and `highlight_y` are ignored.

    highlight_x : iterable object, optional
        An iterable of x-values for highlighted points. Also requires `highlight_y`.

    highlight_y : iterable object, optional
        An iterable of y-values for highlighted points. Also requires `highlight_x`.

    highlight_marker : str, default='x'
        Marker style to be used for highlight points. Accepts any `matplotlib marker`_.

    highlight_size : int, default=90
        Size of the highlight marker.

    highlight_color : string or list of color values, default='k'
        Color of the highlight points.

    highlight_name : str, optional
        Name of the highlights, to be used in the plot legend. If not supplied,
        highlight points will not be included in the legend.

    highlight_alpha : float, default=0.9
        Alpha of the highlight points.

    plot_kwargs : dict, optional
        Dictionary containing keyword arguments that will be passed to ``pyplot.scatter()``.

    legend_on_data : bool, default=False
        Plot legend labels on the data rather than in a separate legend. The X/Y midpoint
        for each legend category is used as the label location.

    legend_fontsize : int or float, default=12
        Fontsize for legend labels.

    legend_fontweight : str, default="normal"
        Fontweight for legend labels. Options are any accepted `Matplotlib text weight`_.

    legend_fontoutline : int or float, default=None
        Width of the outline of legend labels. Only used when `legend_on_data` is ``True``.
        Default is ``None``, which results in no outline.

    legend_marker_alpha : float, default=None
        Opacity for legend markers (or legend labels if `legend_on_data` is ``True``).
        By default, legend markers will use `alpha` and legend labels will be completely
        opaque, equivalent to `legend_marker_alpha` of ``1.0``.

    legend_label_position_offsets : dict, default=None
        A ``dict`` mapping legend labels to ``(x,y)`` coordinates used to offset legend labels.
        Only used when `legend_on_data` is ``True``. Offsets are in relative plot units: ``(0.1, 0.1)``
        would move the label up and to the right by 10% of the overall plot area.

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

    tiny_axis : bool, default=False
        Plots tiny axis lines in the lower left corner of the plot. Typcally used in
        UMAP plots. If ``True``, ticks and tick labels will be hidden.

    tiny_axis_xoffset : float, default=None
        X-axis offset for the tiny axis.

    tiny_axis_yoffset : float, default=None
        Y-axis offset for the tiny axis.

    cbar_width : int, default=35
        Width of the colorbar. Only used for continuous `hue` types.

    cbar_height : int, default=5
        Height of the colorbar. Only used for continuous `hue` types.

    cbar_loc : str or iterable object, default='lower right'
        Location of the colorbar. Accepts `any valid inset_axes() location`_.

    cbar_orientation : str, default='horizontal'
        Orientation of the colorbar. Options are ``'horizontal'`` and ``'vertical'``.

    cbar_bbox_to_anchor : list or tuple, optional
        bbox_to_anchor for the colorbar. Used in combination with `cbar_loc` to provide
        fine-grained positioning of the colorbar.

    cbar_flip_ticks : bool, default=False
        Flips the position of colorbar ticks. Ticks default to the bottom if `cbar_orientation`
        is  ``'horizontal'`` and the left if  `cbar_orientation` is ``'vertical'``.

    cbar_title : str, optional
        Colorbar title. If not provided, `hue` is used.

    cbar_title_fontsize : int or float, default=12
        Fontsize for the colorbar title.

    hide_cbar : bool, default=False.
        If ``True``, the color bar will be hidden on plots with continuous `hue` values.

    equal_axes : bool, default=True
        If ```True```, the the limits of the x- and y-axis will be equal.

    show :bool, default=False
        If ``True``, plot is shown and the plot ``Axes`` object is not returned. Default
        is ``False``, which does not call ``pyplot.show()`` and returns the ``Axes`` object.

    figsize : iterable object, default=[6, 4]
        List containing the figure size (as ``[x-dimension, y-dimension]``) in inches.

    figfile : str, optional
        Path at which to save the figure file. If not provided, the figure is not saved
        and is either shown (if `show` is ``True``) or the ``Axes`` object is returned.


    .. _matplotlib marker:
        https://matplotlib.org/stable/api/markers_api.html

    .. _Matplotlib's set of named colors:
        https://matplotlib.org/stable/gallery/color/named_colors.html

    .. _default Seaborn color palette:
        https://seaborn.pydata.org/generated/seaborn.color_palette.html

    .. _Matplotlib text weight
        https://matplotlib.org/stable/tutorials/text/text_props.html

    .. _any valid inset_axes() location:
        https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.inset_axes.html

    """

    # get x, y and hue data
    # check that the UMAP data exists
    if "X_umap" not in adata.obsm:
        adata = dimensionality_reduction(adata)

    # parse hues (if provided)
    if hue is not None:
        if isinstance(hue, str):
            hues = [hue]
        else:
            hues = hue
    else:
        hues = [None]

    # set some UMAP-specific defaults
    figsize = figsize if figsize is not None else [7, 7]
    if legend_fontsize is not None:
        if legend_kwargs is None:
            legend_kwargs = {"fontsize": legend_fontsize}
        else:
            legend_kwargs["fontsize"] = legend_fontsize

    if len(hues) > 1:
        n_row = int(np.ceil(len(hues) / n_col))
        gridsize = [figsize[0] * n_col, figsize[1] * n_row]
        _, axes = plt.subplots(n_row, n_col, figsize=gridsize)
        axes = axes.flatten()
    else:
        axes = [ax]

    # make plots
    for ax, hue in zip(axes, hues):
        # get X/Y data
        d = {}
        x, y = zip(*adata.obsm["X_umap"])
        d["x"] = x
        d["y"] = y
        if hue is not None:
            d[hue] = get_adata_values(
                adata,
                hue,
                receptor=receptor,
                chain=hue_chain if hue_chain is not None else chain,
            )
            continuous_hue = (
                all([isinstance(h, float) for h in d[hue]])
                and not force_categorical_hue
            )
        if show_title:
            plot_title = title if title is not None else hue
        else:
            plot_title = None
        df = pd.DataFrame(d, index=adata.obs.index)

        ax = abutils.pl.scatter(
            data=df,
            x="x",
            y="y",
            hue=hue,
            marker=marker,
            hue_order=hue_order,
            force_categorical_hue=force_categorical_hue,
            force_continuous_hue=force_continuous_hue,
            palette=palette,
            color=color,
            cmap=cmap,
            hue_min=hue_min,
            hue_max=hue_max,
            under_color=under_color,
            # zero_color=zero_color,
            size=size,
            alpha=alpha,
            highlight_index=highlight_index,
            highlight_x=highlight_x,
            highlight_y=highlight_y,
            highlight_marker=highlight_marker,
            highlight_size=highlight_size,
            highlight_color=highlight_color,
            highlight_name=highlight_name,
            highlight_alpha=highlight_alpha,
            plot_kwargs=plot_kwargs,
            legend_marker_alpha=legend_marker_alpha,
            legend_on_data=legend_on_data,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            legend_fontoutline=legend_fontoutline,
            legend_label_position_offsets=legend_label_position_offsets,
            legend_kwargs=legend_kwargs,
            hide_legend=hide_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=plot_title,
            title_fontsize=title_fontsize,
            title_fontweight=title_fontweight,
            title_loc=title_loc,
            title_pad=title_pad,
            show_title=show_title,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            xtick_labelrotation=xtick_labelrotation,
            ytick_labelrotation=ytick_labelrotation,
            hide_ticks=hide_ticks,
            tiny_axis=tiny_axis,
            tiny_axis_xoffset=tiny_axis_xoffset,
            tiny_axis_yoffset=tiny_axis_yoffset,
            cbar_width=cbar_width,
            cbar_height=cbar_height,
            cbar_loc=cbar_loc,
            cbar_orientation=cbar_orientation,
            cbar_bbox_to_anchor=cbar_bbox_to_anchor,
            cbar_flip_ticks=cbar_flip_ticks,
            cbar_title=cbar_title,
            cbar_title_fontsize=cbar_title_fontsize,
            cbar_title_labelpad=cbar_title_labelpad,
            hide_cbar=hide_cbar,
            equal_axes=equal_axes,
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
        plt.tight_layout()
        plt.show()
    else:
        return ax
