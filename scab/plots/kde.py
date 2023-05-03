#!/usr/bin/env python
# filename: kde.py


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


def kde(
    adata: AnnData,
    x: str,
    y: Optional[str] = None,
    hue: Union[str, Iterable, None] = None,
    marker: str = "o",
    hue_order: Optional[Iterable] = None,
    force_categorical_hue: bool = False,
    force_continuous_hue: bool = False,
    only_scatter_hue: bool = False,
    palette: Union[dict, Iterable, None] = None,
    color: Union[str, Iterable, None] = None,
    cmap: Union[str, mpl.colors.Colormap, None] = None,
    hue_min: Optional[float] = None,
    hue_max: Optional[float] = None,
    under_color: Union[str, Iterable, None] = "whitesmoke",
    receptor: str = "bcr",
    chain: str = "heavy",
    x_chain: Optional[str] = None,
    y_chain: Optional[str] = None,
    hue_chain: Optional[str] = None,
    show_scatter: bool = True,
    scatter_size: Union[int, float] = 5,
    scatter_alpha: float = 0.2,
    thresh: float = 0.1,
    fill: bool = False,
    kde_fill_alpha: float = 0.7,
    kde_line_alpha: float = 1.0,
    highlight_index: Optional[Iterable] = None,
    highlight_x: Optional[Iterable] = None,
    highlight_y: Optional[Iterable] = None,
    highlight_marker: str = "x",
    highlight_size: Union[int, float] = 90,
    highlight_color: Union[str, Iterable] = "k",
    highlight_name: Optional[str] = None,
    highlight_alpha: float = 0.9,
    kde_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
    legend_marker_alpha: Optional[float] = None,
    legend_fontsize: Union[int, float] = 12,
    legend_title: Optional[str] = None,
    legend_title_fontsize: int = 14,
    legend_kwargs: Optional[dict] = None,
    hide_legend: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
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
    hide_ticks: bool = False,
    cbar_width: float = 0.35,
    cbar_height: float = 0.05,
    cbar_loc: str = "lower right",
    cbar_orientation: str = "horizontal",
    cbar_bbox_to_anchor: Optional[Iterable] = None,
    cbar_flip_ticks: bool = False,
    cbar_title: Optional[str] = None,
    cbar_title_fontsize: Union[int, float] = 12,
    cbar_title_loc: Optional[str] = None,
    cbar_title_labelpad: float = 8.0,
    hide_cbar: bool = False,
    equal_axes: bool = True,
    ax: Optional[mpl.axes.Axes] = None,
    show: bool = False,
    figsize: Optional[Iterable] = None,
    figfile: Optional[str] = None,
) -> Optional[mpl.axes.Axes]:
    """
    Produces a kernel density estimate (KDE) plot.

    Parameters
    ----------
    adata : anndata.AnnData
        A ``anndata.AnnData`` object containing the input data.

    x : str
        Name of a column in `adata.obs` or a BCR/TCR annotation field to be plotted on the
        x-axis. BCR/TCR annotations can be further specified using `receptor` and `chain` or,
        if data from different chains is being analyzed, using `x_chain`. BCR/TCR annotation
        fields must contain numerical data. Required.

    y : str, optional
        Name of a column in `adata.obs` or a BCR/TCR annotation field to be plotted on the
        y-axis. BCR/TCR annotations can be further specified using `receptor` and `chain` or,
        if data from different chains is being analyzed, using `y_chain`. BCR/TCR annotation
        fields must contain numerical data. If not provided, a 1-dimensional KDE plot
        will be generated using `x` values only.

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

    only_scatter_hue : bool, default=False
        If ``True``, only the scatter plot will be colored by `hue`. This results in `color`
        being used to color only the points and not the kernel density.

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
            individual RGB(A) values to a hue category. Wrapping the RGB(A) iterable in a list
            will ensure that the color is interpreted correctly.

        Only used if `hue` contains categorical data (`cmap` is used for continuous data). If not
        provided, the `default Seaborn color palette`_ will be used.

    cmap : str or matplotlib.color.Colormap, default='flare'
        Colormap to be used for continuous `hue` data.

    hue_min : float, default=0
        Minimum value for `hue` when `hue` is continuous. Values below `hue_min` will be set to
        `under_color`.

    hue_max : float, default=1
        Maximum value for `hue` when `hue` is continuous. Values at or above `hue_max` will
        all be colored as the maxmum value in `cmap`.

    under_color : str or list of RGB(A) values
        Separate color for values less than `hue_min` when `hue` is continuous. By default, `cmap` is
        used for all values. An example use would be GEx plots for which visualization is
        improved if ``0`` values are more obviously distinguished from low count values.

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


    show_scatter : bool, default=True
        If ``True``, a scatter plot will be added to the KDE plot.

    scatter_size : str or float or iterable object, default=5
        Size of the scatter points. Either a

    scatter_alpha : float, default=0.2
        Alpha of the scatter points.

    thresh: float, default=0.1
        Threshold for the KDE plot. Values below `thresh` will be set to ``0``.

    fill: bool, default=True
        If ``True``, the KDE plot will be filled.

    kde_fill_alpha: float, default=0.7
        Alpha of the KDE fill.

    kde_line_alpha: float, default=1.0
        Alpha of the KDE line.

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

    scatter_kwargs : dict, optional
        Dictionary containing keyword arguments that will be passed to ``pyplot.scatter()``.

    kde_kwargs : dict, optional
        Dictionary containing keyword arguments that will be passed to ``seaborn.kdeplot()``.

    legend_marker_alpha : float, default=None
        Opacity for legend markers (or legend labels if `legend_on_data` is ``True``).
        By default, legend markers will use `alpha` and legend labels will be completely
        opaque, equivalent to `legend_marker_alpha` of ``1.0``.

    legend_fontsize : int or float, default=12
        Fontsize for legend labels.

    legend_title : str, optional
        Title for the plot legend.

    legend_title_fontsize : int or float, default=14
        Fontsize for the legend title.

    legend_kwargs : dict, optional
        Dictionary containing keyword arguments that will be passed to ``ax.legend()``.

    hide_legend : bool, default=False
        By default, a plot legend will be shown if multiple hues are plotted. If ``True``,
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

    ax : mpl.axes.Axes, default=None
        Pre-existing axes for the plot. If not provided, a new axes will be created.

    show : bool, default=False
        If ``True``, plot is shown and the plot ``Axes`` object is not returned. Default
        is ``False``, which does not call ``pyplot.show()`` and returns the ``Axes`` object.

    figsize : iterable object, optional
        List containing the figure size (as ``[x-dimension, y-dimension]``) in inches. If
        `y` is provided (a 2-dimensional KDE plot), the default is ``[6, 6]``. If `y` is
        not provided (a 1-dimensional KDE plot), the default is ``[6, 4]``.

    figfile : str, optional
        Path at which to save the figure file. If not provided, the figure is not saved
        and is either shown (if `show` is ``True``) or the ``Axes`` object is returned.


    Returns
    -------
    ax : mpl.axes.Axes
        If `figfile` is ``None`` and `show` is ``False``, the ``ax`` is returned.
        Otherwise, the return value is ``None``.


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
    d = {}
    d["x"] = get_adata_values(
        adata, x, receptor=receptor, chain=x_chain if x_chain is not None else chain
    )
    if y is not None:
        d["y"] = get_adata_values(
            adata, y, receptor=receptor, chain=y_chain if y_chain is not None else chain
        )
    if hue is not None:
        d[hue] = get_adata_values(
            adata,
            hue,
            receptor=receptor,
            chain=hue_chain if hue_chain is not None else chain,
        )
    df = pd.DataFrame(d, index=adata.obs.index)
    # figsize
    if figsize is None:
        if y is not None:
            figsize = [6, 6]
        else:
            figsize = [6, 4]
    # default x- and y-axis labels
    if xlabel is None:
        xlabel = f"{x} " + "($\mathregular{log_2}$ UMI counts)"
    if ylabel is None:
        if y is None:
            ylabel = "Density"
        else:
            ylabel = f"{y} " + "($\mathregular{log_2}$ UMI counts)"
    # make the plot
    ax = abutils.pl.kde(
        data=df,
        x="x",
        y="y" if y is not None else y,
        hue=hue,
        marker=marker,
        hue_order=hue_order,
        force_categorical_hue=force_categorical_hue,
        force_continuous_hue=force_continuous_hue,
        only_scatter_hue=only_scatter_hue,
        palette=palette,
        color=color,
        cmap=cmap,
        hue_min=hue_min,
        hue_max=hue_max,
        under_color=under_color,
        show_scatter=show_scatter,
        scatter_size=scatter_size,
        scatter_alpha=scatter_alpha,
        thresh=thresh,
        fill=fill,
        kde_fill_alpha=kde_fill_alpha,
        kde_line_alpha=kde_line_alpha,
        highlight_index=highlight_index,
        highlight_x=highlight_x,
        highlight_y=highlight_y,
        highlight_marker=highlight_marker,
        highlight_size=highlight_size,
        highlight_color=highlight_color,
        highlight_name=highlight_name,
        highlight_alpha=highlight_alpha,
        kde_kwargs=kde_kwargs,
        scatter_kwargs=scatter_kwargs,
        legend_title=legend_title,
        legend_title_fontsize=legend_title_fontsize,
        legend_marker_alpha=legend_marker_alpha,
        legend_fontsize=legend_fontsize,
        legend_kwargs=legend_kwargs,
        hide_legend=hide_legend,
        xlabel=xlabel if xlabel is not None else x,
        ylabel=ylabel if ylabel is not None else y,
        title=title,
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
        cbar_width=cbar_width,
        cbar_height=cbar_height,
        cbar_loc=cbar_loc,
        cbar_orientation=cbar_orientation,
        cbar_bbox_to_anchor=cbar_bbox_to_anchor,
        cbar_flip_ticks=cbar_flip_ticks,
        cbar_title=cbar_title,
        cbar_title_fontsize=cbar_title_fontsize,
        cbar_title_loc=cbar_title_loc,
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
        plt.show()
    else:
        return ax
