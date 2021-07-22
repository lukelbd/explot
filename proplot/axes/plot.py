#!/usr/bin/env python3
"""
The second-level axes subclass used for all ProPlot figures.
Implements plotting method overrides.
"""
import functools
import inspect
import re
import sys
from numbers import Integral

import matplotlib.axes as maxes
import matplotlib.cm as mcm
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import numpy.ma as ma

from .. import colors as pcolors
from .. import constructor
from .. import ticker as pticker
from ..config import rc
from ..internals import ic  # noqa: F401
from ..internals import (
    _getattr_flexible,
    _not_none,
    _pop_kwargs,
    _process_props,
    _state_context,
    docstring,
    warnings,
)
from ..utils import edges, edges2d, to_xyz
from . import base

try:
    from cartopy.crs import PlateCarree
except ModuleNotFoundError:
    PlateCarree = object

__all__ = ['PlotAxes']


# Positional args that can be passed as keywords. Should be a comprehensive list
# WARNING: The 'barh' interpretation represents a breaking change from default
# (y, width, height, left) behavior. Want to have consistent interpretation
# of vertical or horizontal bar 'width' with 'width' key or 3rd positional arg.
# May cause errors if we let internal matplotlib commands use proplot wrappers.
KEYWORD_TO_POSITIONAL = {
    'plot': ('x', 'y'),
    'plotx': ('y', 'x'),
    'step': ('x', 'y'),
    'stepx': ('y', 'x'),
    'stem': ('x', 'y'),  # TODO: add 'stemh' method?
    'parametric': ('x', 'y', 'c'),  # TODO: ensure 'values' is synonym
    'vlines': ('x', 'ymin', 'ymax', 'c'),  # TODO: native is 'colors', support synonyms
    'hlines': ('y', 'xmin', 'xmax', 'c'),
    'scatter': ('x', 'y', 's', 'c'),
    'scatterx': ('y', 'x', 's', 'c'),
    'fill_between': ('x', 'y1', 'y2', 'where'),
    'fill_betweenx': ('y', 'x1', 'x2', 'where'),
    'bar': ('x', 'height', 'width', 'bottom'),
    'barh': ('y', 'height', 'width', 'left'),
    'boxplot': ('positions', 'x'),  # NOTE: used as x-coordinate only during wrapping
    'boxploth': ('positions', 'x'),
    'violinplot': ('positions', 'x'),  # NOTE: native for 'x' is 'dataset'
    'violinploth': ('positions', 'x'),
    'pie': ('x',),  # TODO: add 'explode', 'labels', 'colors'
    'hist': ('x',),  # TODO: allow 'weights'
    'histh': ('y',),  # TODO: allow 'weights'
    'hist2d': ('x', 'y'),  # TODO: allow 'weights'
    'hexbin': ('x', 'y', 'c'),
    'contour': ('x', 'y', 'z'),  # TODO: automatic application of legend elements!
    'contourf': ('x', 'y', 'z'),
    'pcolor': ('x', 'y', 'z'),  # TODO: automatic application of legend elements!
    'pcolormesh': ('x', 'y', 'z'),
    'pcolorfast': ('x', 'y', 'z'),
    'streamplot': ('x', 'y', 'u', 'v'),
    'quiver': ('x', 'y', 'u', 'v', 'c'),  # TODO: automatic quiver key!
    'barbs': ('x', 'y', 'u', 'v', 'c'),
    'tripcolor': ('x', 'y', 'z'),  # NOTE: only parse cmap below this point
    'tricontour': ('x', 'y', 'z'),
    'tricontourf': ('x', 'y', 'z'),
    'imshow': ('z',),  # NOTE: native arg is 'X'
    'matshow': ('z',),  # NOTE: native arg is 'Z'
    'spy': ('z',),  # NOTE: native arg is 'Z'
}

# Standardized cycle keyword arguments translated to artist
# arguments to permit cycling over additional properties
CYCLE_ARGS_APPLY = {
    'scatter': {
        'color': 'c',
        'markersize': 's',
        'linewidth': 'linewidths',
        'markeredgewidth': 'linewidths',
        'markeredgecolor': 'edgecolors',
        'alpha': 'alpha',
        'marker': 'marker',
    }
}


# Standardization docstrings
_shared_args_docstring = """
data : dict-like, optional
    A dict-like dataset container (e.g., `~pandas.DataFrame` or `~xarray.DataArray`).
    If passed, positional arguments must be valid `data` keys and the arrays used for
    plotting are retrieved with ``data[key]``. This is a native `matplotlib feature \
<https://matplotlib.org/stable/gallery/misc/keyword_plotting.html>`__.
autoformat : bool, optional
    Whether *x* axis labels, *y* axis labels, axis formatters, axes titles,
    legend labels, and colorbar labels are automatically configured when
    a `~pandas.Series`, `~pandas.DataFrame` or `~xarray.DataArray` is passed
    to the plotting command. Default is :rc:`autoformat`.
"""
_1d_args_docstring = """
*args : ({y},) or ({x}, {y})
    The data passed as positional arguments. Interpreted as follows:

    * If only *{y}* coordinates are passed, try to infer the *{x}* coordinates
      from the `~pandas.Series` or `~pandas.DataFrame` indices or the
      `~xarray.DataArray` coordinate arrays. Otherwise, the *{x}* coordinates
      are ``np.arange(0, {y}.shape[0])``.
    * If the *{y}* coordinates are a 2D array, plot each column of data in succession
      (except where each column of data represents a statistical distribution, as with
      ``boxplot``, ``violinplot``, or when using ``means=True`` or ``medians=True``).

%(plot.shared_args)s
"""
_1d_multi_args_docstring = """
*args : ({y}2,), ({x}, {y}2), or ({x}, {y}1, {y}2)
    The data passed as positional arguments. Interpreted as follows:

    * If only *{y}2* coordinates are passed, try to infer the *{x}* coordinates
      from the `~pandas.Series` or `~pandas.DataFrame` indices or the
      `~xarray.DataArray` coordinate arrays. Otherwise, the *{x}* coordinates
      are ``np.arange(0, {y}2.shape[0])``.
    * If only *{x}* and *{y}2* coordinates are passed, set the
      *{y}1* coordinates to zero. This draws elements originating
      from the zero line.
    * If both `{y}1` and `{y}2` are provided, draw elements between
      these points. If either are 2D, draw elements by iterating over
      each column.
"""
_2d_args_docstring = """
*args : (z1, ...) or (x, y, z1, ...)
    The data passed as positional arguments. Interpreted as follows:

    * If only *z* coordinates are passed, try to infer the *x* and *y* coordinates
      from the `~pandas.DataFrame` indices and columns or the `~xarray.DataArray`
      coordinate arrays. Otherwise, the *y* coordinates are ``np.arange(0, y.shape[0])``
      and the *x* coordinates are ``np.arange(0, y.shape[1])``.
    * For ``pcolor`` and ``pcolormesh``, calculate coordinate *edges* using
      `~proplot.utils.edges` or `~proplot.utils.edges2d` if *centers* were
      provided. For all other methods, calculate coordinate *centers*
      if *edges* were provided.

%(plot.shared_args)s
order : {{'C', 'F'}}, optional
    If ``'C'``, *z* coordinates should be shaped ``(y, x)``. If ``'F'``,
    *z* coordinates should be shaped ``(x, y)``. Default is ``'C'``.
globe : bool, optional
    *For `proplot.axes.GeoAxes` only*. Whether to ensure global coverage.
    Default is ``False``. When set to ``True`` this does the following:

    #. Interpolates input data to the North and South poles by setting the data
       values at the poles to the mean from latitudes nearest each pole.
    #. Makes meridional coverage "circular", i.e. the last longitude coordinate
       equals the first longitude coordinate plus 360\N{DEGREE SIGN}.
    #. *For `~proplot.axes.BasemapAxes` only*. Cycles 1D longitude vectors to fit
       within the map edges. For example, if the projection central longitude
       is 90\N{DEGREE SIGN}, the data is shifted so that it spans -90\N{DEGREE SIGN}
       to 270\N{DEGREE SIGN}.
"""
docstring.snippets['plot.shared_args'] = _shared_args_docstring
docstring.snippets['plot.1d_args'] = docstring.add_snippets(_1d_args_docstring.format(x='x', y='y'))  # noqa: E501
docstring.snippets['plot.1d_argsx'] = docstring.add_snippets(_1d_args_docstring.format(x='y', y='x'))  # noqa: E501
docstring.snippets['plot.1d_multi_args'] = docstring.add_snippets(_1d_multi_args_docstring.format(x='x', y='y'))  # noqa: E501
docstring.snippets['plot.1d_multi_argsx'] = docstring.add_snippets(_1d_multi_args_docstring.format(x='y', y='x'))  # noqa: E501
docstring.snippets['plot.2d_args'] = docstring.add_snippets(_2d_args_docstring)


# Auto colorbar and legend docstring
_colorbar_legend_docstring = """
colorbar, cbar : bool, int, or str, optional
    If not ``None``, this is a location specifying where to draw an *inset*
    or *panel* colorbar from the resulting object(s). If ``True``, the
    default location is used. Valid locations are described in
    `~proplot.axes.Axes.colorbar`.
colorbar_kw, cbar_kw : dict-like, optional
    Ignored if `colorbar` is ``None``. Extra keyword args for the call
    to `~proplot.axes.Axes.colorbar`.
legend, leg : bool, int, or str, optional
    If not ``None``, this is a location specifying where to draw an *inset*
    or *panel* legend from the resulting object(s). If ``True``, the
    default location is used. Valid locations are described in
    `~proplot.axes.Axes.legend`.
legend_kw, leg_kw : dict-like, optional
    Ignored if `legend` is ``None``. Extra keyword args for the call
    to `~proplot.axes.Axes.legend`.
"""
docstring.snippets['plot.colorbar_legend'] = _colorbar_legend_docstring


# Error indication docstrings
_error_means_docstring = """
mean, means : bool, optional
    Whether to plot the means of each column for 2D *{y}* coordinates. Means
    are calculated with `numpy.nanmean`. If no other arguments are specified,
    this also sets ``barstd=True`` (and ``boxstd=True`` for violin plots).
median, medians : bool, optional
    Whether to plot the medians of each column for 2D *{y}* coordinates. Medians
    are calculated with `numpy.nanmedian`. If no other arguments arguments are
    specified, this also sets ``barstd=True`` (and ``boxstd=True`` for violin plots).
"""
_error_bars_docstring = """
barstd, barstds : float, (float, float), or bool, optional
    *Valid only if `mean` or `median` is ``True``*. Standard deviation multiples for
    *thin error bars* with optional whiskers (i.e., caps). If scalar, then +/- that
    multiple is used. If ``True``, the default standard deviation range of +/-3 is
    used. Standard deviations are calculated with `numpy.nanstd`
barpctile, barpctiles : float, (float, float) or bool, optional
    *Valid only if `mean` or `median` is ``True``*. As with `barstd`, but instead
    using *percentiles* for the error bars. If scalar, that percentile range is
    used (e.g., ``90`` shows the 5th to 95th percentiles). If ``True``, the default
    percentile range of 0 to 100 is used. Percentiles are calculated with
    `numpy.nanpercentile`.
bardata : 2 x N array or 1D array, optional
    *Valid only if `mean` and `median` are ``False``*. If shape is 2 x N, these
    are the lower and upper bounds for the thin error bars. If shape is N, these
    are the absolute, symmetric deviations from the central points.
boxstd, boxstds, boxpctile, boxpctiles, boxdata : optional
    As with `barstd`, `barpctile`, and `bardata`, but for *thicker error bars*
    representing a smaller interval than the thin error bars. If `boxstd` is
    ``True``, the default standard deviation range of +/-1 is used. If `boxpctiles`
    is ``True``, the default percentile range of 25 to 75 is used (i.e., the
    interquartile range). When "boxes" and "bars" are combined, this has the
    effect of drawing miniature box-and-whisker plots.
barcolor, boxcolor : color-spec, optional
    Colors for the thin and thick error bars. Default is ``'k'``.
barzorder, boxzorder : float, optional
    The "zorder" for the thin and thick error bars. Default is ``2.5``.
barlinewidth, boxlinewidth, barlw, boxlw : float, optional
    Line widths for the thin and thick error bars, in points. The defaults
    are ``barlw=0.8`` and ``boxlw=4 * barlw``.
boxmarker : bool, optional
    Whether to draw a small marker in the middle of the box denoting the mean or
    median position. Ignored if `boxes` is ``False``.
boxmarkercolor : color-spec, optional
    Color for the `boxmarker` marker. Default is ``'w'``.
capsize : float, optional
    The cap size for thin error bars in points.
"""
_error_shading_docstring = """
shadestd, shadestds, shadepctile, shadepctiles, shadedata : optional
    As with `barstd`, `barpctile`, and `bardata`, but using *shading* to indicate
    the error range. If `shadestds` is ``True``, the default standard deviation
    range of +/-2 is used. If `shadepctiles` is ``True``, the default
    percentile range of 10 to 90 is used.
fadestd, fadestds, fadepctile, fadepctiles, fadedata : optional
    As with `shadestd`, `shadepctile`, and `shadedata`, but for an additional,
    more faded, *secondary* shaded region. If `fadestds` is ``True``, the default
    standard deviation range of +/-3 is used. If `fadepctiles` is ``True``,
    the default percentile range of 0 to 100 is used.
shadecolor, fadecolor : color-spec, optional
    Colors for the different shaded regions. Default is to inherit the parent color.
shadezorder, fadezorder : float, optional
    The "zorder" for the different shaded regions. Default is ``1.5``.
shadealpha, fadealpha : float, optional
    The opacity for the differnt shaded regions. Defaults are ``0.4`` and ``0.2``.
shadelabel, fadelabel : bool or str, optional
    Labels for the shaded regions to be used as separate legend entries. To toggle
    labels "on" and apply a *default* label, use e.g. ``shadelabel=True``. To apply
    a *custom* label, use e.g. ``shadelabel='label'``. Otherwise, the shading is
    drawn underneath the line and/or marker in the legend entry.
"""
docstring.snippets['plot.error_means'] = _error_means_docstring.format(y='y')
docstring.snippets['plot.error_meansx'] = _error_means_docstring.format(y='x')
docstring.snippets['plot.error_bars'] = _error_bars_docstring
docstring.snippets['plot.error_shading'] = _error_shading_docstring


# Color docstrings
_cycle_docstring = """
cycle : cycle-spec, optional
    The cycle specifer, passed to the `~proplot.constructor.Cycle`
    constructor. If the returned list of colors is unchanged from the
    current axes color cycler, the axes cycle will **not** be reset to the
    first position.
cycle_kw : dict-like, optional
    Passed to `~proplot.constructor.Cycle`.
"""
_cmap_norm_docstring = """
cmap : colormap spec, optional
    The colormap specifer, passed to the `~proplot.constructor.Colormap`
    constructor.
cmap_kw : dict-like, optional
    Passed to `~proplot.constructor.Colormap`.
norm : normalizer spec, optional
    The colormap normalizer, used to warp data before passing it
    to `~proplot.colors.DiscreteNorm`. This is passed to the
    `~proplot.constructor.Norm` constructor.
norm_kw : dict-like, optional
    Passed to `~proplot.constructor.Norm`.
extend : {{'neither', 'min', 'max', 'both'}}, optional
    Whether to assign unique colors to out-of-bounds data and draw
    "extensions" (triangles, by default) on the colorbar.
"""
docstring.snippets['plot.cycle'] = _cycle_docstring
docstring.snippets['plot.cmap_norm'] = _cmap_norm_docstring


# Levels docstrings
# NOTE: In some functions we only need some components
_vmin_vmax_docstring = """
vmin, vmax : float, optional
    Used to determine level locations if `levels` or `values` is an integer.
    Actual levels may not fall exactly on `vmin` and `vmax`, but the minimum
    level will be no smaller than `vmin` and the maximum level will be
    no larger than `vmax`. If `vmin` or `vmax` are not provided, the
    minimum and maximum data values are used.
"""
_levels_values_docstring = """
N
    Shorthand for `levels`.
levels : int or list of float, optional
    The number of level edges or a list of level edges. If the former,
    `locator` is used to generate this many level edges at "nice" intervals.
    If the latter, the levels should be monotonically increasing or
    decreasing (note that decreasing levels will only work with ``pcolor``
    plots, not ``contour`` plots). Default is :rc:`image.levels`.
values : int or list of float, optional
    The number of level centers or a list of level centers. If the former,
    `locator` is used to generate this many level centers at "nice" intervals.
    If the latter, levels are inferred using `~proplot.utils.edges`.
    This will override any `levels` input.
discrete : bool, optional
    If ``False``, the `~proplot.colors.DiscreteNorm` is not applied to the
    colormap when ``levels=N`` or ``levels=array_of_values``  are not
    explicitly requested. Instead, the number of levels in the colormap will be
    roughly controlled by :rcraw:`image.lut`. This has a similar effect
    to using `levels=large_number` but it may improve rendering speed.
    By default, this is ``False`` only for `~matplotlib.axes.Axes.imshow`,
    `~matplotlib.axes.Axes.matshow`, `~matplotlib.axes.Axes.spy`,
    `~matplotlib.axes.Axes.hexbin`, and `~matplotlib.axes.Axes.hist2d` plots.
"""
_auto_levels_docstring = """
inbounds : bool, optional
    If ``True`` (the default), when automatically selecting levels in the presence
    of hard *x* and *y* axis limits (i.e., when `~matplotlib.axes.Axes.set_xlim`
    or `~matplotlib.axes.Axes.set_ylim` have been called previously), only the
    in-bounds data is sampled. Default is :rc:`image.inbounds`.
locator : locator-spec, optional
    The locator used to determine level locations if `levels` or `values`
    is an integer and `vmin` and `vmax` were not provided. Passed to the
    `~proplot.constructor.Locator` constructor. Default is
    `~matplotlib.ticker.MaxNLocator` with ``levels`` integer levels.
locator_kw : dict-like, optional
    Passed to `~proplot.constructor.Locator`.
symmetric : bool, optional
    If ``True``, automatically generated levels are symmetric
    about zero.
positive : bool, optional
    If ``True``, automatically generated levels are positive
    with a minimum at zero.
negative : bool, optional
    If ``True``, automatically generated levels are negative
    with a maximum at zero.
nozero : bool, optional
    If ``True``, ``0`` is removed from the level list. This is
    mainly useful for `~matplotlib.axes.Axes.contour` plots.
"""
docstring.snippets['plot.levels_manual'] = _levels_values_docstring
docstring.snippets['plot.levels_auto'] = _auto_levels_docstring
docstring.snippets['plot.vmin_vmax'] = _vmin_vmax_docstring


# Labels docstrings
_1d_labels_docstring = """
label : float or str, optional
    The legend label to be used for this plotted element.
labels, values : list of float or list of str, optional
    Used with 2D input arrays. The legend labels or colorbar coordinates for
    each column in the array. Can be numeric or string, and must match
    the number of columns in the 2D array.
"""
_2d_labels_docstring = """
labels : bool, optional
    For `~matplotlib.axes.Axes.contour`, whether to add contour labels
    with `~matplotlib.axes.Axes.clabel`. For `~matplotlib.axes.Axes.pcolor`
    or `~matplotlib.axes.Axes.pcolormesh`, whether to add labels to the
    center of grid boxes. In the latter case, the text will be black
    when the luminance of the underlying grid box color is >50%%, and
    white otherwise.
labels_kw : dict-like, optional
    Ignored if `labels` is ``False``. Extra keyword args for the labels.
    For `~matplotlib.axes.Axes.contour`, passed to
    `~matplotlib.axes.Axes.clabel`.  For `~matplotlib.axes.Axes.pcolor`
    or `~matplotlib.axes.Axes.pcolormesh`, passed to
    `~matplotlib.axes.Axes.text`.
fmt : format-spec, optional
    Passed to the `~proplot.constructor.Norm` constructor, used to format
    number labels. You can also use the `precision` keyword arg.
precision : int, optional
    Maximum number of decimal places for the number labels. Number labels
    are generated with the `~proplot.ticker.SimpleFormatter` formatter,
    which permits limiting the precision.
"""
docstring.snippets['plot.1d_labels'] = _1d_labels_docstring
docstring.snippets['plot.2d_labels'] = _2d_labels_docstring


# Composite docstrings
_1d_kwargs_docstring = """
%(plot.cycle)s
%(plot.1d_labels)s
%(plot.colorbar_legend)s
"""
_2d_kwargs_docstring = """
%(plot.cmap_norm)s
%(plot.levels_manual)s
%(plot.vmin_vmax)s
%(plot.levels_auto)s
%(plot.2d_labels)s
%(plot.colorbar_legend)s
"""
_levels_all_docstring = """
%(plot.cmap_norm)s
%(plot.levels_manual)s
%(plot.vmin_vmax)s
%(plot.levels_auto)s
%(plot.colorbar_legend)s
"""  # 2d kwargs minus 'labels' for e.g. 'imshow'
docstring.snippets['plot.1d_kwargs'] = docstring.add_snippets(_1d_kwargs_docstring)
docstring.snippets['plot.2d_kwargs'] = docstring.add_snippets(_2d_kwargs_docstring)
docstring.snippets['plot.levels_all'] = docstring.add_snippets(_levels_all_docstring)


# Plot docstring
_plot_docstring = """
Plot lines.

Parameters
----------
%(plot.1d_args{suffix})s

Other parameters
----------------
%(plot.1d_kwargs)s
%(plot.error_means{suffix})s
%(plot.error_bars)s
%(plot.error_shading)s
**kwargs
    Passed to `~matplotlib.axes.Axes.plot`.

See also
--------
PlotAxes.plot
PlotAxes.plotx
matplotlib.axes.Axes.plot
"""
docstring.snippets['plot.plot'] = docstring.add_snippets(
    _plot_docstring.format(suffix='')
)
docstring.snippets['plot.plotx'] = docstring.add_snippets(
    _plot_docstring.format(suffix='x')
)


# Step docstring
# NOTE: Internally matplotlib implements step with thin wrapper of plot
_step_docstring = """
Plot steps.

Parameters
----------
%(plot.1d_args{suffix})s

Other parameters
----------------
%(plot.1d_kwargs)s
**kwargs
    Passed to `~matplotlib.axes.Axes.step`.

See also
--------
PlotAxes.step
PlotAxes.stepx
matplotlib.axes.Axes.step
"""
docstring.snippets['plot.step'] = docstring.add_snippets(
    _step_docstring.format(suffix='')
)
docstring.snippets['plot.stepx'] = docstring.add_snippets(
    _step_docstring.format(suffix='x')
)


# Lines docstrings
_lines_docstring = """
Plot {orientation} lines.

Parameters
----------
%(plot.1d_multi_args{suffix})s

Other parameters
----------------
stack, stacked : bool, optional
    Whether to "stack" successive columns of the *{y}1* array. If this is
    ``True`` and *{y}2* was provided, it will be ignored.
negpos : bool, optional
    Whether to color lines greater than zero with `poscolor` and lines less
    than zero with `negcolor`.
negcolor, poscolor : color-spec, optional
    Colors to use for the negative and positive lines. Ignored if `negpos`
    is ``False``. Defaults are :rc:`negcolor` and :rc:`poscolor`.
color, colors : color-spec or list thereof, optional
    The line color(s).
linestyle, linestyles : linestyle-spec or list thereof, optional
    The line style(s).
lw, linewidth, linewidths : linewidth-spec or list thereof, optional
    The line width(s).
%(plot.1d_kwargs)s
**kwargs
    Passed to `~matplotlib.axes.Axes.{prefix}lines`.

See also
--------
PlotAxes.vlines
PlotAxes.hlines
matplotlib.axes.Axes.vlines
matplotlib.axes.Axes.hlines
"""
docstring.snippets['plot.vlines'] = docstring.add_snippets(
    _lines_docstring.format(y='y', prefix='v', suffix='', orientation='vertical')
)
docstring.snippets['plot.hlines'] = docstring.add_snippets(
    _lines_docstring.format(y='x', prefix='h', suffix='x', orientation='horizontal')
)


# Scatter function docstring
_scatter_docstring = """
Plot markers with flexible keyword arguments.

Parameters
----------
%(plot.1d_args{suffix})s
s, size, markersize : float or list of float, optional
    The marker size(s). The units are optionally scaled by
    `smin` and `smax`.
smin, smax : float, optional
    The minimum and maximum marker size in units ``points^2`` used to scale
    `s`. If not provided, the marker sizes are equivalent to the values in `s`.
c, color, markercolor : color-spec or list thereof, or array, optional
    The marker fill color(s). If this is an array of scalar values, colors
    will be generated using the colormap `cmap` and normalizer `norm`.
%(plot.vmin_vmax)s

Other parameters
----------------
lw, linewidth, linewidths, markeredgewidth, markeredgewidths \
: float or list thereof, optional
    The marker edge width.
edgecolors, markeredgecolor, markeredgecolors \
: color-spec or list thereof, optional
    The marker edge color.
%(plot.cmap_norm)s
%(plot.levels_manual)s
%(plot.levels_auto)s
%(plot.1d_kwargs)s
%(plot.error_means{suffix})s
%(plot.error_bars)s
%(plot.error_shading)s
**kwargs
    Passed to `~matplotlib.axes.Axes.scatter`.

See also
--------
PlotAxes.scatter
PlotAxes.scatterx
matplotlib.axes.Axes.scatter
"""
docstring.snippets['plot.scatter'] = docstring.add_snippets(
    _scatter_docstring.format(suffix='')
)
docstring.snippets['plot.scatterx'] = docstring.add_snippets(
    _scatter_docstring.format(suffix='x')
)


# Bar function docstring
_bar_docstring = """
Plot individual, grouped, or stacked bars.

Parameters
----------
{x}, height, width, {bottom} : float or list of float, optional
    The dimensions of the bars. If the *{x}* coordinates are not provided,
    they are set to ``np.arange(0, len(height))``. The units for width
    are *relative* by default.
%(plot.1d_args{suffix})s
width : float or array-like, optional
    The width(s) of the bars relative to the {x} coordinate step size.
    Can be passed as a third positional argument.
{bottom} : float or array-like, optional
    The coordinate(s) of the {bottom} edge of the bars. Default is
    ``0``. Can be passed as a fourth positinal argument.
absolute_width : bool, optional
    Whether to make the `width` units *absolute*. If ``True``, this
    restores the default matplotlib behavior. Default is ``False``.

Other parameters
----------------
stack, stacked : bool, optional
    Whether to stack columns of the input array or plot the bars
    side-by-side in groups.
negpos : bool, optional
    Whether to shade bars greater than zero with `poscolor` and bars less
    than zero with `negcolor`.
negcolor, poscolor : color-spec, optional
    Colors to use for the negative and positive bars. Ignored if `negpos`
    is ``False``. Defaults are :rc:`negcolor` and :rc:`poscolor`.
lw, linewidth : float, optional
    The edge width for the bar patches.
edgecolor : color-spec, optional
    The edge color for the bar patches.
%(plot.1d_kwargs)s
%(plot.error_means{suffix})s
%(plot.error_bars)s
**kwargs
    Passed to `~matplotlib.axes.Axes.bar{suffix2}`.

See also
--------
PlotAxes.bar
PlotAxes.barh
matplotlib.axes.Axes.bar
matplotlib.axes.Axes.barh
"""
docstring.snippets['plot.bar'] = docstring.add_snippets(
    _bar_docstring.format(x='x', bottom='bottom', suffix='', suffix2='')
)
docstring.snippets['plot.barh'] = docstring.add_snippets(
    _bar_docstring.format(x='y', bottom='left', suffix='x', suffix2='h')
)


# Histogram docstrings
_hist_docstring = """
Plot {orientation} histograms.

Parameters
----------
%(plot.1d_args{suffix})s

Other parameters
----------------
%(plot.1d_kwargs)s
**kwargs
    Passed to `~matplotlib.axes.Axes.hist`.

See also
--------
PlotAxes.hist
PlotAxes.histh
matplotlib.axes.Axes.hist
"""
docstring.snippets['plot.hist'] = docstring.add_snippets(
    _hist_docstring.format(suffix='', orientation='vertical')
)
docstring.snippets['plot.histh'] = docstring.add_snippets(
    _hist_docstring.format(suffix='x', orientation='horizontal')
)


# Area plot docstring
_fill_docstring = """
Plot individual, grouped, or overlaid shading patches.

Parameters
----------
%(plot.1d_multi_args{suffix})s

Other parameters
----------------
stack, stacked : bool, optional
    Whether to "stack" successive columns of the `{y}1` array. If this is
    ``True`` and `{y}2` was provided, it will be ignored.
negpos : bool, optional
    Whether to shade where ``{y}1 >= {y}2`` with `poscolor` and where ``{y}1 < {y}2``
    with `negcolor`. For example, to shade positive values red and negative values
    blue, simply use ``ax.fill_between{suffix}({x}, {y}, negpos=True)``.
negcolor, poscolor : color-spec, optional
    Colors to use for the negative and positive shaded regions. Ignored if `negpos`
    is ``False``. Defaults are :rc:`negcolor` and :rc:`poscolor`.
where : ndarray, optional
    Boolean ndarray mask for points you want to shade. See `this example \
<https://matplotlib.org/stable/gallery/pyplots/whats_new_98_4_fill_between.html>`__.
lw, linewidth : float, optional
    The edge width for the area patches.
edgecolor : color-spec, optional
    The edge color for the area patches.
%(plot.1d_kwargs)s
**kwargs
    Passed to `~matplotlib.axes.Axes.fill_between{suffix}`.

See also
--------
PlotAxes.area
PlotAxes.areax
PlotAxes.fill_between
PlotAxes.fill_betweenx
matplotlib.axes.Axes.fill_between
matplotlib.axes.Axes.fill_betweenx
"""
docstring.snippets['plot.fill_between'] = docstring.add_snippets(
    _fill_docstring.format(x='x', y='y', suffix='')
)
docstring.snippets['plot.fill_betweenx'] = docstring.add_snippets(
    _fill_docstring.format(x='y', y='x', suffix='x')
)


# Box plot docstrings
_boxplot_docstring = """
Plot {orientation} boxes and whiskers with a nice default style.

Parameters
----------
%(plot.1d_args{suffix})s

Other parameters
----------------
mean, means : bool, optional
    If ``True``, this passes ``showmeans=True`` and ``meanline=True`` to
    `~matplotlib.axes.Axes.boxplot`.
fill : bool, optional
    Whether to fill the box with a color.
fillcolor : color-spec, list, optional
    The fill color for the boxes. Default is the next color cycler color. If
    a list, it should be the same length as the number of objects.
fillalpha : float, optional
    The opacity of the boxes. Default is ``0.7``. If a list, should be
    the same length as the number of objects.
lw, linewidth : float, optional
    The linewidth of all objects. Default is ``0.8``.
color, edgecolor : color-spec, list, optional
    The color of all objects. Defalut is ``'black'``. If a list, it should
    be the same length as the number of objects.
meanls, medianls, meanlinestyle, medianlinestyle : line style-spec, optional
    The line style for the mean and median lines drawn horizontally
    across the box.
boxcolor, capcolor, whiskercolor, fliercolor, meancolor, mediancolor \
: color-spec, list, optional
    The color of various boxplot components. If a list, it should be the
    same length as the number of objects. These are shorthands so you don't
    have to pass e.g. a ``boxprops`` dictionary.
boxlw, caplw, whiskerlw, flierlw, meanlw, medianlw, boxlinewidth, caplinewidth,\
meanlinewidth, medianlinewidth, whiskerlinewidth, flierlinewidth : float, optional
    The line width of various boxplot components. These are shorthands so
    you don't have to pass e.g. a ``boxprops`` dictionary.
marker : marker-spec, optional
    Marker style for the 'fliers', i.e. outliers.
markersize : float, optional
    Marker size for the 'fliers', i.e. outliers.
%(plot.1d_kwargs)s
**kwargs
    Passed to `~matplotlib.axes.Axes.boxplot`.

See also
--------
PlotAxes.boxes
PlotAxes.boxesh
PlotAxes.boxplot
PlotAxes.boxploth
matplotlib.axes.Axes.boxplot
"""
docstring.snippets['plot.boxplot'] = docstring.add_snippets(
    _boxplot_docstring.format(orientation='vertical', suffix='')
)
docstring.snippets['plot.boxploth'] = docstring.add_snippets(
    _boxplot_docstring.format(orientation='horizontal', suffix='x')
)


# Violin plot docstrings
_violinplot_docstring = """
Plot {orientation} violins with a nice default style matching `this matplotlib example \
<https://matplotlib.org/stable/gallery/statistics/customized_violin.html>`__.

Parameters
----------
%(plot.1d_args{suffix})s

Other parameters
----------------
fillcolor : color-spec, list, optional
    The violin plot fill color. Default is the next color cycler color. If
    a list, it should be the same length as the number of objects.
fillalpha : float, optional
    The opacity of the violins. Default is ``0.7``. If a list, it
    should be the same length as the number of objects.
lw, linewidth : float, optional
    The linewidth of the line objects. Default is ``0.8``.
color, edgecolor : color-spec, list, optional
    The edge color for the violin patches. Default is ``'black'``. If a
    list, it should be the same length as the number of objects.
%(plot.1d_kwargs)s
%(plot.error_bars)s
**kwargs
    Passed to `~matplotlib.axes.Axes.violinplot`.

Note
----
It is no longer possible to show minima and maxima with whiskers --
while this is useful for `~matplotlib.axes.Axes.boxplot`\\ s it is
redundant for `~matplotlib.axes.Axes.violinplot`\\ s.

See also
--------
PlotAxes.violins
PlotAxes.violinsh
PlotAxes.violinplot
PlotAxes.violinploth
matplotlib.axes.Axes.violinplot
"""
docstring.snippets['plot.violinplot'] = docstring.add_snippets(
    _violinplot_docstring.format(orientation='vertical', suffix='')
)
docstring.snippets['plot.violinploth'] = docstring.add_snippets(
    _violinplot_docstring.format(orientation='horizontal', suffix='x')
)


# Contour docstrings
_contour_docstring = """
Parameters
----------
%(plot.2d_args)s

Other parameters
----------------
lw, linewidth, linewidths
    The width of the contour lines. For `contourf` plots, lines are added
    between the filled contours.
ls, linestyle, linestyles
    The style of the contour lines. For `contourf` plots, lines are added
    between the filled contours.
c, color, colors, ec, edgecolor, edgecolors
    The color for the contour lines. If not passed, the color is
    determined by `cmap` and the `z` data.
edgefix : bool, optional
    Whether to fix an issue where `white lines appear between filled contours \
<https://stackoverflow.com/q/8263769/4970632>`__ in saved vector graphics.
    This can slow down figure rendering. Default is :rc:`image.edgefix`.
"""
docstring.snippets['plot.contour'] = docstring.add_snippets(_contour_docstring)


# Pcolor docstring
_pcolor_docstring = """
Parameters
----------
%(plot.2d_args)s

Other parameters
----------------
%(plot.2d_kwargs)s
lw, linewidth, linewidths
    The width of lines between grid boxes.
ls, linestyle, linestyles
    The style of lines between grid boxes.
c, color, colors, ec, edgecolor, edgecolors
    The color for lines between grid boxes.
edgefix : bool, optional
    Whether to fix an issue where `white lines appear between grid boxes \
<https://stackoverflow.com/q/8263769/4970632>`__ in saved vector graphics.
    This can slow down figure rendering. Default is :rc:`image.edgefix`.
"""
docstring.snippets['plot.pcolor'] = docstring.add_snippets(_pcolor_docstring)


# Flow function docstring
_flow_docstring = """
Parameters
----------
%(plot.2d_args)s

Other parameters
----------------
%(plot.2d_kwargs)s
"""
docstring.snippets['plot.flow'] = docstring.add_snippets(_flow_docstring)


def _load_objects():
    """
    Delay loading expensive modules. We just want to detect if *input
    arrays* belong to these types -- and if this is the case, it means the
    module has already been imported! So, we only try loading these classes
    within autoformat calls. This saves >~500ms of import time.
    """
    global DataArray, DataFrame, Series, Index, ndarray
    ndarray = np.ndarray
    DataArray = getattr(sys.modules.get('xarray', None), 'DataArray', ndarray)
    DataFrame = getattr(sys.modules.get('pandas', None), 'DataFrame', ndarray)
    Series = getattr(sys.modules.get('pandas', None), 'Series', ndarray)
    Index = getattr(sys.modules.get('pandas', None), 'Index', ndarray)


_load_objects()


def _is_number(data):
    """
    Test whether input is numeric array rather than datetime or strings.
    """
    return len(data) and np.issubdtype(_to_numpy_array(data).dtype, np.number)


def _is_string(data):
    """
    Test whether input is array of strings.
    """
    return len(data) and isinstance(_to_numpy_array(data).flat[0], str)


def _to_array_like(data):
    """
    Convert list of lists to array-like type.
    """
    _load_objects()
    if data is None:
        raise ValueError('Cannot convert None data.')
        return None
    if not isinstance(data, (ndarray, DataArray, DataFrame, Series, Index)):
        data = np.asarray(data)
    if not np.iterable(data):
        data = np.atleast_1d(data)
    return data


def _to_numpy_array(data):
    """
    Convert arbitrary input to ndarray cleanly. Returns a masked
    array if input is a masked array.
    """
    return np.atleast_1d(getattr(data, 'values', data))


class _MetaPlotAxes(type):
    """
    Redirect internal plotting calls to native matplotlib methods.
    """
    # NOTE: This obviates the need for basemap-specific redirection to mpl.axes.Axes.
    # WARNING: This will fail if there is no native matplotlib method
    def __new__(cls, name, bases, dct_orig):
        dct = dct_orig.copy()
        for attr in (*KEYWORD_TO_POSITIONAL, 'violin'):
            func = dct_orig.get(attr, None)
            if not callable(func):
                continue
            @functools.wraps(func)  # noqa: E306
            def _redirect_matplotlib(self, *args, _func_override=func, **kwargs):
                _func_name = _func_override.__name__
                _func_native = getattr(super(self, PlotAxes), _func_name, None)
                _internal_call = getattr(self, '_internal_call', None)
                if _internal_call and callable(_func_native):  # noqa: E501
                    return _func_native(self, *args, **kwargs)
                else:
                    return _func_override(self, *args, **kwargs)
        return super().__new__(cls, name, bases, dct)


class PlotAxes(base.Axes, metaclass=_MetaPlotAxes):
    """
    Second lowest-level axes subclass implementing plotting overrides.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs
            Passed to `~proplot.axes.Axes`.

        See also
        --------
        matplotlib.axes.Axes
        proplot.axes.Axes
        proplot.axes.CartesianAxes
        proplot.axes.PolarAxes
        proplot.axes.GeoAxes
        """
        super().__init__(*args, **kwargs)

    def _call_method(self, name, *args, **kwargs):
        """
        Call the plotting method and use context object to redirect internal calls to
        native methods. Also implement automatic colorbars, legends, and labels.
        """
        # Properties used for post-processing
        edgefix = kwargs.pop('edgefix', rc['image.edgefix'])
        contour_kw = kwargs.pop('contour_kw', None) or {}

        # Properties used for automatic colorbar, legend, and labels
        # TODO: Ensure no conflicts with 1D plotting commands
        # TODO: Implement these within individual plotting methods
        kw_labels = _pop_kwargs(kwargs, 'labels', 'labels_kw', 'fmt', 'formatter', 'formatter_kw', 'precision')  # noqa: E501
        kw_colorbar_legend = _pop_kwargs(kwargs, 'legend', 'legend_kw', 'colorbar', 'colorbar_kw')  # noqa: E501

        # Properties added for 'colorbar' function
        ticks = kwargs.pop('ticks', None)
        extend = kwargs.get('extend', None) if 'contour' in name else kwargs.pop('extend', None)  # noqa: E501

        # Safely plot stuff
        # NOTE: Previously allowed internal matplotlib plotting function calls to run
        # through proplot overrides then avoided awkward conflicts in piecemeal fashion
        # (e.g. relative bar widths, sticky plot lines). Now just prevent internal calls
        # from running through proplot overrides altogether.
        with _state_context(self, _internal_call=True):
            # Call method
            if getattr(self, 'name', None) == 'proplot_basemap':
                obj = getattr(self.projection, name)(*args, ax=self, **kwargs)
            else:
                obj = getattr(self, name)(*args, **kwargs)  # NOTE: redirects

            # Automatic stuff
            self._auto_labels(obj, **kw_labels)
            self._auto_colorbar_legend(obj, **kw_colorbar_legend)
            if edgefix:
                self._fix_edges(obj)  # valid for any ScalarMappable
            if contour_kw:
                self.contour(*args, **contour_kw, **kwargs)
            if isinstance(obj, mcm.ScalarMappable):
                obj._colorbar_ticks = ticks  # used by proplot colorbar
                obj._colorbar_extend = extend  # used by proplot colorbar
                # obj._colorbar_label = label  # TODO: add this!!!

        return obj

    def _call_negpos(
        self, name, x, y1, y2, *args,
        negcolor=None, poscolor=None, use_where=False, **kwargs
    ):
        """
        Call the method with allotment for "negative" and "positive" regions.
        """
        # Handle ignored arguments
        for key in ('color', 'colors', 'stack', 'stacked', 'where'):
            value = kwargs.pop(key, None)
            if value is None:
                continue
            warnings._warn_proplot(
                f'{name}() argument {key}={value!r} is incompatible with '
                'negpos=True. Ignoring.'
            )

        # Plot negative component
        color = _not_none(negcolor, rc['negcolor'])
        if use_where:
            y1neg, y2neg, kwargs['where'] = y1, y2, y1 < y2
            kwargs.setdefault('interpolate', True)  # see fill_between docs
        else:
            y1neg, y2neg = self._mask_negpos(y2 < y1, y1, y2)
        negobj = self._call_method(name, x, y1neg, y2neg, *args, color=color, **kwargs)

        # Plot positive component
        if use_where:
            y1pos, y2pos, kwargs['where'] = y1, y2, y2 >= y1
        else:
            y1pos, y2pos = self._mask_negpos(y2 >= y1, *y1, y2)
        color = _not_none(poscolor, rc['poscolor'])
        posobj = self._call_method(name, x, y1pos, y2pos, *args, color=color, **kwargs)

        return (negobj, posobj)

    @staticmethod
    def _mask_negpos(mask, *args):
        """
        Apply the mask to the input arrays. Values matching ``False`` are
        set to `np.nan`.
        """
        invalid = ~mask  # True if invalid
        args_masked = []
        for arg in args:
            if arg.size > 1 and arg.shape != invalid.shape:
                raise ValueError('Shape mismatch between mask and array.')
            arg_masked = arg.astype(np.float64)
            if arg.size == 1:
                pass
            elif invalid.size == 1:
                arg_masked = np.nan if invalid.item() else arg_masked
            elif arg.size > 1:
                arg_masked[invalid] = np.nan
            args_masked.append(arg_masked)
        return args_masked[0] if len(args_masked) == 1 else args_masked

    @staticmethod
    def _get_data(data, *args):
        """
        Try to convert positional `key` arguments to `data[key]`. If argument is string
        it could be a valid positional argument like `fmt` so do not raise error.
        """
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                try:
                    array = data[arg]
                except KeyError:
                    pass
                else:
                    args[i] = array
        return args

    @staticmethod
    def _get_coords(*args, which='x', **kwargs):
        """
        Convert string arrays and lists to index coordinates.
        """
        # NOTE: Why FixedLocator and not IndexLocator? The latter requires plotting
        # lines or else error is raised... very strange.
        # NOTE: Why IndexFormatter and not FixedFormatter? The former ensures labels
        # correspond to indices while the latter can mysteriously truncate labels.
        res = []
        for arg in args:
            arg = _to_array_like(arg)
            if _is_string(arg) and arg.ndim > 1:
                raise ValueError('Non-1D string coordinate input is unsupported.')
            if not _is_string(arg):
                res.append(arg)
                continue
            idx = np.arange(len(arg))
            kwargs.setdefault(which + 'locator', mticker.FixedLocator(idx))
            kwargs.setdefault(which + 'formatter', pticker._IndexFormatter(_to_numpy_array(arg)))  # noqa: E501
            kwargs.setdefault(which + 'minorlocator', mticker.NullLocator())
            res.append(idx)
        return *res, kwargs

    @staticmethod
    def _get_title(data, include_units=True):
        """
        Return the "title" associated with an array-like object with metadata. This
        might be a pandas `DataFrame` `name` or a name constructed from `DataArray`
        attributes. In the latter case we search for `long_name` and `standard_name`,
        preferring the former, and append `(units)` if `units` is ``True``. If no
        names are available but units are available we just use the units string.
        """
        title = None
        _load_objects()
        if isinstance(data, ndarray):
            pass
        # Xarray object with possible long_name, standard_name, and units attributes.
        # Output depends on if units is True
        elif isinstance(data, DataArray):
            title = getattr(data, 'name', None)
            for key in ('standard_name', 'long_name'):
                title = data.attrs.get(key, title)
            if include_units:
                suffix = data.attrs.get('units', None)
                if title and suffix:
                    title = f'{title} ({suffix})'
                elif suffix:
                    title = suffix
        # Pandas object. DataFrame has no native name attribute but user can add one
        # See: https://github.com/pandas-dev/pandas/issues/447
        elif isinstance(data, (DataFrame, Series, Index)):
            title = getattr(data, 'name', None) or None
        # Standardize result
        if title is not None:
            title = str(title).strip()
        return title

    @staticmethod
    def _get_labels(data, axis=0, always=True):
        """
        Return the array-like "labels" along axis `axis` from an array-like
        object. These might be an xarray `DataArray` or pandas `Index`. If
        `always` is ``False`` we return ``None`` for simple ndarray input.
        """
        # NOTE: Previously inferred 'axis 1' metadata of 1D variable using the
        # data values metadata but that is incorrect. The paradigm for 1D plots
        # is we have row coordinates representing x, data values representing y,
        # and column coordinates representing individual series.
        if axis not in (0, 1, 2):
            raise ValueError(f'Invalid axis {axis}.')
        labels = None
        _load_objects()
        if isinstance(data, ndarray):
            if not always:
                pass
            elif axis < data.ndim:
                labels = np.arange(data.shape[axis])
            else:  # requesting 'axis 1' on a 1D array
                labels = np.array([0])
        # Xarray object
        # NOTE: Even if coords not present .coords[dim] auto-generates indices
        elif isinstance(data, DataArray):
            if axis < data.ndim:
                labels = data.coords[data.dims[axis]]
            elif not always:
                pass
            else:
                labels = np.array([0])
        # Pandas object
        elif isinstance(data, (DataFrame, Series, Index)):
            if axis == 0 and isinstance(data, (DataFrame, Series)):
                labels = data.index
            elif axis == 1 and isinstance(data, (DataFrame,)):
                labels = data.columns
            elif not always:
                pass
            else:  # beyond dimensionality
                labels = np.array([0])
        # Everything else
        # NOTE: Ensure data is at least 1D in _to_array_like so this covers everything
        else:
            raise ValueError(f'Unrecognized array type {type(data)}.')
        return labels

    @staticmethod
    def _require_centers(x, y, z):
        """
        Enforce that coordinates are centers. Convert from edges if possible.
        """
        xlen, ylen = x.shape[-1], y.shape[0]
        if z.ndim == 2 and z.shape[1] == xlen - 1 and z.shape[0] == ylen - 1:
            # Get centers given edges
            if all(z.ndim == 1 and z.size > 1 and _is_number(z) for z in (x, y)):
                x = 0.5 * (x[1:] + x[:-1])
                y = 0.5 * (y[1:] + y[:-1])
            else:
                if (
                    x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1
                    and _is_number(x)
                ):
                    x = 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])
                if (
                    y.ndim == 2 and y.shape[0] > 1 and y.shape[1] > 1
                    and _is_number(y)
                ):
                    y = 0.25 * (y[:-1, :-1] + y[:-1, 1:] + y[1:, :-1] + y[1:, 1:])
        elif z.shape[-1] != xlen or z.shape[0] != ylen:
            # Helpful error message
            raise ValueError(
                f'Input shapes x {x.shape} and y {y.shape} '
                f'must match z centers {z.shape} '
                f'or z borders {tuple(i+1 for i in z.shape)}.'
            )
        return x, y

    @staticmethod
    def _require_edges(x, y, z):
        """
        Enforce that coordinates are edges. Convert from centers if possible.
        """
        xlen, ylen = x.shape[-1], y.shape[0]
        if z.ndim == 2 and z.shape[1] == xlen and z.shape[0] == ylen:
            # Get edges given centers
            if all(z.ndim == 1 and z.size > 1 and _is_number(z) for z in (x, y)):
                x = edges(x)
                y = edges(y)
            else:
                if (
                    x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1
                    and _is_number(x)
                ):
                    x = edges2d(x)
                if (
                    y.ndim == 2 and y.shape[0] > 1 and y.shape[1] > 1
                    and _is_number(y)
                ):
                    y = edges2d(y)
        elif z.shape[-1] != xlen - 1 or z.shape[0] != ylen - 1:
            # Helpful error message
            raise ValueError(
                f'Input shapes x {x.shape} and y {y.shape} must match '
                f'array centers {z.shape} or '
                f'array borders {tuple(i + 1 for i in z.shape)}.'
            )
        return x, y

    def _cartopy_2d(self, x, y, *zs, globe=False):
        """
        Fix cartopy 2D geographic data arrays.
        """
        # Fix coordinates
        x, y = self._lon_monotonic(x, y)

        # Fix data
        x_orig, y_orig, zs_orig = x, y, zs
        zs = []
        for z_orig in zs_orig:
            # Bail for 2D coordinates
            if not globe or x_orig.ndim > 1 or y_orig.ndim > 1:
                zs.append(z_orig)
                continue
            # Fix holes over poles by *interpolating* there
            y, z = self._lat_global(y_orig, z_orig)
            # Fix seams by ensuring circular coverage (cartopy can plot over map edges)
            if x_orig[0] % 360 != (x_orig[-1] + 360) % 360:
                x = ma.concatenate((x_orig, [x_orig[0] + 360]))
                z = ma.concatenate((z, z[:, :1]), axis=1)
            zs.append(z)

        return x, y, *zs

    def _basemap_1d(self, x, *ys):
        """
        Fix basemap geographic 1D data arrays.
        """
        xmin, xmax = self.projection.lonmin, self.projection.lonmax
        x_orig, ys_orig = self._lon_monotonic(x), ys
        ys = []
        for y_orig in ys_orig:
            x, y = self._lon_global(x_orig, y_orig, xmin, xmax)
            ys.append(y)
        return x, *ys

    def _basemap_2d(self, x, y, *zs, globe=False):
        """
        Fix basemap 2D geographic data arrays.
        """
        # Fix coordinates
        x = self._lon_monotonic(x)

        # Fix data
        xmin, xmax = self.projection.lonmin, self.projection.lonmax
        x_orig, y_orig, zs_orig = x, y, zs
        zs = []
        for z_orig in zs_orig:
            # Ensure data is within map bounds
            x, z_orig = self._lon_global(x_orig, z_orig, xmin, xmax)
            # Bail for 2D coordinates
            if not globe or x_orig.ndim > 1 or y_orig.ndim > 1:
                zs.append(z_orig)
                continue
            # Fix holes over poles by *interpolating* there
            y, z = self._lat_global(y_orig, z_orig)
            # Fix seams at map boundary
            if x[0] == xmin and x.size - 1 == z.shape[1]:  # scenario 1
                # Edges (e.g. pcolor) fit perfectly against seams. Size is unchanged.
                pass
            elif x.size - 1 == z.shape[1]:  # scenario 2
                # Edges (e.g. pcolor) do not fit perfectly. Size augmented by 1.
                x = ma.append(xmin, x)
                x[-1] = xmin + 360
                z = ma.concatenate((z[:, -1:], z), axis=1)
            elif x.size == z.shape[1]:  # scenario 3
                # Centers (e.g. contour) interpolated to edge. Size augmented by 2.
                xi = np.array([x[-1], x[0] + 360])
                if xi[0] == xi[1]:  # impossible to interpolate
                    pass
                else:
                    zq = ma.concatenate((z[:, -1:], z[:, :1]), axis=1)
                    xq = xmin + 360
                    zq = (zq[:, :1] * (xi[1] - xq) + zq[:, 1:] * (xq - xi[0])) / (xi[1] - xi[0])  # noqa: E501
                    x = ma.concatenate(([xmin], x, [xmin + 360]))
                    z = ma.concatenate((zq, z, zq), axis=1)
            else:
                raise ValueError('Unexpected shapes of coordinates or data arrays.')
            zs.append(z)

        # Convert coordinates
        if x.ndim == 1 and y.ndim == 1:
            x, y = np.meshgrid(x, y)
        x, y = self.projection(x, y)

        return x, y, *zs

    @staticmethod
    def _lon_monotonic(x):
        """
        Ensure longitudes are monotonic and make `~numpy.ndarray` copies so the
        contents can be modified. Ignores 2D coordinate arrays.
        """
        if x.ndim != 1 or all(x < x[0]):  # skip 2D arrays and monotonic backwards data
            return x
        lon1 = x[0]
        filter_ = x < lon1
        while filter_.sum():
            filter_ = x < lon1
            x[filter_] += 360
        return x

    @staticmethod
    def _lon_global(x, y, xmin, xmax):
        """
        Ensure data for basemap plots is restricted between the minimum and
        maximum longitude of the projection. Input is the ``x`` and ``y``
        coordinates. The ``y`` coordinates are rolled along the rightmost axis.
        """
        if x.ndim != 1:
            return x, y

        # Roll in same direction if some points on right-edge extend
        # more than 360 above min longitude; *they* should be on left side
        lonroll = np.where(x > xmin + 360)[0]  # tuple of ids
        if lonroll.size:  # non-empty
            roll = x.size - lonroll.min()
            x = np.roll(x, roll)
            y = np.roll(y, roll, axis=-1)
            x[:roll] -= 360  # make monotonic

        # Set NaN where data not in range xmin, xmax. Must be done
        # for regional smaller projections or get weird side-effects due
        # to having valid data way outside of the map boundaries
        y = y.astype(np.float64)
        if x.size - 1 == y.shape[-1]:  # test western/eastern grid cell edges
            y[..., (x[1:] < xmin) | (x[:-1] > xmax)] = np.nan
        elif x.size == y.shape[-1]:  # test the centers and pad by one for safety
            where = np.where((x < xmin) | (x > xmax))[0]
            y[..., where[1:-1]] = np.nan

        return x, y

    @staticmethod
    def _lat_global(y, z):
        """
        Add data points on the poles as the average of highest latitude data.
        """
        # Get means
        with np.errstate(all='ignore'):
            p1 = z[0, :].mean()  # pole 1, make sure is not 0D DataArray!
            p2 = z[-1, :].mean()  # pole 2
        if hasattr(p1, 'item'):
            p1 = np.asscalar(p1)  # happens with DataArrays
        if hasattr(p2, 'item'):
            p2 = np.asscalar(p2)
        # Concatenate
        ps = (-90, 90) if (y[0] < y[-1]) else (90, -90)
        z1 = np.repeat(p1, z.shape[1])[None, :]
        z2 = np.repeat(p2, z.shape[1])[None, :]
        y = ma.concatenate((ps[:1], y, ps[1:]))
        z = ma.concatenate((z1, z, z2), axis=0)
        return y, z

    def _auto_format_1d(
        self, x, *ys, name='plot', autoformat=False,
        label=None, values=None, labels=None, **kwargs
    ):
        """
        Try to retrieve default coordinates from array-like objects and apply default
        formatting. Also update the keyword arguments.
        """
        # Parse input
        parametric = name in ('parametric',)
        scatter = name in ('scatter',)
        hist = name in ('hist',)
        box = name in ('boxplot', 'violinplot')
        pie = name in ('pie',)
        stem = name in ('stem',)
        nocycle = name in ('stem', 'hexbin', 'hist2d', 'parametric')
        labels = _not_none(
            label=label,
            values=values,
            labels=labels,
            colorbar_kw_values=kwargs.get('colorbar_kw', {}).pop('values', None),
            legend_kw_labels=kwargs.get('legend_kw', {}).pop('labels', None),
        )

        # Retrieve the x coords
        # NOTE: Allow for "ragged array" input to boxplot and violinplot.
        # NOTE: Where columns represent distributions, like for box and violinplot or
        # where we use 'means' or 'medians', columns coords (axis 1) are 'x' coords.
        # Otherwise, columns represent e.g. lines and row coords (axis 0) are 'x' coords
        dists = box or any(kwargs.get(s) for s in ('mean', 'means', 'median', 'medians'))  # noqa: E501
        ragged = any(getattr(y, 'dtype', None) == 'object' for y in ys)
        xaxis = 1 if dists and not ragged else 0
        if x is None and not hist:
            x = self._get_labels(ys[0], axis=xaxis)  # infer from rows or columns

        # Default legend or colorbar labels and title. We want default legend
        # labels if this is an object with 'title' metadata and/or coords are string
        # WARNING: Confusing terminology differences here -- for box and violin plots
        # 'labels' refer to indices along x axis. Get interpreted that way down the line
        if autoformat and not stem:
            # The inferred labels and title
            title = None
            if labels is not None:
                title = self._get_title(labels)
            else:
                yaxis = xaxis if box or pie else xaxis + 1
                labels = self._get_labels(ys[0], axis=yaxis, always=False)
                title = self._get_title(labels)  # e.g. if labels is a Series
                if labels is None:
                    pass
                elif not title and not any(isinstance(_, str) for _ in labels):
                    labels = None
            # Apply the title
            if title:
                kwargs.setdefault('colorbar_kw', {}).setdefault('title', title)
                if not nocycle:
                    kwargs.setdefault('legend_kw', {}).setdefault('title', title)
        # Apply the labels
        if labels is not None:
            if not nocycle:
                kwargs['labels'] = _to_numpy_array(labels)
            elif parametric:
                values, colorbar_kw = self._get_coords(labels, which='')
                kwargs['values'] = _to_numpy_array(values)
                kwargs.setdefault('colorbar_kw', {}).update(colorbar_kw)

        # The basic x and y settings
        # NOTE: We no longer test the function name
        horiz = kwargs.get('vert') is False or kwargs.get('orientation') == 'horizontal'
        sx, sy = 'yx' if horiz else 'xy'
        sy = sx if hist else sy  # histogram 'y' values end up along 'x' axis
        if not hasattr(self, 'projection'):
            # Apply label
            # NOTE: Do not overwrite existing labels!
            kw_format = {}
            if autoformat:  # 'y' axis
                title = self._get_title(ys[0])
                if title and not getattr(self, f'get_{sy}label')():
                    kw_format[sy + 'label'] = title
            if autoformat and not hist:  # 'x' axis
                title = self._get_title(x)
                if title and not getattr(self, f'get_{sx}label')():
                    kw_format[sx + 'label'] = title

            # Handle string-type coordinates
            if not pie and not hist:
                x, kw_format = self._get_coords(x, which=sx, **kw_format)
            if not hist and not box and not pie:
                *ys, kw_format = self._get_coords(*ys, which=sy, **kw_format)
            if not hist and not scatter and not parametric and x.ndim == 1 and x.size > 1 and x[1] < x[0]:  # noqa: E501
                kw_format[sx + 'reverse'] = True  # auto reverse

            # Appply
            if kw_format:
                self.format(**kw_format)

        # Finally strip metadata
        # WARNING: Most methods that accept 2D arrays use columns of data, but when
        # pandas DataFrame specifically is passed to hist, boxplot, or violinplot, rows
        # of data assumed! Converting to ndarray necessary.
        return _to_numpy_array(x), *map(_to_numpy_array, ys), kwargs

    @docstring.add_snippets
    def _standardize_1d(self, name, *args, data=None, autoformat=None, **kwargs):
        """
        Interpret positional arguments for all "1D" plotting commands
        so the syntax is consistent.
        """
        # TODO: Replace with keyword args
        box = name in ('boxplot', 'violinplot')
        parametric = name in ('parametric',)
        onecoord = name in ('hist',)
        twocoords = name in ('vlines', 'hlines', 'fill_between', 'fill_betweenx')
        allowempty = name in ('fill', 'plot', 'plotx',)
        autoformat = _not_none(autoformat, rc['autoformat'])

        # Find and translate input args
        args = list(args)
        keys = KEYWORD_TO_POSITIONAL.get(name, {})
        for idx, key in enumerate(keys):
            if key in kwargs:
                args.insert(idx, kwargs.pop(key))
        if data is not None:
            args = self._get_data(data, *args)
        if not args:
            if allowempty:
                return []  # match matplotlib behavior
            else:
                raise TypeError('Positional arguments are required.')

        # Parse positional args
        if parametric and len(args) == 3:  # allow positional values
            kwargs['values'] = args.pop(2)
        if parametric and 'c' in kwargs:  # handle aliases
            kwargs['values'] = kwargs.pop('c')
        if onecoord or len(args) == 1:  # allow hist() positional bins
            x, ys, args = None, args[:1], args[1:]
        elif twocoords:
            x, ys, args = args[0], args[1:3], args[3:]
        else:
            x, ys, args = args[0], args[1:2], args[2:]
        if x is not None:
            x = _to_array_like(x)
        ys = tuple(map(_to_array_like, ys))

        # Automatic formatting and coordinates
        # NOTE: For 'hist' the 'x' coordinate is None then is ignored in _apply_cycle.
        x, *ys, kwargs = self._auto_format_1d(
            x, *ys, name=name, autoformat=autoformat, **kwargs
        )

        # Ensure data is monotonic and falls within map bounds
        if getattr(self, 'name', None) == 'proplot_basemap' and kwargs.get('latlon', None):  # noqa: E501
            x, *ys = self._basemap_1d(x, *ys)

        # Call function
        if box:
            kwargs.setdefault('positions', x)  # *this* is how 'x' is passed to boxplot
        return x, *ys, *args, kwargs

    def _auto_format_2d(
        self, x, y, *zs, name=None, order='C', autoformat=False, **kwargs
    ):
        """
        Try to retrieve default coordinates from array-like objects and apply default
        formatting. Also apply optional transpose and update the keyword arguments.
        """
        # Retrieve coordinates
        allow1d = name in ('barbs', 'quiver')  # these also allow 1D data
        if x is None and y is None:
            z = zs[0]
            if z.ndim == 1:
                x = self._get_labels(z, axis=0)
                y = np.zeros(z.shape)  # default barb() and quiver() behavior in mpl
            else:
                x = self._get_labels(z, axis=1)
                y = self._get_labels(z, axis=0)
            if order == 'F':
                x, y = y, x

        # Check coordinate and data shapes
        shapes = tuple(z.shape for z in zs)
        if any(len(_) != 2 and not (allow1d and len(_) == 1) for _ in shapes):
            raise ValueError(f'Data arrays must be 2d, but got shapes {shapes}.')
        shapes = set(shapes)
        if len(shapes) > 1:
            raise ValueError(f'Data arrays must have same shape, but got shapes {shapes}.')  # noqa: E501
        if any(_.ndim not in (1, 2) for _ in (x, y)):
            raise ValueError('x and y coordinates must be 1d or 2d.')
        if x.ndim != y.ndim:
            raise ValueError('x and y coordinates must have same dimensionality.')
        if order == 'F':  # TODO: double check this
            x, y = x.T, y.T  # in case they are 2-dimensional
            zs = tuple(z.T for z in zs)

        # The labels and XY axis settings
        if not hasattr(self, 'projection'):
            # Apply labels
            # NOTE: Do not overwrite existing labels!
            kw_format = {}
            if autoformat:
                for s, d in zip('xy', (x, y)):
                    title = self._get_title(d)
                    if title and not getattr(self, f'get_{s}label')():
                        kw_format[s + 'label'] = title

            # Handle string-type coordinates
            x, kw_format = self._get_coords(x, which='x', **kw_format)
            y, kw_format = self._get_coords(y, which='y', **kw_format)
            for s, d in zip('xy', (x, y)):
                if d.size > 1 and d.ndim == 1 and _to_numpy_array(d)[1] < _to_numpy_array(d)[0]:  # noqa: E501
                    kw_format[s + 'reverse'] = True

            # Apply formatting
            if kw_format:
                self.format(**kw_format)

        # Default colorbar label
        # WARNING: This will fail for any funcs wrapped by _standardize_2d but not
        # wrapped by _apply_cmap. So far there are none.
        if autoformat:
            kwargs.setdefault('colorbar_kw', {})
            title = self._get_title(zs[0])
            if title and True:
                kwargs['colorbar_kw'].setdefault('label', title)

        # Finally strip metadata
        return _to_numpy_array(x), _to_numpy_array(y), *map(_to_numpy_array, zs), kwargs

    @docstring.add_snippets
    def _standardize_2d(
        self, name, *args, data=None, autoformat=None, order='C', globe=False, **kwargs
    ):
        """
        Interpret positional arguments for all "2D" plotting commands
        so the syntax is consistent.
        """
        # TODO: Replace name-dependent behavior with keyword args
        method = kwargs.pop('_method')
        name = method.__name__
        pcolor = name in ('pcolor', 'pcolormesh', 'pcolorfast')
        allow1d = name in ('barbs', 'quiver')  # these also allow 1D data
        autoformat = _not_none(autoformat, rc['autoformat'])

        # Find and translate input args
        if data is not None:
            args = self._get_data(data, *args)
        if not args:
            raise TypeError('Positional arguments are required.')

        # Parse input args
        if len(args) > 2:
            x, y, *args = args
        else:
            x = y = None
        if x is not None:
            x = _to_array_like(x)
        if y is not None:
            y = _to_array_like(y)
        zs = tuple(map(_to_array_like, args))

        # Automatic formatting
        x, y, *zs, kwargs = self._auto_format_2d(
            x, y, *zs, name=name, order=order, autoformat=autoformat, **kwargs
        )

        # Standardize coordinates
        if pcolor:
            x, y = self._require_edges(x, y, zs[0])
        else:
            x, y = self._require_centers(x, y, zs[0])

        # Cartopy projection axes
        if (
            not allow1d and getattr(self, 'name', None) == 'proplot_cartopy'
            and isinstance(kwargs.get('transform', None), PlateCarree)
        ):
            x, y, *zs = self._cartopy_2d(x, y, *zs, globe=globe)

        # Basemap projection axes
        elif (
            not allow1d and getattr(self, 'name', None) == 'proplot_basemap'
            and kwargs.get('latlon', None)
        ):
            x, y, *zs = self._basemap_2d(x, y, *zs, globe=globe)
            kwargs['latlon'] = False

        # Call function
        return x, y, *zs, kwargs

    def _error_distribution(
        self, x, y, *args,
        mean=None, means=None, median=None, medians=None, **kwargs
    ):
        """
        Take means or medians.
        """
        # Get means or medians along columns
        # TODO: Permit 3D array with error dimension coming first
        # NOTE: Previously went to great pains to preserve metadata but now retrieval
        # of default legend handles moved to _auto_format_1d so can strip.
        x, y, *args = args
        data = y
        means = _not_none(mean=mean, means=means)
        medians = _not_none(median=median, medians=medians)
        if means and medians:
            warnings._warn_proplot('Cannot have both means=True and medians=True. Using former.')  # noqa: E501
            medians = None
        if means or medians:
            if data.ndim != 2:
                raise ValueError(f'Expected 2D array for means=True. Got {data.ndim}D.')
            if means:
                y = np.nanmean(data, axis=0)
            elif medians:
                y = np.nanmedian(data, axis=0)

        # Save argument passed to _error_bars
        kwargs['distribution'] = data
        return x, y, *args, kwargs

    @staticmethod
    def _error_data(
        y, stds=None, pctiles=None, errdata=None, distribution=None,
        stds_default=None, pctiles_default=None,
        absolute=False, label=False,
    ):
        """
        Return values that can be passed to the `~matplotlib.axes.Axes.errorbar`
        `xerr` and `yerr` keyword args.
        """
        # Parse stds arguments
        # NOTE: Have to guard against "truth value of an array is ambiguous" errors
        if stds is True:
            stds = stds_default
        elif stds is False or stds is None:
            stds = None
        else:
            stds = np.atleast_1d(stds)
            if stds.size == 1:
                stds = sorted((-stds.item(), stds.item()))
            elif stds.size != 2:
                raise ValueError('Expected scalar or length-2 stdev specification.')

        # Parse pctiles arguments
        if pctiles is True:
            pctiles = pctiles_default
        elif pctiles is False or pctiles is None:
            pctiles = None
        else:
            pctiles = np.atleast_1d(pctiles)
            if pctiles.size == 1:
                delta = (100 - pctiles.item()) / 2.0
                pctiles = sorted((delta, 100 - delta))
            elif pctiles.size != 2:
                raise ValueError('Expected scalar or length-2 pctiles specification.')

        # Incompatible settings
        if distribution is None and any(_ is not None for _ in (stds, pctiles)):
            raise ValueError(
                'To automatically compute standard deviations or percentiles on '
                'columns of data you must pass means=True or medians=True.'
            )
        if stds is not None and pctiles is not None:
            warnings._warn_proplot(
                'You passed both a standard deviation range and a percentile range for '
                'error indicators. Using the former.'
            )
            pctiles = None
        if distribution is not None and errdata is not None:
            stds = pctiles = None
            warnings._warn_proplot(
                'You explicitly provided the error bounds but also requested '
                'automatically calculating means or medians on data columns. '
                'It may make more sense to use the "stds" or "pctiles" keyword args '
                'and have *proplot* calculate the error bounds.'
            )

        # Compute error data in format that can be passed to maxes.Axes.errorbar()
        # NOTE: Include option to pass symmetric deviation from central points
        if errdata is not None:
            # Manual error data
            if y.ndim != 1:
                raise ValueError(
                    'errdata with 2D y coordinates is not yet supported.'
                )
            label_default = 'uncertainty'
            err = _to_numpy_array(errdata)
            if (
                err.ndim not in (1, 2)
                or err.shape[-1] != y.size
                or err.ndim == 2 and err.shape[0] != 2
            ):
                raise ValueError(
                    f'errdata must have shape (2, {y.shape[-1]}), but got {err.shape}.'
                )
            if err.ndim == 1:
                abserr = err
                err = np.empty((2, err.size))
                err[0, :] = y - abserr  # translated back to absolute deviations below
                err[1, :] = y + abserr
        elif stds is not None:
            # Standard deviations
            label_default = fr'{abs(stds[1])}$\sigma$ range'
            stds = _to_numpy_array(stds)[:, None]
            err = y + stds * np.nanstd(distribution, axis=0)
        elif pctiles is not None:
            # Percentiles
            label_default = f'{pctiles[1] - pctiles[0]}% range'
            err = np.nanpercentile(distribution, pctiles, axis=0)
        else:
            raise ValueError('You must provide error bounds.')

        # Return label possibly
        if label is True:
            label = label_default
        elif not label:
            label = None

        # Make relative data for maxes.Axes.errorbar() ingestion
        if not absolute:
            err = err - y
            err[0, :] *= -1  # absolute deviations from central points

        # Return data with legend entry
        return err, label

    def _error_bars(
        self, x, y, *_, distribution=None, vert=True, bars=False, violins=False,
        barstd=None, barstds=None, barpctile=None, barpctiles=None, bardata=None,
        boxstd=None, boxstds=None, boxpctile=None, boxpctiles=None, boxdata=None,
        boxcolor=None, barcolor=None, boxzorder=2.5, barzorder=2.5,
        boxlinewidth=None, boxlw=None, barlinewidth=None, barlw=None, capsize=None,
        boxmarker=None, boxmarkercolor='white', **kwargs,
    ):
        """
        Add error bars.
        """
        # NOTE: Ensure 'vert' and 'violins' is passed here!
        # TODO: Do not add auto error bars if user enabled shading!
        bars = any(_ is not None for _ in (bardata, barstds, barpctiles))
        boxes = any(_ is not None for _ in (boxdata, boxstds, boxpctiles))
        barstds = _not_none(barstd=barstd, barstds=barstds)
        boxstds = _not_none(boxstd=boxstd, boxstds=boxstds)
        barpctiles = _not_none(barpctile=barpctile, barpctiles=barpctiles)
        boxpctiles = _not_none(boxpctile=boxpctile, boxpctiles=boxpctiles)
        if distribution is not None and not bars and not boxes:
            barstds = True
            if violins:
                boxstds = True

        # Parse args
        # NOTE: Should not use plot() 'linewidth' for bar elements
        getter = kwargs.pop if violins else kwargs.get if bars else lambda *args: None
        boxmarker = _not_none(boxmarker, True if violins else False)
        capsize = _not_none(capsize, 3.0)
        linewidth = _not_none(getter('linewidth', None), 1.0)  # aliases should be ok
        barlinewidth = _not_none(barlinewidth=barlinewidth, barlw=barlw, default=linewidth)  # noqa: E501
        boxlinewidth = _not_none(boxlinewidth=boxlinewidth, boxlw=boxlw, default=4 * barlinewidth)  # noqa: E501
        edgecolor = _not_none(getter('edgecolor', None), 'k')
        barcolor = _not_none(barcolor, edgecolor)
        boxcolor = _not_none(boxcolor, barcolor)

        # Draw thin or thick error bars from distributions or explicit errdata
        sy = 'y' if vert else 'x'  # yerr
        ex, ey = (x, y) if vert else (y, x)
        eobjs = []
        if boxes:
            edata, _ = self._error_data(
                y, distribution=distribution,
                stds=boxstds, pctiles=boxpctiles, errdata=boxdata,
                stds_default=(-1, 1), pctiles_default=(25, 75),
            )
            if boxmarker:
                self.scatter(
                    ex, ey, s=boxlinewidth, marker='o', color=boxmarkercolor, zorder=5
                )
            obj = self.errorbar(
                ex, ey, color=boxcolor, linewidth=boxlinewidth, linestyle='none',
                capsize=0, zorder=boxzorder, **{sy + 'err': edata}
            )
            eobjs.append(obj)
        if bars:  # now impossible to make thin bar width different from cap width!
            edata, _ = self._error_data(
                y, distribution=distribution,
                stds=barstds, pctiles=barpctiles, errdata=bardata,
                stds_default=(-3, 3), pctiles_default=(0, 100),
            )
            obj = self.errorbar(
                ex, ey, color=barcolor, linewidth=barlinewidth, linestyle='none',
                markeredgecolor=barcolor, markeredgewidth=barlinewidth,
                capsize=capsize, zorder=barzorder, **{sy + 'err': edata}
            )
            eobjs.append(obj)

        return *eobjs, kwargs

    def _error_shading(
        self, x, y, *_, distribution=None, vert=True, violins=False, bars=False,
        shadestd=None, shadestds=None, shadepctile=None, shadepctiles=None, shadedata=None,  # noqa: E501
        fadestd=None, fadestds=None, fadepctile=None, fadepctiles=None, fadedata=None,
        shadecolor=None, fadecolor=None, shadealpha=0.4, fadealpha=0.2,
        shadezorder=1.5, fadezorder=1.5, shadelabel=False, fadelabel=False, **kwargs
    ):
        """
        Add error shading.
        """
        shadestds = _not_none(shadestd=shadestd, shadestds=shadestds)
        fadestds = _not_none(fadestd=fadestd, fadestds=fadestds)
        shadepctiles = _not_none(shadepctile=shadepctile, shadepctiles=shadepctiles)
        fadepctiles = _not_none(fadepctile=fadepctile, fadepctiles=fadepctiles)
        shade = any(_ is not None for _ in (shadedata, shadestds, shadepctiles))
        fade = any(_ is not None for _ in (fadedata, fadestds, fadepctiles))

        # Parse keyword args and apply defaults
        shadecolor = _not_none(shadecolor, kwargs.get('facecolor'), kwargs.get('color'))
        if shade and shadecolor is None:
            key = 'facecolor' if violins or bars else 'color'
            parser = self._get_patches_for_fill if violins or bars else self._get_lines
            shadecolor = kwargs[key] = parser.get_next_color()
        fadecolor = _not_none(fadecolor, shadecolor)

        # Draw dark and light shading from distributions or explicit errdata
        eobjs = []
        fill = self.fill_between if vert else self.fill_betweenx
        if fade:
            edata, label = self._error_data(
                y, distribution=distribution,
                stds=fadestds, pctiles=fadepctiles, errdata=fadedata,
                stds_default=(-3, 3), pctiles_default=(0, 100),
                label=fadelabel, absolute=True,
            )
            eobj = fill(
                x, *edata, linewidth=0, label=label,
                color=fadecolor, alpha=fadealpha, zorder=fadezorder,
            )
            eobjs.append(eobj)
        if shade:
            edata, label = self._error_data(
                y, distribution=distribution,
                stds=shadestds, pctiles=shadepctiles, errdata=shadedata,
                stds_default=(-2, 2), pctiles_default=(10, 90),
                label=shadelabel, absolute=True,
            )
            eobj = fill(
                x, *edata, linewidth=0, label=label,
                color=shadecolor, alpha=shadealpha, zorder=shadezorder,
            )
            eobjs.append(eobj)

        return *eobjs, kwargs

    @staticmethod
    def _parse_vert(
        vert=None, orientation=None,
        default_vert=None, default_orientation=None,
        **kwargs
    ):
        """
        Interpret both 'vert' and 'orientation' and add to keyword args
        if a default is provided.
        """
        # NOTE: Users should only pass these to hist, boxplot, or violinplot. To change
        # the plot, scatter, area, or bar orientation users should use the differently
        # named functions. Internally, however, they use these keyword args.
        if default_vert is not None:
            kwargs['vert'] = _not_none(
                vert=vert,
                orientation=None if orientation is None else orientation == 'vertical',
                default=default_vert,
            )
        if default_orientation is not None:
            kwargs['orientation'] = _not_none(
                orientation=orientation,
                vert=None if vert is None else 'vertical' if vert else 'horizontal',
                default=default_orientation,
            )
        if kwargs.get('orientation', None) not in (None, 'horizontal', 'vertical'):
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'.")
        return kwargs

    def _from_cycle(self, name, **kwargs):
        """
        Return the properties that should be used by cycler.
        """
        # Return dict of cycle props and function keys for those unspecified by user
        # NOTE: Matplotlib uses the property cycler in _get_patches_for_fill for
        # scatter() plots. It only ever inherits color from that cycler. We instead
        # use _get_lines to help overarching goal of unifying plot() and scatter().
        # Now shading/bars loop over one cycle, plot/scatter along another.
        parser = self._get_lines  # the _process_plot_var_args instance
        prop_keys = set(parser._prop_keys) - {'color', 'linestyle', 'dashes'}
        get_from_cycle = {}  # which keys to apply from property cycler
        for prop, key in CYCLE_ARGS_APPLY.get(name, {}).items():
            value = kwargs.get(key, None)  # an _update_cycle argument
            if prop in prop_keys and value is None:  # key in cycler and prop unset
                get_from_cycle[prop] = key
        if get_from_cycle:
            kwargs['get_from_cycle'] = get_from_cycle
        return kwargs

    @staticmethod
    def _parse_cycle(self, N=None, cycle=None, cycle_kw=None, **kwargs):
        """
        Update the property cycle.
        """
        # Update the property cycler
        # TODO: Automatically use appropriate 'N'
        if cycle is None and not cycle_kw:
            return kwargs
        cycle_kw = cycle_kw or {}
        cycle_kw.setdefault('N', _not_none(N, 10))
        cycle_args = () if cycle is None else (cycle,)
        cycle = constructor.Cycle(*cycle_args, **cycle_kw)

        # Get the original property cycle
        # NOTE: Matplotlib saves an itertools.cycle(cycler) object under the _get_lines
        # and _get_patches_for_fill parsers. We must build up the original input cycler
        # by iterating over prop_cycler with next() then comparing those keys and values
        # to the input cycler keys and values. Only reset if the keys and values differ
        i = 0
        by_key = {}
        cycle_orig = self._get_lines.prop_cycler
        for i in range(len(cycle)):  # use the cycler object length as a guess
            prop = next(cycle_orig)  # returns to original pos if they were identical
            for key, value in prop.items():
                if key not in by_key:
                    by_key[key] = set()
                if isinstance(value, (list, np.ndarray)):
                    value = tuple(value)
                by_key[key].add(value)

        # Reset property cycler if it differs
        reset = set(by_key) != set(cycle.by_key())
        if not reset:  # test individual entries
            for key, value in cycle.by_key().items():
                if key in by_key and by_key[key] != set(value):
                    reset = True
                    break
        if reset:
            self.set_prop_cycle(cycle)  # update _get_lines and _get_patches_for_fill

        return kwargs

    def _iter_columns(
        self, *args, label=None, labels=None, values=None, get_from_cycle=None, **kwargs
    ):
        """
        Retrieve the positional and keyword arguments for successively
        calling the plotting function.
        """
        # Handle legend labels
        # NOTE: Must convert to ndarray or can get singleton DataArrays
        ncols = max(1 if _.ndim == 1 else _ for _ in args if isinstance(_, np.ndarray))
        labels = _not_none(label=label, values=values, labels=labels)
        if not np.iterable(labels) or isinstance(labels, str):
            labels = [labels] * ncols
        if len(labels) != ncols:
            raise ValueError(f'Array has {ncols} columns but got {len(labels)} labels.')  # noqa: E501
        if labels is not None:
            labels = [str(_not_none(label, '')) for label in _to_numpy_array(labels)]  # noqa: E501
        else:
            labels = [None] * ncols

        # Plot successive columns
        get_from_cycle = get_from_cycle or {}
        for i in range(ncols):
            # Keyword args for column
            kw = kwargs.copy()
            if get_from_cycle:
                props = next(self._get_lines.prop_cycler)
            for prop, key in get_from_cycle.items():
                kw[key] = props[prop]
            kw['label'] = labels[i] or None
            # Positional args for column
            a = tuple(
                a if not isinstance(a, np.ndaray) or a.ndim == 1 else a[:, i]
                for a in args
            )
            # Yield column
            yield *a, kw

    def _auto_labels(
        self, obj, *args, labels=False, labels_kw=None,
        fmt=None, formatter=None, formatter_kw, precision=None,
    ):
        """
        Add automatic labels.
        """
        # Apply labels
        # TODO: Add quiverkey to this!
        if not labels:
            return
        labels_kw = labels_kw or {}
        formatter_kw = formatter_kw or {}
        formatter = _not_none(
            fmt_labels_kw=labels_kw.pop('fmt', None),
            formatter_labels_kw=labels_kw.pop('formatter', None),
            fmt=fmt,
            formatter=formatter,
            default='simple'
        )
        formatter_kw.setdefault('precision', precision)
        formatter = constructor.Formattter(formatter, **formatter_kw)
        if isinstance(obj, mcontour.ContourSet):
            self._label_contours(obj, *args, fmt=formatter, **labels_kw)
        elif isinstance(obj, (mcollections.PolyCollection, mcollections.QuadMesh)):
            self._label_gridboxes(obj, fmt=formatter, **labels_kw)
        else:
            raise RuntimeError('Not possible to add labels to object {obj!r}.')

    def _label_contours(self, obj, *args, fmt=None, **kwargs):
        """
        Add labels to contours with support for shade-dependent filled contour labels
        flexible keyword arguments. Requires original arguments passed to contour func.
        """
        # Parse input args
        text_kw = {}
        for key in (*kwargs,):  # allow dict to change size
            if key not in (
                'levels', 'colors', 'fontsize', 'inline', 'inline_spacing',
                'manual', 'rightside_up', 'use_clabeltext',
            ):
                text_kw[key] = kwargs.pop(key)
        kwargs.setdefault('inline_spacing', 3)
        kwargs.setdefault('fontsize', rc['text.labelsize'])
        fmt = _not_none(fmt, pticker.SimpleFormatter())

        # Draw hidden additional contour for filled contour labels
        cobj = obj
        colors = None
        if _getattr_flexible(obj, 'filled'):  # guard against changes?
            cobj = self.contour(*args, levels=obj.levels, linewidths=0)
            lums = (to_xyz(obj.cmap(obj.norm(level)), 'hcl')[2] for level in obj.levels)
            colors = ['w' if lum < 50 else 'k' for lum in lums]
        kwargs.setdefault('colors', colors)

        # Draw labels
        labs = cobj.clabel(fmt=fmt, **kwargs)
        if labs is not None:  # returns None if no contours
            for lab in labs:
                lab.update(text_kw)

        return labs

    def _label_gridboxes(self, obj, fmt=None, **kwargs):
        """
        Add labels to pcolor boxes with support for shade-dependent text colors.
        """
        # Parse input args and populate _facecolors, which is initially unfilled
        # See: https://stackoverflow.com/a/20998634/4970632
        fmt = _not_none(fmt, pticker.SimpleFormatter())
        labels_kw = {'size': rc['text.labelsize'], 'ha': 'center', 'va': 'center'}
        labels_kw.update(kwargs)
        obj.update_scalarmappable()  # populate _facecolors

        # Get positions and contour colors
        array = obj.get_array()
        paths = obj.get_paths()
        colors = _to_numpy_array(obj.get_facecolors())
        edgecolors = _to_numpy_array(obj.get_edgecolors())
        if len(colors) == 1:  # weird flex but okay
            colors = np.repeat(colors, len(array), axis=0)
        if len(edgecolors) == 1:
            edgecolors = np.repeat(edgecolors, len(array), axis=0)

        # Apply colors
        labs = []
        for i, (color, path, num) in enumerate(zip(colors, paths, array)):
            if not np.isfinite(num):
                edgecolors[i, :] = 0
                continue
            bbox = path.get_extents()
            x = (bbox.xmin + bbox.xmax) / 2
            y = (bbox.ymin + bbox.ymax) / 2
            if 'color' not in kwargs:
                _, _, lum = to_xyz(color, 'hcl')
                if lum < 50:
                    color = 'w'
                else:
                    color = 'k'
                labels_kw['color'] = color
            lab = self.text(x, y, fmt(num), **labels_kw)
            labs.append(lab)
        obj.set_edgecolors(edgecolors)

        return labs

    @staticmethod
    def _fix_edges(obj, linewidth=0.3):
        """
        Fix white lines between between filled contours and mesh and fix issues with
        colormaps that are transparent.
        """
        # See: https://github.com/jklymak/contourfIssues
        # See: https://stackoverflow.com/q/15003353/4970632
        # 0.3pt is thick enough to hide lines but thin enough to not add "dots"
        # in corner of pcolor plots so good compromise.
        if not isinstance(obj, mcm.ScalarMappable):
            return
        cmap = obj.cmap
        if not cmap._isinit:
            cmap._init()
        if all(cmap._lut[:-1, 3] == 1):  # skip for cmaps with transparency
            edgecolor = 'face'
        else:
            edgecolor = 'none'
        # Contour fixes
        # NOTE: This also covers TriContourSet returned by tricontour
        if isinstance(obj, mcontour.ContourSet):
            for contour in obj.collections:
                contour.set_edgecolor(edgecolor)
                contour.set_linewidth(linewidth)
                contour.set_linestyle('-')
        # Pcolor fixes
        # NOTE: This ignores AxesImage and PcolorImage sometimes returned by pcolorfast
        elif isinstance(obj, (mcollections.PolyCollection, mcollections.QuadMesh)):
            if hasattr(obj, 'set_linewidth'):  # not always true for pcolorfast
                obj.set_linewidth(linewidth)
            if hasattr(obj, 'set_edgecolor'):  # not always true for pcolorfast
                obj.set_edgecolor(edgecolor)

    @staticmethod
    def _pop_unused_args(kwargs):
        """
        Remove args specific to `_auto_vmin_vmax` and `_auto_levels_locator`
        in case user passed their own levels and did not reach these blocks.
        """
        # NOTE: 'nozero' only makes sense when we are creating levels without
        # a colormap i.e. only in _auto_levels_locator.
        for key in (
            'vmin', 'vmax', 'locator', 'locator_kw',
            'symmetric', 'positive', 'negative', 'nozero',
            'counts', 'centers', 'inbounds', 'cmap_kw', 'norm_kw',
        ):
            value = kwargs.pop(key, None)
            if value is not None:
                warnings._warn_proplot(f'Ignoring unused argument {key}={value!r}.')

    def _adjust_inbounds(self, x, y, z, *, centers=False):
        """
        Adjust the smaple based on the axis limits.
        """
        # Get masks
        # TODO: Expand this to selecting x-limits giving scales y-limits, vice versa?
        # NOTE: X and Y coordinates were sanitized by _standardize_2d when passed here.
        xmask = ymask = None
        if any(_ is None or _.ndim not in (1, 2) for _ in (x, y, z)):
            return z
        if centers and z.ndim == 2:
            x, y = self._require_centers(x, y, z)
        if not self.get_autoscalex_on():
            xlim = self.get_xlim()
            xmask = (x >= xlim[0]) & (x <= xlim[1])
        if not self.get_autoscaley_on():
            ylim = self.get_ylim()
            ymask = (y >= ylim[0]) & (y <= ylim[1])

        # Subsample
        if xmask is not None and ymask is not None:
            z = z[np.ix_(ymask, xmask)] if z.ndim == 2 and xmask.ndim == 1 else z[ymask & xmask]  # noqa: E501
        elif xmask is not None:
            z = z[:, xmask] if z.ndim == 2 and xmask.ndim == 1 else z[xmask]
        elif ymask is not None:
            z = z[ymask, :] if z.ndim == 2 and ymask.ndim == 1 else z[ymask]
        return z

    def _auto_vmin_vmax(
        self, *args, vmin=None, vmax=None, counts=False, centers=False, inbounds=None,
        **kwargs,
    ):
        """
        Return a suitable colormap range based on the input data.

        Parameters
        ----------
        *args
            The x, y, z, data.
        vmin, vmax : float, optional
            The user input minimum and maximum.
        inbounds : bool, optional
            Whether to filter to in-bounds data.
        centers : bool, optional
            Whether to convert coordinates to 'centers'.
        counts : bool, optional
            Whether to guesstimate histogram counts rather than use data.

        Returns
        -------
        vmin, vmax : float
            The minimum and maximum.
        """
        # Which bounds we need to get
        automin = vmin is None
        automax = vmax is None
        inbounds = _not_none(inbounds, rc['image.inbounds'])
        if not automin and not automax:
            return vmin, vmax, kwargs

        # Get sample data
        # NOTE: Critical to use _to_array_like here because some commands
        # are unstandardized.
        # NOTE: Try to get reasonable *count* levels for hexbin/hist2d, but in general
        # have no way to select nice ones a priori (why we disable discretenorm).
        x = y = None
        if len(args) < 3:
            zs = map(_to_array_like, args)
        else:
            x, y, *zs = map(_to_array_like, args)
        vmins, vmaxs = [], []
        for z in zs:
            # Restrict to in-bounds data
            # Use catch-all exception because it really isn't mission-critical.
            if z.ndim > 2:
                continue  # 3D imshow plots will ignore the cmap we give it
            if counts:
                z = np.array([0, z.size]) // 10
            elif inbounds:
                # WARNING: Experimental, seems robust but this is not
                # mission-critical so keep this try-except clause for now.
                try:
                    z = self._adjust_inbounds(x, y, z, centers=centers)
                except Exception:
                    warnings._warn_proplot(
                        'Failed to adjust bounds for automatic colormap normalization.'  # noqa: E501
                    )
            # Mask invalid data
            z = ma.masked_invalid(z, copy=False)
            if automin:
                vmin = float(z.min())
            if automax:
                vmax = float(z.max())
            if vmin == vmax or ma.is_masked(vmin) or ma.is_masked(vmax):
                vmin, vmax = 0, 1
            vmins.append(vmin)
            vmaxs.append(vmax)

        # Return
        if vmins:
            return min(vmins), max(vmaxs), kwargs
        else:
            return 0, 1, kwargs  # fallback

    def _auto_levels_locator(
        self, *args, levels=None, norm=None, norm_kw=None, extend='neither',
        vmin=None, vmax=None, locator=None, locator_kw=None,
        symmetric=False, positive=False, negative=False, nozero=False, **kwargs
    ):
        """
        Automatically generate level locations based on the input data, the
        input locator, and the input normalizer.

        Parameters
        ----------
        *args
            The x, y, z, data.
        levels : int, optional
            The (approximate) number of levels to create.
        norm, norm_kw
            Passed to `~proplot.constructor.Norm`. Used to determine suitable
            level locations if `locator` is not passed.
        extend : str, optional
            The extend setting.
        vmin, vmax : float, optional
            The data limits.
        locator, locator_kw
            Passed to `~proplot.constructor.Locator`. Used to determine suitable
            level locations.
        symmetric, positive, negative, nozero : bool, optional
            Whether the automatic levels should be symmetric, should be all positive
            with a minimum at zero, should be all negative with a maximum at zero,
            or should exclude the value zero.

        Returns
        -------
        levels : ndarray
            The levels.
        locator : ndarray or `matplotlib.ticker.Locator`
            The locator used for colorbar tick locations.
        """
        # Input args
        # NOTE: Some of this is adapted from the hidden contour.ContourSet._autolev
        norm_kw = norm_kw or {}
        locator_kw = locator_kw or {}
        levels = _not_none(levels, rc['image.levels'])

        if np.iterable(levels):
            # Included so we can use this to apply positive, negative, nozero
            levels = ticks = levels
        else:
            # Get default locator from input norm
            norm = constructor.Norm(norm or 'linear', **norm_kw)
            if positive and negative:
                raise ValueError('Incompatible options positive=True and negative=True')
            if locator is not None:
                locator = ticks = constructor.Locator(locator, **locator_kw)
            elif isinstance(norm, mcolors.LogNorm):
                locator = ticks = mticker.LogLocator(**locator_kw)
            elif isinstance(norm, mcolors.SymLogNorm):
                locator_kw.setdefault('base', _getattr_flexible(norm, 'base', 10))
                locator_kw.setdefault('linthresh', _getattr_flexible(norm, 'linthresh', 1))  # noqa: E501
                locator = ticks = mticker.SymmetricalLogLocator(**locator_kw)
            else:
                nbins = levels * 2 if positive or negative else levels
                locator_kw.setdefault('symmetric', symmetric or positive or negative)
                locator = mticker.MaxNLocator(nbins, min_n_ticks=1, **locator_kw)
                ticks = None  # do not necessarily prefer ticks at these levels'

            # Get default level locations
            automin = vmin is None
            automax = vmax is None
            vmin, vmax, kwargs = self._auto_vmin_vmax(
                *args, vmin=vmin, vmax=vmax, **kwargs
            )
            levels_orig = levels
            try:
                levels = locator.tick_values(vmin, vmax)
            except RuntimeError:  # too-many-ticks error
                levels = np.linspace(vmin, vmax, levels)  # TODO: _autolev used N+1

            # Trim excess levels the locator may have supplied
            # NOTE: This part is mostly copied from matplotlib _autolev
            if not locator_kw.get('symmetric', None):
                i0, i1 = 0, len(levels)  # defaults
                under, = np.where(levels < vmin)
                if len(under):
                    i0 = under[-1]
                    if not automin or extend in ('min', 'both'):
                        i0 += 1  # permit out-of-bounds data
                over, = np.where(levels > vmax)
                if len(over):
                    i1 = over[0] + 1 if len(over) else len(levels)
                    if not automax or extend in ('max', 'both'):
                        i1 -= 1  # permit out-of-bounds data
                if i1 - i0 < 3:
                    i0, i1 = 0, len(levels)  # revert
                levels = levels[i0:i1]

            # Compare the no. of levels we *got* (levels) to what we *wanted* (N)
            # If we wanted more than 2 times the result, then add nn - 1 extra
            # levels in-between the returned levels *in normalized space*.
            # Example: A LogNorm gives too few levels, so we select extra levels
            # here, but use the locator for determining tick locations.
            nn = levels_orig // len(levels)
            if nn >= 2:
                olevels = norm(levels)
                nlevels = []
                for i in range(len(levels) - 1):
                    l1, l2 = olevels[i], olevels[i + 1]
                    nlevels.extend(np.linspace(l1, l2, nn + 1)[:-1])
                nlevels.append(olevels[-1])
                levels = norm.inverse(nlevels)

        # Filter the remaining contours
        if nozero and 0 in levels:
            levels = levels[levels != 0]
        if positive:
            levels = levels[levels >= 0]
        if negative:
            levels = levels[levels <= 0]

        # Use auto-generated levels for ticks if still None
        ticks = _not_none(ticks, levels)
        return levels, ticks, kwargs

    def _auto_discrete_norm(
        self, *args, levels=None, values=None, cmap=None, norm=None, norm_kw=None,
        extend='neither', vmin=None, vmax=None, minlength=2, **kwargs,
    ):
        """
        Build a `~proplot.colors.DiscreteNorm` or `~proplot.colors.BoundaryNorm`
        from the input arguments. This automatically calculates "nice" level
        boundaries if they were not provided.

        Parameters
        ----------
        *args
            The data.
        cmap : `matplotlib.colors.Colormap`, optional
            The colormap. Passed to `DiscreteNorm`.
        norm, norm_kw
            Passed to `~proplot.constructor.Norm` and then to `DiscreteNorm`.
        extend : str, optional
            The extend setting.
        levels, N, values : ndarray, optional
            The explicit boundaries.
        vmin, vmax : float, optional
            The minimum and maximum values for the normalizer.
        minlength : int, optional
            The minimum length for level lists.
        **kwargs
            Passed to `_auto_levels_locator`.

        Returns
        -------
        norm : `matplotlib.colors.Normalize`
            The normalizer.
        ticks : `numpy.ndarray` or `matplotlib.locator.Locator`
            The axis locator or the tick location candidates.
        """
        # Parse flexible keyword args
        levels = _not_none(
            levels=levels,
            norm_kw_levels=norm_kw.pop('levels', None),
            default=rc['image.levels']
        )
        vmin = _not_none(vmin=vmin, norm_kw_vmin=norm_kw.pop('vmin', None))
        vmax = _not_none(vmax=vmax, norm_kw_vmax=norm_kw.pop('vmax', None))
        norm_kw = norm_kw or {}
        if norm == 'segments':  # TODO: remove
            norm = 'segmented'

        # NOTE: Matplotlib colorbar algorithm *cannot* handle descending levels
        # so this function reverses them and adds special attribute to the
        # normalizer. Then colorbar_extras reads this attribute and flips the
        # axis and the colormap direction.
        # Check input levels and values
        for key, val in (('levels', levels), ('values', values)):
            if not np.iterable(val):
                continue
            if len(val) < minlength or len(val) >= 2 and any(
                np.sign(np.diff(val)) != np.sign(val[1] - val[0])
            ):
                raise ValueError(
                    f'{key!r} must be monotonically increasing or decreasing '
                    f'and at least length {minlength}, got {val}.'
                )

        # Get level edges from level centers
        locator = None
        if isinstance(values, Integral):
            levels = values + 1
        elif np.iterable(values) and len(values) == 1:
            levels = [values[0] - 1, values[0] + 1]  # weird but why not
        elif np.iterable(values) and len(values) > 1:
            # Try to generate levels such that a LinearSegmentedNorm will
            # place values ticks at the center of each colorbar level.
            # utils.edges works only for evenly spaced values arrays.
            # We solve for: (x1 + x2)/2 = y --> x2 = 2*y - x1
            # with arbitrary starting point x1. We also start the algorithm
            # on the end with *smaller* differences.
            if norm is None or norm == 'segmented':
                reverse = abs(values[-1] - values[-2]) < abs(values[1] - values[0])
                if reverse:
                    values = values[::-1]
                levels = [values[0] - (values[1] - values[0]) / 2]
                for val in values:
                    levels.append(2 * val - levels[-1])
                if reverse:
                    levels = levels[::-1]
                if any(np.sign(np.diff(levels)) != np.sign(levels[1] - levels[0])):
                    levels = edges(values)  # backup plan, weird tick locations
            # Generate levels by finding in-between points in the
            # normalized numeric space, e.g. LogNorm space.
            else:
                inorm = constructor.Norm(norm, **norm_kw)
                levels = inorm.inverse(edges(inorm(values)))
        elif values is not None:
            raise ValueError(
                f'Unexpected input values={values!r}. '
                'Must be integer or list of numbers.'
            )

        # Get default normalizer
        # Only use LinearSegmentedNorm if necessary, because it is slow
        descending = False
        if np.iterable(levels):
            if len(levels) == 1:
                if norm is None:
                    norm = mcolors.Normalize(vmin=levels[0] - 1, vmax=levels[0] + 1)
            else:
                levels, descending = pcolors._check_levels(levels)
                if norm is None and len(levels) > 2:
                    steps = np.abs(np.diff(levels))
                    eps = np.mean(steps) / 1e3
                    if np.any(np.abs(np.diff(steps)) >= eps):
                        norm = 'segmented'
        if norm == 'segmented':
            if not np.iterable(levels):
                norm = 'linear'  # same result with improved speed
            else:
                norm_kw['levels'] = levels
        norm = constructor.Norm(norm or 'linear', **norm_kw)

        # Figure out correct levels
        # NOTE: Subsequently, we *only* use the locator to determine ticks if
        # *levels* and *values* were not passed.
        if isinstance(norm, mcolors.BoundaryNorm):
            # Get levels from bounds
            # NOTE: No warning because we get here internally?
            levels = norm.boundaries
        elif np.iterable(values):
            # Prefer ticks in center, but subsample if necessary
            locator = _to_numpy_array(values)
        elif np.iterable(levels):
            # Prefer ticks on level edges, but subsample if necessary
            locator = _to_numpy_array(levels)
        else:
            # Determine levels automatically
            levels, locator, kwargs = self._auto_levels_locator(
                *args, levels=levels, norm=norm, vmin=vmin, vmax=vmax, extend=extend, **kwargs  # noqa: E501
            )

        # Generate DiscreteNorm and update "child" norm with vmin and vmax from
        # levels. This lets the colorbar set tick locations properly!
        if not isinstance(norm, mcolors.BoundaryNorm) and len(levels) > 1:
            norm = pcolors.DiscreteNorm(
                levels, cmap=cmap, norm=norm, descending=descending, unique=extend,
            )
        if descending:
            cmap = cmap.reversed()
        return norm, cmap, levels, locator, kwargs

    @staticmethod
    def _parse_labels(
        labels=False, labels_kw=None,
        fmt=None, formatter=None, precision=2, **kwargs
    ):
        """
        Parse arguments for grid box and contour labels.
        """
        # Keyword args used with labels
        kw_labels = {
            'labels': labels,
            'labels_kw': labels_kw or {},
            'precision': precision,
            'formatter': formatter,
            'fmt': fmt,
        }
        return kw_labels, kwargs

    def _parse_edges(name, **kwargs):
        """
        Parse arguments related to pcolor or contour 'edges'.
        """
        add_contours = name in ('contourf', 'tricontourf') and (
            kwargs.get('linewidths') is not None or kwargs.get('linestyles') is not None
        )
        no_cmap = kwargs.get('colors', None) is not None and (
            name in ('contour', 'triccontour')
            or not add_contours and name in ('contourf', 'tricontourf')
        )
        kwargs['no_cmap'] = no_cmap
        if add_contours:
            kwargs['contour_kw'] = {
                'colors': _not_none(kwargs.pop('colors', None), 'k'),
                'linewidths': kwargs.pop('linewidths', None),
                'linestyles': kwargs.pop('linestyles', None),
            }
            kwargs['edgefix'] = False

        return kwargs

    @warnings._rename_kwargs('0.6', centers='values')
    @docstring.add_snippets
    def _parse_cmap(
        self, *args,
        cmap=None, cmap_kw=None, norm=None, norm_kw=None, extend='neither',
        N=None, levels=None, values=None, vmin=None, vmax=None, discrete=None,
        no_cmap=False, keep_levels=False, keep_values=False,
        default_cmap=None, default_discrete=True, **kwargs
    ):
        """
        Parse colormap-related arguments.
        """
        # Parse keyword args
        # NOTE: For now when drawing contour or contourf plots with no colormap,
        # cannot use 'values' to specify level centers or level center count.
        # NOTE: For now need to duplicate 'levels' parsing here and in
        # _auto_discrete_norm so that it works with contour plots with no cmap.
        cmap_kw = cmap_kw or {}
        norm_kw = norm_kw or {}
        norm_kw = norm_kw or {}
        discrete = _not_none(discrete, rc['image.discrete'], default_discrete)
        levels = _not_none(
            N=N,
            levels=levels,
            norm_kw_levels=norm_kw.pop('levels', None),
            default=rc['image.levels'] if discrete else None
        )

        # Get colormap, but do not use cmap when 'colors' are passed to contour()
        # or to contourf() -- the latter only when 'linewidths' and 'linestyles'
        # are also *not* passed. This wrapper lets us add "edges" to contourf
        # plots by calling contour() after contourf() if 'linewidths' or
        # 'linestyles' are explicitly passed, but do not want to disable the
        # native matplotlib feature for manually coloring filled contours.
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contourf
        if no_cmap:
            if cmap is not None:
                warnings._warn_proplot(f'Ignoring colormap cmap={cmap!r}.')
            cmap = None
        else:
            if cmap is None:
                cmap = _not_none(default_cmap, rc['image.cmap'])
            cmap = constructor.Colormap(cmap, **cmap_kw)
        if getattr(cmap, '_cyclic', None) and extend != 'neither':
            warnings._warn_proplot(
                'Cyclic colormap requires extend="neither". '
                f'Overriding user input extend={extend!r}.'
            )
            extend = 'neither'

        # Build colormap normalizer
        # NOTE: This ensures contour() and tricontour() use the same default
        # levels whether or not colormap is active.
        kwargs.update({
            'norm': norm, 'norm_kw': norm_kw, 'extend': extend,
            'vmin': vmin, 'vmax': vmax,
        })
        ticks = None
        if levels is None:
            if norm is not None:
                norm = constructor.Norm(norm, **norm_kw)
        elif no_cmap:
            if not np.iterable(levels):
                levels, ticks, kwargs = self._auto_levels_locator(
                    *args, levels=levels, **kwargs
                )
        else:
            norm, cmap, levels, ticks, kwargs = self._auto_discrete_norm(
                *args, cmap=cmap, levels=levels, values=values, **kwargs  # noqa: E501
            )
        self._pop_unused_args(kwargs)

        # Keyword args passed to function
        kwargs['ticks'] = ticks  # handled by _call_method
        kwargs['extend'] = extend  # handled by _call_method
        if cmap is not None:
            kwargs['cmap'] = cmap
        if norm is not None:
            kwargs['norm'] = norm
        if levels is None:  # i.e. no DiscreteNorm was used
            kwargs['vmin'] = vmin
            kwargs['vmax'] = vmax
        if keep_levels:
            kwargs['levels'] = levels
        if keep_values:
            kwargs['values'] = values

        return kwargs

    def _plot_pairs(self, *args):
        """
        Support ``x1, y1, fmt1, x2, y2, fmt2, ...`` input.
        """
        # NOTE: Copied from _process_plot_var_args.__call__ to avoid relying
        # on public API. ProPlot already supports passing extra positional
        # arguments beyond x, y so can feed (x1, y1, fmt) through wrappers.
        # Instead represent (x2, y2, fmt, ...) as successive calls to plot().
        args = list(args)
        while args:
            iargs, args = args[:2], args[2:]
            if args and isinstance(args[0], str):
                iargs.append(args[0])
                args = args[1:]
            yield iargs

    def _apply_plot(self, *args, vert=True, **kwargs):
        """
        Plot the lines.
        """
        # Plot the lines
        _process_props(kwargs, 'line')
        objs = []
        for args in self._plot_pairs(*args):
            *a, kw = self._standardize_1d('plot', *args, **kwargs)  # TODO: add 'N' key
            *a, kw = self._error_distribution(*a, **kw)
            kw = self._parse_cycle(**kw)
            for x, y, *a, kw in self._iter_columns(*a, **kw):
                x, y = (x, y) if vert else (y, x)
                *eb, kw = self._error_bars(x, y, *a, **kw)
                *es, kw = self._error_shading(x, y, *a, **kw)
                obj = self._call_method('plot', x, y, *a, **kw)
                if eb or es:
                    objs.append((*eb, *es, *obj))
                else:
                    objs.extend(obj)

        # Add sticky edges
        axis = 'x' if vert else 'y'
        for obj in objs:
            if not isinstance(obj, mlines.Line2D):
                continue  # TODO: still looks good with error caps???
            data = getattr(obj, 'get_' + axis + 'data')()
            if not data.size:
                continue
            convert = getattr(self, 'convert_' + axis + 'units')
            edges = getattr(obj.sticky_edges, axis)
            edges.append(convert(min(data)))
            edges.append(convert(max(data)))

        return objs  # always return list to match matplotlib behavior

    @docstring.concatenate
    @docstring.add_snippets
    def plot(self, *args, **kwargs):
        """
        %(plot.plot)s
        """
        kwargs = self._parse_vert(default_vert=True, **kwargs)
        return self._apply_plot(*args, **kwargs)

    @docstring.add_snippets
    def plotx(self, *args, **kwargs):
        """
        %(plot.plotx)s
        """
        kwargs = self._parse_vert(default_vert=False, **kwargs)
        return self._apply_plot(*args, **kwargs)

    def _apply_step(self, *args, vert=True, where='pre', **kwargs):
        """
        Plot the steps.
        """
        # Plot the steps
        # NOTE: Internally matplotlib plot() calls step() so we could rely on
        # that but better to disable 'error bar' feature where it makes no sense.
        _process_props(kwargs, 'line')
        if where not in ('pre', 'post', 'mid'):
            raise ValueError(f"Invalid where={where!r}. Options are 'pre', 'post', 'mid'.")  # noqa: E501
        kwargs.setdefault('drawstyle', 'steps-' + where)
        objs = []
        for args in self._plot_pairs(*args):
            *a, kw = self._standardize_1d('step', *args, **kwargs)  # TODO: add 'N' key
            kw = self._parse_cycle(**kw)
            for x, y, *a, kw in self._iter_columns(*a, **kw):
                x, y = (x, y) if vert else (y, x)
                obj = self._call_method('step', x, y, *a, **kw)
                objs.extend(obj)

        return objs

    @docstring.concatenate
    @docstring.add_snippets
    def step(self, *args, **kwargs):
        """
        %(plot.step)s
        """
        kwargs = self._parse_vert(default_vert=True, **kwargs)
        return self._apply_step(*args, **kwargs)

    @docstring.add_snippets
    def stepx(self, *args, **kwargs):
        """
        %(plot.stepx)s
        """
        kwargs = self._parse_vert(default_vert=False, **kwargs)
        return self._apply_step(*args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def stem(self, *args, linefmt=None, markerfmt=None, basefmt=None, **kwargs):
        """
        Plot stem lines.

        Parameters
        ----------
        %(plot.1d_args)s

        Other parameters
        ----------------
        **kwargs
            Passed to `~matplotlib.axes.Axes.stem`.
        """
        # Set default colors
        # NOTE: 'fmt' strings can only be 2 to 3 characters and include color
        # shorthands like 'r' or cycle colors like 'C0'. Cannot use full color names.
        # NOTE: Matplotlib defaults try to make a 'reddish' color the base and 'bluish'
        # color the stems. To make this more robust we temporarily replace the cycler.
        fmts = (linefmt, basefmt, markerfmt)
        if not any(isinstance(_, str) and re.match(r'\AC[0-9]', _) for _ in fmts):
            cycle = constructor.Cycle((rc['negcolor'], rc['poscolor']), name='_no_name')
            kwargs.setdefault('cycle', cycle)

        # Stem lines with bluish stem color and reddish base color
        # NOTE: Here we are careful to only apply cycle *temporarily*
        kwargs['basefmt'] = _not_none(basefmt, 'C1-')  # red base
        kwargs['linefmt'] = linefmt = _not_none(linefmt, 'C0-')  # blue stems
        kwargs['markerfmt'] = _not_none(markerfmt, linefmt[:-1] + 'o')  # blue marker
        sig = inspect.signature(maxes.Axes.stem)
        if 'use_line_collection' in sig.parameters:
            kwargs.setdefault('use_line_collection', True)

        # Call function then restore property cycle
        cycle = rc['axes.prop_cycle']  # always return to default
        *args, kwargs = self._standardize_1d(*args, **kwargs)
        kwargs = self._parse_cycle(**kwargs)  # allow re-application
        obj = self._call_method('stem', *args, **kwargs)
        self.set_prop_cycle(cycle)
        return obj

    @docstring.add_snippets
    def parametric(
        self, x, y, c=None, *, values=None, interp=0, cmap=None, norm=None,
        scalex=True, scaley=True, **kwargs
    ):
        """
        Plot a parametric line.

        Parameters
        ----------
        *args : (y,), (x, y), or (x, y, c)
            The coordinates. If `x` is not provided, it is inferred from `y`.
            The parametric coordinate can be indicated as a third positional
            argument or with the `c`, `values`, or `labels` keywords. The
            parametric coordinate can be numeric or an array of string labels.
        c, values, labels : array-like, optional
            The parametric coordinates passed as a keyword argument. They
            can also be passed as a third positional argument.

        Other parameters
        ----------------
        %(plot.cmap_norm)s
        %(plot.vmin_vmax)s
        %(plot.colorbar_legend)s
        scalex, scaley : bool, optional
            Whether the view limits are adapted to the data limits. The values are
            passed on to `~matplotlib.axes.Axes.autoscale_view`.
        interp : int, optional
            If greater than ``0``, we interpolate to additional points
            between the `values` coordinates. The number corresponds to the
            number of additional color levels between the line joints
            and the halfway points between line joints.
        **kwargs
            Valid `~matplotlib.collections.LineCollection` properties.

        Returns
        -------
        `~matplotlib.collections.LineCollection`
            The parametric line. See `this matplotlib example \
<https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line>`__.

        See also
        --------
        PlotAxes.plot
        PlotAxes.plotx
        matplotlib.collections.LineCollection
        """
        # Standardize arguments
        _process_props(kwargs, 'lines')
        x, y, kwargs = self._standardize_1d(x, y, c, values=values, **kwargs)
        kwargs = self._parse_cmap(c, keep_values=True, **kwargs)
        c = kwargs.pop('values', None)

        # Parse input
        # NOTE: Critical to put this here instead of parametric() so that the
        # interpolated 'values' are used to select colormap levels in _apply_cmap.
        c = _not_none(c=c, values=values)
        if c is None:
            raise ValueError('Values must be provided.')
        c = _to_numpy_array(c)
        ndim = tuple(_.ndim for _ in (x, y, c))
        size = tuple(_.size for _ in (x, y, c))
        if any(_ != 1 for _ in ndim):
            raise ValueError(f'Input coordinates must be 1D. Instead got dimensions {ndim}.')  # noqa: E501
        if any(_ != size[0] for _ in size):
            raise ValueError(f'Input coordinates must have identical size. Instead got sizes {size}.')  # noqa: E501

        # Interpolate values to allow for smooth gradations between values
        # (interp=False) or color switchover halfway between points
        # (interp=True). Then optionally interpolate the colormap values.
        # NOTE: The 'extras' wrapper handles input before ingestion by other wrapper
        # functions. *This* method is analogous to a native matplotlib method.
        if interp > 0:
            x_orig, y_orig, v_orig = x, y, c
            x, y, c = [], [], []
            for j in range(x_orig.shape[0] - 1):
                idx = slice(None)
                if j + 1 < x_orig.shape[0] - 1:
                    idx = slice(None, -1)
                x.extend(np.linspace(x_orig[j], x_orig[j + 1], interp + 2)[idx].flat)
                y.extend(np.linspace(y_orig[j], y_orig[j + 1], interp + 2)[idx].flat)
                c.extend(np.linspace(v_orig[j], v_orig[j + 1], interp + 2)[idx].flat)  # noqa: E501
            x, y, c = np.array(x), np.array(y), np.array(c)

        # Get coordinates and values for points to the 'left' and 'right' of joints
        c = _not_none(c=c, values=values)
        coords = []
        levels = edges(c)
        for i in range(y.shape[0]):
            icoords = np.empty((3, 2))
            for j, arr in enumerate((x, y)):
                icoords[0, j] = arr[0] if i == 0 else 0.5 * (arr[i - 1] + arr[i])
                icoords[1, j] = arr[i]
                icoords[2, j] = arr[-1] if i + 1 == y.shape[0] else 0.5 * (arr[i + 1] + arr[i])  # noqa: E501
            coords.append(icoords)
        coords = np.array(coords)

        # Create LineCollection and update with values
        # NOTE: Default capstyle is butt but this may look weird with vector graphics
        obj = mcollections.LineCollection(
            coords, cmap=cmap, norm=norm,
            linestyles='-', capstyle='butt', joinstyle='miter',
        )
        obj.set_array(c)  # the ScalarMappable method
        obj.update({
            key: value for key, value in kwargs.items()
            if key not in ('color',)
        })

        # Add collection with some custom attributes
        # NOTE: Modern API uses self._request_autoscale_view but this is
        # backwards compatible to earliest matplotlib versions.
        self.add_collection(obj)
        self.autoscale_view(scalex=scalex, scaley=scaley)
        obj.values = c
        obj.levels = levels  # needed for other functions

        return obj

    def _apply_lines(
        self, *args, vert=True,
        stack=None, stacked=None, negpos=False, **kwargs
    ):
        """
        Apply hlines or vlines command. Support default "minima" at zero.
        """
        # Parse input arguments
        name = 'vlines' if vert else 'hlines'
        stack = _not_none(stack=stack, stacked=stacked)
        _process_props(kwargs, 'lines')
        linewidths = kwargs.pop('linewidths', None)  # mpl does not handle this
        linestyles = kwargs.pop('linestyles', None)
        *args, kwargs = self._standardize_1d(*args, vert=vert, **kwargs)
        args = list(args)
        if len(args) > 3:
            raise ValueError(f'Expected 1-3 positional args, got {len(args)}.')
        if len(args) == 3 and stack:
            warnings._warn_proplot(
                f'{name}() cannot have three positional arguments with stack=True. '
                'Ignoring second argument.'
            )
        if len(args) == 2:  # empty possible
            args.insert(1, np.array([0.0]))  # default base

        # Support "negative" and "positive" lines
        y0 = 0
        objs = []
        kwargs['stack'] = stack
        for x, y1, y2, *args, kwargs in self._iter_columns(*args, **kwargs):
            # Apply stacking
            if stack:
                y1 += y0
                y2 += y0
                y0 += y2 - y1  # irrelevant that we added y0 to both
            # Plot basic lines or negative-positive lines
            if negpos:
                obj = self._call_negpos(name, x, y1, y2, *args, **kwargs)
            else:
                obj = self._call_method(name, x, y1, y2, *args, **kwargs)
            # Apply formatting unavailable in matplotlib
            for o in (obj if isinstance(obj, tuple) else (obj,)):
                linewidths = kwargs.get('linewidths', None)
                if linewidths is not None:
                    o.set_linewidth(linewidths)  # LineCollection setters
                linestyles = kwargs.get('linestyles', None)
                if linestyles is not None:
                    o.set_linestyle(linestyles)
            objs.append(obj)

        return objs[0] if len(objs) == 1 else objs

    @docstring.add_snippets
    def vlines(self, *args, **kwargs):
        """
        %(plot.vlines)s
        """
        kwargs = self._parse_vert(default_vert=True, **kwargs)
        return self._apply_lines(*args, **kwargs)

    @docstring.add_snippets
    def hlines(self, *args, **kwargs):
        """
        %(plot.hlines)s
        """
        kwargs = self._parse_vert(default_vert=False, **kwargs)
        return self._apply_lines(*args, **kwargs)

    def _apply_scatter(self, *args, smin=None, smax=None, vert=True, **kwargs):
        """
        Apply scatter or scatterx markers. Permit up to 4 positional arguments
        including `s` and `c`.
        """
        # Manage input arguments
        # TODO: Carefully manage props
        args = list(args)
        _process_props(kwargs, 'line')
        kwargs = self._parse_cycle(**kwargs)
        if len(args) > 4:
            raise ValueError(f'Expected 1-4 positional arguments, got {len(args)}.')
        c = _not_none(
            color_positional=args.pop(-1) if len(args) == 4 else None,
            color=kwargs.pop('color', None)
        )
        s = _not_none(
            markersize_positional=args.pop(-1) if len(args) == 3 else None,
            markersize=kwargs.pop('markersize', None)
        )
        *a, s, c, kw = self._standardize_1d(*args, c, s, vert=vert, **kwargs)
        if all(_.squeeze().ndim > 1 for _ in (*a, s, c) if isinstance(_, np.ndarray)):
            *a, s, c = (np.ravel(_) if isinstance(_, np.ndarray) else _ for _ in (*a, s, c))  # noqa: E501

        # Scale s array
        if np.iterable(s) and (smin is not None or smax is not None):
            s = _to_numpy_array(s)
            smin_true, smax_true = np.min(s), np.max(s)
            if smin is None:
                smin = smin_true
            if smax is None:
                smax = smax_true
            factor = (s - smin_true) / (smax_true - smin_true)
            s = smin + (smax - smin) * factor

        # Scale c array
        if (
            isinstance(c, np.ndaray) and np.issubdtype(c.dtype, np.number)
            and not (c.ndim == 2 and c.shape[1] in (3, 4))
        ):
            kwargs.setdefault('cmap_kw', {}).setdefault('luminance', 90)
            norm, cmap, levels, ticks, kwargs = self._auto_discrete_norm(c.ravel(), **kwargs)  # noqa: E501
            kwargs.update({'cmap': cmap, 'norm': norm, 'ticks': ticks})
        self._pop_unused_args(kwargs)

        # Iterate using keys
        objs = []
        kwargs = self._from_cycle('scatter', **kwargs)
        *a, kw = self._error_distribution(*a, **kw)
        for x, y, *a, kw in self._iter_columns(*args, **kwargs):
            x, y = (x, y) if vert else (y, x)
            *eb, kw = self._error_bars(x, y, *a, **kw)
            *es, kw = self._error_shading(x, y, *a, **kw)
            obj = self._call_method('scatter', x, y, *args, c=c, s=s, **kwargs)
            if eb or es:
                objs.append((*eb, *es, obj))
            else:
                objs.append(obj)
            objs.append(obj if not eb and not es else (*eb, *es, obj))

        return objs[0] if len(objs) == 1 else objs

    @docstring.concatenate
    @docstring.add_snippets
    def scatter(self, *args, **kwargs):
        """
        %(plot.scatter)s
        """
        kwargs = self._parse_vert(default_vert=True, **kwargs)
        return self._apply_scatter(*args, **kwargs)

    @docstring.add_snippets
    def scatterx(self, *args, **kwargs):
        """
        %(plot.scatterx)s
        """
        kwargs = self._parse_vert(default_vert=False, **kwargs)
        return self._apply_scatter(*args, **kwargs)

    def _apply_fill_between(
        self, *args, where=None, negpos=None, negcolor=None, poscolor=None,
        lw=None, linewidth=None, color=None, facecolor=None,
        stack=None, stacked=None, vert=True, **kwargs
    ):
        """
        Apply area shading.
        """
        # Parse input arguments
        name = 'fill_between' if vert else 'fill_betweenx'
        stack = _not_none(stack=stack, stacked=stacked)
        color = _not_none(color=color, facecolor=facecolor)
        linewidth = _not_none(lw=lw, linewidth=linewidth, default=0)
        *args, kwargs = self._standardize_1d(*args, vert=vert, **kwargs)
        kwargs = self._parse_cycle(**kwargs)
        if len(args) > 4:
            raise ValueError(f'Expected 1-4 positional args, got {len(args)}.')
        if len(args) == 4:
            where = _not_none(where_positional=args.pop(3), where=where)
        if len(args) == 3 and stack:
            warnings._warn_proplot(
                f'{name}() cannot have three positional arguments with stack=True. '
                'Ignoring second argument.'
            )
        if len(args) == 2:  # empty possible
            args.insert(1, np.array([0.0]))  # default base

        # Draw patches with default edge width zero
        y0 = 0
        objs, xsides, ysides = [], [], []
        for x, y1, y2, kw in self._iter_columns(*args, **kwargs):
            # Apply stacking
            if stack:
                y1 += y0
                y2 += y0
                y0 += y2 - y1  # irrelevant that we added y0 to both
            # Call basic patches or negative-positive patches
            kwargs.update({
                'linewidth': linewidth,
                'color': color,
                'where': where,
                'stack': stack,
            })
            kwargs['linewidth'] = linewidth
            if negpos:
                obj = self._call_negpos(name, x, y1, y2, use_where=True, **kwargs)
            else:
                obj = self._call_method(name, x, y1, y2, **kwargs)
            # Document sticky sides
            xsides.extend((np.min(x), np.max(x)))
            for y in (y1, y2):
                if y.size == 1:
                    ysides.append(y.item())
            objs.append(obj)

        # Add sticky edges in x-direction, and sticky edges in y-direction
        # *only* if one of the y limits is scalar. This should satisfy most users.
        # NOTE: Could also retrieve data from PolyCollection but that's tricky.
        # NOTE: Standardize function guarantees ndarray input by now
        iter_ = (obj for _ in objs for obj in (_ if isinstance(_, tuple) else (_,)))
        for obj in iter_:
            for s, sides in zip('xy' if vert else 'yx', (xsides, ysides)):
                convert = getattr(self, 'convert_' + s + 'units')
                edges = getattr(obj.sticky_edges, s)
                edges.extend(convert(sides))

        return objs[0] if len(objs) == 1 else objs

    @docstring.add_snippets
    def area(self, *args, **kwargs):
        """
        %(plot.fill_between)s
        """
        return self.fill_between(*args, **kwargs)

    @docstring.add_snippets
    def areax(self, *args, **kwargs):
        """
        %(plot.fill_betweenx)s
        """
        return self.fill_betweenx(*args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def fill_between(self, *args, **kwargs):
        """
        %(plot.fill_between)s
        """
        kwargs = self._parse_vert(default_vert=True, **kwargs)
        return self._apply_fill_between(*args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def fill_betweenx(self, *args, **kwargs):
        """
        %(plot.fill_betweenx)s
        """
        # NOTE: The 'horizontal' orientation will be inferred by downstream
        # wrappers using the function name.
        kwargs = self._parse_vert(default_vert=False, **kwargs)
        return self._apply_fill_between(*args, **kwargs)

    @staticmethod
    def _convert_bar_width(x, width=1):
        """
        Convert bar plot widths from relative to coordinate spacing. Relative
        widths are much more convenient for users.
        """
        # WARNING: This will fail for non-numeric non-datetime64 singleton
        # datatypes but this is good enough for vast majority of cases.
        x_test = np.atleast_1d(_to_numpy_array(x))
        if len(x_test) >= 2:
            x_step = x_test[1:] - x_test[:-1]
            x_step = np.concatenate((x_step, x_step[-1:]))
        elif x_test.dtype == np.datetime64:
            x_step = np.timedelta64(1, 'D')
        else:
            x_step = np.array(0.5)
        if np.issubdtype(x_test.dtype, np.datetime64):
            # Avoid integer timedelta truncation
            x_step = x_step.astype('timedelta64[ns]')
        return width * x_step

    def _apply_bar(
        self, *args, stack=None, stacked=None,
        lw=None, linewidth=None, color=None, facecolor=None, edgecolor='black',
        negpos=False, negcolor=None, poscolor=None, absolute_width=False, **kwargs
    ):
        """
        Apply bar or barh command. Support default "minima" at zero.
        """
        # Parse args
        # TODO: Stacked feature is implemented in `_update_cycle`, but makes more
        # sense do document here. Figure out way to move it here?
        name = 'barh' if kwargs.get('orientation') == 'horizontal' else 'bar'
        stack = _not_none(stack=stack, stacked=stacked)
        color = _not_none(color=color, facecolor=facecolor)
        linewidth = _not_none(lw=lw, linewidth=linewidth, default=rc['patch.linewidth'])
        kwargs.update({'linewidth': linewidth, 'edgecolor': edgecolor})
        *args, kwargs = self._standardize_1d(*args, **kwargs)
        if len(args) > 4:
            raise ValueError(f'Expected 1-4 positional args, got {len(args)}.')
        if len(args) == 4 and stack:
            warnings._warn_proplot(
                f'{name}() cannot have four positional arguments with stack=True. '
                'Ignoring fourth argument.'  # i.e. ignore default 'bottom'
            )
        if len(args) == 2:
            args.append(np.array([0.8]))  # default width
        if len(args) == 3:
            args.append(np.array([0.0]))  # default base

        # Call func after converting bar width
        b0 = 0
        objs = []
        args, kwargs = self._indicate_error(*args, **kwargs)
        ncols = max(_.ndim for _ in args)
        for i, (x, h, w, b, kw) in enumerate(self._iter_columns(*args, stack=stack, **kwargs)):  # noqa: E501
            # Adjust x or y coordinates for grouped and stacked bars
            if not absolute_width:
                w = self._convert_bar_width(x, w)
            if not stack:
                o = 0.5 * (ncols - 1)
                x += w * (i - o) / ncols
            else:
                b += b0
                b0 += h
            # Draw simple bars
            if not negpos:
                if color is not None:
                    kw['color'] = color
                obj = self._call_method(name, x, h, w, b, **kw)
            # Draw negative and positive bars
            else:
                self._warn_negpos(name, stack=stack, color=color)
                if x.ndim > 1 or h.ndim > 1:
                    raise ValueError(f'{name}() arguments with negpos=True must be 1D.')
                hneg = self._mask_negpos(h < b, h)
                color = _not_none(negcolor, rc['negcolor'])
                negobj = self._call_method(name, x, hneg, w, b, facecolor=color, **kw)
                hpos = self._mask_negpos(h >= b, h)
                color = _not_none(poscolor, rc['poscolor'])
                posobj = self._call_method(name, x, hpos, w, b, facecolor=color, **kw)
                obj = (negobj, posobj)
            objs.append(obj)

        return obj

    @docstring.concatenate
    @docstring.add_snippets
    def bar(self, *args, **kwargs):
        """
        %(plot.bar)s
        """
        kwargs = self._parse_vert(default_orientation='vertical', **kwargs)
        return self._apply_bar('bar', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def barh(self, *args, **kwargs):
        """
        %(plot.barh)s
        """
        kwargs = self._parse_vert(default_orientation='horizontal', **kwargs)
        return self._apply_bar('barh', *args, **kwargs)

    @docstring.concatenate
    def pie(self, *args, **kwargs):
        """
        Parameters
        ----------
        %(plot.1d_args)s

        Other parameters
        ----------------
        %(plot.1d_kwargs)s

        See also
        --------
        matplotlib.axes.Axes.pie
        """
        *args, kwargs = self._standardize_1d(*args, **kwargs)
        kwargs = self._parse_cycle(**kwargs)
        self._call_method('pie', *args, **kwargs)

    def _apply_boxplot(
        self, *args,
        mean=None, means=None,
        fill=True, fillcolor=None, fillalpha=None,
        lw=None, linewidth=None,
        color=None, edgecolor=None,
        boxcolor=None, boxlw=None, boxlinewidth=None,
        capcolor=None, caplw=None, caplinewidth=None,
        whiskercolor=None, whiskerlw=None, whiskerlinewidth=None,
        fliercolor=None, flierlw=None, flierlinewidth=None,  # fliers have no line width
        meancolor=None, meanlw=None, meanlinewidth=None,
        mediancolor=None, medianlw=None, medianlinewidth=None,
        meanls=None, meanlinestyle=None, medianls=None, medianlinestyle=None,
        marker=None, markersize=None,
        **kwargs
    ):
        """
        Apply the box plot.
        """
        # Parse keyword args
        # NOTE: For some reason native violinplot() uses _get_lines for
        # property cycler. We dot he same here.
        fill = fill is True or fillcolor is not None or fillalpha is not None
        fillalpha = _not_none(fillalpha, default=0.7)
        if fill and fillcolor is None:
            cycler = next(self._get_lines.prop_cycler)
            fillcolor = cycler.get('color', None)
        color = _not_none(color=color, edgecolor=edgecolor, default='black')
        linewidth = _not_none(lw=lw, linewidth=linewidth, default=0.8)
        boxlinewidth = _not_none(boxlw=boxlw, boxlinewidth=boxlinewidth)
        caplinewidth = _not_none(caplw=caplw, caplinewidth=caplinewidth)
        whiskerlinewidth = _not_none(whiskerlw=whiskerlw, whiskerlinewidth=whiskerlinewidth)  # noqa: E501
        flierlinewidth = _not_none(flierlw=flierlw, flierlinewidth=flierlinewidth)
        meanlinewidth = _not_none(meanlw=meanlw, meanlinewidth=meanlinewidth)
        medianlinewidth = _not_none(medianlw=medianlw, medianlinewidth=medianlinewidth)
        meanlinestyle = _not_none(meanls=meanls, meanlinestyle=meanlinestyle)
        medianlinestyle = _not_none(medianls=medianls, medianlinestyle=medianlinestyle)
        means = _not_none(mean=mean, means=means, showmeans=kwargs.get('showmeans'))
        if means:
            kwargs['showmeans'] = kwargs['meanline'] = True

        # Call function
        obj = self._call_method('boxplot', *args, **kwargs)

        # Modify artist settings
        # TODO: Pass props keyword args instead? Maybe does not matter.
        for key, icolor, ilinewidth, ilinestyle in (
            ('boxes', boxcolor, boxlinewidth, None),
            ('caps', capcolor, caplinewidth, None),
            ('whiskers', whiskercolor, whiskerlinewidth, None),
            ('fliers', fliercolor, flierlinewidth, None),
            ('means', meancolor, meanlinewidth, meanlinestyle),
            ('medians', mediancolor, medianlinewidth, medianlinestyle),
        ):
            if key not in obj:  # possible if not rendered
                continue
            artists = obj[key]
            icolor = _not_none(icolor, color)
            ilinewidth = _not_none(ilinewidth, linewidth)
            if not isinstance(fillalpha, list):
                fillalpha = [fillalpha] * len(artists)
            if not isinstance(fillcolor, list):
                fillcolor = [fillcolor] * len(artists)
            for i, artist in enumerate(artists):
                # Lines used for boxplot components
                jcolor = icolor
                if isinstance(icolor, list):
                    jcolor = icolor[i // 2 if key in ('caps', 'whiskers') else i]
                if ilinestyle is not None:
                    artist.set_linestyle(ilinestyle)
                if ilinewidth is not None:
                    artist.set_linewidth(ilinewidth)
                    artist.set_markeredgewidth(ilinewidth)
                if jcolor is not None:
                    artist.set_color(jcolor)
                    artist.set_markeredgecolor(jcolor)
                # "Filled" boxplot by adding patch beneath line path
                if fill and key == 'boxes':
                    patch = mpatches.PathPatch(
                        artist.get_path(), linewidth=0,
                        facecolor=fillcolor[i], alpha=fillalpha[i]
                    )
                    self.add_artist(patch)
                # Outlier markers
                if key == 'fliers':
                    if marker is not None:
                        artist.set_marker(marker)
                    if markersize is not None:
                        artist.set_markersize(markersize)

        return obj

    @docstring.add_snippets
    def box(self, *args, **kwargs):
        """
        %(plot.boxplot)s
        """
        return self.boxplot(*args, **kwargs)

    @docstring.add_snippets
    def boxh(self, *args, **kwargs):
        """
        %(plot.boxploth)s
        """
        return self.boxploth(*args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def boxplot(self, *args, **kwargs):
        """
        %(plot.boxplot)s
        """
        kwargs = self._parse_vert(default_vert=True, **kwargs)
        return self._apply_boxplot(*args, **kwargs)

    @docstring.add_snippets
    def boxploth(self, *args, **kwargs):
        """
        %(plot.boxploth)s
        """
        kwargs = self._parse_vert(default_vert=False, **kwargs)
        return self._apply_boxplot(*args, **kwargs)

    def _apply_violinplot(
        self, *args, fillcolor=None, fillalpha=None,
        lw=None, linewidth=None, color=None, edgecolor=None, **kwargs
    ):
        """
        Apply the violinplot.
        """
        # Parse keyword args
        color = _not_none(color=color, edgecolor=edgecolor, default='black')
        linewidth = _not_none(lw=lw, linewidth=linewidth, default=0.8)
        fillalpha = _not_none(fillalpha, default=0.7)
        kwargs.setdefault('capsize', 0)  # caps are redundant for violin plots
        kwargs.setdefault('means', kwargs.pop('showmeans', None))  # for _indicate_error
        kwargs.setdefault('medians', kwargs.pop('showmedians', None))
        if kwargs.pop('showextrema', None):
            warnings._warn_proplot('Ignoring showextrema=True.')

        # Call function
        *a, kw = self._standardize_1d('violinplot', *args, **kwargs)
        *a, kw, errobjs = self._indicate_error('violinplot', *a, **kw)
        kw.update({'linewidth': linewidth})
        kw.update({'showmeans': False, 'showmedians': False, 'showextrema': False})
        kw.pop('labels', None)  # already applied in _standardize_1d
        obj = self._call_method('violinplot', *args, **kwargs)

        # Modify body settings
        artists = (obj or {}).get('bodies', ())
        if not isinstance(fillalpha, list):
            fillalpha = [fillalpha] * len(artists)
        if not isinstance(fillcolor, list):
            fillcolor = [fillcolor] * len(artists)
        if not isinstance(color, list):
            color = [color] * len(artists)
        for i, artist in enumerate(artists):
            artist.set_linewidths(linewidth)
            if fillalpha[i] is not None:
                artist.set_alpha(fillalpha[i])
            if fillcolor[i] is not None:
                artist.set_facecolor(fillcolor[i])
            if color[i] is not None:
                artist.set_edgecolor(color[i])

        return obj

    @docstring.add_snippets
    def violin(self, *args, **kwargs):
        """
        %(plot.violinplot)s
        """
        # WARNING: This disables use of 'violin' by users but
        # probably very few people use this anyway.
        return self.violinplot(*args, **kwargs)

    @docstring.add_snippets
    def violinh(self, *args, **kwargs):
        """
        %(plot.violinploth)s
        """
        return self.violinploth(*args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def violinplot(self, *args, **kwargs):
        """
        %(plot.violinplot)s
        """
        kwargs = self._parse_vert(default_vert=True, **kwargs)
        return self._apply_violinplot(*args, **kwargs)

    @docstring.add_snippets
    def violinploth(self, *args, **kwargs):
        """
        %(plot.violinploth)s
        """
        kwargs = self._parse_vert(default_vert=False, **kwargs)
        return self._apply_violinplot(*args, **kwargs)

    def _apply_hist(self, *args, **kwargs):
        """
        Apply the histogram.
        """
        *args, kwargs = self._standardize_1d(*args, **kwargs)
        kwargs = self._parse_cycle(**kwargs)
        for *a, kw in self._iter_columns(*args, **kwargs):
            self._call_method('hist', *a, **kw)

    @docstring.concatenate
    @docstring.add_snippets
    def hist(self, *args, **kwargs):
        """
        %(plot.hist)s
        """
        kwargs = self._parse_vert(default_orientation='vertical', **kwargs)
        self._apply_hist(*args, **kwargs)

    @docstring.add_snippets
    def histh(self, *args, **kwargs):
        """
        %(plot.histh)s
        """
        kwargs = self._parse_vert(default_orientation='horizontal', **kwargs)
        self._apply_hist(*args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def hexbin(self, *args, **kwargs):
        """
        Parameters
        ----------
        %(plot.1d_args)s

        Other parameters
        ----------------
        %(plot.2d_kwargs)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.hexbin`.

        See also
        --------
        PlotAxes.hist2d
        matplotlib.axes.Axes.hexbin
        """
        _process_props(kwargs, 'lines')  # takes LineCollection props
        if 'colors' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('colors')
        *args, kwargs = self._standardize_1d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, counts=True, default_discrete=False, **kwargs)
        return self._call_method('hexbin', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def hist2d(self, *args, **kwargs):
        """
        Parameters
        ----------
        %(plot.1d_args)s

        Other parameters
        ----------------
        %(plot.2d_kwargs)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.hexbin`.

        See also
        --------
        PlotAxes.hist2d
        matplotlib.axes.Axes.hexbin
        """
        _process_props(kwargs, 'lines')  # takes LineCollection props
        if 'colors' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('colors')
        *args, kwargs = self._standardize_1d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, counts=True, default_discrete=False, **kwargs)
        return self._call_method('hist2d', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def contour(self, *args, **kwargs):
        """
        Plot contour lines.

        %(plot.contour)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.contour`.

        See also
        --------
        PlotAxes.contourf
        matplotlib.axes.Axes.contour
        """
        _process_props(kwargs, 'lines')
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, minlength=1, keep_levels=True, **kwargs)
        return self._call_method('pcolor', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def contourf(self, *args, **kwargs):
        """
        Plot filled contour lines.

        %(plot.contour)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.contourf`.

        See also
        --------
        PlotAxes.contour
        matplotlib.axes.Axes.contourf
        """
        _process_props(kwargs, 'lines')
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, keep_levels=True, **kwargs)
        return self._call_method('pcolor', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def pcolor(self, *args, **kwargs):
        """
        Plot irregular grid boxes.

        %(plot.pcolor)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.pcolor`.

        See also
        --------
        PlotAxes.pcolormesh
        PlotAxes.pcolorfast
        matplotlib.axes.Axes.pcolor
        """
        _process_props(kwargs, 'line')
        if 'color' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('color')
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, centers=True, **kwargs)
        return self._call_method('pcolor', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def pcolormesh(self, *args, **kwargs):
        """
        Plot regular grid boxes.

        %(plot.pcolor)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.pcolormesh`.

        See also
        --------
        PlotAxes.pcolor
        PlotAxes.pcolorfast
        PlotAxes.heatmap
        matplotlib.axes.Axes.pcolormesh
        """
        _process_props(kwargs, 'line')
        if 'color' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('color')
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, centers=True, **kwargs)
        return self._call_method('pcolormesh', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def pcolorfast(self, *args, **kwargs):
        """
        Plot grid boxes quickly.

        %(plot.pcolor)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.pcolorfast`.

        See also
        --------
        PlotAxes.pcolor
        PlotAxes.pcolormesh
        PlotAxes.heatmap
        matplotlib.axes.Axes.pcolorfast
        """
        _process_props(kwargs, 'line')
        if 'color' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('color')
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, centers=True, **kwargs)
        return self._call_method('pcolorfast', *args, **kwargs)

    @docstring.add_snippets
    def heatmap(self, *args, aspect=None, **kwargs):
        """
        Plot grid boxes with formatting suitable for heatmaps. Ensures square
        grid boxes, adds major ticks to the center of each grid box, and
        disables minor ticks and gridlines.

        %(plot.pcolor)s
        aspect : {'equal', 'auto'} or float, optional
            Controls the aspect ratio of the axes. The aspect is of particular
            relevance for heatmaps since it may distort the heatmap, i.e. a grid box
            will not be square. This parameter is a shortcut for explicitly calling
            `~matplotlib.axes.set_aspect`.

            The default is :rc:`image.heatmap`. The options are:

            * ``'equal'``:
              Ensures an aspect ratio of 1. Grid boxes will be square.
            * ``'auto'``:
              The axes is kept fixed and the aspect is adjusted so that the data fit
              in the axes. In general, this will result in non-square boxes.

        **kwargs
            Passed to `~matplotlib.axes.Axes.pcolormesh`.

        See also
        --------
        PlotAxes.pcolor
        PlotAxes.pcolormesh
        PlotAxes.pcolorfast
        matplotlib.axes.Axes.pcolormesh
        """
        obj = self.pcolormesh(*args, **kwargs)
        aspect = _not_none(aspect, rc['image.aspect'])
        from .cartesian import CartesianAxes
        if not isinstance(self, CartesianAxes):
            warnings._warn_proplot(
                'Cannot adjust aspect ratio or ticks for non-Cartesian heatmap plot. '
                'Consider using pcolormesh() or pcolor() instead.'
            )
        else:
            xlocator = ylocator = None
            if hasattr(obj, '_coordinates'):
                coords = obj._coordinates
                coords = 0.5 * (coords[1:, ...] + coords[:-1, ...])
                coords = 0.5 * (coords[:, 1:, :] + coords[:, :-1, :])
                xlocator, ylocator = coords[0, :, 0], coords[:, 0, 1]
            self.format(
                aspect=aspect, xlocator=xlocator, ylocator=ylocator,
                xgrid=False, ygrid=False, xtickminor=False, ytickminor=False,
            )
        return obj

    @docstring.concatenate
    @docstring.add_snippets
    def barbs(self, *args, **kwargs):
        """
        Plot wind barbs.

        %(plot.flow)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.barbs`.

        See also
        --------
        PlotAxes.quiver
        PlotAxes.streamplot
        matplotlib.axes.Axes.barbs
        """
        _process_props(kwargs, 'line')  # applied to barbs
        if 'color' in kwargs:  # more intuitive for 'color' to apply to barbs
            kwargs['barbcolor'] = kwargs.pop('color')
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, default_discrete=False, **kwargs)
        return self._call_method('barbs', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def quiver(self, *args, **kwargs):
        """
        Plot quiver arrows.

        %(plot.flow)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.quiver`.

        See also
        --------
        PlotAxes.barbs
        PlotAxes.streamplot
        matplotlib.axes.Axes.quiver
        """
        _process_props(kwargs, 'line')  # applied to outline
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, default_discrete=False, **kwargs)
        return self._call_method('quiver', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def streamplot(self, *args, **kwargs):
        """
        Plot streamlines.

        %(plot.flow)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.streamplot`.

        See also
        --------
        PlotAxes.barbs
        PlotAxes.quiver
        matplotlib.axes.Axes.streamplot
        """
        _process_props(kwargs, 'line')  # applied to lines
        *args, kwargs = self._standardize_2d(*args, **kwargs)
        kwargs = self._parse_cmap(*args, default_discrete=False, **kwargs)
        return self._call_method('streamplot', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def tricontour(self, *args, **kwargs):
        """
        Plot contour lines from irregular points.

        Other parameters
        ----------------
        %(plot.levels_all)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.tricontour`.

        See also
        --------
        matplotlib.axes.Axes.tricontour
        """
        _process_props(kwargs, 'lines')
        kwargs = self._parse_cmap(minlength=1, keep_levels=True, **kwargs)
        self._call_method('tricontour', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def tricontourf(self, *args, **kwargs):
        """
        Plot filled contour lines from irregular points.

        Other parameters
        ----------------
        %(plot.levels_all)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.tricontourf`.

        See also
        --------
        matplotlib.axes.Axes.tricontourf
        """
        _process_props(kwargs, 'lines')
        kwargs = self._parse_cmap(keep_levels=True, **kwargs)
        self._call_method('tricontourf', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def tripcolor(self, *args, **kwargs):
        """
        Plot grid boxes from irregular points.

        Other parameters
        ----------------
        %(plot.levels_all)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.tripcolor`.

        See also
        --------
        matplotlib.axes.Axes.tripcolor
        """
        _process_props(kwargs, 'line')
        if 'color' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('color')
        kwargs = self._parse_cmap(**kwargs)
        self._call_method('tripcolor', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def imshow(self, *args, **kwargs):
        """
        Plot an image.

        Other parameters
        ----------------
        %(plot.levels_all)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.imshow`.

        See also
        --------
        matplotlib.axes.Axes.imshow
        """
        kwargs = self._parse_cmap(default_discrete=False, **kwargs)
        self._call_method('imshow', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def matshow(self, *args, **kwargs):
        """
        Plot a matrix.

        Other parameters
        ----------------
        %(plot.levels_all)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.matshow`.

        See also
        --------
        matplotlib.axes.Axes.matshow
        """
        kwargs = self._parse_cmap(**kwargs)
        self._call_method('imshow', *args, **kwargs)

    @docstring.concatenate
    @docstring.add_snippets
    def spy(self, *args, **kwargs):
        """
        Plot a sparcity pattern.

        Other parameters
        ----------------
        %(plot.levels_all)s
        **kwargs
            Passed to `~matplotlib.axes.Axes.spy`.

        See also
        --------
        matplotlib.axes.Axes.spy
        """
        _process_props(kwargs, 'line')  # takes valid Line2D properties
        default_cmap = pcolors.ListedColormap(['w', 'k'], '_no_name')
        kwargs = self._parse_cmap(default_cmap=default_cmap, **kwargs)
        self._call_method('spy', *args, **kwargs)

    # Rename the shorthands
    boxes = warnings._rename_objs('0.8', boxes=box)
    violins = warnings._rename_objs('0.8', violins=violin)
