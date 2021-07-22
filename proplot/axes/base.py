#!/usr/bin/env python3
"""
The first-level axes subclass used for all ProPlot figures.
Implements basic shared functionality.
"""
import copy
import re
from numbers import Integral, Number

import matplotlib.axes as maxes
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.container as mcontainer
import matplotlib.contour as mcontour
import matplotlib.font_manager as mfonts
import matplotlib.legend as mlegend
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.projections as mprojections
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np

from .. import constructor
from .. import gridspec as pgridspec
from ..config import rc
from ..internals import ic  # noqa: F401
from ..internals import _not_none, _pop_props, docstring, rcsetup, warnings
from ..utils import to_rgb, units

__all__ = ['Axes']

ABC_STRING = 'abcdefghijklmnopqrstuvwxyz'
KEYS_INNER = (
    'border', 'borderwidth', 'bbox', 'bboxpad', 'bboxcolor', 'bboxstyle', 'bboxalpha',
)
LOC_SIDES = {  # translate 'filled' legends to location
    'left': 'center right',
    'right': 'center left',
    'top': 'lower center',
    'bottom': 'upper center',
}
LOC_TRANSLATE = {  # for inset colorbars and legends TODO: also as text locations
    'inset': 'best',
    'i': 'best',
    0: 'best',
    1: 'upper right',
    2: 'upper left',
    3: 'lower left',
    4: 'lower right',
    5: 'center left',
    6: 'center right',
    7: 'lower center',
    8: 'upper center',
    9: 'center',
    'l': 'left',
    'r': 'right',
    'b': 'bottom',
    't': 'top',
    'c': 'center',
    'ur': 'upper right',
    'ul': 'upper left',
    'll': 'lower left',
    'lr': 'lower right',
    'cr': 'center right',
    'cl': 'center left',
    'uc': 'upper center',
    'lc': 'lower center',
}


_format_other_docstring = """
rc_kw : dict, optional
    Dictionary containing `~proplot.config.rc` settings applied to
    this axes using `~proplot.config.RcConfigurator.context`.
**kwargs
    Passed to `Axes.format` or passed to `~proplot.config.RcConfigurator.context`
    and used to update axes `~proplot.config.rc` settings. For example,
    ``abcstyle='A.'`` modifies the :rcraw:`abc.style` setting.
"""
docstring.snippets['axes.format_other'] = _format_other_docstring

_patch_kw_docstring = """
patch_kw : dict-like, optional
    Keyword arguments used to update the background patch. This can
    be used e.g. to apply background hatching with ``patch_kw={'hatch': 'xxx'}``.
"""
docstring.snippets['axes.patch_kw'] = _patch_kw_docstring

_inset_docstring = """
Return an inset `CartesianAxes`. This is similar to the builtin
`~matplotlib.axes.Axes.inset_axes` but includes some extra options.

Parameters
----------
bounds : list of float
    The bounds for the inset axes, listed as ``(x, y, width, height)``.
transform : {'data', 'axes', 'figure'} or `~matplotlib.transforms.Transform`, optional
    The transform used to interpret `bounds`. Can be a
    `~matplotlib.transforms.Transform` object or a string representing
    the `~matplotlib.axes.Axes.transData`,
    `~matplotlib.axes.Axes.transAxes`,
    or `~matplotlib.figure.Figure.transFigure` transforms. Default is
    ``'axes'``, i.e. `bounds` is in axes-relative coordinates.
proj, projection : str, `cartopy.crs.Projection`, or `~mpl_toolkits.basemap.Basemap`
    The map projection specification(s). If not provided, the inset axes
    projection is identical to the current axes projection. If ``'cartesian'``,
    a `~proplot.axes.CartesianAxes` inset is created. If ``'polar'``, a
    `~proplot.axes.PolarAxes` inset is created. Otherwise, the argument is
    interpreted by `~proplot.constructor.Proj`, and the result is used
    to make a `~proplot.axes.GeoAxes` (in this case the argument can be
    a `cartopy.crs.Projection` instance, a `~mpl_toolkits.basemap.Basemap`
    instance, or a projection name listed in :ref:`this table <proj_table>`).
proj_kw, projection_kw : dict-like, optional
    Keyword arguments passed to `~mpl_toolkits.basemap.Basemap` or
    cartopy `~cartopy.crs.Projection` classes on instantiation.
basemap : bool or dict-like, optional
    Whether to use `~mpl_toolkits.basemap.Basemap` or
    `~cartopy.crs.Projection` for map projections. Default is ``False``.
zorder : float, optional
    The `zorder <https://matplotlib.org/stable/gallery/misc/zorder_demo.html>`__
    of the axes, should be greater than the zorder of
    elements in the parent axes. Default is ``4``.
zoom : bool, optional
    Whether to draw lines indicating the inset zoom using
    `~Axes.indicate_inset_zoom`. The lines will automatically
    adjust whenever the parent axes or inset axes limits are changed.
    Default is ``True``.
zoom_kw : dict, optional
    Passed to `~Axes.indicate_inset_zoom`.

Other parameters
----------------
**kwargs
    Passed to `CartesianAxes`.
"""
docstring.snippets['axes.inset'] = _inset_docstring

_panel_docstring = """
Return a panel drawn along the edge of this axes.

Parameters
----------
side : str, optional
    The panel location. The following location keys are valid:

    ==========  =====================
    Location    Valid keys
    ==========  =====================
    left        ``'left'``, ``'l'``
    right       ``'right'``, ``'r'``
    bottom      ``'bottom'``, ``'b'``
    top         ``'top'``, ``'t'``
    ==========  =====================

width : float or str or list thereof, optional
    The panel width. Units are interpreted by `~proplot.utils.units`.
    Default is :rc:`subplots.panelwidth`.
space : float or str or list thereof, optional
    Empty space between the main subplot and the panel.
    When :rcraw:`tight` is ``True``, this is adjusted automatically.
    Otherwise, the default is :rc:`subplots.panelpad`.
share : bool, optional
    Whether to enable axis sharing between the *x* and *y* axes of the
    main subplot and the panel long axes for each panel in the stack.
    Sharing between the panel short axis and other panel short axes
    is determined by figure-wide `sharex` and `sharey` settings.

Returns
-------
`~proplot.axes.CartesianAxes`
    The panel axes.
"""
docstring.snippets['axes.panel'] = _panel_docstring


# Colorbar docstrings
_colorbar_args_docstring = """
mappable : mappable, list of plot handles, list of color-spec, or colormap-spec
    There are four options here:

    1. A mappable object. Basically, any object with a ``get_cmap`` method,
       like the objects returned by `~matplotlib.axes.Axes.contourf` and
       `~matplotlib.axes.Axes.pcolormesh`.
    2. A list of "plot handles". Basically, any object with a ``get_color``
       method, like `~matplotlib.lines.Line2D` instances. A colormap will
       be generated from the colors of these objects, and colorbar levels
       will be selected using `values`.  If `values` is ``None``, we try
       to infer them by converting the handle labels returned by
       `~matplotlib.artist.Artist.get_label` to `float`. Otherwise, it is
       set to ``np.linspace(0, 1, len(mappable))``.
    3. A list of hex strings, color string names, or RGB tuples. A colormap
       will be generated from these colors, and colorbar levels will be
       selected using `values`. If `values` is ``None``, it is set to
       ``np.linspace(0, 1, len(mappable))``.
    4. A `~matplotlib.colors.Colormap` instance. In this case, a colorbar
       will be drawn using this colormap and with levels determined by
       `values`. If `values` is ``None``, it is set to
       ``np.linspace(0, 1, cmap.N)``.

values : list of float, optional
    Ignored if `mappable` is a mappable object. This maps each color or
    plot handle in the `mappable` list to numeric values, from which a
    colormap and normalizer are constructed.
"""
_colorbar_kwargs_docstring = """
extend : {None, 'neither', 'both', 'min', 'max'}, optional
    Direction for drawing colorbar "extensions" (i.e. references to
    out-of-bounds data with a unique color). These are triangles by
    default. If ``None``, we try to use the ``extend`` attribute on the
    mappable object. If the attribute is unavailable, we use ``'neither'``.
extendsize : float or str, optional
    The length of the colorbar "extensions" in *physical units*.
    If float, units are inches. If string, units are interpreted
    by `~proplot.utils.units`. Default is :rc:`colorbar.insetextend`
    for inset colorbars and :rc:`colorbar.extend` for outer colorbars.
frame, frameon : bool, optional
    For inset colorbars only. Indicates whether to draw a "frame", just
    like `~matplotlib.axes.Axes.legend`. Default is :rc:`colorbar.frameon`.
alpha, linewidth, edgecolor, facecolor : optional
    For inset colorbars only. Controls the transparency, edge width, edge color,
    and face color of the frame. Defaults are :rc:`colorbar.framealpha`,
    :rc:`axes.linewidth`, :rc:`axes.edgecolor`, and :rc:`axes.facecolor`.
norm : normalizer spec, optional
    Ignored if `values` is ``None``. The normalizer for converting `values`
    to colormap colors. Passed to `~proplot.constructor.Norm`.
norm_kw : dict-like, optional
    The normalizer settings. Passed to `~proplot.constructor.Norm`.
reverse : bool, optional
    Whether to reverse the direction of the colorbar.
tickloc, ticklocation : {'bottom', 'top', 'left', 'right'}, optional
    Where to draw tick marks on the colorbar.
tickdir, tickdirection : {'out', 'in', 'inout'}, optional
    Direction that major and minor tick marks point.
tickminor : bool, optional
    Whether to add minor ticks to the colorbar with
    `~matplotlib.colorbar.ColorbarBase.minorticks_on`.
grid : bool, optional
    Whether to draw "gridlines" between each level of the colorbar.
    Default is :rc:`colorbar.grid`.
label, title : str, optional
    The colorbar label. The `title` keyword is also accepted for
    consistency with `~matplotlib.axes.Axes.legend`.
locator, ticks : locator spec, optional
    Used to determine the colorbar tick positions. Passed to the
    `~proplot.constructor.Locator` constructor function.
maxn : int, optional
    Used if `locator` is ``None``. Determines the maximum number of levels
    that are ticked. Default depends on the colorbar length relative
    to the font size. The keyword name "maxn" is meant to mimic
    the `~matplotlib.ticker.MaxNLocator` class name.
locator_kw : dict-like, optional
    The locator settings. Passed to `~proplot.constructor.Locator`.
minorlocator, minorticks, maxn_minor, minorlocator_kw
    As with `locator`, `maxn`, and `locator_kw`, but for the minor ticks.
format, formatter, ticklabels : formatter spec, optional
    The tick label format. Passed to the `~proplot.constructor.Formatter`
    constructor function.
formatter_kw : dict-like, optional
    The formatter settings. Passed to `~proplot.constructor.Formatter`.
rotation : float, optional
    The tick label rotation. Default is ``0``.
edgecolor, linewidth : optional
    The edge color and line width for the colorbar outline.
labelsize, labelweight, labelcolor : optional
    The font size, weight, and color for colorbar label text.
ticklabelsize, ticklabelweight, ticklabelcolor : optional
    The font size, weight, and color for colorbar tick labels.
orientation : {{None, 'horizontal', 'vertical'}}, optional
    The colorbar orientation. By default this depends on the "side"
    of the subplot or figure where the colorbar is drawn. Inset
    colorbars are always horizontal.
**kwargs
    Passed to `~matplotlib.figure.Figure.colorbar`.
"""
docstring.snippets['axes.colorbar_args'] = _colorbar_args_docstring
docstring.snippets['axes.colorbar_kwargs'] = _colorbar_kwargs_docstring


# Legend docstrings
_legend_args_docstring = """
handles : list of `~matplotlib.artist.Artist`, optional
    List of artists instances, or list of lists of artist instances (see
    the `center` keyword). If ``None``, the artists are retrieved with
    `~matplotlib.axes.Axes.get_legend_handles_labels`.
labels : list of str, optional
    Matching list of string labels, or list of lists of string labels (see
    the `center` keywod). If ``None``, the labels are retrieved by calling
    `~matplotlib.artist.Artist.get_label` on each
    `~matplotlib.artist.Artist` in `handles`.
"""
_legend_kwargs_docstring = """
frame, frameon : bool, optional
    Toggles the legend frame. For centered-row legends, a frame
    independent from matplotlib's built-in legend frame is created.
ncol, ncols : int, optional
    The number of columns. `ncols` is an alias, added
    for consistency with `~matplotlib.pyplot.subplots`.
order : {'C', 'F'}, optional
    Whether legend handles are drawn in row-major (``'C'``) or column-major
    (``'F'``) order. Analagous to `numpy.array` ordering. For some reason
    ``'F'`` was the original matplotlib default. Default is ``'C'``.
center : bool, optional
    Whether to center each legend row individually. If ``True``, we
    actually draw successive single-row legends stacked on top of each
    other. If ``None``, we infer this setting from `handles`. By default,
    `center` is set to ``True`` if `handles` is a list of lists (each
    sublist is used as a row in the legend).
title, label : str, optional
    The legend title. The `label` keyword is also accepted, for consistency
    with `~matplotlib.figure.Figure.colorbar`.
fontsize, fontweight, fontcolor : optional
    The font size, weight, and color for the legend text. The default
    font size is :rcraw:`legend.fontsize`.
color, lw, linewidth, m, marker, ls, linestyle, dashes, ms, markersize \
: property-spec, optional
    Properties used to override the legend handles. For example, for a
    legend describing variations in line style ignoring variations in color, you
    might want to use ``color='k'``. For now this does not include `facecolor`,
    `edgecolor`, and `alpha`, because `~matplotlib.axes.Axes.legend` uses these
    keyword args to modify the frame properties.
**kwargs
    Passed to `~matplotlib.axes.Axes.legend`.
"""
docstring.snippets['axes.legend_args'] = _legend_args_docstring
docstring.snippets['axes.legend_kwargs'] = _legend_kwargs_docstring


class Axes(maxes.Axes):
    """
    Lowest-level axes subclass.
    """
    def __init__(self, *args, number=None, main=False, _subplotspec=None, **kwargs):
        """
        Parameters
        ----------
        number : int
            The subplot number, used for a-b-c labeling. See `~Axes.format`
            for details. Note the first axes is ``1``, not ``0``.
        main : bool, optional
            Used internally, indicates whether this is a "main axes" rather
            than a twin, panel, or inset axes.

        Other parameters
        ----------------
        *args, **kwargs
            Passed to `~matplotlib.axes.Axes`.

        See also
        --------
        matplotlib.axes.Axes
        proplot.axes.PlotAxes
        proplot.axes.CartesianAxes
        proplot.axes.PolarAxes
        proplot.axes.GeoAxes
        """
        super().__init__(*args, **kwargs)

        # Ensure isDefault_minloc enabled at start, needed for dual axes
        self.xaxis.isDefault_minloc = self.yaxis.isDefault_minloc = True

        # Properties
        # TODO: Why are some of these not set in __init__?
        if main:
            self.figure._subplots_main.append(self)
        self.number = number  # for a-b-c numbering
        self._auto_format = None  # manipulated by wrapper functions
        self._abc_loc = None
        self._abc_text = None
        self._abc_border_kwargs = {}
        self._abc_pad = rc['abc.titlepad']
        self._title_loc = None
        self._title_border_kwargs = {}  # title border properties
        self._title_above = rc['title.above']
        self._title_pad = rc['title.pad']
        self._title_pad_current = None
        self._tight_bbox = None  # bounding boxes are saved
        self._panel_hidden = False  # True when "filled" with cbar/legend
        self._panel_parent = None
        self._panel_share = False
        self._panel_sharex_group = False
        self._panel_sharey_group = False
        self._panel_side = None
        self._inset_parent = None
        self._inset_zoom = False
        self._inset_zoom_data = None

        # Axes panels
        d = self._panel_dict = {}
        d['left'] = []  # NOTE: panels will be sorted inside-to-outside
        d['right'] = []
        d['bottom'] = []
        d['top'] = []

        # Axes titles
        # Record the original positions to account for offsetting
        d = self._title_dict = {}
        ta = self.transAxes
        d['abc'] = self.text(0, 0, '', transform=ta)
        d['left'] = self._left_title  # WARNING: track in case mpl changes this
        d['center'] = self.title
        d['right'] = self._right_title
        d['upper left'] = self.text(0, 0, '', va='top', ha='left', transform=ta)
        d['upper center'] = self.text(0, 0, '', va='top', ha='center', transform=ta)
        d['upper right'] = self.text(0, 0, '', va='top', ha='right', transform=ta)
        d['lower left'] = self.text(0, 0, '', va='bottom', ha='left', transform=ta)
        d['lower center'] = self.text(0, 0, '', va='bottom', ha='center', transform=ta)
        d['lower right'] = self.text(0, 0, '', va='bottom', ha='right', transform=ta)

        # Axes row and column labels
        # NOTE: Most of these sit empty for most subplots
        # TODO: Implement this with EdgeStack, avoid creating silly empty objects
        d = self._label_dict = {}
        tf = self.figure.transFigure
        tc = mtransforms.blended_transform_factory(ta, tf)
        tr = mtransforms.blended_transform_factory(tf, ta)
        d['left'] = self.text(0, 0.5, '', va='center', ha='right', transform=tr)
        d['right'] = self.text(0, 0.5, '', va='center', ha='left', transform=tr)
        d['bottom'] = self.text(0.5, 0, '', va='top', ha='center', transform=tc)
        d['top'] = self.text(0.5, 0, '', va='bottom', ha='center', transform=tc)
        d = self._label_pad = {}
        d['left'] = rc['leftlabel.pad']
        d['right'] = rc['rightlabel.pad']
        d['bottom'] = rc['bottomlabel.pad']
        d['top'] = rc['toplabel.pad']

        # Axes colorbars and legends
        self._colorbar_dict = {}
        self._legend_dict = {}

        # Subplot spec
        # WARNING: For mpl>=3.4.0 subplotspec assigned *after* initialization using
        # set_subplotspec. Tried to defer to setter but really messes up both format()
        # and _auto_share_setup(). Instead use workaround: Have Figure.add_subplot pass
        # subplotspec as a hidden keyword arg. Non-subplots don't need this arg.
        # See https://github.com/matplotlib/matplotlib/pull/18564
        if _subplotspec is not None:
            self.set_subplotspec(_subplotspec)

        # Default sharing and formatting
        # TODO: Apply specific setters instead of format()
        self._auto_share_setup()
        self.format(rc_mode=1)  # rc_mode == 1 applies the custom proplot params

    def _auto_share_setup(self):
        """
        Automatically configure axis sharing based on the horizontal and
        vertical extent of subplots in the figure gridspec.
        """
        # Panel axes sharing, between main subplot and its panels
        # NOTE: _panel_share means "include this panel in the axis sharing group"
        # while _panel_sharex_group indicates the group itself and may include main axes
        def shared(paxs):
            return [pax for pax in paxs if not pax._panel_hidden and pax._panel_share]

        # Internal axis sharing, share stacks of panels and main axes with each other
        # NOTE: *This* block is why, even though share[xy] are figure-wide
        # settings, we still need the axes-specific _share[xy]_override attr
        if not self._panel_side:  # this is a main axes
            # Top and bottom
            bottom = self
            paxs = shared(self._panel_dict['bottom'])
            if paxs:
                bottom = paxs[-1]
                bottom._panel_sharex_group = False
                for iax in (self, *paxs[:-1]):
                    iax._panel_sharex_group = True
                    iax._sharex_setup(bottom)  # parent is bottom-most
            paxs = shared(self._panel_dict['top'])
            for iax in paxs:
                iax._panel_sharex_group = True
                iax._sharex_setup(bottom)
            # Left and right
            # NOTE: Order of panel lists is always inside-to-outside
            left = self
            paxs = shared(self._panel_dict['left'])
            if paxs:
                left = paxs[-1]
                left._panel_sharey_group = False
                for iax in (self, *paxs[:-1]):
                    iax._panel_sharey_group = True
                    iax._sharey_setup(left)  # parent is left-most
            paxs = shared(self._panel_dict['right'])
            for iax in paxs:
                iax._panel_sharey_group = True
                iax._sharey_setup(left)

        # External axes sharing, sometimes overrides panel axes sharing
        # NOTE: This can get very repetitive, but probably minimal impact?
        # Share x axes
        parent, *children = self._get_extent_axes('x')
        for child in children:
            child._sharex_setup(parent)
        # Share y axes
        parent, *children = self._get_extent_axes('y')
        for child in children:
            child._sharey_setup(parent)

    def _get_extent_axes(self, x, panels=False):
        """
        Return the axes whose horizontal or vertical extent in the main
        gridspec matches the horizontal or vertical extend of this axes.
        The lefmost or bottommost axes are at the start of the list.
        """
        if not hasattr(self, 'get_subplotspec'):
            return [self]
        y = 'y' if x == 'x' else 'x'
        idx = 0 if x == 'x' else 1
        argfunc = np.argmax if x == 'x' else np.argmin
        irange = self._range_gridspec(x)
        if panels:
            axs = self.figure._iter_axes(hidden=False, children=False)
        else:
            axs = self.figure._subplots_main
        axs = [ax for ax in axs if ax._range_gridspec(x) == irange]
        if not axs:
            return [self]
        else:
            pax = axs.pop(argfunc([ax._range_gridspec(y)[idx] for ax in axs]))
            return [pax, *axs]

    def _get_side_axes(self, side, panels=False):
        """
        Return the axes whose left, right, top, or bottom sides abutt
        against the same row or column as this axes.
        """
        if side not in ('left', 'right', 'bottom', 'top'):
            raise ValueError(f'Invalid side {side!r}.')
        if not hasattr(self, 'get_subplotspec'):
            return [self]
        x = 'x' if side in ('left', 'right') else 'y'
        idx = 0 if side in ('left', 'top') else 1  # which side to test
        coord = self._range_gridspec(x)[idx]  # side for a particular axes
        if panels:
            axs = self.figure._iter_axes(hidden=False, children=False)
        else:
            axs = self.figure._subplots_main
        axs = [ax for ax in axs if ax._range_gridspec(x)[idx] == coord]
        if not axs:
            return [self]
        else:
            return axs

    def _get_transform(self, transform):
        """
        Translates user input transform. Also used in an axes method.
        """
        cartopy = getattr(self, 'name', None) == 'proplot_cartopy'
        if cartopy:
            from cartopy.crs import CRS, PlateCarree
        else:
            CRS = PlateCarree = object
        if (
            isinstance(transform, mtransforms.Transform)
            or cartopy and isinstance(transform, CRS)
        ):
            return transform
        elif transform == 'figure':
            return self.figure.transFigure
        elif transform == 'axes':
            return self.transAxes
        elif transform == 'data':
            return PlateCarree() if cartopy else self.transData
        elif transform == 'map' and cartopy:
            return self.transData
        else:
            raise ValueError(f'Unknown transform {transform!r}.')

    def _hide_panel(self):
        """
        Hide axes contents but do *not* make the entire axes invisible. This is used to
        fill "panels" surreptitiously added to the gridspec for the purpose of drawing
        outer colorbars and legends.
        """
        # NOTE: Do not run self.clear in case we want to add a subplot title
        # above a colorbar on a top panel (see _reassign_title).
        for s in self.spines.values():
            s.set_visible(False)
        self.xaxis.set_visible(False)
        self.yaxis.set_visible(False)
        self.patch.set_alpha(0)
        self._panel_hidden = True

    def _is_panel(self):
        """
        Return whether the current axes is a panel.
        """
        return bool(self._panel_parent)

    def _is_panel_group_member(self, other):
        """
        Return whether the axes are related.
        """
        return (
            self._panel_parent is other  # child
            or other._panel_parent is self   # parent
            or other._panel_parent is self._panel_parent  # sibling
        )

    def _loc_translate(self, loc, mode=None):
        """
        Return the location string `loc` translated into a standardized form.
        """
        if mode == 'legend':
            options = tuple(LOC_TRANSLATE.values())
        elif mode == 'panel':
            options = ('left', 'right', 'top', 'bottom')
        elif mode == 'colorbar':
            options = (
                'best', 'left', 'right', 'top', 'bottom',
                'upper left', 'upper right', 'lower left', 'lower right',
            )
        elif mode in ('abc', 'title'):
            options = (
                'left', 'center', 'right',
                'upper left', 'upper center', 'upper right',
                'lower left', 'lower center', 'lower right',
            )
        else:
            raise ValueError(f'Invalid mode {mode!r}.')
        loc_translate = {
            key: value
            for short, long in LOC_TRANSLATE.items()
            for key, value in ((long, long), (short, long))
            if long in options
        }
        if loc in (None, True):
            context = mode in ('abc', 'title')
            loc = rc.get(mode + '.loc', context=context)
            if loc is not None:
                loc = self._loc_translate(loc, mode)
        elif isinstance(loc, (str, Integral)):
            try:
                loc = loc_translate[loc]
            except KeyError:
                raise KeyError(
                    f'Invalid {mode} location {loc!r}. Options are: '
                    + ', '.join(map(repr, loc_translate)) + '.'
                )
        elif (
            mode == 'legend'
            and np.iterable(loc)
            and len(loc) == 2
            and all(isinstance(l, Number) for l in loc)
        ):
            loc = tuple(loc)
        else:
            raise KeyError(f'Invalid {mode} location {loc!r}.')
        if mode == 'colorbar' and loc == 'best':  # white lie
            loc = 'lower right'
        return loc

    def _make_inset_locator(self, bounds, trans):
        """
        Return a locator that determines inset axes bounds.
        """
        def inset_locator(ax, renderer):
            bbox = mtransforms.Bbox.from_bounds(*bounds)
            bb = mtransforms.TransformedBbox(bbox, trans)
            tr = self.figure.transFigure.inverted()
            bb = mtransforms.TransformedBbox(bb, tr)
            return bb
        return inset_locator

    def _range_gridspec(self, x):
        """
        Return the column or row gridspec range for the axes.
        """
        if not hasattr(self, 'get_subplotspec'):
            raise RuntimeError('Axes is not a subplot.')
        ss = self.get_subplotspec()
        if hasattr(ss, 'get_active_rows_columns'):
            func = ss.get_active_rows_columns
        else:
            func = ss.get_rows_columns
        if x == 'x':
            _, _, _, _, col1, col2 = func()
            return col1, col2
        else:
            _, _, row1, row2, _, _ = func()
            return row1, row2

    def _range_tightbbox(self, x):
        """
        Return the tight bounding box span from the cached bounding box.
        `~proplot.axes.Axes.get_tightbbox` caches bounding boxes when
        `~Figure.get_tightbbox` is called.
        """
        # TODO: Better testing for axes visibility
        bbox = self._tight_bbox
        if bbox is None:
            return np.nan, np.nan
        if x == 'x':
            return bbox.xmin, bbox.xmax
        else:
            return bbox.ymin, bbox.ymax

    def _reassign_label(self, side):
        """
        Reassign the column and row labels to the relevant panel if present.
        This is called by `~proplot.figure.Figure._align_subplot_figure_labels`.
        """
        # NOTE: Since panel axes are "children" main axes is always drawn first.
        paxs = self._panel_dict[side]
        if not paxs:
            return self
        kw = {}
        pax = paxs[-1]  # outermost
        cobj = self._label_dict[side]
        pobj = pax._label_dict[side]
        for key in ('text', 'color', 'fontproperties'):
            kw[key] = getattr(cobj, 'get_' + key)()
        pobj.update(kw)
        cobj.set_text('')
        return pax

    def _reassign_title(self):
        """
        Re-assign the title to the first upper panel if present. We cannot
        simply add the upper panel as a child axes, because then the title will
        be offset but still belong to main axes, which messes up the tight
        bounding box.
        """
        # NOTE: Since panel axes are "children" main axes is always drawn first.
        taxs = self._panel_dict['top']
        if not taxs or not self._title_above:
            return
        tax = taxs[-1]  # outermost
        tax._title_pad = self._title_pad
        for loc in ('abc', 'left', 'center', 'right'):
            kw = {}
            cobj = self._title_dict[loc]
            tobj = tax._title_dict[loc]  # WARNING: Careful to use 'abc' here
            if loc == 'abc':
                loc = tax._abc_loc = self._abc_loc
            if loc not in ('left', 'center', 'right'):
                continue
            text = cobj.get_text()
            if not text:
                continue
            for key in ('color', 'fontproperties'):
                kw[key] = getattr(cobj, 'get_' + key)()
            tobj.update(kw)
            tobj.set_text(text)
            cobj.set_text('')

    def _sharex_setup(self, sharex):
        """
        Configure x-axis sharing for panels. Main axis sharing is done in
        `~CartesianAxes._sharex_setup`.
        """
        self._share_short_axis(sharex, 'left')  # x axis of left panels
        self._share_short_axis(sharex, 'right')
        self._share_long_axis(sharex, 'bottom')  # x axis of bottom panels
        self._share_long_axis(sharex, 'top')

    def _sharey_setup(self, sharey):
        """
        Configure y-axis sharing for panels. Main axis sharing is done in
        `~CartesianAxes._sharey_setup`.
        """
        self._share_short_axis(sharey, 'bottom')  # y axis of bottom panels
        self._share_short_axis(sharey, 'top')
        self._share_long_axis(sharey, 'left')  # y axis of left panels
        self._share_long_axis(sharey, 'right')

    def _share_short_axis(self, share, side):
        """
        Share the "short" axes of panels belonging to this subplot
        with panels belonging to an external subplot.
        """
        if share is None or self._panel_side:
            return  # if this is a panel
        axis = 'x' if side in ('left', 'right') else 'y'
        caxs = self._panel_dict[side]
        paxs = share._panel_dict[side]
        caxs = [pax for pax in caxs if not pax._panel_hidden]
        paxs = [pax for pax in paxs if not pax._panel_hidden]
        for cax, pax in zip(caxs, paxs):  # may be uneven
            getattr(cax, '_share' + axis + '_setup')(pax)

    def _share_long_axis(self, share, side):
        """
        Share the "long" axes of panels belonging to this subplot
        with panels belonging to an external subplot.
        """
        # NOTE: We do not check _panel_share because that only controls
        # sharing with main subplot, not other subplots
        if share is None or self._panel_side:
            return  # if this is a panel
        axis = 'x' if side in ('top', 'bottom') else 'y'
        paxs = self._panel_dict[side]
        paxs = [pax for pax in paxs if not pax._panel_hidden]
        for pax in paxs:
            getattr(pax, '_share' + axis + '_setup')(share)

    def _update_abc(self):
        """
        Whether to update the label.
        """
        abc = False
        if self._panel_side:
            return

        # Properties
        # NOTE: Border props only apply for "inner" title locations so we
        # need to store on the axes whenever they are modified and always
        # re-apply the ones stored on the axes.
        kw = rc.fill(
            {
                'fontsize': 'abc.size',
                'weight': 'abc.weight',
                'color': 'abc.color',
                'fontfamily': 'font.family',
            },
            context=True
        )
        kwb = rc.fill(
            {
                'border': 'abc.border',
                'borderwidth': 'abc.borderwidth',
                'bbox': 'abc.bbox',
                'bboxpad': 'abc.bboxpad',
                'bboxcolor': 'abc.bboxcolor',
                'bboxstyle': 'abc.bboxstyle',
                'bboxalpha': 'abc.bboxalpha',
            },
            context=True,
        )
        self._abc_border_kwargs.update(kwb)
        kw.update(self._abc_border_kwargs)

        # A-b-c labels. Build as a...z...aa...zz...aaa...zzz
        style = rc.get('abc.style', context=True)  # 1st run, or changed
        if style and self.number is not None:
            if not isinstance(style, str) or 'a' not in style and 'A' not in style:
                raise ValueError(
                    f'Invalid abcstyle {style!r}. Must include letter "a" or "A".'
                )
            nabc, iabc = divmod(self.number - 1, 26)
            old = re.search('[aA]', style).group()  # return the *first* 'a'
            new = (nabc + 1) * ABC_STRING[iabc]
            new = new.upper() if old == 'A' else new
            self._abc_text = style.replace(old, new, 1)

        # Apply a-b-c text
        abc = rc.get('abc', context=True)
        aobj = self._title_dict['abc']
        if abc is not None:
            aobj.set_text(self._abc_text if bool(abc) else '')

        # Apply a-b-c settings
        loc = self._loc_translate(None, 'abc')
        loc_prev = self._abc_loc
        if loc is None:
            loc = loc_prev
        if loc in ('left', 'right', 'center'):
            for key in KEYS_INNER:
                kw.pop(key, None)
        aobj.update(kw)
        self._abc_loc = loc

    def _update_super(self, suptitle, **kwargs):
        """
        Update super title and row and column labels.
        """
        # NOTE: These are actually *figure-wide* settings, but that line gets
        # blurred where we have shared axes, spanning labels, and whatnot. May result
        # in redundant assignments if formatting more than one axes, but operations
        # are fast so some redundancy is nbd.
        # NOTE: Below kludge prevents changed *figure-wide* settings from getting
        # overwritten when user makes a new panels or insets. Funky limnitation but
        # kind of makes sense if these are inaccessible from panels.
        fig = self.figure
        ignore = self not in fig._subplots_main
        kw = {} if ignore else rc.fill(
            {
                'fontsize': 'suptitle.size',
                'weight': 'suptitle.weight',
                'color': 'suptitle.color',
                'fontfamily': 'font.family'
            },
            context=True,
        )
        if suptitle or kw:
            fig._update_super_title(suptitle, **kw)

        # Labels
        for side, labels in kwargs.items():
            kw = {} if ignore else rc.fill(
                {
                    'color': side + 'label.color',
                    'rotation': side + 'label.rotation',
                    'fontsize': side + 'label.size',
                    'weight': side + 'label.weight',
                    'fontfamily': 'font.family'
                },
                context=True,
            )
            if labels or kw:
                fig._update_super_labels(self, side, labels, **kw)

    def _update_title_all(self, title=None, **kwargs):
        """
        Update the titles.
        """
        # Titles, with two workflows here:
        # 1. title='name' and titleloc='position'
        # 2. ltitle='name', rtitle='name', etc., arbitrarily many titles
        # NOTE: Matplotlib added axes.titlecolor in version 3.2 but we
        # still use custom title.size, title.weight, title.color
        # properties for retroactive support in older matplotlib versions.
        # First get params and update kwargs
        kw = rc.fill(
            {
                'fontsize': 'title.size',
                'weight': 'title.weight',
                'color': 'title.color',
                'fontfamily': 'font.family',
            },
            context=True
        )
        if 'color' in kw and kw['color'] == 'auto':
            del kw['color']  # WARNING: matplotlib permits invalid color here
        kwb = rc.fill(
            {
                'border': 'title.border',
                'borderwidth': 'title.borderwidth',
                'bbox': 'title.bbox',
                'bboxpad': 'title.bboxpad',
                'bboxcolor': 'title.bboxcolor',
                'bboxstyle': 'title.bboxstyle',
                'bboxalpha': 'title.bboxalpha',
            },
            context=True,
        )
        self._title_border_kwargs.update(kwb)
        kw.update(self._title_border_kwargs)

        # Workflow 2, want this to come first so workflow 1 gets priority
        for iloc, ititle in kwargs.items():
            ikw = kw.copy()
            iloc = self._loc_translate(iloc, 'title')
            if iloc in ('left', 'center', 'right'):
                for key in KEYS_INNER:
                    ikw.pop(key, None)
            iobj = self._title_dict[iloc]
            iobj.update(ikw)
            if ititle is not None:
                iobj.set_text(ititle)

        # Workflow 1, make sure that if user calls ax.format(title='Title')
        # *then* ax.format(titleloc='left') it copies over the text.
        # Get current and previous location, prevent overwriting abc label
        loc = self._loc_translate(None, 'title')
        loc_prev = self._title_loc
        if loc is None:  # never None first run
            loc = loc_prev  # never None on subsequent runs

        # Remove previous text
        if loc_prev is not None and loc != loc_prev:
            tobj_prev = self._title_dict[loc_prev]
            if title is None:
                title = tobj_prev.get_text()
            tobj_prev.set_text('')

        # Add new text and settings
        kw = kw.copy()
        if loc in ('left', 'center', 'right'):
            for key in KEYS_INNER:
                kw.pop(key, None)
        tobj = self._title_dict[loc]
        tobj.update(kw)
        if title is not None:
            tobj.set_text(title)
        self._title_loc = loc  # assigns default loc on first run

    def _update_title_position(self, renderer):
        """
        Update the position of proplot inset titles and builtin matplotlib
        titles. This is called by matplotlib at drawtime.
        """
        # Update title positions
        # NOTE: Critical to do this every time in case padding changes or
        # we added or removed an a-b-c label in the same position as a title
        width, height = self.get_size_inches()
        x_pad = self._title_pad / (72 * width)
        y_pad = self._title_pad / (72 * height)
        for loc, obj in self._title_dict.items():
            x, y = (0, 1)
            if loc == 'abc':  # redirect
                loc = self._abc_loc
            if loc == 'left':
                x = 0
            elif loc == 'center':
                x = 0.5
            elif loc == 'right':
                x = 1
            if loc in ('upper center', 'lower center'):
                x = 0.5
            elif loc in ('upper left', 'lower left'):
                x = x_pad
            elif loc in ('upper right', 'lower right'):
                x = 1 - x_pad
            if loc in ('upper left', 'upper right', 'upper center'):
                y = 1 - y_pad
            elif loc in ('lower left', 'lower right', 'lower center'):
                y = y_pad
            obj.set_position((x, y))

        # Push title above tick marks, since builtin algorithm seems to ignore them.
        # This is known matplotlib problem but especially annoying with top panels.
        # NOTE: See axis.get_ticks_position for inspiration
        pad = self._title_pad
        if self.xaxis.get_visible() and any(
            tick.tick2line.get_visible() and not tick.label2.get_visible()
            for tick in self.xaxis.majorTicks
        ):
            pad += self.xaxis.get_tick_padding()

        # Avoid applying padding on every draw in case it is expensive to change
        # the title Text transforms every time.
        pad_current = self._title_pad_current
        if pad_current is None or not np.isclose(pad, pad_current):
            self._title_pad_current = pad
            self._set_title_offset_trans(pad)

        # Adjust the above-axes positions with builtin algorithm
        # WARNING: Make sure the name of this private function doesn't change
        super()._update_title_position(renderer)

        # Sync the title positiona with the a-b-c label position
        aobj = self._title_dict['abc']
        tobj = self._title_dict[self._abc_loc]
        aobj.set_ha(tobj.get_ha())
        aobj.set_va(tobj.get_va())
        aobj.set_position(tobj.get_position())
        aobj.set_transform(tobj.get_transform())

        # Offset title away from a-b-c label
        # NOTE: Title texts all use axes transform in x-direction
        # TODO: Make empirical padding of '0.4' em tunable?
        if not tobj.get_text() or not aobj.get_text():
            return
        awidth, twidth = (
            obj.get_window_extent(renderer).transformed(self.transAxes.inverted()).width
            for obj in (aobj, tobj)
        )
        apad, tpad = (
            (self._abc_pad / 72) / self.get_size_inches()[0]
            for obj in (aobj, tobj)
        )
        ha = aobj.get_ha()
        aoffset = toffset = 0
        if ha == 'left':
            toffset = awidth + apad
        elif ha == 'right':
            aoffset = -(twidth + tpad)
        else:  # guaranteed center, there are others
            toffset = 0.5 * (awidth + apad)
            aoffset = -0.5 * (twidth + tpad)
        aobj.set_x(aobj.get_position()[0] + aoffset)
        tobj.set_x(tobj.get_position()[0] + toffset)

    @staticmethod
    @warnings._rename_kwargs('0.6', mode='rc_mode')
    def _parse_format(rc_kw=None, rc_mode=None, **kwargs):
        """
        Separate `~proplot.config.rc` setting name value pairs from
        `~Axes.format` keyword arguments.
        """
        kw = {}
        rc_kw = rc_kw or {}
        rc_mode = _not_none(rc_mode, 2)
        for key, value in kwargs.items():
            key_fixed = rcsetup._rc_nodots.get(key, None)
            if key_fixed is None:
                kw[key] = value
            else:
                rc_kw[key_fixed] = value
        return rc_kw, rc_mode, kw

    @docstring.add_snippets
    def format(
        self, *, title=None,
        figtitle=None, suptitle=None, rowlabels=None, collabels=None,
        leftlabels=None, rightlabels=None, toplabels=None, bottomlabels=None,
        llabels=None, rlabels=None, tlabels=None, blabels=None,
        ltitle=None, ctitle=None, rtitle=None,
        ultitle=None, uctitle=None, urtitle=None,
        lltitle=None, lctitle=None, lrtitle=None,
    ):
        """
        Modify the axes title(s), the a-b-c label, row and column labels, and
        the figure title. Called by the `~proplot.axes.CartesianAxes`,
        `~proplot.axes.PolarAxes`, and `~proplot.axes.GeoAxes` ``format``
        methods.

        Parameters
        ----------
        title : str, optional
            The axes title.
        abc : bool, optional
            Whether to apply "a-b-c" subplot labeling based on the subplot
            `~Axes.number`. If `~Axes.number` is greater than 26, the labels
            will loop around to a, ..., z, aa, ..., zz, aaa, ..., zzz, etc.
            Default is :rc:`abc`.
        abcstyle : str, optional
            String denoting the format of a-b-c labels containing the character
            ``a`` or ``A``. ``'a'`` is the default, but e.g. ``'a.'``,
            ``'a)'``, or ``'A'`` might also be desirable. Default is
            :rc:`abc.style`.
        abcloc, titleloc : str, optional
            Strings indicating the location for the a-b-c label and
            main title. The following locations keys are valid (defaults are
            :rc:`abc.loc` and :rc:`title.loc`):

            .. _title_table:

            ========================  ============================
            Location                  Valid keys
            ========================  ============================
            center above axes         ``'center'``, ``'c'``
            left above axes           ``'left'``, ``'l'``
            right above axes          ``'right'``, ``'r'``
            lower center inside axes  ``'lower center'``, ``'lc'``
            upper center inside axes  ``'upper center'``, ``'uc'``
            upper right inside axes   ``'upper right'``, ``'ur'``
            upper left inside axes    ``'upper left'``, ``'ul'``
            lower left inside axes    ``'lower left'``, ``'ll'``
            lower right inside axes   ``'lower right'``, ``'lr'``
            ========================  ============================

        ltitle, ctitle, rtitle, ultitle, uctitle, urtitle, lltitle, lctitle, lrtitle \
: str, optional
            Axes titles in specific positions. Works as an alternative to
            ``ax.format(title='title', titleloc='loc')`` and lets you specify
            multiple title-like labels in a single subplot.
        abcborder, titleborder : bool, optional
            Whether to draw a white border around titles and a-b-c labels
            positioned inside the axes. This can help them stand out on top
            of artists plotted inside the axes. Defaults are
            :rc:`abc.border` and :rc:`title.border`
        abcbbox, titlebbox : bool, optional
            Whether to draw a white bbox around titles and a-b-c labels
            positioned inside the axes. This can help them stand out on top
            of artists plotted inside the axes. Defaults are
            :rc:`abc.bbox` and :rc:`title.bbox`
        titlepad : float, optional
            The padding for the inner and outer titles and a-b-c labels in
            arbitrary units (default is points). Default is :rc:`title.pad`.
        abctitlepad : float, optional
            The horizontal padding between the a-b-c label and title when they are
            in the same location. Default is :rc:`abc.titlepad`.
        titleabove : bool, optional
            Whether to try to put outer titles and a-b-c labels above panels,
            colorbars, or legends that are above the axes. Default is :rc:`title.above`.
        leftlabels, toplabels, rightlabels, bottomlabels : list of str, optional
            Labels for the subplots lying along the left, top, right, and
            bottom edges of the figure. The length of each list must match
            the number of subplots along the corresponding edge.
        rowlabels, collabels, llabels, tlabels, rlabels, blabels : list of str, optional
            Aliases for `leftlabels`, `toplabels`, `leftlabels`, `toplabels`,
            `rightlabels`, and `bottomlabels`.
        leftlabelpad, toplabelpad, rightlabelpad, bottomlabelpad : float, optional
            The padding between the labels and the axes content in arbitrary units
            (default is points). Defaults are :rcraw:`leftlabel.pad`,
            :rcraw:`toplabel.pad`, :rcraw:`rightlabel.pad`, and
            :rcraw:`bottomlabel.pad`
        suptitle, figtitle : str, optional
            The figure "super" title, centered between the left edge of
            the lefmost column of subplots and the right edge of the rightmost
            column of subplots, and automatically offset above figure titles.
            This is an improvement on matplotlib's "super" title, which just
            centers the text between figure edges.
        suptitlepad : float, optional
            The padding between the super title and the axes content in arbitrary
            units (default is points). Default is :rcraw:`suptitle.pad`.

        Other parameters
        ----------------
        %(axes.format_other)s

        Important
        ---------
        The `abc`, `abcstyle`, `abcloc`, `titleloc`, `titleabove`, `titlepad`,
        `abctitlepad`, `leftlabelpad`, `toplabelpad`, `rightlabelpad`, and
        `bottomlabelpad` keywords are :ref:`configuration settings <ug_config>`.
        We explicitly document these arguments here because it is very common to change
        them. But many :ref:`other configuration settings <ug_format>` can be passed
        to ``format`` too.

        See also
        --------
        proplot.config.RcConfigurator.context
        proplot.axes.CartesianAxes.format
        proplot.axes.PolarAxes.format
        proplot.axes.GeoAxes.format
        """
        # General figure settings
        # TODO: Work out awkward situation where "figure" settings applied on axes
        kw = rc.fill({'facecolor': 'figure.facecolor'}, context=True)
        self.figure.patch.update(kw)

        # General axes settings
        cycle = rc.get('axes.prop_cycle', context=True)
        if cycle is not None:
            self.set_prop_cycle(cycle)

        # Text positioning adjustments, applied at drawtime
        above = rc.get('title.above', context=True)
        if above is not None:
            self._title_above = above
        pad = rc.get('abc.titlepad', context=True)
        if pad is not None:
            self._abc_pad = pad
        pad = rc.get('title.pad', context=True)  # title
        if pad is not None:
            self._set_title_offset_trans(pad)
            self._title_pad = pad
        pad = rc.get('suptitle.pad', context=True)
        if pad is not None:
            self.figure._suptitle_pad = pad
        for side in tuple(self._label_pad):  # labels
            pad = rc.get(side + 'label.pad', context=True)
            if pad is not None:
                self._label_pad[side] = pad

        # Update a-b-c label and title(s)
        self._update_abc()
        self._update_title_all(
            title, l=ltitle, r=rtitle, c=ctitle,
            ul=ultitle, uc=uctitle, ur=urtitle,
            ll=lltitle, lc=lctitle, lr=lrtitle,
        )

        # Update 'super' labels
        suptitle = _not_none(figtitle=figtitle, suptitle=suptitle)
        rlabels = _not_none(rightlabels=rightlabels, rlabels=rlabels)
        blabels = _not_none(bottomlabels=bottomlabels, blabels=blabels)
        llabels = _not_none(rowlabels=rowlabels, leftlabels=leftlabels, llabels=llabels)
        tlabels = _not_none(collabels=collabels, toplabels=toplabels, tlabels=tlabels)
        self._update_super(
            suptitle, left=llabels, right=rlabels, top=tlabels, bottom=blabels
        )

    def draw(self, renderer=None, *args, **kwargs):
        # Perform extra post-processing steps
        # NOTE: Used to have _reassign_title here (maybe _reassign_label too?)
        # but figured out it needs to get called by Figure spacing algorithm.
        super().draw(renderer, *args, **kwargs)

    def get_size_inches(self):
        # Return the width and height of the axes in inches.
        width, height = self.figure.get_size_inches()
        bbox = self.get_position()
        width = width * abs(bbox.width)
        height = height * abs(bbox.height)
        return np.array([width, height])

    def get_tightbbox(self, renderer, *args, **kwargs):
        # Perform extra post-processing steps and cache the bounding
        # box as an attribute.
        bbox = super().get_tightbbox(renderer, *args, **kwargs)
        self._tight_bbox = bbox
        return bbox

    @docstring.add_snippets
    def inset(self, *args, **kwargs):
        """
        %(axes.inset)s
        """
        return self.inset_axes(*args, **kwargs)

    @docstring.add_snippets
    def inset_axes(
        self, bounds, transform=None, zorder=4,
        zoom=None, zoom_kw=None,
        proj=None, proj_kw=None, projection=None, projection_kw=None, basemap=None,
        **kwargs
    ):
        """
        %(axes.inset)s
        """
        # Carbon copy with my custom axes
        if not transform:
            transform = self.transAxes
        else:
            transform = self._get_transform(transform)
        label = kwargs.pop('label', 'inset_axes')
        proj = _not_none(proj=proj, projection=projection)
        proj_kw = _not_none(proj_kw=proj_kw, projection_kw=projection_kw, default={})
        if basemap is not None:
            proj_kw['basemap'] = basemap

        # Inherit from current axes
        if proj is None:
            proj = self.name  # will have 'proplot_' prefix
            if proj_kw:
                warnings._warn_proplot(
                    'Inheriting projection from the main axes. '
                    f'Ignoring proj_kw keyword args: {proj_kw}'
                )
            if proj in ('proplot_cartopy', 'proplot_basemap'):
                m = copy.copy(self.projection)
                kwargs.setdefault('map_projection', m)

        # Create new projection
        elif proj == 'cartesian':
            proj = 'proplot_cartesian'
        elif proj == 'polar':
            proj = 'proplot_polar'
        else:
            m = constructor.Proj(proj, **proj_kw)
            kwargs.setdefault('map_projection', m)
            proj = 'proplot_' + m._proj_package

        # This puts the rectangle into figure-relative coordinates.
        locator = self._make_inset_locator(bounds, transform)
        cls = mprojections.get_projection_class(proj)
        bb = locator(None, None)
        ax = cls(self.figure, bb.bounds, zorder=zorder, label=label, **kwargs)

        # The following locator lets the axes move if we used data coordinates,
        # is called by ax.apply_aspect()
        zoom = _not_none(zoom, self.name == ax.name)  # only zoom when same projection
        ax.set_axes_locator(locator)
        self.add_child_axes(ax)
        ax._inset_zoom = zoom
        ax._inset_parent = self

        # Zoom indicator (NOTE: Requires version >=3.0)
        if zoom:
            zoom_kw = zoom_kw or {}
            ax.indicate_inset_zoom(**zoom_kw)
        return ax

    def indicate_inset_zoom(
        self, alpha=None,
        lw=None, linewidth=None, zorder=3.5,
        ls=None, linestyle=None,
        color=None, edgecolor=None, **kwargs
    ):
        """
        Draw lines indicating the zoom range of the inset axes. This is similar
        to the builtin `~matplotlib.axes.Axes.indicate_inset_zoom` except
        lines are *refreshed* at draw-time. This is also called automatically
        when ``zoom=True`` is passed to `~Axes.inset_axes`. Note this method
        must be called from the *inset* axes and not the parent axes.

        Parameters
        ----------
        alpha : float, optional
            The transparency of the zoom box fill.
        lw, linewidth : float, optional
            The width of the zoom lines and box outline in points.
        ls, linestyle : linestyle-spec, optional
            The line style for the zoom lines and box outline.
        color, edgecolor : color-spec, optional
            The color of the zoom lines and box outline.
        capstyle : {'butt', 'round', 'projecting'}
            The cap style for the zoom lines and box outline.
        zorder : float, optional
            The `zorder <https://matplotlib.org/stable/gallery/misc/zorder_demo.html>`__
            of the zoom lines. Should be greater than the zorder of
            elements in the parent axes. Default is ``3.5``.

        Other parameters
        ----------------
        **kwargs
            Passed to `~matplotlib.axes.Axes.indicate_inset`.
        """
        # Should be called from the inset axes
        parent = self._inset_parent
        alpha = alpha or 1.0
        kwargs.setdefault('capstyle', 'round')  # match zoom capstyle
        linestyle = _not_none(ls=ls, linestyle=linestyle, default='-')
        linewidth = _not_none(
            lw=lw, linewidth=linewidth, default=rc['axes.linewidth'],
        )
        edgecolor = _not_none(
            color=color, edgecolor=edgecolor, default=rc['axes.edgecolor'],
        )
        if not parent:
            raise ValueError(f'{self} is not an inset axes.')
        xlim, ylim = self.get_xlim(), self.get_ylim()
        rect = (xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0])

        # Call indicate_inset
        props = {
            'linestyle': linestyle, 'linewidth': linewidth,
            'edgecolor': edgecolor, 'alpha': alpha,
        }
        rectpatch, connects = parent.indicate_inset(
            rect, self, zorder=zorder, **props, **kwargs
        )

        # Update zoom or adopt properties from old one
        if self._inset_zoom_data is None:
            for line in connects:
                line.update(props)
        else:
            rectpatch_prev, connects_prev = self._inset_zoom_data
            rectpatch.update_from(rectpatch_prev)
            rectpatch_prev.set_visible(False)
            for line, line_prev in zip(connects, connects_prev):
                visible = line.get_visible()
                line.update_from(line_prev)
                line.set_visible(visible)
                line_prev.set_visible(False)
        self._inset_zoom_data = (rectpatch, connects)
        return rectpatch, connects

    @docstring.add_snippets
    def panel(self, side, **kwargs):
        """
        %(axes.panel)s
        """
        return self.panel_axes(side, **kwargs)

    @docstring.add_snippets
    def panel_axes(self, side, **kwargs):
        """
        %(axes.panel)s
        """
        side = self._loc_translate(side, 'panel')
        return self.figure._add_axes_panel(self, side, **kwargs)

    def _add_colorbar_legend(self, type_, loc, obj, legend=False, **kwargs):
        """
        Queue up or replace objects for legends and list-of-artist style colorbars.
        """
        # Initial stuff
        d = self._legend_dict if legend else self._colorbar_dict
        if type_ not in ('legend', 'colorbar'):
            raise TypeError(f'Invalid type {type_!r}.')
        if loc == 'fill':  # should have already been indexed in the *parent*
            return

        # Remove previous instances
        # NOTE: No good way to remove inset colorbars right now until the bounding
        # box and axes are merged into some colorbar subclass. Fine for now.
        if loc in d and not isinstance(d[loc], tuple):
            obj_prev = d.pop(loc)  # possibly pop a queued object
            if hasattr(self, 'legend_') and self.legend_ is obj_prev:
                self.legend_ = None  # was never added as artist
            elif legend:
                obj_prev.remove()  # remove legends and inner colorbars

        # Replace with instance or update the queue
        if not isinstance(obj, tuple) or any(isinstance(_, mlegend.Legend) for _ in obj):  # noqa: E501
            d[loc] = obj
        else:
            handles, labels = obj
            handles_full, labels_full, kwargs_full = d.setdefault(loc, ([], [], {}))
            handles_full.extend(_not_none(handles, []))
            labels_full.extend(_not_none(labels, []))
            kwargs_full.update(kwargs)

    def _auto_colorbar_legend(
        self, objs, colorbar=None, colorbar_kw=None, legend=None, legend_kw=None,
    ):
        """
        Add automatic legend.
        """
        # Add colorbar
        # NOTE: Colorbar will get the labels from the artists. Don't need to extract
        # them because can't have multiple-artist entries like for legend()
        if colorbar:
            colorbar_kw = colorbar_kw or {}
            self.colorbar(objs, loc=colorbar, queue=True, **colorbar_kw)

        # Add legend
        if legend:
            legend_kw = legend_kw or {}
            try:
                objs, labels = list(zip(*self._auto_legend_pairs(objs)))
            except ValueError:
                objs = labels = ()
            self.legend(objs, labels, loc=legend, queue=True, **legend_kw)

    def _auto_legend_pairs(self, objs):
        """
        Retrieve the (object, label) pairs for objects with actual labels from
        nested lists and tuples of objects. Used for auto legends only.
        """
        # Return only valid labels
        def _valid_legend_label(obj):
            label = getattr(obj, 'get_label', lambda: None)()
            label = label if label is None else str(label)
            return None if label is None or label[:1] == '_' else label
        # Account for return values like (bins, values, patches). Also *expand*
        # tuples of multiple artists if they have non-matching non-placeholder
        # labels. This is used to interpret (shading, line) style input.
        label = _valid_legend_label(objs)
        if label is not None:
            yield (objs, label)
        elif isinstance(objs, list):
            # Add from list of plotted columns
            for obj in objs:
                yield from self._auto_legend_pairs(obj)
        elif isinstance(objs, tuple) and not isinstance(objs, mcontainer.Container):
            # Add from grouped objects
            # TODO: Ignore error bar objects?
            labels = tuple(map(_valid_legend_label, objs))
            labels_valid = set(label for label in labels if label is not None)
            if len(labels_valid) > 1:
                for obj, label in zip(objs, labels):
                    if label is not None:
                        yield obj, label
            elif labels_valid:
                yield (objs, labels_valid.pop())
        else:
            raise ValueError(f'Unrecognized legend object {objs!r}.')

    def _draw_colorbars_legends(self):
        """
        Draw the queued-up legends and colorbars. Wrapper funcs and legend func let
        user add handles to location lists with successive calls.
        """
        # Draw colorbars
        for loc, colorbar in self._colorbar_dict.items():
            if not isinstance(colorbar, tuple):
                continue
            handles, labels, kwargs = colorbar
            self.colorbar(handles, labels or None, loc=loc, **kwargs)

        # Draw legends
        # WARNING: Passing empty list labels=[] to legend causes matplotlib
        # _parse_legend_args to search for everything. Ensure None if empty.
        for loc, legend in self._legend_dict.items():
            if not isinstance(legend, tuple) or any(isinstance(_, mlegend.Legend) for _ in legend):  # noqa: E501
                continue
            handles, labels, kwargs = legend
            self.legend(handles, labels or None, loc=loc, **kwargs)

    def _fill_colorbar_axes(self, length=None, **kwargs):
        """
        Return the axes and adjusted keyword args for a panel-filling colorbar.
        """
        # Get subplotspec for colorbar axes
        side = self._panel_side
        length = _not_none(length, rc['colorbar.length'])
        subplotspec = self.get_subplotspec()
        if length <= 0 or length > 1:
            raise ValueError(
                f'Panel colorbar length must satisfy 0 < length <= 1, '
                f'got length={length!r}.'
            )
        if side in ('bottom', 'top'):
            gridspec = pgridspec._GridSpecFromSubplotSpec(
                nrows=1, ncols=3, wspace=0,
                subplot_spec=subplotspec,
                width_ratios=((1 - length) / 2, length, (1 - length) / 2),
            )
            subplotspec = gridspec[1]
        else:
            gridspec = pgridspec._GridSpecFromSubplotSpec(
                nrows=3, ncols=1, hspace=0,
                subplot_spec=subplotspec,
                height_ratios=((1 - length) / 2, length, (1 - length) / 2),
            )
            subplotspec = gridspec[1]

        # Draw colorbar axes within this one
        self._hide_panel()
        with self.figure._context_authorize_add_subplot():
            ax = self.figure.add_subplot(subplotspec, projection='proplot_cartesian')  # noqa: E501
        self.add_child_axes(ax)

        # Location
        if side is None:  # manual
            orientation = kwargs.pop('orientation', None)
            if orientation == 'vertical':
                side = 'left'
            else:
                side = 'bottom'
        if side in ('bottom', 'top'):
            outside, inside = 'bottom', 'top'
            if side == 'top':
                outside, inside = inside, outside
            ticklocation = outside
            orientation = 'horizontal'
        else:
            outside, inside = 'left', 'right'
            if side == 'right':
                outside, inside = inside, outside
            ticklocation = outside
            orientation = 'vertical'

        # Update default keyword args
        iorientation = kwargs.get('orientation', None)
        if iorientation and iorientation != orientation:
            warnings._warn_proplot(f'Overriding input orientation={iorientation!r}.')
        ticklocation = _not_none(
            ticklocation=kwargs.pop('ticklocation', None),
            tickloc=kwargs.pop('tickloc', None),
            default=ticklocation,
        )
        kwargs.update({
            'orientation': orientation,
            'ticklocation': ticklocation
        })

        return ax, kwargs

    def _inset_colorbar_axes(
        self, loc=None, width=None, length=None, pad=None, frame=None, frameon=None,
        alpha=None, linewidth=None, edgecolor=None, facecolor=None, **kwargs
    ):
        """
        Return the axes and adjusted keyword args for an inset colorbar.
        """
        # Default properties
        cbwidth, cblength = width, length
        width, height = self.get_size_inches()
        frame = _not_none(frame=frame, frameon=frameon, default=rc['colorbar.frameon'])
        cbwidth = units(_not_none(cbwidth, rc['colorbar.insetwidth'])) / height
        cblength = units(_not_none(cblength, rc['colorbar.insetlength'])) / width
        extend = units(_not_none(kwargs.get('extendsize', None), rc['colorbar.insetextend']))  # noqa: E501
        pad = units(_not_none(pad, rc['colorbar.insetpad']))
        xpad, ypad = pad / width, pad / height

        # Get location in axes-relative coordinates
        # Bounds are x0, y0, width, height in axes-relative coordinates
        xspace = rc['xtick.major.size'] / 72
        if kwargs.get('label', None) or kwargs.get('title', None):
            xspace += 2.4 * rc['font.size'] / 72
        else:
            xspace += 1.2 * rc['font.size'] / 72
        xspace /= height  # space for labels
        if loc == 'upper right':
            ibounds = (1 - xpad - cblength, 1 - ypad - cbwidth)
            fbounds = (1 - 2 * xpad - cblength, 1 - 2 * ypad - cbwidth - xspace)
        elif loc == 'upper left':
            ibounds = (xpad, 1 - ypad - cbwidth)
            fbounds = (0, 1 - 2 * ypad - cbwidth - xspace)
        elif loc == 'lower left':
            ibounds = (xpad, ypad + xspace)
            fbounds = (0, 0)
        else:
            ibounds = (1 - xpad - cblength, ypad + xspace)
            fbounds = (1 - 2 * xpad - cblength, 0)
        ibounds = (*ibounds, cblength, cbwidth)  # inset axes
        fbounds = (*fbounds, 2 * xpad + cblength, 2 * ypad + cbwidth + xspace)  # frame

        # Make frame
        # NOTE: We do not allow shadow effects or fancy edges effect.
        # Also keep zorder same as with legend.
        if frame:
            xmin, ymin, width, height = fbounds
            patch = mpatches.Rectangle(
                (xmin, ymin), width, height,
                snap=True, zorder=4, transform=self.transAxes
            )
            alpha = _not_none(alpha, rc['colorbar.framealpha'])
            linewidth = _not_none(linewidth, rc['axes.linewidth'])
            edgecolor = _not_none(edgecolor, rc['axes.edgecolor'])
            facecolor = _not_none(facecolor, rc['axes.facecolor'])
            patch.update({
                'alpha': alpha,
                'linewidth': linewidth,
                'edgecolor': edgecolor,
                'facecolor': facecolor
            })
            self.add_artist(patch)

        # Make axes
        from .cartesian import CartesianAxes
        locator = self._make_inset_locator(ibounds, self.transAxes)
        bbox = locator(None, None)
        ax = CartesianAxes(self.figure, bbox.bounds, zorder=5)
        ax.set_axes_locator(locator)
        self.add_child_axes(ax)

        # Default keyword args
        orient = kwargs.pop('orientation', None)
        if orient is not None and orient != 'horizontal':
            warnings._warn_proplot(
                f'Orientation for inset colorbars must be horizontal, '
                f'ignoring orient={orient!r}.'
            )
        ticklocation = kwargs.pop('tickloc', None)
        ticklocation = kwargs.pop('ticklocation', None) or ticklocation
        if ticklocation is not None and ticklocation != 'bottom':
            warnings._warn_proplot('Inset colorbars can only have ticks on the bottom.')
        kwargs.update({'orientation': 'horizontal', 'ticklocation': 'bottom'})
        kwargs.setdefault('maxn', 5)
        kwargs.setdefault('extendsize', extend)
        kwargs.update({'edgecolor': edgecolor, 'linewidth': linewidth})  # cbar edge

        return ax, kwargs

    def _parse_colorbar_ticks(
        self, mappable, orientation='horizontal',
        format=None, formatter=None, ticklabels=None, formatter_kw=None,
        ticks=None, locator=None, locator_kw=None,
        minorticks=None, minorlocator=None, minorlocator_kw=None,
        maxn=None, maxn_minor=None, tickminor=False, fontsize=None, **kwargs,
    ):
        """
        Get the default locator for colorbar ticks.
        """
        locator = _not_none(ticks=ticks, locator=locator)
        minorlocator = _not_none(minorticks=minorticks, minorlocator=minorlocator)
        locator = _not_none(locator, getattr(mappable, '_colorbar_ticks', None))
        formatter = _not_none(ticklabels=ticklabels, formatter=formatter, format=format)

        # Get colorbar locator
        # NOTE: Do not necessarily want minor tick locations at logminor for LogNorm!
        # In _build_discrete_norm we sometimes select evenly spaced levels in log-space
        # *between* powers of 10, so logminor ticks would be misaligned with levels.
        if isinstance(locator, mticker.Locator):
            pass

        elif locator is None:
            # This should only happen if user calls plotting method on native mpl axes
            if isinstance(mappable.norm, mcolors.LogNorm):
                locator = 'log'
            elif isinstance(mappable.norm, mcolors.SymLogNorm):
                locator = 'symlog'
                locator_kw.setdefault('linthresh', mappable.norm.linthresh)
            else:
                locator = 'auto'

        else:
            # Get default maxn, try to allot 2em squares per label maybe?
            # NOTE: Cannot use Axes.get_size_inches because this is a
            # native matplotlib axes
            width, height = self.figure.get_size_inches()
            if orientation == 'horizontal':
                scale = 3  # em squares alotted for labels
                length = width * abs(self.get_position().width)
                fontsize = _not_none(fontsize, rc['xtick.labelsize'])
            else:
                scale = 1
                length = height * abs(self.get_position().height)
                fontsize = _not_none(fontsize, rc['ytick.labelsize'])
            fontsize = rc._scale_font(fontsize)
            maxn = _not_none(maxn, int(length / (scale * fontsize / 72)))
            maxn_minor = _not_none(maxn_minor, int(length / (0.5 * fontsize / 72)))

            # Get locator
            if tickminor and minorlocator is None:
                step = 1 + len(locator) // max(1, maxn_minor)
                minorlocator = locator[::step]
            step = 1 + len(locator) // max(1, maxn)
            locator = locator[::step]

        # Return locators
        locator = constructor.Locator(locator, **locator_kw)
        formatter = constructor.Formatter(formatter, **formatter_kw)
        minorlocator = constructor.Locator(minorlocator, **minorlocator_kw)
        return locator, formatter, minorlocator, kwargs

    def _parse_colorbar_mappable(
        self, mappable, values=None, *, orientation=None,
        locator=None, formatter=None, rotation=None, norm=None, norm_kw=None,
        **kwargs,
    ):
        """
        Generate a mappable from flexible non-mappable input. Useful in bridging
        the gap between legends and colorbars (e.g., creating colorbars from line
        objects whose data values span a natural colormap range).
        """
        # Special case where auto colorbar is generated from 1d methods, a list is
        # always passed, but some 1d methods (scatter) do have colormaps.
        if (
            np.iterable(mappable)
            and len(mappable) == 1
            and isinstance(mappable[0], mcm.ScalarMappable)
        ):
            mappable = mappable[0]
        if isinstance(mappable, mcm.ScarlarMappable):
            return mappable, locator, formatter, rotation, kwargs

        # For container objects, we just assume color is the same for every item.
        # Works for ErrorbarContainer, StemContainer, BarContainer.
        if (
            np.iterable(mappable)
            and len(mappable) > 0
            and all(isinstance(obj, mcontainer.Container) for obj in mappable)
        ):
            mappable = [obj[0] for obj in mappable]

        # A colormap instance
        # TODO: Pass remaining arguments through Colormap()? This is really
        # niche usage so maybe not necessary.
        orientation = _not_none(orientation, 'horizontal')
        if isinstance(mappable, mcolors.Colormap):
            # NOTE: 'Values' makes no sense if this is just a colormap. Just
            # use unique color for every segmentdata / colors color.
            cmap = mappable
            values = np.linspace(0, 1, cmap.N)

        # List of colors
        elif np.iterable(mappable) and all(
            isinstance(obj, str) or (np.iterable(obj) and len(obj) in (3, 4))
            for obj in mappable
        ):
            cmap = mcolors.ListedColormap(list(mappable), '_no_name')
            if values is None:
                values = np.arange(len(mappable))
            locator = _not_none(locator, values)  # tick *all* values by default

        # List of artists
        # NOTE: Do not check for isinstance(Artist) in case it is an mpl collection
        elif np.iterable(mappable) and all(
            hasattr(obj, 'get_color') or hasattr(obj, 'get_facecolor')
            for obj in mappable
        ):
            # Generate colormap from colors and infer tick labels
            colors = []
            for obj in mappable:
                if hasattr(obj, 'get_color'):
                    color = obj.get_color()
                else:
                    color = obj.get_facecolor()
                if isinstance(color, np.ndarray):
                    color = color.squeeze()  # e.g. scatter plot
                    if color.ndim != 1:
                        raise ValueError(
                            'Cannot make colorbar from list of artists '
                            f'with more than one color: {color!r}.'
                        )
                colors.append(to_rgb(color))

            # Try to infer tick values and tick labels from Artist labels
            cmap = mcolors.ListedColormap(colors, '_no_name')
            if values is None:
                # Get object labels and values
                labels = []
                values = []
                for obj in mappable:
                    label = self._get_label(obj)  # could be None
                    try:
                        value = float(label)  # could be float(None)
                    except (TypeError, ValueError):
                        value = None
                    labels.append(label)
                    values.append(value)

                # Use default values if labels are non-numeric (numeric labels are
                # common when making on-the-fly colorbars). Try to use object labels
                # for ticks with default vertical rotation, like datetime axes.
                if any(value is None for value in values):
                    values = np.arange(len(mappable))
                    if formatter is None and any(_ is not None for _ in labels):
                        formatter = labels  # use these fixed values for ticks
                        if orientation == 'horizontal':
                            rotation = _not_none(rotation, 90)
            locator = _not_none(locator, values)  # tick *all* values by default

        else:
            raise ValueError(
                'Input mappable must be a matplotlib artist, '
                'list of objects, list of colors, or colormap. '
                f'Got {mappable!r}.'
            )

        # Build ad hoc ScalarMappable object from colors
        if np.iterable(mappable) and len(values) != len(mappable):
            raise ValueError(
                f'Passed {len(values)} values, but only {len(mappable)} '
                f'objects or colors.'
            )
        norm, *_ = self._build_discrete_norm(
            cmap=cmap,
            norm=norm,
            norm_kw=norm_kw,
            extend='neither',
            values=values,
        )
        mappable = mcm.ScalarMappable(norm, cmap)

        return mappable, locator, formatter, rotation, kwargs

    @docstring.add_snippets
    def colorbar(
        self, mappable, values=None, *, loc=None, length=None, shrink=None, width=None,
        space=None, pad=None, queue=False, extend=None, extendsize=None,
        reverse=False, tickloc=None, ticklocation=None,
        tickdir=None, tickdirection=None, tickminor=None,
        grid=None, title=None, label=None, rotation=None,
        norm=None, norm_kw=None, orientation=None,
        edgecolor=None, linewidth=None,
        labelsize=None, labelweight=None, labelcolor=None,
        ticklabelsize=None, ticklabelweight=None, ticklabelcolor=None,
        **kwargs
    ):
        """
        Add an *inset* colorbar or *outer* colorbar along the outside edge of
        the axes.

        Parameters
        ----------
        %(axes.colorbar_args)s
        loc : str, optional
            The colorbar location. Default is :rc:`colorbar.loc`. The
            following location keys are valid:

            .. _colorbar_table:

            ==================  =======================================
            Location            Valid keys
            ==================  =======================================
            outer left          ``'left'``, ``'l'``
            outer right         ``'right'``, ``'r'``
            outer bottom        ``'bottom'``, ``'b'``
            outer top           ``'top'``, ``'t'``
            default inset       ``'best'``, ``'inset'``, ``'i'``, ``0``
            upper right inset   ``'upper right'``, ``'ur'``, ``1``
            upper left inset    ``'upper left'``, ``'ul'``, ``2``
            lower left inset    ``'lower left'``, ``'ll'``, ``3``
            lower right inset   ``'lower right'``, ``'lr'``, ``4``
            "filled"            ``'fill'``
            ==================  =======================================

        space : float or str, optional
            For outer colorbars only. The space between the colorbar and the
            main axes. Units are interpreted by `~proplot.utils.units`.
            When :rcraw:`tight` is ``True``, this is adjusted automatically.
            Otherwise, the default is :rc:`subplots.panelpad`.
        pad : float or str, optional
            For inset colorbars only. The space between the axes edge and the colorbar.
            Units are interpreted by `~proplot.utils.units`.
            Default is :rc:`colorbar.insetpad`.
        length : float or str, optional
            The colorbar length. For outer colorbars, units are relative to the
            axes width or height. Default is :rc:`colorbar.length`. For inset
            colorbars, units are interpreted by `~proplot.utils.units`. Default
            is :rc:`colorbar.insetlength`.
        shrink : float, optional
            Alias for `length`. This is included for consistency with
            `matplotlib.figure.Figure.colorbar`.
        width : float or str, optional
            The colorbar width. Units are interpreted by `~proplot.utils.units`.
            For outer colorbars, default is :rc:`colorbar.width`. For inset colorbars,
            default is :rc:`colorbar.insetwidth`.

        Other parameters
        ----------------
        %(axes.colorbar_kwargs)s

        See also
        --------
        proplot.figure.Figure.colorbar
        matplotlib.figure.Figure.colorbar
        """
        # TODO: Add option to pad the frame away from the axes edge
        # TODO: Get the 'best' inset colorbar location using the legend algorithm.
        # NOTE: There is a weird problem with colorbars when simultaneously
        # passing levels and norm object to a mappable; fixed by passing vmin/vmax
        # instead of levels. (see: https://stackoverflow.com/q/40116968/4970632).
        # NOTE: Often want levels instead of vmin/vmax, while simultaneously
        # using a Normalize (for example) to determine colors between the levels
        # (see: https://stackoverflow.com/q/42723538/4970632). Workaround makes
        # sure locators are in vmin/vmax range exclusively; cannot match values.
        length = _not_none(length=length, shrink=shrink)
        if loc != 'fill':
            loc = self._loc_translate(loc, 'colorbar')

        # Optionally add to queue
        if queue:
            obj = (mappable, values)
            kwargs.update({'width': width, 'length': length, 'space': space, 'pad': pad})  # noqa: E501
            self._add_colorbar_legend('colorbar', loc, obj, **kwargs)
            return

        # Generate panel
        if loc in ('left', 'right', 'top', 'bottom'):
            ax = self.panel_axes(loc, width=width, space=space, filled=True)
            obj = ax.colorbar(mappable, values, loc='fill', length=length, **kwargs)
            self._add_colorbar_legend('colorbar', loc, obj)
            return obj

        # Generate colorbar axes
        if loc == 'fill':
            ax, kwargs = self._fill_colorbar_axes(length=length, **kwargs)
        else:
            ax, kwargs = self._inset_colorbar_axes(loc=loc, width=width, length=length, pad=pad, **kwargs)  # noqa: E501

        # Parse input args
        norm_kw = norm_kw or {}
        label = _not_none(title=title, label=label)
        ticklocation = _not_none(tickloc=tickloc, ticklocation=ticklocation)
        tickdirection = _not_none(tickdir=tickdir, tickdirection=tickdirection)

        # Colorbar kwargs
        grid = _not_none(grid, rc['colorbar.grid'])
        orientation = _not_none(orientation, 'horizontal')
        kwargs.update({
            'cax': self,
            'use_gridspec': True,
            'orientation': orientation,
            'spacing': 'uniform',
        })
        kwargs.setdefault('drawedges', grid)

        # Test if we were given a mappable, or iterable of stuff. Note
        # Container and PolyCollection matplotlib classes are iterable.
        mappable, locator, formatter, rotation, kwargs = self._parse_colorbar_mappable(
            mappable, values, orientation=orientation,
        )

        # Define text property keyword args
        kw_label = {}
        for key, value in (
            ('size', labelsize),
            ('weight', labelweight),
            ('color', labelcolor),
        ):
            if value is not None:
                kw_label[key] = value
        kw_ticklabels = {}
        for key, value in (
            ('size', ticklabelsize),
            ('weight', ticklabelweight),
            ('color', ticklabelcolor),
            ('rotation', rotation),
        ):
            if value is not None:
                kw_ticklabels[key] = value

        # Try to get tick locations from *levels* or from *values* rather than
        # random points along the axis.
        locator, formatter, minorlocator, kwargs = self._parse_colorbar_ticks(
            mappable, fontsize=ticklabelsize, tickminor=tickminor, **kwargs,
        )

        # Get extend triangles in physical units
        width, height = self.figure.get_size_inches()
        if orientation == 'horizontal':
            scale = width * abs(self.get_position().width)
        else:
            scale = height * abs(self.get_position().height)
        extendsize = units(_not_none(extendsize, rc['colorbar.extend']))
        extendsize = extendsize / (scale - 2 * extendsize)

        # Draw the colorbar
        # NOTE: Set default formatter here because we optionally apply a FixedFormatter
        # using *labels* from handle input.
        extend = _not_none(extend, getattr(mappable, '_colorbar_extend', 'neither'))
        kwargs.update({
            'ticks': locator,
            'format': formatter,
            'ticklocation': ticklocation,
            'extendfrac': extendsize,
        })
        if isinstance(mappable, mcontour.ContourSet):
            mappable.extend = extend  # required in mpl >= 3.3, else optional
        else:
            kwargs['extend'] = extend
        cb = self.figure.colorbar(mappable, **kwargs)
        axis = self.xaxis if orientation == 'horizontal' else self.yaxis

        # The minor locator
        # TODO: Document the improved minor locator functionality!
        # NOTE: Colorbar._use_auto_colorbar_locator() is never True because we use
        # the custom DiscreteNorm normalizer. Colorbar._ticks() always called.
        if minorlocator is None:
            if tickminor:
                cb.minorticks_on()
            else:
                cb.minorticks_off()
        elif not hasattr(cb, '_ticker'):
            warnings._warn_proplot(
                'Matplotlib colorbar API has changed. '
                f'Cannot use custom minor tick locator {minorlocator!r}.'
            )
            cb.minorticks_on()  # at least turn them on
        else:
            # Set the minor ticks just like matplotlib internally sets the
            # major ticks. Private API is the only way!
            ticks, *_ = cb._ticker(minorlocator, mticker.NullFormatter())
            axis.set_ticks(ticks, minor=True)
            axis.set_ticklabels([], minor=True)

        # Label and tick label settings
        # WARNING: Must use colorbar set_label to set text, calling set_text on
        # the axis will do nothing!
        if label is not None:
            cb.set_label(label)
        axis.label.update(kw_label)
        for obj in axis.get_ticklabels():
            obj.update(kw_ticklabels)

        # Ticks consistent with rc settings and overrides
        s = axis.axis_name
        for which in ('minor', 'major'):
            kw = rc.category(s + 'tick.' + which)
            kw.pop('visible', None)
            if tickdirection:
                kw['direction'] = tickdirection
            if edgecolor:
                kw['color'] = edgecolor
            if linewidth:
                kw['width'] = linewidth
            axis.set_tick_params(which=which, **kw)
        axis.set_ticks_position(ticklocation)

        # Fix alpha-blending issues. Cannot set edgecolor to 'face' because blending
        # will occur, get colored lines instead of white ones. Need manual blending.
        # NOTE: For some reason cb solids uses listed colormap with always 1.0 alpha,
        # then alpha is applied after. See: https://stackoverflow.com/a/35672224/4970632
        cmap = cb.cmap
        blend = 'pcolormesh.snap' not in rc or not rc['pcolormesh.snap']
        if not cmap._isinit:
            cmap._init()
        if blend and any(cmap._lut[:-1, 3] < 1):
            warnings._warn_proplot(
                f'Using manual alpha-blending for {cmap.name!r} colorbar solids.'
            )
            # Generate "secret" copy of the colormap!
            lut = cmap._lut.copy()
            cmap = mcolors.Colormap('_no_name', N=cmap.N)
            cmap._isinit = True
            cmap._init = lambda: None
            # Manually fill lookup table with alpha-blended RGB colors!
            for i in range(lut.shape[0] - 1):
                alpha = lut[i, 3]
                lut[i, :3] = (1 - alpha) * 1 + alpha * lut[i, :3]  # blend *white*
                lut[i, 3] = 1
            cmap._lut = lut
            # Update colorbar
            cb.cmap = cmap
            cb.draw_all()

        # Fix colorbar outline
        kw_outline = {
            'edgecolor': _not_none(edgecolor, rc['axes.edgecolor']),
            'linewidth': _not_none(linewidth, rc['axes.linewidth']),
        }
        if cb.outline is not None:
            cb.outline.update(kw_outline)
        if cb.dividers is not None:
            cb.dividers.update(kw_outline)

        # *Never* rasterize because it causes misalignment with border lines
        if cb.solids:
            cb.solids.set_rasterized(False)
            cb.solids.set_linewidth(0.4)
            cb.solids.set_edgecolor('face')

        # Invert the axis if descending DiscreteNorm
        norm = mappable.norm
        if getattr(norm, '_descending', None):
            axis.set_inverted(True)
        if reverse:  # potentially double reverse, although that would be weird...
            axis.set_inverted(True)

        # Return after registering location
        self._add_colorbar_legend('colorbar', loc, cb)  # possibly replace another
        return obj

    def _legend_children(self, children):
        """
        Iterate recursively through `_children` attributes of various `HPacker`,
        `VPacker`, and `DrawingArea` classes.
        """
        for obj in children:
            if hasattr(obj, '_children'):
                yield from self._legend_children(obj._children)
            else:
                yield obj

    def _single_legend(self, pairs, ncol=None, order=None, **kwargs):
        """
        Draw an individual legend with support for changing legend-entries
        between column-major and row-major.
        """
        # Optionally change order
        # See: https://stackoverflow.com/q/10101141/4970632
        # Example: If 5 columns, but final row length 3, columns 0-2 have
        # N rows but 3-4 have N-1 rows.
        ncol = _not_none(ncol, 3)
        if order == 'C':
            nbase = len(pairs) // ncol + 1
            split = [pairs[i * ncol:(i + 1) * ncol] for i in range(nbase)]
            pairs = []
            nrows_max = len(split)  # max possible row count
            ncols_final = len(split[-1])  # columns in final row
            nrows = [nrows_max] * ncols_final + [nrows_max - 1] * (ncol - ncols_final)
            for col, nrow in enumerate(nrows):  # iterate through cols
                pairs.extend(split[row][col] for row in range(nrow))

        # Draw legend
        return mlegend.Legend(self, *zip(*pairs), ncol=ncol, **kwargs)

    def _multiple_legend(
        self, pairs, *, fontsize, loc=None, ncol=None, order=None, **kwargs
    ):
        """
        Draw "legend" with centered rows by creating separate legends for
        each row. The label spacing/border spacing will be exactly replicated.
        """
        # Message when overriding some properties
        legs = []
        overridden = []
        frameon = kwargs.pop('frameon', None)  # then add back later!
        for override in ('bbox_transform', 'bbox_to_anchor'):
            prop = kwargs.pop(override, None)
            if prop is not None:
                overridden.append(override)
        if ncol is not None:
            warnings._warn_proplot(
                'Detected list of *lists* of legend handles. '
                'Ignoring user input property "ncol".'
            )
        if overridden:
            warnings._warn_proplot(
                'Ignoring user input properties '
                + ', '.join(map(repr, overridden))
                + ' for centered-row legend.'
            )

        # Determine space we want sub-legend to occupy as fraction of height
        # NOTE: Empirical testing shows spacing fudge factor necessary to
        # exactly replicate the spacing of standard aligned legends.
        width, height = self.get_size_inches()
        spacing = kwargs.get('labelspacing', None) or rc['legend.labelspacing']
        if pairs:
            interval = 1 / len(pairs)  # split up axes
            interval = (((1 + spacing * 0.85) * fontsize) / 72) / height

        # Iterate and draw
        # NOTE: We confine possible bounding box in *y*-direction, but do not
        # confine it in *x*-direction. Matplotlib will automatically move
        # left-to-right if you request this.
        ymin = ymax = None
        if order == 'F':
            raise NotImplementedError(
                'When center=True, ProPlot vertically stacks successive '
                "single-row legends. Column-major (order='F') ordering "
                'is un-supported.'
            )
        loc = _not_none(loc, 'upper center')
        if not isinstance(loc, str):
            raise ValueError(
                f'Invalid location {loc!r} for legend with center=True. '
                'Must be a location *string*.'
            )
        elif loc == 'best':
            warnings._warn_proplot(
                'For centered-row legends, cannot use "best" location. '
                'Using "upper center" instead.'
            )

        # Iterate through sublists
        for i, ipairs in enumerate(pairs):
            if i == 1:
                title = kwargs.pop('title', None)
            if i >= 1 and title is not None:
                i += 1  # add extra space!

            # Legend position
            if 'upper' in loc:
                y1 = 1 - (i + 1) * interval
                y2 = 1 - i * interval
            elif 'lower' in loc:
                y1 = (len(pairs) + i - 2) * interval
                y2 = (len(pairs) + i - 1) * interval
            else:  # center
                y1 = 0.5 + interval * len(pairs) / 2 - (i + 1) * interval
                y2 = 0.5 + interval * len(pairs) / 2 - i * interval
            ymin = min(y1, _not_none(ymin, y1))
            ymax = max(y2, _not_none(ymax, y2))

            # Draw legend
            bbox = mtransforms.Bbox([[0, y1], [1, y2]])
            leg = mlegend.Legend(
                self, *zip(*ipairs), loc=loc, ncol=len(ipairs),
                bbox_transform=self.transAxes, bbox_to_anchor=bbox,
                frameon=False, **kwargs
            )
            legs.append(leg)

        # Simple cases
        if not frameon:
            return legs
        if len(legs) == 1:
            legs[0].set_frame_on(True)
            return legs

        # Draw manual fancy bounding box for un-aligned legend
        # WARNING: The matplotlib legendPatch transform is the default transform, i.e.
        # universal coordinates in points. Means we have to transform mutation scale
        # into transAxes sizes.
        # WARNING: Tempting to use legendPatch for everything but for some reason
        # coordinates are messed up. In some tests all coordinates were just result
        # of get window extent multiplied by 2 (???). Anyway actual box is found in
        # _legend_box attribute, which is accessed by get_window_extent.
        width, height = self.get_size_inches()
        renderer = self.figure._get_renderer()
        bboxs = [
            leg.get_window_extent(renderer).transformed(self.transAxes.inverted())
            for leg in legs
        ]
        xmin = min(bbox.xmin for bbox in bboxs)
        xmax = max(bbox.xmax for bbox in bboxs)
        ymin = min(bbox.ymin for bbox in bboxs)
        ymax = max(bbox.ymax for bbox in bboxs)
        fontsize = (fontsize / 72) / width  # axes relative units
        fontsize = renderer.points_to_pixels(fontsize)

        # Draw and format patch
        patch = mpatches.FancyBboxPatch(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            snap=True, zorder=4.5,
            mutation_scale=fontsize,
            transform=self.transAxes
        )
        if kwargs.get('fancybox', rc['legend.fancybox']):
            patch.set_boxstyle('round', pad=0, rounding_size=0.2)
        else:
            patch.set_boxstyle('square', pad=0)
        patch.set_clip_on(False)
        self.add_artist(patch)

        # Add shadow
        # TODO: This does not work, figure out
        if kwargs.get('shadow', rc['legend.shadow']):
            shadow = mpatches.Shadow(patch, 20, -20)
            self.add_artist(shadow)

        # Add patch to list
        return patch, *legs

    @docstring.add_snippets
    def legend(
        self, handles=None, labels=None, *,
        loc=None, width=None, space=None, queue=False,
        frame=None, frameon=None, ncol=None, ncols=None,
        center=None, order='C', label=None, title=None,
        fontsize=None, fontweight=None, fontcolor=None,
        **kwargs
    ):
        """
        Add an *inset* legend or *outer* legend along the edge of the axes.

        Parameters
        ----------
        %(axes.legend_args)s
        loc : int or str, optional
            The legend location. The following location keys are valid:

            .. _legend_table:

            ==================  =======================================
            Location            Valid keys
            ==================  =======================================
            outer left          ``'left'``, ``'l'``
            outer right         ``'right'``, ``'r'``
            outer bottom        ``'bottom'``, ``'b'``
            outer top           ``'top'``, ``'t'``
            "best" inset        ``'best'``, ``'inset'``, ``'i'``, ``0``
            upper right inset   ``'upper right'``, ``'ur'``, ``1``
            upper left inset    ``'upper left'``, ``'ul'``, ``2``
            lower left inset    ``'lower left'``, ``'ll'``, ``3``
            lower right inset   ``'lower right'``, ``'lr'``, ``4``
            center left inset   ``'center left'``, ``'cl'``, ``5``
            center right inset  ``'center right'``, ``'cr'``, ``6``
            lower center inset  ``'lower center'``, ``'lc'``, ``7``
            upper center inset  ``'upper center'``, ``'uc'``, ``8``
            center inset        ``'center'``, ``'c'``, ``9``
            "filled"            ``'fill'``
            ==================  =======================================

        queue : bool, optional
            If ``True`` and `loc` is the same as an existing legend, the `handles`
            and `labels` are added to a queue and this function returns ``None``.
            This is used to "update" the same legend with successive ``ax.legend(...)``
            calls. If ``False`` (the default) and `loc` is the same as an existing
            legend, this function returns a `~matplotlib.legend.Legend` instance
            and the old legend is removed from the axes.
        space : float or str, optional
            For outer legends only. The space between the axes and the legend
            box. Units are interpreted by `~proplot.utils.units`.
            When :rcraw:`tight` is ``True``, this is adjusted automatically.
            Otherwise, the default is :rc:`subplots.panelpad`.
        width : float or str, optional
            For outer legends only. The space allocated for the legend box.
            This does nothing if :rcraw:`tight` is ``True``. Units are
            interpreted by `~proplot.utils.units`.

        Other parameters
        ----------------
        %(axes.legend_kwargs)s

        See also
        --------
        proplot.figure.Figure.legend
        matplotlib.axes.Axes.legend
        """
        if loc != 'fill':
            loc = self._loc_translate(loc, 'legend')
        if isinstance(loc, np.ndarray):
            loc = loc.tolist()

        # Optionally add to queue
        if queue:
            obj = (handles, labels)
            kwargs.update({'width': width, 'space': space})
            self._add_colorbar_legend('legend', loc, obj, **kwargs)
            return

        # Generate panel
        if loc in ('left', 'right', 'top', 'bottom'):
            ax = self.panel_axes(loc, width=width, space=space, filled=True)
            obj = ax.legend(handles, labels, loc='fill', **kwargs)
            self._add_colorbar_legend('legend', loc, obj)  # add to *this* axes
            return obj

        # Adjust settings
        if loc == 'fill':
            # Try to make handles and stuff flush against the axes edge
            self._hide_panel()
            kwargs.setdefault('borderaxespad', 0)
            frameon = _not_none(kwargs.get('frame'), kwargs.get('frameon'), rc['legend.frameon'])  # noqa: E501
            if not frameon:
                kwargs.setdefault('borderpad', 0)
            # Adjust location
            side = self._panel_side
            if side in LOC_SIDES:
                loc = LOC_SIDES[side]
            else:
                raise ValueError(f'Invalid panel side {side!r}.')

        # Parse input args
        # TODO: Legend entries for colormap or scatterplot objects! Idea is we
        # pass a scatter plot or contourf or whatever, and legend is generated by
        # drawing patch rectangles or markers using data values and their
        # corresponding cmap colors! For scatterplots just test get_facecolor()
        # to see if it contains more than one color.
        # TODO: It is *also* often desirable to label a colormap object with
        # one data value. Maybe add a legend option for the *number of samples*
        # or the *sample points* when drawing legends for colormap objects.
        # Look into "legend handlers", might just want to add own handlers by
        # passing handler_map to legend() and get_legend_handles_labels().
        if order not in ('F', 'C'):
            raise ValueError(
                f'Invalid order {order!r}. Choose from '
                '"C" (row-major, default) and "F" (column-major).'
            )
        ncol = _not_none(ncols=ncols, ncol=ncol)
        title = _not_none(label=label, title=title)
        frameon = _not_none(frame=frame, frameon=frameon, default=rc['legend.frameon'])
        if handles is not None and not np.iterable(handles):  # e.g. a mappable object
            handles = [handles]
        if labels is not None and (not np.iterable(labels) or isinstance(labels, str)):
            labels = [labels]
        if title is not None:
            kwargs['title'] = title
        if frameon is not None:
            kwargs['frameon'] = frameon
        fontsize = kwargs.get('fontsize', None) or rc['legend.fontsize']
        if fontsize is None:
            pass
        elif fontsize in mfonts.font_scalings:
            kwargs['fontsize'] = rc._scale_font(fontsize)
        else:
            kwargs['fontsize'] = units(fontsize, 'pt')

        # Handle and text properties that are applied after-the-fact
        # NOTE: Set solid_capstyle to 'butt' so line does not extend past error bounds
        # shading in legend entry. This change is not noticable in other situations.
        kw_text = {}
        for key, value in (('color', fontcolor), ('weight', fontweight)):
            if value is not None:
                kw_text[key] = value
        kw_handle = _pop_props(kwargs, 'lines')
        kw_handle['solid_capstyle'] = 'butt'

        # Get axes for legend handle detection
        # TODO: Update this when no longer use "filled panels" for outer legends
        axs = [self]
        if self._panel_hidden:
            if self._panel_parent:  # axes panel
                axs = list(self._panel_parent._iter_axes(hidden=False, children=True))
            else:
                axs = list(self.figure._iter_axes(hidden=False, children=True))

        # Handle list of lists (centered row legends)
        # NOTE: Avoid very common plot() error where users draw individual lines
        # with plot() and add singleton tuples to a list of handles. If matplotlib
        # gets a list like this but gets no 'labels' argument, it raises error.
        list_of_lists = False
        if handles is not None:
            handles = [h[0] if isinstance(h, tuple) and len(h) == 1 else h for h in handles]  # noqa: E501
            list_of_lists = any(isinstance(h, (list, np.ndarray)) for h in handles)
        if list_of_lists:
            if any(not np.iterable(_) for _ in handles):
                raise ValueError(f'Invalid handles={handles!r}.')
            if not labels:
                labels = [None] * len(handles)
            elif not all(np.iterable(_) and not isinstance(_, str) for _ in labels):
                # e.g. handles=[obj1, [obj2, obj3]] requires labels=[lab1, [lab2, lab3]]
                raise ValueError(f'Invalid labels={labels!r} for handles={handles!r}.')

        # Parse handles and legends with native matplotlib parser
        if not list_of_lists:
            if isinstance(handles, np.ndarray):
                handles = handles.tolist()
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            handles, labels, *_ = mlegend._parse_legend_args(
                axs, handles=handles, labels=labels,
            )
            pairs = list(zip(handles, labels))
        else:
            pairs = []
            for ihandles, ilabels in zip(handles, labels):
                if isinstance(ihandles, np.ndarray):
                    ihandles = ihandles.tolist()
                if isinstance(ilabels, np.ndarray):
                    ilabels = ilabels.tolist()
                ihandles, ilabels, *_ = mlegend._parse_legend_args(
                    axs, handles=ihandles, labels=ilabels,
                )
                pairs.append(list(zip(ihandles, ilabels)))

        # Manage pairs in context of 'center' option
        center = _not_none(center, list_of_lists)
        if not center and list_of_lists:  # standardize format based on input
            list_of_lists = False  # no longer is list of lists
            pairs = [pair for ipairs in pairs for pair in ipairs]
        elif center and not list_of_lists:
            list_of_lists = True
            ncol = _not_none(ncol, 3)
            pairs = [pairs[i * ncol:(i + 1) * ncol] for i in range(len(pairs))]
            ncol = None
        if list_of_lists:  # remove empty lists, pops up in some examples
            pairs = [ipairs for ipairs in pairs if ipairs]

        # Bail if no pairs
        if not pairs:
            objs = [mlegend.Legend(self, [], [], loc=loc, ncol=ncol, **kwargs)]
        # Multiple-legend pseudo-legend
        elif center:
            objs = self._multiple_legend(pairs, loc=loc, ncol=ncol, order=order, **kwargs)  # noqa: E501
        # Individual legend
        else:
            objs = [self._single_legend(pairs, loc=loc, ncol=ncol, order=order, **kwargs)]  # noqa: E501

        # Add legends manually so matplotlib does not remove old ones
        for obj in objs:
            if isinstance(obj, mpatches.FancyBboxPatch):
                continue
            if hasattr(self, 'legend_') and self.legend_ is None:
                self.legend_ = obj  # set *first* legend accessible with get_legend()
            else:
                self.add_artist(obj)

        # Apply legend box properties
        outline = rc.fill({
            'linewidth': 'axes.linewidth',
            'edgecolor': 'axes.edgecolor',
            'facecolor': 'axes.facecolor',
            'alpha': 'legend.framealpha',
        })
        for key in (*outline,):
            if key != 'linewidth':
                if kwargs.get(key, None):
                    outline.pop(key, None)
        for obj in objs:
            if isinstance(obj, mpatches.FancyBboxPatch):
                obj.update(outline)  # the multiple-legend bounding box
            else:
                obj.legendPatch.update(outline)  # no-op if frame is off

        # Apply *overrides* to legend elements
        # WARNING: legendHandles only contains the *first* artist per legend because
        # HandlerBase.legend_artist() called in Legend._init_legend_box() only
        # returns the first artist. Instead we try to iterate through offset boxes.
        # TODO: Remove this feature? Idea was this lets users create *categorical*
        # legends in clunky way, e.g. entries denoting *colors* and entries denoting
        # *markers*. But would be better to add capacity for categorical labels in a
        # *single* legend like seaborn rather than multiple legends.
        for obj in objs:
            try:
                children = obj._legend_handle_box._children
            except AttributeError:  # older versions maybe?
                children = []
            for obj in self._legend_children(children):
                # Account for mixed legends, e.g. line on top of error bounds shading
                if isinstance(obj, mtext.Text):
                    obj.update(kw_text)
                else:
                    for key, value in kw_handle.items():
                        getattr(obj, 'set_' + key, lambda value: None)(value)

        # Return after registering location
        for obj in objs:
            obj.set_clip_on(False)  # critical for tight bounding box calcs
        if isinstance(objs[0], mpatches.FancyBboxPatch):
            objs = objs[1:]
        obj = objs[0] if len(objs) == 1 else tuple(objs)
        self._add_colorbar_legend('legend', loc, obj)  # possibly replace another
        return obj

    @staticmethod
    def _text_update(text, props=None, **kwargs):
        """
        Monkey patch that adds pseudo "border" and "bbox" properties to text
        objects without wrapping the entire class. Overrides update to
        facilitate updating inset titles.
        """
        props = props or {}
        props = props.copy()  # shallow copy
        props.update(kwargs)

        # Update border
        border = props.pop('border', None)
        bordercolor = props.pop('bordercolor', 'w')
        borderinvert = props.pop('borderinvert', False)
        borderwidth = props.pop('borderwidth', 2)
        if border:
            facecolor, bgcolor = text.get_color(), bordercolor
            if borderinvert:
                facecolor, bgcolor = bgcolor, facecolor
            kwargs = {
                'linewidth': borderwidth,
                'foreground': bgcolor,
                'joinstyle': 'miter',
            }
            text.update({
                'color': facecolor,
                'path_effects': [mpatheffects.Stroke(**kwargs), mpatheffects.Normal()],
            })
        elif border is False:
            text.update({
                'path_effects': None,
            })

        # Update bounding box
        # NOTE: We use '_title_pad' and '_title_above' for both titles and a-b-c labels
        # because always want to keep them aligned.
        # NOTE: For some reason using pad / 10 results in perfect alignment. Matplotlib
        # docs are vague about bounding box units, maybe they are tens of points?
        bbox = props.pop('bbox', None)
        bboxcolor = props.pop('bboxcolor', 'w')
        bboxstyle = props.pop('bboxstyle', 'round')
        bboxalpha = props.pop('bboxalpha', 0.5)
        bboxpad = _not_none(props.pop('bboxpad', None), text.axes._title_pad / 10)
        if isinstance(bbox, dict):  # *native* matplotlib usage
            props['bbox'] = bbox
        elif bbox:
            text.set_bbox({
                'edgecolor': 'black',
                'facecolor': bboxcolor,
                'boxstyle': bboxstyle,
                'alpha': bboxalpha,
                'pad': bboxpad,
            })
        elif bbox is False:
            text.set_bbox(None)  # disables the bbox

        return type(text).update(text, props)

    @docstring.concatenate
    def text(
        self, x=0, y=0, s='', transform='data', *,
        family=None, fontfamily=None, fontname=None, fontsize=None, size=None,
        border=False, bordercolor='w', borderwidth=2, borderinvert=False,
        bbox=False, bboxcolor='w', bboxstyle='round', bboxalpha=0.5, bboxpad=None,
        **kwargs
    ):
        """
        Support specifying the coordinate `tranform` with a string name and
        drawing white borders and bounding boxes around the text.

        Parameters
        ----------
        x, y : float
            The *x* and *y* coordinates for the text.
        s : str
            The text string.
        transform \
: {{'data', 'axes', 'figure'}} or `~matplotlib.transforms.Transform`, optional
            The transform used to interpret `x` and `y`. Can be a
            `~matplotlib.transforms.Transform` object or a string corresponding to
            `~matplotlib.axes.Axes.transData`, `~matplotlib.axes.Axes.transAxes`,
            or `~matplotlib.figure.Figure.transFigure` transforms. Default is
            ``'data'``, i.e. the text is positioned in data coordinates.
        fontsize, size : float or str, optional
            The font size. If float, units are inches. If string, units are
            interpreted by `~proplot.utils.units`.
        fontname, fontfamily, family : str, optional
            The font name (e.g. ``'Fira Math'``) or font family name (e.g. ``'serif'``).
            Matplotlib falls back to the system default if not found.
        fontweight, weight, fontstyle, style, fontvariant, variant : str, optional
            Additional font properties. See `~matplotlib.text.Text` for details.
        border : bool, optional
            Whether to draw border around text.
        borderwidth : float, optional
            The width of the text border. Default is ``2`` points.
        bordercolor : color-spec, optional
            The color of the text border. Default is ``'w'``.
        borderinvert : bool, optional
            If ``True``, the text and border colors are swapped.
        bbox : bool, optional
            Whether to draw a bounding box around text.
        bboxcolor : color-spec, optional
            The color of the text bounding box. Default is ``'w'``.
        bboxstyle : boxstyle, optional
            The style of the bounding box. Default is ``'round'``.
        bboxalpha : float, optional
            The alpha for the bounding box. Default is ``'0.5'``.
        bboxpad : float, optional
            The padding for the bounding box. Default is :rc:`title.bboxpad`.

        Other parameters
        ----------------
        **kwargs
            Passed to `~matplotlib.axes.Axes.text`.

        See also
        --------
        matplotlib.axes.Axes.text
        """
        # Parse input args
        # NOTE: Previously issued warning if fontname did not match any of names
        # in ttflist but this would result in warning for e.g. family='sans-serif'.
        # Matplotlib font API makes it very difficult to inject warning in
        # correct place. Simpler to just
        # NOTE: Do not emit warning if user supplied conflicting properties
        # because matplotlib has like 100 conflicting text properties for which
        # it doesn't emit warnings. Prefer not to fix all of them.
        fontsize = _not_none(fontsize, size)
        fontfamily = _not_none(fontname, fontfamily, family)
        if fontsize is not None:
            if fontsize in mfonts.font_scalings:
                fontsize = rc._scale_font(fontsize)
            else:
                fontsize = units(fontsize, 'pt')
            kwargs['fontsize'] = fontsize
        if fontfamily is not None:
            kwargs['fontfamily'] = fontfamily
        if not transform:
            transform = self.transData
        else:
            transform = self._get_transform(transform)

        # Apply monkey patch to text object
        # TODO: Why only support this here, and not in arbitrary places throughout
        # rest of matplotlib API? Units engine needs better implementation.
        obj = super().text(x, y, s, transform=transform, **kwargs)
        obj.update = self._text_update.__get__(obj)
        obj.update({
            'border': border,
            'bordercolor': bordercolor,
            'borderinvert': borderinvert,
            'borderwidth': borderwidth,
            'bbox': bbox,
            'bboxcolor': bboxcolor,
            'bboxstyle': bboxstyle,
            'bboxalpha': bboxalpha,
            'bboxpad': bboxpad,
        })
        return obj

    def _iter_axes(self, panels=None, hidden=False, children=False):
        """
        Return a list of axes and child panel axes.

        Parameters
        ----------
        panels : tuple, optional
            Tuple of panels to select.
        hidden : bool, optional
            Whether to include "hidden" panels.
        children : bool, optional
            Whether to include children.
        """
        panels = _not_none(panels, ('left', 'right', 'bottom', 'top'))
        if not set(panels) <= {'left', 'right', 'bottom', 'top'}:
            raise ValueError(f'Invalid sides {panels!r}.')
        for iax in (self, *(jax for side in panels for jax in self._panel_dict[side])):
            if not hidden and iax._panel_hidden:
                continue  # ignore hidden panel and its colorbar/legend child
            for jax in ((iax, *iax.child_axes) if children else (iax,)):
                if not jax.get_visible():
                    continue  # safety first
                yield jax

    @property
    def number(self):
        """
        The axes number. This controls the order of a-b-c labels and the
        order of appearence in the `~proplot.ui.SubplotsContainer` returned by
        `~proplot.ui.subplots`.
        """
        return self._number

    @number.setter
    def number(self, num):
        if num is not None and (not isinstance(num, Integral) or num < 1):
            raise ValueError(f'Invalid number {num!r}. Must be integer >=1.')
        self._number = num
