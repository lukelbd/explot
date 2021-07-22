#!/usr/bin/env python3
"""
Polar axes using azimuth and radius instead of *x* and *y*.
"""
import matplotlib.projections as mproj
import matplotlib.ticker as mticker
import numpy as np

from .. import constructor
from .. import ticker as pticker
from ..config import rc
from ..internals import ic  # noqa: F401
from ..internals import _not_none, docstring, warnings
from . import plot

__all__ = ['PolarAxes']


class PolarAxes(plot.PlotAxes, mproj.PolarAxes):
    """
    Axes subclass for plotting in polar coordinates. Adds the `~PolarAxes.format`
    method and overrides several existing methods.
    """
    #: The registered projection name.
    name = 'proplot_polar'

    def __init__(self, *args, **kwargs):
        """
        See also
        --------
        proplot.ui.subplots
        proplot.axes.Axes
        proplot.axes.PlotAxes
        matplotlib.projections.PolarAxes
        """
        # Set tick length to zero so azimuthal labels are not too offset
        # Change default radial axis formatter but keep default theta one
        super().__init__(*args, **kwargs)
        formatter = pticker.AutoFormatter()
        self.yaxis.set_major_formatter(formatter)
        self.yaxis.isDefault_majfmt = True
        for axis in (self.xaxis, self.yaxis):
            axis.set_tick_params(which='both', size=0)

    @docstring.add_snippets
    def format(
        self, *args,
        r0=None, theta0=None, thetadir=None,
        thetamin=None, thetamax=None, thetalim=None,
        rmin=None, rmax=None, rlim=None,
        rlabelpos=None, rscale=None, rborder=None,
        thetalocator=None, rlocator=None, thetalines=None, rlines=None,
        thetaformatter=None, rformatter=None,
        thetalabels=None, rlabels=None,
        thetalocator_kw=None, rlocator_kw=None,
        thetaformatter_kw=None, rformatter_kw=None,
        patch_kw=None,
        **kwargs
    ):
        """
        Modify radial gridline locations, gridline labels, limits, and more.
        Unknown keyword arguments are passed to `Axes.format` and
        `~proplot.config.RcConfigurator.context`. All ``theta`` arguments are
        specified in *degrees*, not radians. The below parameters are specific
        to `PolarAxes`.

        Parameters
        ----------
        r0 : float, optional
            The radial origin.
        theta0 : {'N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'}
            The zero azimuth location.
        thetadir : {1, -1, 'anticlockwise', 'counterclockwise', 'clockwise'}, optional
            The positive azimuth direction. Clockwise corresponds to ``-1``
            and anticlockwise corresponds to ``1``. Default is ``1``.
        thetamin, thetamax : float, optional
            The lower and upper azimuthal bounds in degrees. If
            ``thetamax != thetamin + 360``, this produces a sector pplt.
        thetalim : (float, float), optional
            Specifies `thetamin` and `thetamax` at once.
        rmin, rmax : float, optional
            The inner and outer radial limits. If ``r0 != rmin``, this
            produces an annular pplt.
        rlim : (float, float), optional
            Specifies `rmin` and `rmax` at once.
        rborder : bool, optional
            Toggles the polar axes border on and off. Visibility of the "inner"
            radial spine and "start" and "end" azimuthal spines is controlled
            automatically be matplotlib.
        thetalocator, rlocator : float or list of float, optional
            Used to determine the azimuthal and radial gridline positions.
            Passed to the `~proplot.constructor.Locator` constructor. Can be
            float, list of float, string, or `matplotlib.ticker.Locator` instance.
        thetalines, rlines
            Aliases for `thetalocator`, `rlocator`.
        thetalocator_kw, rlocator_kw : dict-like, optional
            The azimuthal and radial locator settings. Passed to
            `~proplot.constructor.Locator`.
        rlabelpos : float, optional
            The azimuth at which radial coordinates are labeled.
        thetaformatter, rformatter : formatter spec, optional
            Used to determine the azimuthal and radial label format.
            Passed to the `~proplot.constructor.Formatter` constructor.
            Can be string, list of string, or `matplotlib.ticker.Formatter`
            instance. Use ``[]`` or ``'null'`` for no ticks.
        thetalabels, rlabels : optional
            Aliases for `thetaformatter`, `rformatter`.
        thetaformatter_kw, rformatter_kw : dict-like, optional
            The azimuthal and radial label formatter settings. Passed to
            `~proplot.constructor.Formatter`.
        %(axes.patch_kw)s

        Other parameters
        ----------------
        %(axes.format_other)s

        See also
        --------
        proplot.config.RcConfigurator.context
        proplot.axes.Axes.format
        """
        rc_kw, rc_mode, kwargs = self._parse_format(**kwargs)
        with rc.context(rc_kw, mode=rc_mode):
            # Background patch
            kw_face = rc.fill(
                {
                    'facecolor': 'axes.facecolor',
                    'alpha': 'axes.alpha'
                },
                context=True,
            )
            patch_kw = patch_kw or {}
            kw_face.update(patch_kw)
            self.patch.update(kw_face)

            # Not mutable default args
            thetalocator_kw = thetalocator_kw or {}
            thetaformatter_kw = thetaformatter_kw or {}
            rlocator_kw = rlocator_kw or {}
            rformatter_kw = rformatter_kw or {}

            # Flexible input
            if rlim is not None:
                if rmin is not None or rmax is not None:
                    warnings._warn_proplot(
                        f'Conflicting keyword args rmin={rmin}, rmax={rmax}, '
                        f'and rlim={rlim}. Using "rlim".'
                    )
                rmin, rmax = rlim
            if thetalim is not None:
                if thetamin is not None or thetamax is not None:
                    warnings._warn_proplot(
                        f'Conflicting keyword args thetamin={thetamin}, '
                        f'thetamax={thetamax}, and thetalim={thetalim}. '
                        f'Using "thetalim".'
                    )
                thetamin, thetamax = thetalim
            thetalocator = _not_none(
                thetalines=thetalines, thetalocator=thetalocator,
            )
            thetaformatter = _not_none(
                thetalabels=thetalabels, thetaformatter=thetaformatter,
            )
            rlocator = _not_none(
                rlines=rlines, rlocator=rlocator,
            )
            rformatter = _not_none(
                rlabels=rlabels, rformatter=rformatter,
            )

            # Special radius settings
            if r0 is not None:
                self.set_rorigin(r0)
            if rlabelpos is not None:
                self.set_rlabel_position(rlabelpos)
            if rscale is not None:
                self.set_rscale(rscale)
            if rborder is not None:
                self.spines['polar'].set_visible(bool(rborder))

            # Special azimuth settings
            if theta0 is not None:
                self.set_theta_zero_location(theta0)
            if thetadir is not None:
                self.set_theta_direction(thetadir)

            # Iterate
            for (
                x, r, axis,
                min_, max_,
                locator, formatter,
                locator_kw, formatter_kw,
            ) in zip(
                ('x', 'y'), ('theta', 'r'), (self.xaxis, self.yaxis),
                (thetamin, rmin), (thetamax, rmax),
                (thetalocator, rlocator), (thetaformatter, rformatter),
                (thetalocator_kw, rlocator_kw),
                (thetaformatter_kw, rformatter_kw)
            ):
                # Axis limits
                # Try to use public API where possible
                if min_ is not None:
                    getattr(self, 'set_' + r + 'min')(min_)
                else:
                    min_ = getattr(self, 'get_' + r + 'min')()
                if max_ is not None:
                    getattr(self, 'set_' + r + 'max')(max_)
                else:
                    max_ = getattr(self, 'get_' + r + 'max')()

                # Spine settings
                kw = rc.fill(
                    {
                        'linewidth': 'axes.linewidth',
                        'color': 'axes.edgecolor',
                    },
                    context=True,
                )
                sides = ('inner', 'polar') if r == 'r' else ('start', 'end')
                spines = [self.spines[side] for side in sides]
                for spine, side in zip(spines, sides):
                    spine.update(kw)

                # Grid and grid label settings
                # NOTE: Not sure if polar lines inherit tick or grid props
                kw = rc.fill(
                    {
                        'color': x + 'tick.color',
                        'labelcolor': 'tick.labelcolor',  # new props
                        'labelsize': 'tick.labelsize',
                        'grid_color': 'grid.color',
                        'grid_alpha': 'grid.alpha',
                        'grid_linewidth': 'grid.linewidth',
                        'grid_linestyle': 'grid.linestyle',
                    },
                    context=True,
                )
                axis.set_tick_params(which='both', **kw)
                # Label settings that can't be controlled with set_tick_params
                kw = rc.fill(
                    {
                        'fontfamily': 'font.family',
                        'weight': 'tick.labelweight'
                    },
                    context=True,
                )
                for t in axis.get_ticklabels():
                    t.update(kw)

                # Tick locator, which in this case applies to gridlines
                # NOTE: Must convert theta locator input to radians, then back
                # to degrees.
                if locator is not None:
                    if r == 'theta' and (
                            not isinstance(locator, (str, mticker.Locator))):
                        # real axis limts are rad
                        locator = np.deg2rad(locator)
                    locator = constructor.Locator(locator, **locator_kw)
                    locator.set_axis(axis)  # this is what set_locator does
                    grids = np.array(locator())
                    if r == 'r':
                        grids = grids[(grids >= min_) & (grids <= max_)]
                        self.set_rgrids(grids)
                    else:
                        grids = np.rad2deg(grids)
                        grids = grids[(grids >= min_) & (grids <= max_)]
                        if grids[-1] == min_ + 360:  # exclusive if 360 degrees
                            grids = grids[:-1]
                        self.set_thetagrids(grids)
                # Tick formatter and toggling
                if formatter is not None:
                    formatter = constructor.Formatter(formatter, **formatter_kw)  # noqa: E501
                    axis.set_major_formatter(formatter)

            # Parent method
            super().format(*args, **kwargs)
