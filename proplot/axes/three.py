#!/usr/bin/env python3
"""
The "3D" axes class.
"""
from ..config import rc
from . import plot

try:
    from mpl_toolkits.mplot3d import Axes3D as Axes3DBase
except ImportError:
    Axes3DBase = object


class Axes3D(plot.PlotAxes, Axes3DBase):
    """
    Simple mix-in of `proplot.axes.PlotAxes` with `~mpl_toolkits.mplot3d.Axes3D`.
    """
    #: The registered projection name.
    name = 'proplot_3d'

    def __init__(self, *args, **kwargs):
        # No additions for now
        import mpl_toolkits.mplot3d  # noqa: F401 verify package is available
        # Initialize axes
        super().__init__(*args, **kwargs)

    def format(self, **kwargs):
        # No additions for now
        rc_kw, rc_mode, kwargs = self._parse_format(**kwargs)
        with rc.context(rc_kw, mode=rc_mode):
            return super().format(**kwargs)
