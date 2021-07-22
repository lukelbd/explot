#!/usr/bin/env python3
"""
The axes classes used for all ProPlot figures.
"""
import matplotlib.projections as mproj

from .base import Axes  # noqa: F401
from .cartesian import CartesianAxes
from .geo import GeoAxes  # noqa: F401
from .geo import BasemapAxes, CartopyAxes
from .plot import PlotAxes  # noqa: F401
from .polar import PolarAxes
from .three import Axes3D  # noqa: F401

mproj.register_projection(CartesianAxes)
mproj.register_projection(PolarAxes)
mproj.register_projection(BasemapAxes)
mproj.register_projection(CartopyAxes)
mproj.register_projection(Axes3D)

# Prevent importing module names and set order of appearance for objects
__all__ = [
    'Axes',
    'PlotAxes',
    'CartesianAxes',
    'PolarAxes',
    'GeoAxes',
    'CartopyAxes',
    'BasemapAxes',
    'Axes3D',
]
