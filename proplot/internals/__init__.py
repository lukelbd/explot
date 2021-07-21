#!/usr/bin/env python3
"""
Utilities used internally by proplot.
"""
import matplotlib

from . import docstring, rcsetup, timers, warnings  # noqa: F401

try:
    import cartopy
except ImportError:
    cartopy = None

try:  # print debugging
    from icecream import ic
except ImportError:  # graceful fallback if IceCream isn't installed
    ic = lambda *args: print(*args)  # noqa: E731


# Aliases. This package only works with a subset of available artists
# and keywords so we simply create our own system rather than working
# with matplotlib's normalize_kwargs and _alias_maps.
ALIASES = {
    'rgba': {
        'red': ('r',),
        'green': ('g',),
        'blue': ('b',),
        'alpha': ('a',),
    },
    'hsla': {
        'hue': ('h',),
        'saturation': ('s', 'c', 'chroma'),
        'luminance': ('l',),
        'alpha': ('a',),
    },
    'line': {  # copied from lines.py but expanded to include plurals
        'antialiased': ('aa',),
        'alpha': ('a', 'alphas'),
        'color': ('c', 'colors'),
        'linewidth': ('lw', 'linewidths'),
        'linestyle': ('ls', 'linestyles'),
        'drawstyle': ('ds', 'drawstyles'),
        'dashes': ('d',),
        'marker': ('m', 'markers'),
        'markersize': ('s', 'ms', 'markersizes'),
        'markeredgecolor': ('mec', 'markeredgecolors'),
        'markeredgewidth': ('mew', 'markeredgewidths'),
        'markerfacecolor': ('mfc', 'markerfacecolors'),
    },
    'lines': {
        'colors': ('c', 'color', 'ec', 'edgecolor', 'edgecolors'),
        'linewidths': ('lw', 'linewidth'),
        'linestyles': ('ls', 'linestyle'),
    },
}


def _getattr_flexible(obj, attr, default=None):
    """
    Search for attribute ``attr`` and ``_attr``. This crudely guards against
    upstream matplotlib changes.
    """
    if hasattr(obj, attr) and hasattr(obj, '_' + attr):
        warnings._warn_proplot(
            f"Object {obj!r} has both {attr!r} and {'_' + attr!r} attributes."
            'Using former.'
        )
    return getattr(obj, attr, getattr(obj, '_' + attr, default))


def _translate_props(input, output, *categories):
    """
    Driver function.
    """
    for category in categories:
        if category not in ALIASES:
            raise ValueError(f'Invalid alias category {category!r}.')
        for key, aliases in ALIASES[category].items():
            if isinstance(aliases, str):
                aliases = (aliases,)
            opts = {alias: input.pop(alias, None) for alias in (key, *aliases)}
            prop = _not_none(**opts)
            if prop is not None:
                output[key] = prop
    return output


def _translate_kwargs(input, output, *keys, **aliases):
    """
    Similar to `_pop_props` but instead aliases are passed by the user
    and property values are yielded in that order.
    """
    aliases.update({key: () for key in keys})
    for key, aliases in aliases.items():
        aliases = (aliases,) if isinstance(aliases, str) else aliases
        opts = {key: input.pop(key, None) for key in (key, *aliases)}
        output[key] = _not_none(**opts)
    return output


def _pop_kwargs(kwargs, *keys, **aliases):
    """
    Pop out input properties and return them in a new dictionary.
    """
    return _translate_kwargs(kwargs, {}, *keys, **aliases)


def _process_kwargs(kwargs, *keys, **aliases):
    """
    Translate input properties and add translated names to the original dictionary.
    """
    return _translate_kwargs(kwargs, kwargs, *keys, **aliases)


def _pop_props(kwargs, *categories):
    """
    Pop out registered properties and return them in a new dictionary.
    """
    return _translate_props(kwargs, {}, *categories)


def _process_props(kwargs, *categories):
    """
    Translate registered properties and add translated names to the original dictionary.
    """
    return _translate_props(kwargs, kwargs, *categories)


def _not_none(*args, default=None, **kwargs):
    """
    Return the first non-``None`` value. This is used with keyword arg aliases and
    for setting default values. Use `kwargs` to issue warnings when multiple
    non-``None`` values were passed. Use `args` to just ignore extra values.
    """
    first = default
    if args and kwargs:
        raise ValueError('_not_none can only be used with args or kwargs.')
    elif args:
        for arg in args:
            if arg is not None:
                first = arg
                break
    elif kwargs:
        for name, arg in list(kwargs.items()):
            if arg is not None:
                first = arg
                break
        kwargs = {name: arg for name, arg in kwargs.items() if arg is not None}
        if len(kwargs) > 1:
            warnings._warn_proplot(
                f'Got conflicting or duplicate keyword args: {kwargs}. '
                'Using the first one.'
            )
    return first


class _dummy_context(object):
    """
    Dummy context manager.
    """
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):  # noqa: U100
        pass


class _state_context(object):
    """
    Temporarily modify attribute(s) for an arbitrary object.
    """
    def __init__(self, obj, **kwargs):
        self._obj = obj
        self._attrs_new = kwargs
        self._attrs_prev = {
            key: getattr(obj, key) for key in kwargs if hasattr(obj, key)
        }

    def __enter__(self):
        for key, value in self._attrs_new.items():
            setattr(self._obj, key, value)

    def __exit__(self, *args):  # noqa: U100
        for key in self._attrs_new.keys():
            if key in self._attrs_prev:
                setattr(self._obj, key, self._attrs_prev[key])
            else:
                delattr(self._obj, key)


class _version(list):
    """
    Casual version parser for MAJOR.MINOR version strings. Do not want to add
    'packaging' dependency and only care about major and minor tags.
    """
    def __repr__(self):
        return f'version({self._version})'

    def __init__(self, version):
        try:
            major, minor, *_ = version.split('.')
            major, minor = int(major), int(minor)
        except (ValueError, AttributeError):
            warnings._warn_proplot(
                f"Invalid version {version!r}. Defaulting to '0.0'."
            )
            major = minor = 0
        self._version = version
        super().__init__([major, minor])  # then use builtin python list sorting


_version_mpl = _version(matplotlib.__version__)
if cartopy is None:
    _version_cartopy = [0, 0]
else:
    _version_cartopy = _version(cartopy.__version__)
