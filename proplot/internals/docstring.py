#!/usr/bin/env python3
"""
Utilities for modifying proplot docstrings.
"""
import inspect
import re

import matplotlib.axes as maxes
from matplotlib import rcParams

#: Dictionary of docstring snippets.
snippets = {}


def add_snippets(obj):
    """
    Decorator that dedents docstrings with `inspect.getdoc` and adds
    un-indented snippets from the global `snippets` dictionary. This function
    uses ``%(name)s`` substitution rather than `str.format` substitution so
    that the `snippets` keys can be invalid variable names.
    """
    parts = {key: value.strip() for key, value in snippets.items()}
    if isinstance(obj, str):
        obj %= parts
    else:
        obj.__doc__ = inspect.getdoc(obj)
        if obj.__doc__:
            obj.__doc__ %= parts
    return obj


def concatenate(func, prepend_summary=False):
    """
    Concatenate docstrings from a matplotlib axes method with a ProPlot axes
    method and obfuscate the call signature to avoid misleading users.

    Warning
    -------
    This is not yet used but will be in the future.
    """
    # NOTE: Originally had idea to use numpydoc.docscrape.NumpyDocString to
    # interpolate docstrings but *enormous* number of assupmtions would go into
    # this. And simple is better than complex.
    # Get matplotlib axes func
    name = func.__name__
    func_orig = getattr(maxes.Axes, name, None)
    doc = inspect.getdoc(func) or ''  # also dedents
    doc_orig = inspect.getdoc(func_orig)
    if not doc_orig:  # should never happen
        return func

    # Prepend summary
    if prepend_summary:
        regex = re.search(r'\.( | *\n|\Z)', doc_orig)
        if regex:
            doc = doc_orig[:regex.start() + 1] + '\n\n' + doc

    # Exit if this is documentation generated for website
    if rcParams['docstring.hardcopy']:
        func.__doc__ = doc
        return func

    # Obfuscate signature by converting to *args **kwargs. Note this does
    # not change behavior of function! Copy parameters from a dummy function
    # because I'm too lazy to figure out inspect.Parameters API
    # See: https://stackoverflow.com/a/33112180/4970632
    sig = inspect.signature(func)
    sig_obfuscated = inspect.signature(lambda *args, **kwargs: None)
    func.__signature__ = sig.replace(
        parameters=tuple(sig_obfuscated.parameters.values())
    )

    # Concatenate docstrings and copy summary
    # Make sure different sections are very visible
    pad = '=' * len(name)
    doc = f"""
===================================={pad}
proplot.axes.PlotAxes.{name} documentation
===================================={pad}
{doc}

==================================={pad}
matplotlib.axes.Axes.{name} documentation
==================================={pad}
{doc_orig}
"""
    func.__doc__ = inspect.cleandoc(doc)  # dedents and trims whitespace

    # Return
    return func
