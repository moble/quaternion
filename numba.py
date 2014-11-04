from __future__ import division, print_function, absolute_import

## Allow the code to function without numba, but discourage it
## strongly.
try:
    from numbapro import njit, jit
    from numba.utils import IS_PY3
except ImportError:
    try:
        from numba import njit, jit
        from numba.utils import IS_PY3
    except ImportError:
        import warnings, sys
        warning_text = \
            "\n\n" + "!"*53 + "\n" + \
            "Could not import from either numbapro or numba.\n" + \
            "This means that the code will run MUCH more slowly.\n" + \
            "You probably REALLY want to install numba / numbapro." + \
            "\n" + "!"*53 + "\n"
        warnings.warn(warning_text)
        def _identity_decorator_outer(*args, **kwargs):
            def _identity_decorator_inner(fn):
                return fn
            return _identity_decorator_inner
        njit = _identity_decorator_outer
        jit = _identity_decorator_outer
        IS_PY3 = (sys.version_info[:2] >= (3, 0))
