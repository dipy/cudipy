"""Vendored cupyx.ndimage internals

To avoid relying on private implementation details of CuPy, we vendor here
a copy of n-dimensional kernel code from cupyx.scipy.ndimage with only very
minimal modifications.

The key function in this folder is _generate_nd_kernel which is used
internally by most of CuPy's ndimage filtering and morphology functions.
"""
