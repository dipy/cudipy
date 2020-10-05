from cupy import get_array_module


# May eventually replace this cupy.get_array_module with
# NumPy's module dispatch with NEP-37:
# https://numpy.org/neps/nep-0037-array-module.html
# Currently requires https://github.com/seberg/numpy-dispatch
# CuPy will have to implement the __array_module__ protocol for this to work

__all__ = ['get_array_module']
