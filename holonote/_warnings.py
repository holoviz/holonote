import inspect
import os
import warnings

import holoviews as hv
import param

__all__ = (
    "find_stack_level",
    "warn",
)


def warn(message, category=None, stacklevel=None):
    if stacklevel is None:
        stacklevel = find_stack_level()

    warnings.warn(message, category, stacklevel=stacklevel)


def find_stack_level():
    """
    Find the first place in the stack that is not inside Holoviews and Param.
    Inspired by: pandas.util._exceptions.find_stack_level
    """

    import holonote as hn

    pkg_dir = os.path.dirname(hn.__file__)
    test_dir = os.path.join(pkg_dir, "tests")
    hv_dir = os.path.dirname(hv.__file__)
    param_dir = os.path.dirname(param.__file__)

    try:
        frame = inspect.currentframe()
        stacklevel = 0
        while frame:
            fname = inspect.getfile(frame)
            if fname.startswith((pkg_dir, hv_dir, param_dir)) and not fname.startswith(test_dir):
                frame = frame.f_back
                stacklevel += 1
            else:
                break
    finally:
        del frame

    return stacklevel
