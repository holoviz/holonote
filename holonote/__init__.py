import os

from . import annotate, editor

# Define '__version__'
try:
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", ".git")):
        # If setuptools_scm is installed (e.g. in a development environment with
        # an editable install), then use it to determine the version dynamically.
        from setuptools_scm import get_version

        # This will fail with LookupError if the package is not installed in
        # editable mode or if Git is not installed.
        __version__ = get_version(root="..", relative_to=__file__)
    else:
        raise FileNotFoundError
except Exception:
    # As a fallback, use the version that is hard-coded in the file.
    try:
        from ._version import __version__
    except ModuleNotFoundError:
        # The user is probably trying to run this without having installed
        # the package.
        __version__ = "0.0.0+unknown"

__all__ = ("__version__", "annotate", "editor")
