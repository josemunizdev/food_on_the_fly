"""Food on the Fly.

A food delivery prediction AI for estimated delivery times
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("food_on_the_fly")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "YesChef"
__email__ = "jose.muniz@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
