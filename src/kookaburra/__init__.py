from . import flux
from . import data
from . import plot

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = 'unknown'
