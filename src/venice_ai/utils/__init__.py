"""
Venice AI SDK Utility Functions
==============================

Utility categories:
    * **Sentinel types**: ``NotGivenType`` / ``NOT_GIVEN`` for optional params
    * **Model filtering**: ``get_filtered_models``
    * **Chat helpers**: ``build_model_id``
    * **Form helpers**: ``serialize_form_value``
    * **Parsing helpers**: ``safe_int`` / ``safe_float``
    * **Error wrapping**: ``wrap_aiohttp_errors``, ``map_aiohttp_error``, ``read_body``

All public symbols are re-exported here for backward compatibility so that
``from venice_ai.utils import <symbol>`` continues to work.
"""

from .errors import map_aiohttp_error, read_body, wrap_aiohttp_errors
from .form import serialize_form_value
from .models import _apply_model_filters, build_model_id, get_filtered_models
from .parsing import safe_float, safe_int
from .sentinel import NOT_GIVEN, NotGiven, NotGivenType

__all__ = [
    "NotGivenType",
    "NOT_GIVEN",
    "NotGiven",
    "serialize_form_value",
    "get_filtered_models",
    "_apply_model_filters",
    "build_model_id",
    "safe_int",
    "safe_float",
    "map_aiohttp_error",
    "read_body",
    "wrap_aiohttp_errors",
]
