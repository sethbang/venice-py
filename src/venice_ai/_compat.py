"""Typing constructs whose standard-library form needs a newer Python than the SDK's floor.

On Python 3.12+ these are the ``typing`` objects themselves, so
``isinstance(alias, typing.TypeAliasType)`` and ``typing.is_typeddict`` behave
as they would for code written against the standard library. On 3.11 the
``typing_extensions`` backports stand in: pydantic only accepts
``typing_extensions.TypedDict`` there, and ``typing.TypeAliasType`` does not exist.
"""

import sys

if sys.version_info >= (3, 12):
    from typing import TypeAliasType, TypedDict
else:
    from typing_extensions import TypeAliasType, TypedDict

__all__ = ["TypeAliasType", "TypedDict"]
