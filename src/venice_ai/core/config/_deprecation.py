"""
Deprecation support for configuration fields that no longer have an effect.

A deprecated field stays accepted so existing configurations keep loading.
Setting it to a value other than its default emits a :class:`FutureWarning`
(shown to end users by default) attributed to the caller's own code rather
than to pydantic or SDK internals.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

from pydantic import BaseModel


def _internal_prefixes() -> tuple[str, ...]:
    import pydantic

    import venice_ai

    prefixes = [
        os.path.dirname(pydantic.__file__) + os.sep,
        os.path.dirname(venice_ai.__file__) + os.sep,
    ]
    try:
        import pydantic_settings
    except ImportError:  # pragma: no cover - optional dependency
        pass
    else:
        prefixes.append(os.path.dirname(pydantic_settings.__file__) + os.sep)
    return tuple(prefixes)


def warn_user(message: str, category: type[Warning] = UserWarning) -> None:
    """Emit ``message`` attributed to the first stack frame outside the SDK and pydantic."""
    warnings.warn(message, category, skip_file_prefixes=_internal_prefixes())


def warn_on_deprecated_fields(model: BaseModel, names: Iterable[str]) -> None:
    """Warn for each deprecated field in ``names`` holding a non-default value.

    Values are read from the instance ``__dict__`` so that pydantic's own
    access-time deprecation warning is not triggered.
    """
    fields = type(model).model_fields
    for name in names:
        info = fields[name]
        value = model.__dict__.get(name)
        if value == info.get_default(call_default_factory=True):
            continue
        reason = info.deprecated if isinstance(info.deprecated, str) else "it has no effect"
        warn_user(
            f"{type(model).__name__}.{name} is deprecated and will be removed in the next "
            f"major release: {reason}",
            FutureWarning,
        )
