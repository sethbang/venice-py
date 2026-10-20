"""
Sentinel type for distinguishing "not provided" from explicit ``None``.

Exports:
    * :class:`NotGivenType` — the sentinel class
    * :data:`NOT_GIVEN` — the singleton instance
    * :data:`NotGiven` — convenience type alias (``NotGivenType | None``)
"""

from __future__ import annotations

from .._compat import TypeAliasType


class NotGivenType:
    """
    Sentinel type for distinguishing between explicit None and missing parameters.

    This class implements the "sentinel object" pattern to differentiate between
    a parameter that was explicitly set to None versus a parameter that was not
    provided at all. This is particularly useful in API functions where None
    might be a valid value that differs from "not specified".

    The sentinel pattern is commonly used in libraries where:
    * None is a valid parameter value
    * You need to detect when a parameter wasn't provided
    * Default behavior differs from None behavior

    Example:
        >>> from venice_ai.utils import NOT_GIVEN, NotGiven
        >>>
        >>> def api_call(param: NotGiven[str] = NOT_GIVEN):
        ...     if param is NOT_GIVEN:
        ...         print("Parameter not provided")
        ...     elif param is None:
        ...         print("Parameter explicitly set to None")
        ...     else:
        ...         print(f"Parameter value: {param}")
        >>>
        >>> api_call()  # "Parameter not provided"
        >>> api_call(None)  # "Parameter explicitly set to None"
        >>> api_call("value")  # "Parameter value: value"
    """

    def __repr__(self) -> str:
        return "NOT_GIVEN"


#: Singleton instance of NotGivenType for parameter default values
NOT_GIVEN = NotGivenType()

#: Type alias for parameters that can be NotGiven, None, or a specific type
NotGiven = TypeAliasType("NotGiven", NotGivenType | None)
