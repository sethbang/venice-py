"""
Shared helpers for Venice AI chat examples.

- ``states_number`` and ``states_reading`` decide whether a model's final
  answer is grounded in what a tool returned: it states the tool's values in
  the form they were returned. That means the exact signed number (not one
  that merely contains its digits, and not the same digits with the opposite
  sign) and a temperature together with its unit. A bare substring test would
  accept "2024" for 24, "5480" for 548 or "-24°C" for 24°C.
"""

import re

# A temperature as the example tools format it: ``24°C``, ``-5°F``.
_READING = re.compile(r"(-?\d+)\s*°\s*([CF])")
_UNIT_NAMES = {"C": "celsius", "F": "fahrenheit"}
_UNIT_SIGNS = {"C": "℃", "F": "℉"}

# Characters written as a degree mark: the degree sign, the masculine ordinal
# indicator (a common stand-in on keyboards that lack °) and the ring above.
_DEGREE_MARKS = "°º˚"

# Ways an answer writes a negative sign: ASCII hyphen-minus or the Unicode
# minus sign (but not right after a digit, where ``18-24`` is a range or a
# subtraction), or the word "minus" followed by any whitespace.
_NEGATIVE = r"(?:(?<![\d.,])[-−]|\bminus\s+)"
_ENDS_WITH_NEGATIVE = re.compile(rf"{_NEGATIVE}\Z", re.IGNORECASE)


def _magnitude_pattern(value: float) -> str:
    """A regex for ``abs(value)`` as a standalone number, without its sign.

    For 548: ``548`` and ``548.0`` match; ``5480`` and ``1548`` do not.
    """
    magnitude = abs(value)
    text = str(int(magnitude)) if float(magnitude).is_integer() else str(magnitude)
    return rf"(?<![\d.,]){re.escape(text)}(?:\.0+)?(?![\d]|[.,]\d)"


def _states_signed(answer: str, value: float, suffix: str = "") -> bool:
    """True if ``answer`` states ``value`` with its sign, followed by ``suffix``.

    For -5: ``-5``, ``−5`` and ``minus 5`` match; ``5`` does not. For 5:
    ``5`` matches; ``-5``, ``−5``, ``minus 5`` and ``minus\n5`` do not. A sign
    character right after a digit (``18-24``) is a range or a subtraction, not
    a sign, so ``18-24`` states 24, not -24.
    """
    if value < 0:
        pattern = rf"{_NEGATIVE}{_magnitude_pattern(value)}{suffix}"
        return re.search(pattern, answer, re.IGNORECASE) is not None
    # A positive value is stated by any occurrence the text before does not
    # negate. The negative forms vary in width, so they are tested against the
    # preceding text rather than in a fixed-width lookbehind.
    pattern = rf"{_magnitude_pattern(value)}{suffix}"
    return any(
        _ENDS_WITH_NEGATIVE.search(answer, 0, match.start()) is None
        for match in re.finditer(pattern, answer, re.IGNORECASE)
    )


def states_number(answer: str, value: float) -> bool:
    """True if ``answer`` states ``value`` as a number of its own, sign included."""
    return _states_signed(answer, value)


def states_reading(answer: str, tool_output: str) -> bool:
    """True if ``answer`` states the temperature in ``tool_output`` with its unit.

    ``tool_output`` is what a ``get_weather`` tool returned (for example
    ``"Tokyo: 24°C, clear"``). For 24°C, ``24°C``, ``24 °C``, ``24ºC``,
    ``24℃``, ``24 C``, ``24 Celsius`` and ``24 degrees Celsius`` all count;
    ``24`` on its own, ``24 hours``, ``2024``, ``24°F`` or ``-24°C`` does not.
    """
    match = _READING.search(tool_output)
    if match is None:
        return False
    value, unit = int(match.group(1)), match.group(2)
    name, sign = _UNIT_NAMES[unit], _UNIT_SIGNS[unit]
    unit_word = rf"(?:{name}|{unit})\b"
    unit_forms = rf"[{_DEGREE_MARKS}]\s*{unit_word}|{sign}|(?:degrees?\s+)?{unit_word}"
    return _states_signed(answer, value, rf"\s*(?:{unit_forms})")
