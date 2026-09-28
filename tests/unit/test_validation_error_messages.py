"""Validation-error bodies must surface the server's per-field reasons.

Venice reports request-validation failures with a Zod-shaped envelope::

    {
      "error": "Invalid request parameters",
      "details": {"_errors": [...], "<field>": {"_errors": [...]}},
      "issues": [{"code": ..., "path": [...], "message": ...}, ...]
    }

The top-level ``error`` string is identical for every validation failure, so
the only actionable information is in ``issues`` / ``details``. These tests
feed bodies recorded from the live API through both exception constructors and
check that ``str(exc)`` names the offending field and the server's reason.

They also pin that the two transports agree: the JSON client path hands the
factory a parsed ``dict`` while the multipart path hands it the raw response
text, and a caller must not get a different message (or lose ``exc.code``)
depending on which endpoint raised.
"""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from venice_ai.exceptions import (
    APIError,
    APIStatusError,
    InvalidRequestError,
    UnprocessableEntityError,
    _make_status_error,
)

GENERIC = "Invalid request parameters"

# Bodies recorded from the live API (chat, models, image upscale), bodies in the
# same envelope carrying the reasons returned by video queue and embeddings, and
# the documented ``details``-only variant with empty ``issues``.
# Each entry: (id, body, substrings that must appear in str(exc)).
VALIDATION_BODIES: list[tuple[str, dict[str, Any], tuple[str, ...]]] = [
    (
        "chat-unrecognized-key",
        {
            "details": {"_errors": ["Unrecognized key(s) in object: 'logit_bias'"]},
            "error": GENERIC,
            "issues": [
                {
                    "code": "unrecognized_keys",
                    "keys": ["logit_bias"],
                    "path": [],
                    "message": "Unrecognized key(s) in object: 'logit_bias'",
                }
            ],
        },
        ("logit_bias",),
    ),
    (
        "models-invalid-union",
        {
            "details": {
                "_errors": [],
                "type": {
                    "_errors": [
                        "Invalid enum value. Expected 'asr' | 'embedding' | 'image' | "
                        "'music' | 'text' | 'tts' | 'upscale' | 'inpaint' | 'video', "
                        "received 'invalid_type_xyz'",
                        "Invalid enum value. Expected 'all' | 'code', received 'invalid_type_xyz'",
                    ]
                },
            },
            "error": GENERIC,
            "issues": [
                {
                    "code": "invalid_union",
                    "unionErrors": [
                        {
                            "issues": [
                                {
                                    "received": "invalid_type_xyz",
                                    "code": "invalid_enum_value",
                                    "options": [
                                        "asr",
                                        "embedding",
                                        "image",
                                        "music",
                                        "text",
                                        "tts",
                                        "upscale",
                                        "inpaint",
                                        "video",
                                    ],
                                    "path": ["type"],
                                    "message": (
                                        "Invalid enum value. Expected 'asr' | "
                                        "'embedding' | 'image' | 'music' | 'text' | "
                                        "'tts' | 'upscale' | 'inpaint' | 'video', "
                                        "received 'invalid_type_xyz'"
                                    ),
                                }
                            ],
                            "name": "ZodError",
                        },
                        {
                            "issues": [
                                {
                                    "received": "invalid_type_xyz",
                                    "code": "invalid_enum_value",
                                    "options": ["all", "code"],
                                    "path": ["type"],
                                    "message": (
                                        "Invalid enum value. Expected 'all' | 'code', "
                                        "received 'invalid_type_xyz'"
                                    ),
                                }
                            ],
                            "name": "ZodError",
                        },
                    ],
                    "path": ["type"],
                    "message": "Invalid input",
                }
            ],
        },
        ("type", "Invalid enum value"),
    ),
    (
        "image-upscale-corrupt-image",
        {
            "details": {
                "_errors": [],
                "image": {
                    "_errors": [
                        "Invalid or corrupt image. Must be a valid image format "
                        "(jpeg, png, webp, heif, heic, or avif) less than 25MB."
                    ]
                },
            },
            "error": GENERIC,
            "issues": [
                {
                    "code": "custom",
                    "message": (
                        "Invalid or corrupt image. Must be a valid image format "
                        "(jpeg, png, webp, heif, heic, or avif) less than 25MB."
                    ),
                    "path": ["image"],
                }
            ],
        },
        ("image", "Invalid or corrupt image"),
    ),
    (
        "video-audio-not-configurable",
        {
            "details": {
                "_errors": [],
                "audio": {"_errors": ["This model does not support audio configuration"]},
            },
            "error": GENERIC,
            "issues": [
                {
                    "code": "custom",
                    "message": "This model does not support audio configuration",
                    "path": ["audio"],
                }
            ],
        },
        ("audio", "does not support audio configuration"),
    ),
    (
        "embeddings-token-limit",
        {
            "details": {
                "_errors": [],
                "input": {"_errors": ["Input text exceeds the maximum token limit of 8192 tokens"]},
            },
            "error": GENERIC,
            "issues": [
                {
                    "code": "custom",
                    "message": "Input text exceeds the maximum token limit of 8192 tokens",
                    "path": ["input"],
                }
            ],
        },
        ("8192 tokens",),
    ),
    (
        "details-only-empty-issues",
        {
            "error": GENERIC,
            "details": {"_errors": [], "field": {"_errors": ["Field is required"]}},
            "issues": [],
        },
        ("field", "Field is required"),
    ),
]

# Statuses that route through the factory to a validation-style exception.
VALIDATION_STATUSES = (400, 413, 415, 422)

# Non-validation bodies carrying a top-level machine-readable ``code``.
CODED_BODIES: list[tuple[str, int, dict[str, Any]]] = [
    (
        "insufficient-balance",
        402,
        {
            "error": "Insufficient USD or Diem balance to complete request.",
            "code": "INSUFFICIENT_BALANCE",
        },
    ),
    (
        "invalid-model",
        400,
        {"error": "Invalid model", "code": "INVALID_MODEL"},
    ),
]


def _response(status: int) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.headers = {}
    return response


def _factory(body: Any, status: int) -> APIError:
    return _make_status_error(
        f"API request failed with status {status}",
        request=None,
        body=body,
        response=_response(status),
    )


# Fragments of the raw envelope. Their presence means the body was dumped into
# the message rather than rendered as field/reason text.
RAW_ENVELOPE_MARKERS = ("_errors", '"path"', '"code"', "unionErrors", '"issues"')


def _assert_names_field_and_reason(message: str, expected: tuple[str, ...]) -> None:
    assert GENERIC in message
    missing = [s for s in expected if s not in message]
    assert not missing, f"str(exc) omits {missing!r}; got {message!r}"
    leaked = [m for m in RAW_ENVELOPE_MARKERS if m in message]
    assert not leaked, f"str(exc) contains raw envelope fragments {leaked!r}; got {message!r}"


def _ids(rows: list[tuple[Any, ...]]) -> list[str]:
    return [row[0] for row in rows]


def test_recorded_body_set_is_non_empty_and_well_formed() -> None:
    assert len(VALIDATION_BODIES) > 0
    assert len(CODED_BODIES) > 0
    for _id, body, expected in VALIDATION_BODIES:
        assert body["error"] == GENERIC
        assert "issues" in body or "details" in body
        assert expected


@pytest.mark.parametrize("status", VALIDATION_STATUSES)
@pytest.mark.parametrize(
    ("case_id", "body", "expected"), VALIDATION_BODIES, ids=_ids(VALIDATION_BODIES)
)
def test_factory_message_names_field_and_reason(
    case_id: str, body: dict[str, Any], expected: tuple[str, ...], status: int
) -> None:
    original = copy.deepcopy(body)

    exc = _factory(body, status)

    assert isinstance(exc, (InvalidRequestError, UnprocessableEntityError))
    _assert_names_field_and_reason(str(exc), expected)
    assert exc.body == original


@pytest.mark.parametrize(
    ("case_id", "body", "expected"), VALIDATION_BODIES, ids=_ids(VALIDATION_BODIES)
)
def test_api_status_error_message_names_field_and_reason(
    case_id: str, body: dict[str, Any], expected: tuple[str, ...]
) -> None:
    original = copy.deepcopy(body)

    exc = APIStatusError(response=_response(400), body=body, request=None)

    _assert_names_field_and_reason(str(exc), expected)
    assert exc.body == original


@pytest.mark.parametrize("status", VALIDATION_STATUSES)
@pytest.mark.parametrize(
    ("case_id", "body", "expected"), VALIDATION_BODIES, ids=_ids(VALIDATION_BODIES)
)
def test_parsed_and_raw_text_validation_bodies_render_identically(
    case_id: str, body: dict[str, Any], expected: tuple[str, ...], status: int
) -> None:
    text = json.dumps(body, indent=2)
    from_dict = _factory(body, status)
    from_text = _factory(text, status)

    assert type(from_dict) is type(from_text)
    assert from_text.body == text
    assert str(from_text) == str(from_dict)
    assert from_text.code == from_dict.code


@pytest.mark.parametrize(("case_id", "status", "body"), CODED_BODIES, ids=_ids(CODED_BODIES))
def test_raw_text_body_keeps_top_level_code(
    case_id: str, status: int, body: dict[str, Any]
) -> None:
    from_dict = _factory(body, status)
    from_text = _factory(json.dumps(body), status)

    assert from_dict.code == body["code"]
    assert from_text.code == body["code"], (
        f"code lost when the body arrives as response text; str(exc)={str(from_text)!r}"
    )
    assert str(from_text) == str(from_dict)
