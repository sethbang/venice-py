"""Integration guard: each CLI command resolves its default model with the
CORRECT ``kind`` when ``--model`` is omitted.

The per-command unit tests use fixtures that supply a model key, so
``resolve_default_model`` short-circuits on config and the API-resolve path is
never driven through a command. These tests close that gap: they patch each
command module's ``resolve_default_model`` to capture the ``kind`` argument and
immediately halt (raising a sentinel before any network call), then assert the
command passed the expected ``kind``. A wrong-``kind`` wiring regression (e.g.
the video command passing ``"image"``) would fail here.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli


class _StopAfterResolve(Exception):
    """Raised by the fake resolver to halt the command before any API call."""


def _capturing_resolve(captured: dict):
    async def _fake(client, config, kind, explicit=None):
        captured["kind"] = kind
        raise _StopAfterResolve

    return _fake


@pytest.fixture
def runner():
    return CliRunner()


def _invoke_capture(
    runner, patch_target: str, argv: list[str], files: dict[Path, bytes] | None = None
) -> dict:
    """Invoke ``argv`` with ``patch_target`` (the fully-qualified
    ``resolve_default_model`` reference for this command) patched to capture the
    ``kind``. Returns the captured dict.

    Commands that import ``resolve_default_model`` at module top level must be
    patched on their own module; commands that import it inside the function
    body must be patched on the source module (``_model_defaults``).
    """
    captured: dict = {}
    env = {"VENICE_API_KEY": "test-key-1234567890"}
    for path, data in (files or {}).items():
        path.write_bytes(data)
    with patch(patch_target, _capturing_resolve(captured)):
        runner.invoke(cli, argv, env=env)
    return captured


# Source module — used by commands that import resolve_default_model in-function.
_SRC = "venice_ai.cli._model_defaults.resolve_default_model"


def test_embeddings_resolves_embedding_kind(runner):
    captured = _invoke_capture(runner, _SRC, ["embeddings", "hello world"])
    assert captured.get("kind") == "embedding"


def test_chat_resolves_chat_kind(runner):
    captured = _invoke_capture(
        runner, "venice_ai.cli.commands.chat.resolve_default_model", ["chat", "start", "hello"]
    )
    assert captured.get("kind") == "chat"


def test_audio_speak_resolves_tts_kind(runner):
    captured = _invoke_capture(runner, _SRC, ["audio", "speak", "hello"])
    assert captured.get("kind") == "tts"


def test_audio_transcribe_resolves_stt_kind(runner, tmp_path):
    sample = tmp_path / "sample.mp3"
    captured = _invoke_capture(
        runner,
        _SRC,
        ["audio", "transcribe", str(sample)],
        files={sample: b"\x00\x01\x02\x03"},
    )
    assert captured.get("kind") == "stt"


def test_video_generate_resolves_video_t2v_kind(runner):
    captured = _invoke_capture(runner, _SRC, ["video", "generate", "a cat surfing"])
    assert captured.get("kind") == "video_t2v"


def test_video_from_image_resolves_video_i2v_kind(runner, tmp_path):
    frame = tmp_path / "frame.png"
    captured = _invoke_capture(
        runner,
        _SRC,
        ["video", "from-image", str(frame)],
        files={frame: b"\x89PNG\r\n\x1a\n" + b"\x00" * 32},
    )
    assert captured.get("kind") == "video_i2v"


def test_image_generate_resolves_image_kind(runner):
    captured = _invoke_capture(
        runner,
        "venice_ai.cli.commands.image.generate.resolve_default_model",
        ["image", "generate", "a cat"],
    )
    assert captured.get("kind") == "image"
