"""A command that did not do what it was asked exits 1, so a script can tell.

Each case runs the real command through ``CliRunner`` against a stubbed client.
An empty listing is not a failure, so those commands keep exiting 0.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner, Result

from venice_ai.cli.cli import cli
from venice_ai.exceptions import AuthenticationError
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoFailedStatus,
    VideoProcessingStatus,
)

QUEUED = SimpleNamespace(queue_id="job-1", model="video-model", download_url=None)
FAILED = VideoFailedStatus(status="FAILED", error="out of capacity")
NOTHING_TO_SAVE = VideoCompletedStatus(status="COMPLETED")
PROCESSING = VideoProcessingStatus(
    status="PROCESSING", average_execution_time=1, execution_duration=1
)
INLINE = VideoCompletedStatus(status="COMPLETED")
INLINE._set_data(b"VIDEO")


def _invoke(client: Any, args: list[str]) -> Result:
    with (
        patch("venice_ai.VeniceClient") as sdk_client,
        patch("venice_ai.cli.commands.models.command.VeniceClient") as models_client,
        patch("venice_ai.cli.commands.image.generate.VeniceClient") as generate_client,
        patch("venice_ai.cli.commands.image.wizard.VeniceClient") as wizard_client,
        patch("venice_ai.cli.commands.chat.VeniceClient") as chat_client,
        patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        for stub in (sdk_client, models_client, generate_client, wizard_client, chat_client):
            stub.return_value.__aenter__ = AsyncMock(return_value=client)
            stub.return_value.__aexit__ = AsyncMock(return_value=None)
        return CliRunner().invoke(cli, args)


def _video_client(status: Any) -> AsyncMock:
    client = AsyncMock()
    client.video.submit = AsyncMock(return_value=QUEUED)
    client.video.retrieve = AsyncMock(return_value=status)
    return client


def _video_commands(tmp_path: Path) -> dict[str, list[str]]:
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    out = str(tmp_path / "out.mp4")
    return {
        "generate": ["video", "generate", "a prompt", "-m", "video-model", "-o", out],
        "from-image": ["video", "from-image", str(image), "-m", "video-model", "-o", out],
    }


@pytest.mark.parametrize("command", ["generate", "from-image"])
@pytest.mark.parametrize("status", [FAILED, NOTHING_TO_SAVE], ids=["failed", "nothing-to-download"])
def test_video_command_exits_1_when_there_is_no_video(
    tmp_path: Path, command: str, status: Any
) -> None:
    result = _invoke(_video_client(status), _video_commands(tmp_path)[command])

    assert result.exit_code == 1, result.output
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.parametrize("command", ["generate", "from-image"])
def test_video_command_exits_1_when_polling_times_out(tmp_path: Path, command: str) -> None:
    with patch("venice_ai.cli.commands.video._poll_and_save", new=AsyncMock(return_value=False)):
        result = _invoke(_video_client(PROCESSING), _video_commands(tmp_path)[command])

    assert result.exit_code == 1, result.output


@pytest.mark.parametrize("command", ["generate", "from-image"])
def test_video_command_exits_0_when_the_video_is_saved(tmp_path: Path, command: str) -> None:
    result = _invoke(_video_client(INLINE), _video_commands(tmp_path)[command])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out.mp4").read_bytes() == b"VIDEO"


@pytest.mark.parametrize("command", ["generate", "from-image"])
def test_video_command_with_no_poll_exits_0(tmp_path: Path, command: str) -> None:
    args = [*_video_commands(tmp_path)[command], "--no-poll"]

    assert _invoke(_video_client(PROCESSING), args).exit_code == 0


@pytest.mark.parametrize(
    ("status", "output", "expected"),
    [
        (FAILED, True, 1),
        (NOTHING_TO_SAVE, True, 1),
        (PROCESSING, True, 0),
        (INLINE, True, 0),
        (FAILED, False, 0),
        (NOTHING_TO_SAVE, False, 0),
        (PROCESSING, False, 0),
    ],
    ids=[
        "failed-output",
        "nothing-output",
        "processing-output",
        "saved-output",
        "failed",
        "nothing",
        "processing",
    ],
)
def test_video_status_exits_1_only_when_a_requested_save_cannot_happen(
    tmp_path: Path, status: Any, output: bool, expected: int
) -> None:
    args = ["video", "status", "job-1", "-m", "video-model"]
    if output:
        args += ["-o", str(tmp_path / "out.mp4")]

    assert _invoke(_video_client(status), args).exit_code == expected


def _image_client(**create: Any) -> AsyncMock:
    client = AsyncMock()
    client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))
    client.image.list_styles = AsyncMock(return_value=SimpleNamespace(data=[]))
    client.image.create = AsyncMock(**create)
    return client


def test_image_generate_exits_1_when_no_image_comes_back(tmp_path: Path) -> None:
    client = _image_client(return_value=SimpleNamespace(images=[], timing=None))

    result = _invoke(
        client, ["image", "generate", "a prompt", "-m", "m", "--save-dir", str(tmp_path)]
    )

    assert result.exit_code == 1, result.output


PNG_BASE64 = "UE5H"
ONE_IMAGE = SimpleNamespace(images=[PNG_BASE64], timing=None)


@pytest.mark.parametrize("plain", [True, False], ids=["plain", "rich"])
@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ([ONE_IMAGE, ONE_IMAGE], 0),
        ([ONE_IMAGE, RuntimeError("boom")], 1),
        ([RuntimeError("boom"), RuntimeError("boom")], 1),
        ([ONE_IMAGE, SimpleNamespace(images=[], timing=None)], 1),
    ],
    ids=["all-saved", "one-raised", "all-raised", "one-empty"],
)
def test_image_batch_exits_1_when_any_prompt_fails(
    tmp_path: Path, answers: list[Any], expected: int, plain: bool
) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first\nsecond\n")
    args = ["image", "batch", "-f", str(prompts), "--save-dir", str(tmp_path)]
    if plain:
        args = ["--plain", *args]

    result = _invoke(_image_client(side_effect=answers), args)

    assert result.exit_code == expected, result.output
    assert "Successful" in result.output


@pytest.mark.parametrize(
    "models",
    [{"return_value": SimpleNamespace(data=[])}, {"side_effect": RuntimeError("boom")}],
    ids=["no-models", "fetch-failed"],
)
def test_image_wizard_exits_1_when_it_has_no_models(models: dict[str, Any]) -> None:
    client = AsyncMock()
    client.models.list = AsyncMock(**models)

    result = _invoke(client, ["image", "generate", "--interactive"])

    assert result.exit_code == 1, result.output


def test_chat_exits_1_when_the_conversation_to_continue_is_missing() -> None:
    client = AsyncMock()
    with patch("venice_ai.cli.commands.chat.load_conversation", return_value=None):
        result = _invoke(client, ["chat", "start", "--model", "m", "--continue-from", "missing"])

    assert result.exit_code == 1, result.output
    client.chat.completions.create.assert_not_called()


def test_chat_history_delete_exits_1_when_the_conversation_is_missing() -> None:
    with patch("venice_ai.cli.conversation.delete_conversation", return_value=False):
        result = _invoke(MagicMock(), ["chat", "history", "--delete", "missing"])

    assert result.exit_code == 1, result.output


def test_chat_history_delete_exits_0_when_it_deleted() -> None:
    with patch("venice_ai.cli.conversation.delete_conversation", return_value=True):
        result = _invoke(MagicMock(), ["chat", "history", "--delete", "there"])

    assert result.exit_code == 0, result.output


def _auth_error() -> AuthenticationError:
    return AuthenticationError(
        "API request failed with status 401: bad key", response=SimpleNamespace(status=401)
    )


def test_models_exits_1_and_says_why_when_every_listing_failed() -> None:
    client = AsyncMock()
    client.models.list = AsyncMock(side_effect=_auth_error())

    result = _invoke(client, ["models"])

    assert result.exit_code == 1, result.output
    assert "bad key" in result.output
    assert "No models found" not in result.output


def test_models_exits_0_when_the_listing_is_empty() -> None:
    client = AsyncMock()
    client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

    result = _invoke(client, ["models"])

    assert result.exit_code == 0, result.output
    assert "No models found" in result.output


TEXT_MODEL = SimpleNamespace(
    id="model-text",
    type="text",
    created=1,
    model_spec=SimpleNamespace(
        name="Text Model",
        traits=[],
        offline=False,
        beta=False,
        availableContextTokens=1,
        pricing=None,
        capabilities=None,
        constraints=None,
    ),
)


def _models_client(failing: set[str], listed: dict[str, list[Any]] | None = None) -> AsyncMock:
    async def list_models(type: str) -> SimpleNamespace:
        if type in failing:
            raise _auth_error()
        return SimpleNamespace(data=(listed or {}).get(type, []))

    client = AsyncMock()
    client.models.list = list_models
    return client


def test_models_exits_1_when_one_listing_failed_and_the_others_are_empty() -> None:
    result = _invoke(_models_client({"text"}), ["models"])

    assert result.exit_code == 1, result.output
    assert "bad key" in result.output
    assert "No models found" not in result.output


def test_models_exits_1_when_a_type_named_in_type_failed() -> None:
    client = _models_client({"video"}, {"text": [TEXT_MODEL]})

    result = _invoke(client, ["models", "--type", "video"])

    assert result.exit_code == 1, result.output
    # Rich wraps long lines at the terminal width.
    output = " ".join(result.output.split())
    assert "Could not list video models" in output
    assert "bad key" in output
    assert "No models match" not in output


def test_models_exits_1_when_one_of_several_named_types_failed() -> None:
    client = _models_client({"video"}, {"text": [TEXT_MODEL]})

    result = _invoke(client, ["models", "--type", "text", "--type", "video"])

    assert result.exit_code == 1, result.output
    assert "model-text" not in result.stdout


def test_models_shows_the_table_and_warns_on_stderr_for_another_type() -> None:
    client = _models_client({"video"}, {"text": [TEXT_MODEL]})

    result = _invoke(client, ["models"])

    assert result.exit_code == 0, result.output
    assert "model-text" in result.stdout
    assert "video" in result.stderr
    assert "bad key" in result.stderr
    assert "WARNING" not in result.stdout


def test_models_filtered_to_a_type_that_worked_warns_about_the_other() -> None:
    client = _models_client({"video"}, {"text": [TEXT_MODEL]})

    result = _invoke(client, ["models", "--type", "text"])

    assert result.exit_code == 0, result.output
    assert "model-text" in result.stdout
    assert "video" in result.stderr


def test_models_json_stays_on_stdout_alone_when_another_type_failed() -> None:
    client = _models_client({"video"})
    client.models.list = _models_client({"video"}, {"text": [TEXT_MODEL]}).models.list

    with patch(
        "venice_ai.cli.commands.models.command.ModelFormatter.format_json", return_value="[]"
    ):
        result = _invoke(client, ["models", "--json"])

    assert result.exit_code == 0, result.output
    assert "[]" in result.stdout
    assert "video" not in result.stdout
    assert "video" in result.stderr


def test_models_exits_1_when_filters_leave_nothing_and_a_listing_failed() -> None:
    client = _models_client({"video"}, {"text": [TEXT_MODEL]})

    result = _invoke(client, ["models", "--search", "no-such-model"])

    assert result.exit_code == 1, result.output
    assert "bad key" in result.output


def test_models_exits_0_when_a_named_type_worked_and_filters_leave_nothing() -> None:
    client = _models_client({"video"}, {"text": [TEXT_MODEL]})

    result = _invoke(client, ["models", "--type", "text", "--search", "no-such-model"])

    assert result.exit_code == 0, result.output
    assert "No models match" in result.output
    assert "video" in result.stderr


def test_models_exits_0_when_filters_leave_nothing_and_every_listing_worked() -> None:
    client = _models_client(set(), {"text": [TEXT_MODEL]})

    result = _invoke(client, ["models", "--search", "no-such-model"])

    assert result.exit_code == 0, result.output
    assert "No models match" in result.output


@pytest.mark.parametrize("answer", [{"success": False}, SimpleNamespace(success=False)])
def test_api_keys_delete_exits_1_when_the_server_says_it_did_not_delete(answer: Any) -> None:
    client = AsyncMock()
    client.api_keys.delete = AsyncMock(return_value=answer)

    result = _invoke(client, ["api-keys", "delete", "key-1", "--yes"])

    assert result.exit_code == 1, result.output
    assert "Failed to delete" in result.output


def test_image_generate_exits_1_before_spending_when_the_preset_is_unknown(tmp_path: Path) -> None:
    client = _image_client(return_value=ONE_IMAGE)
    args = ["image", "generate", "a prompt", "-m", "m", "--preset", "missing"]

    with patch("venice_ai.cli.commands.image.generate._load_preset_config", return_value=None):
        result = _invoke(client, [*args, "--save-dir", str(tmp_path)])

    assert result.exit_code == 1, result.output
    assert "missing" in result.output
    client.image.create.assert_not_called()
