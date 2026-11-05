"""
Tests for cli/commands/video.py

Covers:
- video() click group command
- _image_file_to_data_uri() - reads image file, base64-encodes to data URI
- _poll_and_save() - polls for video completion, all branches
- _determine_output_path() - resolves output filename
- generate() click command entrypoint
- _generate_async() - text-to-video logic
- from_image() click command entrypoint
- _from_image_async() - image-to-video logic
- status() click command entrypoint
- _status_async() - check video job status
"""

import base64
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from click.testing import CliRunner

from venice_ai import RetryOptions, VeniceClient
from venice_ai.cli.commands.video import (
    _determine_output_path,
    _from_image_async,
    _generate_async,
    _image_file_to_data_uri,
    _poll_and_save,
    _status_async,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(plain: bool = False):
    """Return a minimal mock Click context."""
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": plain}
    return ctx


def _setup_client_patch(MockVeniceClient, mock_client):
    """Configure MockVeniceClient to act as async context manager."""
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


def _make_queue_response(queue_id="job-123", model="wan-2.6-text-to-video"):
    """Create a fake queue response object."""
    return SimpleNamespace(queue_id=queue_id, model=model)


def _make_status_completed_data(data=b"fake video bytes"):
    """Create a fake status response with inline binary data."""
    return SimpleNamespace(status="COMPLETED", data=data, url=None)


def _make_status_completed_no_data():
    """Create a fake status response with neither data nor URL."""
    return SimpleNamespace(status="COMPLETED", data=None, url=None)


def _make_status_processing(progress=50, remaining_ms=30000):
    """Create a fake status response in PROCESSING state."""
    return SimpleNamespace(
        status="PROCESSING",
        progress_percent=progress,
        estimated_remaining_ms=remaining_ms,
    )


def _make_status_processing_no_remaining(progress=25):
    """Create a fake status response in PROCESSING state without remaining_ms."""
    return SimpleNamespace(
        status="PROCESSING",
        progress_percent=progress,
        estimated_remaining_ms=None,
    )


def _make_status_failed(error="Generation failed"):
    """Create a fake status response in FAILED state."""
    return SimpleNamespace(status="FAILED", error=error)


# ---------------------------------------------------------------------------
# video() group tests
# ---------------------------------------------------------------------------


class TestVideoGroup:
    """Tests for the video() click group."""

    def test_video_group_help(self):
        """Check --help output for the video group."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["video", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output or "from-image" in result.output

    def test_video_group_shows_subcommands(self):
        """video group lists generate, from-image, status subcommands."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["video", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "from-image" in result.output
        assert "status" in result.output


# ---------------------------------------------------------------------------
# _image_file_to_data_uri tests
# ---------------------------------------------------------------------------


class TestImageFileToDataUri:
    """Tests for _image_file_to_data_uri()."""

    @pytest.mark.asyncio
    async def test_png_file(self, tmp_path):
        """PNG file returns data:image/png;base64,... URI."""
        img_file = tmp_path / "test.png"
        img_data = b"fake png data"
        img_file.write_bytes(img_data)

        result = await _image_file_to_data_uri(str(img_file))
        b64 = base64.b64encode(img_data).decode("ascii")
        assert result == f"data:image/png;base64,{b64}"

    @pytest.mark.asyncio
    async def test_jpg_file(self, tmp_path):
        """JPG file returns data:image/jpeg;base64,... URI."""
        img_file = tmp_path / "test.jpg"
        img_data = b"fake jpg data"
        img_file.write_bytes(img_data)

        result = await _image_file_to_data_uri(str(img_file))
        b64 = base64.b64encode(img_data).decode("ascii")
        assert result == f"data:image/jpeg;base64,{b64}"

    @pytest.mark.asyncio
    async def test_jpeg_file(self, tmp_path):
        """JPEG file returns data:image/jpeg;base64,... URI."""
        img_file = tmp_path / "test.jpeg"
        img_data = b"fake jpeg data"
        img_file.write_bytes(img_data)

        result = await _image_file_to_data_uri(str(img_file))
        b64 = base64.b64encode(img_data).decode("ascii")
        assert result == f"data:image/jpeg;base64,{b64}"

    @pytest.mark.asyncio
    async def test_gif_file(self, tmp_path):
        """GIF file returns data:image/gif;base64,... URI."""
        img_file = tmp_path / "test.gif"
        img_data = b"fake gif data"
        img_file.write_bytes(img_data)

        result = await _image_file_to_data_uri(str(img_file))
        b64 = base64.b64encode(img_data).decode("ascii")
        assert result == f"data:image/gif;base64,{b64}"

    @pytest.mark.asyncio
    async def test_webp_file(self, tmp_path):
        """WebP file returns data:image/webp;base64,... URI."""
        img_file = tmp_path / "test.webp"
        img_data = b"fake webp data"
        img_file.write_bytes(img_data)

        result = await _image_file_to_data_uri(str(img_file))
        b64 = base64.b64encode(img_data).decode("ascii")
        assert result == f"data:image/webp;base64,{b64}"

    @pytest.mark.asyncio
    async def test_unknown_extension_defaults_to_jpeg(self, tmp_path):
        """Unknown extension defaults to image/jpeg MIME type."""
        img_file = tmp_path / "test.bmp"
        img_data = b"fake bmp data"
        img_file.write_bytes(img_data)

        result = await _image_file_to_data_uri(str(img_file))
        b64 = base64.b64encode(img_data).decode("ascii")
        assert result == f"data:image/jpeg;base64,{b64}"

    @pytest.mark.asyncio
    async def test_uppercase_extension(self, tmp_path):
        """Uppercase extension is normalized to lowercase for MIME type lookup."""
        img_file = tmp_path / "test.PNG"
        img_data = b"fake png data"
        img_file.write_bytes(img_data)

        result = await _image_file_to_data_uri(str(img_file))
        b64 = base64.b64encode(img_data).decode("ascii")
        assert result == f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# _determine_output_path tests
# ---------------------------------------------------------------------------


class TestDetermineOutputPath:
    """Tests for _determine_output_path()."""

    def test_no_output_uses_default(self):
        """No output specified → default video_<timestamp>.mp4 filename."""
        result = _determine_output_path(None, "/tmp/videos")
        assert result.startswith("/tmp/videos/video_")
        assert result.endswith(".mp4")

    def test_output_with_extension(self):
        """Output path with extension is kept as-is when it has dir separators."""
        result = _determine_output_path("/output/myvideo.mp4", "/tmp/videos")
        assert result == "/output/myvideo.mp4"

    def test_output_without_extension_adds_mp4(self):
        """Output path without extension gets .mp4 added."""
        result = _determine_output_path("myvideo", "/tmp/videos")
        assert result == "/tmp/videos/myvideo.mp4"

    def test_output_bare_filename_put_in_save_dir(self):
        """Bare filename (no dir) is placed in save_dir."""
        result = _determine_output_path("myvideo.mp4", "/tmp/videos")
        assert result == "/tmp/videos/myvideo.mp4"

    def test_output_with_custom_ext(self):
        """Custom ext parameter changes default extension."""
        result = _determine_output_path(None, "/tmp/videos", ext="webm")
        assert result.endswith(".webm")

    def test_output_path_with_subdirs(self):
        """Output path with subdirectories is returned as absolute path."""
        result = _determine_output_path("subdir/myvideo.mp4", "/tmp/videos")
        assert result == "subdir/myvideo.mp4"

    def test_output_path_with_directory_separator(self):
        """Output path with / separator is treated as absolute."""
        result = _determine_output_path("/some/path/myvideo", "/tmp")
        assert result == "/some/path/myvideo.mp4"


# ---------------------------------------------------------------------------
# _poll_and_save tests
# ---------------------------------------------------------------------------


class TestPollAndSave:
    """Tests for _poll_and_save()."""

    @pytest.mark.asyncio
    async def test_completed_with_data_plain(self, tmp_path):
        """COMPLETED with inline data: saves file and returns True (plain mode)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        video_data = b"fake video data content"
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=video_data)
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=True,
            )

        assert result is True
        assert tmp_path.joinpath("video.mp4").read_bytes() == video_data
        mock_om.success.assert_called_once()
        assert "saved" in mock_om.success.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_completed_with_data_rich(self, tmp_path):
        """COMPLETED with inline data: saves file and returns True (rich mode)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        video_data = b"fake video data content"
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=video_data)
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=False,
            )

        assert result is True
        mock_om.success.assert_called_once()
        assert "Video saved" in mock_om.success.call_args[0][0]

    @pytest.mark.asyncio
    async def test_completed_no_data_no_url_plain(self, tmp_path):
        """COMPLETED with neither data nor URL: logs warning, returns False (plain)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(return_value=_make_status_completed_no_data())

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=True,
            )

        assert result is False
        mock_om.warning.assert_called_once()
        assert "no data" in mock_om.warning.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_completed_no_data_no_url_rich(self, tmp_path):
        """COMPLETED with neither data nor URL: logs warning, returns False (rich)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(return_value=_make_status_completed_no_data())

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=False,
            )

        assert result is False
        mock_om.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_status_plain(self, tmp_path):
        """FAILED status: logs error, returns False (plain)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(return_value=_make_status_failed("Bad prompt"))

        with patch("venice_ai.cli.commands.video.OutputManager") as mock_om:
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=True,
            )

        assert result is False
        mock_om.error.assert_called_once()
        assert "Bad prompt" in mock_om.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_failed_status_rich(self, tmp_path):
        """FAILED status: logs error, returns False (rich)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(return_value=_make_status_failed("Bad prompt"))

        with patch("venice_ai.cli.commands.video.OutputManager") as mock_om:
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=False,
            )

        assert result is False
        mock_om.error.assert_called_once()
        assert "Bad prompt" in mock_om.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_failed_status_no_error_attr(self, tmp_path):
        """FAILED status with no error attribute uses 'unknown error'."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(return_value=SimpleNamespace(status="FAILED"))

        with patch("venice_ai.cli.commands.video.OutputManager") as mock_om:
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=True,
            )

        assert result is False
        assert "unknown error" in mock_om.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_processing_with_remaining_plain(self, tmp_path):
        """PROCESSING with remaining_ms: shows progress (plain)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        # First call returns PROCESSING, second returns COMPLETED with data
        mock_client.video.retrieve = AsyncMock(
            side_effect=[
                _make_status_processing(progress=50, remaining_ms=30000),
                _make_status_completed_data(data=b"video"),
            ]
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=True,
                poll_interval=0.01,
            )

        assert result is True
        mock_om.progress.assert_called()
        progress_calls = "".join(str(c) for c in mock_om.progress.call_args_list)
        assert "30" in progress_calls  # ~30s remaining
        # pct=50 passed as keyword arg
        assert mock_om.progress.call_args_list[0][1]["pct"] == 50

    @pytest.mark.asyncio
    async def test_processing_without_remaining_plain(self, tmp_path):
        """PROCESSING without remaining_ms: shows progress without time (plain)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            side_effect=[
                _make_status_processing_no_remaining(progress=25),
                _make_status_completed_data(data=b"video"),
            ]
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=True,
                poll_interval=0.01,
            )

        assert result is True
        mock_om.progress.assert_called()
        assert mock_om.progress.call_args_list[0][1]["pct"] == 25

    @pytest.mark.asyncio
    async def test_processing_with_remaining_rich(self, tmp_path):
        """PROCESSING with remaining_ms: shows progress (rich)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            side_effect=[
                _make_status_processing(progress=75, remaining_ms=15000),
                _make_status_completed_data(data=b"video"),
            ]
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=False,
                poll_interval=0.01,
            )

        assert result is True
        mock_om.progress.assert_called()
        progress_msg = mock_om.progress.call_args_list[0][0][0]
        assert "15" in progress_msg  # ~15s remaining
        assert mock_om.progress.call_args_list[0][1]["pct"] == 75

    @pytest.mark.asyncio
    async def test_processing_without_remaining_rich(self, tmp_path):
        """PROCESSING without remaining_ms: shows progress (rich)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            side_effect=[
                _make_status_processing_no_remaining(progress=10),
                _make_status_completed_data(data=b"video"),
            ]
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=False,
                poll_interval=0.01,
            )

        assert result is True
        mock_om.progress.assert_called()
        assert mock_om.progress.call_args_list[0][1]["pct"] == 10

    @pytest.mark.asyncio
    async def test_timeout_plain(self, tmp_path):
        """Timeout after max_polls: returns False and logs timeout (plain)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        # Always returns PROCESSING — will timeout
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_processing(progress=10, remaining_ms=5000)
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=True,
                poll_interval=0.01,
                max_polls=2,
            )

        assert result is False
        mock_om.warning.assert_called()
        assert "Timed out" in mock_om.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_timeout_rich(self, tmp_path):
        """Timeout after max_polls: returns False and logs timeout (rich)."""
        output_file = str(tmp_path / "video.mp4")
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_processing(progress=10, remaining_ms=5000)
        )

        with (
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _poll_and_save(
                mock_client,
                model="wan-2.6-text-to-video",
                queue_id="job-123",
                output_path=output_file,
                plain=False,
                poll_interval=0.01,
                max_polls=2,
            )

        assert result is False
        mock_om.warning.assert_called()
        assert "Timed out" in mock_om.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# generate() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestGenerateCLI:
    """Tests for the generate() click command entrypoint."""

    def test_generate_help(self):
        """Check --help output for generate command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["video", "generate", "--help"])
        assert result.exit_code == 0
        assert "prompt" in result.output.lower() or "PROMPT" in result.output

    def test_generate_invokes_asyncio_run(self):
        """generate() calls asyncio.run with _generate_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.video.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["video", "generate", "A sunset"])
            assert mock_run.called

    def test_generate_with_all_options(self):
        """generate() passes all options through to asyncio.run."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.video.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                [
                    "video",
                    "generate",
                    "A sunset",
                    "--model",
                    "wan-2.6-text-to-video",
                    "--duration",
                    "10s",
                    "--resolution",
                    "1080p",
                    "--aspect-ratio",
                    "16:9",
                    "--negative-prompt",
                    "blur",
                    "--output",
                    "out.mp4",
                    "--save-dir",
                    "/tmp",
                    "--no-poll",
                ],
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _generate_async tests
# ---------------------------------------------------------------------------


class TestGenerateAsync:
    """Tests for _generate_async()."""

    @pytest.mark.asyncio
    async def test_generate_async_no_poll_plain(self):
        """no_poll=True: queues job and returns job ID without polling (plain)."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-abc"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        mock_client.video.submit.assert_called_once()
        all_calls = "".join(str(c) for c in mock_om.success.call_args_list)
        all_calls += "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "job-abc" in all_calls
        assert "queued" in all_calls.lower()

    @pytest.mark.asyncio
    async def test_generate_async_no_poll_rich(self):
        """no_poll=True: queues job and returns job ID without polling (rich)."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-abc"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        mock_client.video.submit.assert_called_once()
        all_calls = "".join(str(c) for c in mock_om.success.call_args_list)
        all_calls += "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "job-abc" in all_calls

    @pytest.mark.asyncio
    async def test_generate_async_polls_and_saves(self, tmp_path):
        """Normal flow: queues, polls until COMPLETED, saves file."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=b"video data")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output="output.mp4",
                save_dir=str(tmp_path),
                no_poll=False,
                auto_open=False,
            )

        mock_client.video.submit.assert_called_once()
        mock_client.video.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_async_auto_open_on_success(self, tmp_path):
        """auto_open=True: calls open_file after successful save."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=b"video data")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("venice_ai.cli.commands.video.open_file") as mock_open_f,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output="output.mp4",
                save_dir=str(tmp_path),
                no_poll=False,
                auto_open=True,
            )

        mock_open_f.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_async_auto_open_not_called_on_failure(self, tmp_path):
        """auto_open=True: NOT called when poll_and_save returns False."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )
        mock_client.video.retrieve = AsyncMock(return_value=_make_status_failed("error"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("venice_ai.cli.commands.video.open_file") as mock_open_f,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            with pytest.raises(SystemExit) as exit_info:
                await _generate_async(
                    ctx,
                    prompt="A sunset",
                    model="wan-2.6-text-to-video",
                    duration="5s",
                    resolution=None,
                    aspect_ratio="16:9",
                    negative_prompt=None,
                    output="output.mp4",
                    save_dir=str(tmp_path),
                    no_poll=False,
                    auto_open=True,
                )
            assert exit_info.value.code == 1

        mock_open_f.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_async_with_resolution_and_negative_prompt(self, tmp_path):
        """Resolution and negative_prompt are passed to API when provided."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=b"video data")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution="1080p",
                aspect_ratio="16:9",
                negative_prompt="blur noise",
                output="output.mp4",
                save_dir=str(tmp_path),
                no_poll=False,
                auto_open=False,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["resolution"] == "1080p"
        assert call_kwargs["negative_prompt"] == "blur noise"

    @pytest.mark.asyncio
    async def test_generate_async_ctx_obj_none(self, tmp_path):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
            )

        assert mock_om.info.called or mock_om.success.called

    @pytest.mark.asyncio
    async def test_generate_async_long_prompt_truncated_rich(self, tmp_path):
        """Long prompt is truncated in rich mode display."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            long_prompt = "A" * 100  # 100 chars — over limit of 80
            ctx = _make_ctx(plain=False)
            await _generate_async(
                ctx,
                prompt=long_prompt,
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
            )

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "..." in echo_calls

    @pytest.mark.asyncio
    async def test_generate_async_rich_polls_shows_polling_message(self, tmp_path):
        """Rich mode shows 'Polling for completion...' before polling."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=b"video data")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output="output.mp4",
                save_dir=str(tmp_path),
                no_poll=False,
                auto_open=False,
            )

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "Polling" in echo_calls

    @pytest.mark.asyncio
    async def test_generate_async_no_aspect_ratio(self, tmp_path):
        """aspect_ratio=None: not included in queue kwargs."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-xyz", model="wan-2.6-text-to-video")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio=None,
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert "aspect_ratio" not in call_kwargs


# ---------------------------------------------------------------------------
# from_image() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestFromImageCLI:
    """Tests for the from_image() click command entrypoint."""

    def test_from_image_help(self):
        """Check --help output for from-image command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["video", "from-image", "--help"])
        assert result.exit_code == 0
        assert "INPUT_FILE" in result.output or "input" in result.output.lower()

    def test_from_image_invokes_asyncio_run(self, tmp_path):
        """from_image() calls asyncio.run with _from_image_async coroutine."""
        from venice_ai.cli.cli import cli

        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.video.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["video", "from-image", str(img_file)])
            assert mock_run.called

    def test_from_image_with_all_options(self, tmp_path):
        """from_image() passes all options through."""
        from venice_ai.cli.cli import cli

        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.video.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                [
                    "video",
                    "from-image",
                    str(img_file),
                    "--prompt",
                    "Gentle breeze",
                    "--model",
                    "wan-2.6-image-to-video",
                    "--duration",
                    "10s",
                    "--resolution",
                    "720p",
                    "--output",
                    "out.mp4",
                    "--save-dir",
                    "/tmp",
                    "--no-poll",
                ],
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _from_image_async tests
# ---------------------------------------------------------------------------


class TestFromImageAsync:
    """Tests for _from_image_async()."""

    @pytest.mark.asyncio
    async def test_from_image_async_no_poll_plain(self, tmp_path):
        """no_poll=True: encodes image, queues job, returns ID (plain)."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-1"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        mock_client.video.submit.assert_called_once()
        call_kwargs = mock_client.video.submit.call_args[1]
        assert "image_url" in call_kwargs
        assert call_kwargs["image_url"].startswith("data:image/png;base64,")
        all_calls = "".join(str(c) for c in mock_om.success.call_args_list)
        all_calls += "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "job-img-1" in all_calls

    @pytest.mark.asyncio
    async def test_from_image_async_no_poll_rich(self, tmp_path):
        """no_poll=True: encodes image, queues job, returns ID (rich)."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake jpg data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-2"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Animate gently",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        all_calls = "".join(str(c) for c in mock_om.success.call_args_list)
        all_calls += "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "job-img-2" in all_calls

    @pytest.mark.asyncio
    async def test_from_image_async_empty_prompt_uses_default(self, tmp_path):
        """Empty prompt uses default 'Animate this image with natural motion'."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-3"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="",  # empty prompt
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["prompt"] == "Animate this image with natural motion"

    @pytest.mark.asyncio
    async def test_from_image_async_with_resolution(self, tmp_path):
        """Resolution is passed to queue when provided."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-4"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution="720p",
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["resolution"] == "720p"

    @pytest.mark.asyncio
    async def test_from_image_async_polls_and_saves(self, tmp_path):
        """Normal flow: encodes image, queues, polls until COMPLETED, saves."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-img-5", model="wan-2.6-image-to-video")
        )
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=b"video data")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output="output.mp4",
                save_dir=str(tmp_path),
                no_poll=False,
                auto_open=False,
            )

        mock_client.video.submit.assert_called_once()
        mock_client.video.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_image_async_auto_open(self, tmp_path):
        """auto_open=True: calls open_file after successful save."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-img-5", model="wan-2.6-image-to-video")
        )
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=b"video data")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("venice_ai.cli.commands.video.open_file") as mock_open_f,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output="output.mp4",
                save_dir=str(tmp_path),
                no_poll=False,
                auto_open=True,
            )

        mock_open_f.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_image_async_ctx_obj_none(self, tmp_path):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-6"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        assert mock_om.info.called or mock_om.success.called

    @pytest.mark.asyncio
    async def test_from_image_async_with_prompt_shown_rich(self, tmp_path):
        """Rich mode shows motion prompt when provided."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-7"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Wind blowing through trees",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=".",
                no_poll=True,
                auto_open=False,
            )

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "Wind" in echo_calls or "Motion prompt" in echo_calls

    @pytest.mark.asyncio
    async def test_from_image_async_rich_polls_shows_polling_message(self, tmp_path):
        """Rich mode shows 'Polling for completion...' before polling."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(
            return_value=_make_queue_response("job-img-8", model="wan-2.6-image-to-video")
        )
        mock_client.video.retrieve = AsyncMock(
            return_value=_make_status_completed_data(data=b"video data")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output="output.mp4",
                save_dir=str(tmp_path),
                no_poll=False,
                auto_open=False,
            )

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "Polling" in echo_calls


# ---------------------------------------------------------------------------
# status() CLI entrypoint tests
# ---------------------------------------------------------------------------


class TestStatusCLI:
    """Tests for the status() click command entrypoint."""

    def test_status_help(self):
        """Check --help output for status command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["video", "status", "--help"])
        assert result.exit_code == 0
        assert "JOB_ID" in result.output or "job" in result.output.lower()

    def test_status_invokes_asyncio_run(self):
        """status() calls asyncio.run with _status_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.video.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["video", "status", "abc123", "--model", "m"])
            assert mock_run.called

    def test_status_requires_the_model(self):
        """Venice looks a job up by model and queue ID, so a queue ID alone is a usage error."""
        from venice_ai.cli.cli import cli

        with patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"):
            result = CliRunner().invoke(cli, ["video", "status", "abc123"])

        assert result.exit_code == 2
        assert "--model" in result.output

    def test_status_with_model_option(self):
        """status() passes --model option through."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.video.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                ["video", "status", "abc123", "--model", "wan-2.6-image-to-video"],
            )
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _status_async tests
# ---------------------------------------------------------------------------


class TestStatusAsync:
    """Tests for _status_async()."""

    @pytest.mark.asyncio
    async def test_status_completed_plain(self):
        """COMPLETED status displayed correctly in plain mode."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="COMPLETED", url=None, data=None)
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _status_async(ctx, job_id="job-done", model="wan-2.6-text-to-video")

        mock_om.success.assert_called_once()
        assert "COMPLETED" in mock_om.success.call_args[0][0]
        assert "job-done" in mock_om.success.call_args[0][0]

    @pytest.mark.asyncio
    async def test_status_completed_rich(self):
        """COMPLETED status displayed correctly in rich mode."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="COMPLETED", url=None, data=None)
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _status_async(ctx, job_id="job-done", model="wan-2.6-text-to-video")

        mock_om.success.assert_called_once()
        assert "COMPLETED" in mock_om.success.call_args[0][0]
        assert "job-done" in mock_om.success.call_args[0][0]

    @pytest.mark.asyncio
    async def test_status_completed_with_inline_data_rich(self):
        """COMPLETED with inline data shows size info in rich mode."""
        mock_client = AsyncMock()
        inline_data = b"x" * 1024
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="COMPLETED",
                url=None,
                data=inline_data,
            )
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _status_async(ctx, job_id="job-data", model="wan-2.6-text-to-video")

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "1024" in echo_calls or "bytes" in echo_calls

    @pytest.mark.asyncio
    async def test_status_failed_plain(self):
        """FAILED status displayed correctly in plain mode."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="FAILED", error="Out of resources")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _status_async(ctx, job_id="job-fail", model="wan-2.6-text-to-video")

        mock_om.error.assert_called_once()
        assert "FAILED" in mock_om.error.call_args[0][0]
        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "Out of resources" in echo_calls

    @pytest.mark.asyncio
    async def test_status_failed_rich(self):
        """FAILED status displayed correctly in rich mode."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="FAILED", error="Out of resources")
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _status_async(ctx, job_id="job-fail", model="wan-2.6-text-to-video")

        mock_om.error.assert_called_once()
        assert "FAILED" in mock_om.error.call_args[0][0]
        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "Out of resources" in echo_calls

    @pytest.mark.asyncio
    async def test_status_failed_no_error_attr(self):
        """FAILED with no error attribute shows 'unknown error'."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(return_value=SimpleNamespace(status="FAILED"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _status_async(ctx, job_id="job-fail", model="wan-2.6-text-to-video")

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "unknown error" in echo_calls

    @pytest.mark.asyncio
    async def test_status_processing_with_remaining_plain(self):
        """PROCESSING with remaining_ms shown correctly in plain mode."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="PROCESSING",
                progress_percent=60,
                estimated_remaining_ms=20000,
            )
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _status_async(ctx, job_id="job-proc", model="wan-2.6-text-to-video")

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "PROCESSING" in echo_calls
        mock_om.progress.assert_called_once()
        assert mock_om.progress.call_args[1]["pct"] == 60
        assert "20" in mock_om.progress.call_args[0][0]

    @pytest.mark.asyncio
    async def test_status_processing_without_remaining_plain(self):
        """PROCESSING without remaining_ms shown correctly in plain mode."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="PROCESSING",
                progress_percent=30,
                estimated_remaining_ms=None,
            )
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _status_async(ctx, job_id="job-proc", model="wan-2.6-text-to-video")

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "PROCESSING" in echo_calls
        mock_om.progress.assert_called_once()
        assert mock_om.progress.call_args[1]["pct"] == 30

    @pytest.mark.asyncio
    async def test_status_processing_with_remaining_rich(self):
        """PROCESSING with remaining_ms shown correctly in rich mode."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="PROCESSING",
                progress_percent=40,
                estimated_remaining_ms=45000,
            )
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _status_async(ctx, job_id="job-proc", model="wan-2.6-text-to-video")

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "PROCESSING" in echo_calls
        mock_om.progress.assert_called_once()
        assert mock_om.progress.call_args[1]["pct"] == 40
        assert "45" in mock_om.progress.call_args[0][0]

    @pytest.mark.asyncio
    async def test_status_processing_without_remaining_rich(self):
        """PROCESSING without remaining_ms shown in rich mode (no time estimate)."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                status="PROCESSING",
                progress_percent=20,
                estimated_remaining_ms=None,
            )
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _status_async(ctx, job_id="job-proc", model="wan-2.6-text-to-video")

        echo_calls = "".join(str(c) for c in mock_om.echo.call_args_list)
        assert "PROCESSING" in echo_calls
        mock_om.progress.assert_called_once()
        assert mock_om.progress.call_args[1]["pct"] == 20

    @pytest.mark.asyncio
    async def test_status_processing_no_progress_attr(self):
        """PROCESSING with no progress_percent attr defaults to 0."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(return_value=SimpleNamespace(status="PROCESSING"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _status_async(ctx, job_id="job-proc", model="wan-2.6-text-to-video")

        mock_om.progress.assert_called_once()
        assert mock_om.progress.call_args[1]["pct"] == 0

    @pytest.mark.asyncio
    async def test_status_ctx_obj_none(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="COMPLETED", url=None, data=None)
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _status_async(ctx, job_id="job-done", model="wan-2.6-text-to-video")

        mock_om.success.assert_called_once()


# ---------------------------------------------------------------------------
# High-value flags (--audio, --reference-image-urls,
# --reference-video-urls, --end-image-url)
# ---------------------------------------------------------------------------


class TestGenerateAsyncNewFlags:
    """Tests for the new generate() flags wired to submit()."""

    @pytest.mark.asyncio
    async def test_generate_async_forwards_reference_image_urls_and_audio(self, tmp_path):
        """generate forwards --reference-image-urls (as list) and --audio to submit()."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-ref"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=True,
                reference_image_urls=("https://example.com/a.png", "https://example.com/b.png"),
                reference_video_urls=(),
                end_image_url=None,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["audio"] is True
        assert call_kwargs["reference_image_urls"] == [
            "https://example.com/a.png",
            "https://example.com/b.png",
        ]
        # Empty multiple tuple must not be forwarded.
        assert "reference_video_urls" not in call_kwargs
        assert "end_image_url" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_async_audio_none_not_forwarded(self, tmp_path):
        """audio=None (flag not set) is not forwarded to submit()."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-noaudio"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=None,
                reference_image_urls=(),
                reference_video_urls=(),
                end_image_url=None,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert "audio" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_async_forwards_no_audio_and_reference_videos(self, tmp_path):
        """--no-audio (audio=False) and --reference-video-urls forwarded."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-rv"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=False,
                reference_image_urls=(),
                reference_video_urls=("https://example.com/clip.mp4",),
                end_image_url=None,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["audio"] is False
        assert call_kwargs["reference_video_urls"] == ["https://example.com/clip.mp4"]


class TestGenerateCLINewFlags:
    """CliRunner tests that exercise the click-layer tri-state for --audio."""

    def test_generate_cli_no_audio_flag_not_forwarded(self):
        """No --audio / --no-audio on the command line → audio omitted from submit()."""
        from venice_ai.cli.cli import cli

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-cli"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "video",
                    "generate",
                    "A sunset",
                    "--model",
                    "wan-2.6-text-to-video",
                    "--no-poll",
                ],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_client.video.submit.call_args[1]
        assert "audio" not in call_kwargs

    def test_generate_cli_audio_flag_forwarded(self):
        """--audio on the command line → audio=True forwarded to submit()."""
        from venice_ai.cli.cli import cli

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-cli2"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "video",
                    "generate",
                    "A sunset",
                    "--model",
                    "wan-2.6-text-to-video",
                    "--audio",
                    "--reference-image-urls",
                    "https://example.com/a.png",
                    "--no-poll",
                ],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["audio"] is True
        assert call_kwargs["reference_image_urls"] == ["https://example.com/a.png"]

    def test_generate_cli_no_audio_flag_forwarded_false(self):
        """--no-audio on the command line → audio=False forwarded to submit()."""
        from venice_ai.cli.cli import cli

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-cli3"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "video",
                    "generate",
                    "A sunset",
                    "--model",
                    "wan-2.6-text-to-video",
                    "--no-audio",
                    "--no-poll",
                ],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["audio"] is False


class TestFromImageAsyncNewFlags:
    """Tests for the new from-image() flags wired to submit()."""

    @pytest.mark.asyncio
    async def test_from_image_async_forwards_end_image_url(self, tmp_path):
        """from-image forwards --end-image-url to submit() unchanged (not data-encoded)."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-end"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=None,
                reference_image_urls=(),
                reference_video_urls=(),
                end_image_url="https://example.com/end.png",
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["end_image_url"] == "https://example.com/end.png"

    @pytest.mark.asyncio
    async def test_from_image_async_forwards_audio_and_reference_images(self, tmp_path):
        """from-image forwards --audio and --reference-image-urls."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-ref"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=True,
                reference_image_urls=("https://example.com/ref.png",),
                reference_video_urls=(),
                end_image_url=None,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["audio"] is True
        assert call_kwargs["reference_image_urls"] == ["https://example.com/ref.png"]
        assert "end_image_url" not in call_kwargs


class TestFromImageCLINewFlags:
    """CliRunner test for from-image --end-image-url forwarding."""

    def test_from_image_cli_end_image_url_forwarded(self, tmp_path):
        """--end-image-url on the command line forwarded to submit()."""
        from venice_ai.cli.cli import cli

        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-cli-end"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "video",
                    "from-image",
                    str(img_file),
                    "--model",
                    "wan-2.6-image-to-video",
                    "--end-image-url",
                    "https://example.com/end.png",
                    "--no-poll",
                ],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["end_image_url"] == "https://example.com/end.png"


# ---------------------------------------------------------------------------
# CLI-B-03: --reference-audio-urls (R2V donor audio, up to 3) wired to submit()
# ---------------------------------------------------------------------------


class TestReferenceAudioUrls:
    """Tests for the --reference-audio-urls flag on generate and from-image."""

    @pytest.mark.asyncio
    async def test_generate_async_forwards_reference_audio_urls(self, tmp_path):
        """generate forwards --reference-audio-urls (as list) to submit()."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-ra"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=None,
                reference_image_urls=(),
                reference_video_urls=(),
                reference_audio_urls=(
                    "https://example.com/a.mp3",
                    "https://example.com/b.mp3",
                ),
                end_image_url=None,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["reference_audio_urls"] == [
            "https://example.com/a.mp3",
            "https://example.com/b.mp3",
        ]

    @pytest.mark.asyncio
    async def test_generate_async_empty_reference_audio_urls_not_forwarded(self, tmp_path):
        """An empty --reference-audio-urls tuple is not forwarded to submit()."""
        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-ra-empty"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _generate_async(
                ctx,
                prompt="A sunset",
                model="wan-2.6-text-to-video",
                duration="5s",
                resolution=None,
                aspect_ratio="16:9",
                negative_prompt=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=None,
                reference_image_urls=(),
                reference_video_urls=(),
                reference_audio_urls=(),
                end_image_url=None,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert "reference_audio_urls" not in call_kwargs

    @pytest.mark.asyncio
    async def test_from_image_async_forwards_reference_audio_urls(self, tmp_path):
        """from-image forwards --reference-audio-urls (as list) to submit()."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-img-ra"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.video.OutputManager"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _from_image_async(
                ctx,
                input_file=str(img_file),
                prompt="Gentle motion",
                model="wan-2.6-image-to-video",
                duration="5s",
                resolution=None,
                output=None,
                save_dir=str(tmp_path),
                no_poll=True,
                auto_open=False,
                audio=None,
                reference_image_urls=(),
                reference_video_urls=(),
                reference_audio_urls=("https://example.com/donor.mp3",),
                end_image_url=None,
            )

        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["reference_audio_urls"] == ["https://example.com/donor.mp3"]

    def test_generate_cli_reference_audio_urls_forwarded(self):
        """--reference-audio-urls on the command line forwarded to submit()."""
        from venice_ai.cli.cli import cli

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-cli-ra"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "video",
                    "generate",
                    "A sunset",
                    "--model",
                    "wan-2.6-text-to-video",
                    "--reference-audio-urls",
                    "https://example.com/a.mp3",
                    "--reference-audio-urls",
                    "https://example.com/b.mp3",
                    "--no-poll",
                ],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_client.video.submit.call_args[1]
        assert call_kwargs["reference_audio_urls"] == [
            "https://example.com/a.mp3",
            "https://example.com/b.mp3",
        ]


# ---------------------------------------------------------------------------
# from-image --image-strength (0-1, omitted unless set)
# ---------------------------------------------------------------------------


class TestFromImageCLIImageStrength:
    def _invoke(self, tmp_path, extra_args):
        from venice_ai.cli.cli import cli

        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"fake png data")

        mock_client = AsyncMock()
        mock_client.video.submit = AsyncMock(return_value=_make_queue_response("job-strength"))

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("os.makedirs"),
        ):
            _setup_client_patch(MockClient, mock_client)
            result = CliRunner().invoke(
                cli,
                [
                    "video",
                    "from-image",
                    str(img_file),
                    "--model",
                    "some-i2v-model",
                    "--no-poll",
                    *extra_args,
                ],
            )
        return result, mock_client

    def test_image_strength_forwarded(self, tmp_path):
        result, mock_client = self._invoke(tmp_path, ["--image-strength", "0.6"])
        assert result.exit_code == 0, result.output
        assert mock_client.video.submit.call_args[1]["image_strength"] == 0.6

    def test_image_strength_omitted_when_unset(self, tmp_path):
        result, mock_client = self._invoke(tmp_path, [])
        assert result.exit_code == 0, result.output
        assert "image_strength" not in mock_client.video.submit.call_args[1]

    def test_image_strength_out_of_range_rejected(self, tmp_path):
        result, mock_client = self._invoke(tmp_path, ["--image-strength", "1.5"])
        assert result.exit_code != 0
        mock_client.video.submit.assert_not_called()


# ---------------------------------------------------------------------------
# Downloads over real loopback requests
# ---------------------------------------------------------------------------

IN_PATH = "PATHSECRET"
IN_QUERY = "QUERYSECRET"
LINK_TAIL = f"/{IN_PATH}/video.mp4?sig={IN_QUERY}"

Handler = Callable[[web.Request], Any]


@asynccontextmanager
async def _server(handler: Handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


class _Storage:
    """A storage host: answers the given statuses in turn, then serves the file."""

    def __init__(self, *statuses: int) -> None:
        self.statuses = list(statuses)
        self.requests: list[str] = []

    async def __call__(self, request: web.Request) -> web.Response:
        self.requests.append(request.method)
        if request.method == "DELETE":
            return web.Response(status=204)
        if self.statuses:
            return web.Response(status=self.statuses.pop(0), reason="Refused")
        return web.Response(body=b"VIDEO", content_type="video/mp4")


class _Api:
    """The parts of the Venice API that the video commands call."""

    def __init__(self, retrieve: Any, queue: dict[str, Any] | None = None) -> None:
        self.retrieve = retrieve
        self.queue = {"model": "m", "queue_id": "q", **(queue or {})}
        self.completed = 0

    async def __call__(self, request: web.Request) -> web.Response:
        if request.path.endswith("/video/queue"):
            return web.json_response(self.queue)
        if request.path.endswith("/video/retrieve"):
            return web.json_response(self.retrieve)
        if request.path.endswith("/video/complete"):
            self.completed += 1
            return web.json_response({"success": True})
        return web.json_response({"object": "list", "data": []})


def _real_client(api: str) -> VeniceClient:
    return VeniceClient(api_key="sk-fake", base_url=f"{api}/api/v1")


def _assert_no_link(*texts: str) -> None:
    for text in texts:
        assert IN_PATH not in text
        assert IN_QUERY not in text


@asynccontextmanager
async def _stack(api_handler: _Api, storage_handler: _Storage) -> AsyncIterator[tuple[str, str]]:
    """Run both hosts, with the CLI's client kwargs pointing at the API host."""
    async with _server(api_handler) as api, _server(storage_handler) as storage:
        with patch(
            "venice_ai.cli.config.get_client_kwargs",
            return_value={"api_key": "sk-fake", "base_url": f"{api}/api/v1"},
        ):
            yield api, storage


class TestPollAndSaveDownloads:
    """_poll_and_save() goes through the SDK: its session, its retries and its error handling."""

    @pytest.mark.parametrize("plain", [True, False], ids=["plain", "rich"])
    async def test_url_job_is_saved_and_released_without_printing_the_link(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], plain: bool
    ) -> None:
        storage_host = _Storage()
        async with _server(storage_host) as storage:
            api_host = _Api({"status": "COMPLETED", "url": f"{storage}{LINK_TAIL}"})
            async with _server(api_host) as api, _real_client(api) as client:
                saved = await _poll_and_save(
                    client,
                    model="m",
                    queue_id="q",
                    output_path=str(tmp_path / "video.mp4"),
                    plain=plain,
                )

        assert saved is True
        assert (tmp_path / "video.mp4").read_bytes() == b"VIDEO"
        assert api_host.completed == 1
        assert storage_host.requests == ["GET"]
        out = capsys.readouterr()
        assert "Video saved" in out.out
        assert "Server cleanup: OK" in out.out
        _assert_no_link(out.out, out.err)

    async def test_link_from_the_queue_response_is_used_when_retrieve_has_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        storage_host = _Storage()
        api_host = _Api({"status": "COMPLETED"})
        async with (
            _server(storage_host) as storage,
            _server(api_host) as api,
            _real_client(api) as client,
        ):
            saved = await _poll_and_save(
                client,
                model="m",
                queue_id="q",
                output_path=str(tmp_path / "video.mp4"),
                plain=True,
                download_url=f"{storage}{LINK_TAIL}",
            )

        assert saved is True
        assert (tmp_path / "video.mp4").read_bytes() == b"VIDEO"
        assert storage_host.requests == ["GET", "DELETE"]
        out = capsys.readouterr()
        _assert_no_link(out.out, out.err)

    async def test_download_follows_the_clients_retry_policy(self, tmp_path: Path) -> None:
        storage_host = _Storage(503)
        async with (
            _server(storage_host) as storage,
            _server(_Api({"status": "COMPLETED", "url": f"{storage}{LINK_TAIL}"})) as api,
            VeniceClient(
                api_key="sk-fake", base_url=f"{api}/api/v1", retry_options=_FAST_RETRIES
            ) as client,
        ):
            saved = await _poll_and_save(
                client,
                model="m",
                queue_id="q",
                output_path=str(tmp_path / "video.mp4"),
                plain=True,
            )

        assert saved is True
        assert storage_host.requests == ["GET", "GET"]

    @pytest.mark.parametrize("link_from", ["retrieve", "queue"])
    async def test_refused_download_raises_without_the_link(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], link_from: str
    ) -> None:
        async with _server(_Storage(410)) as storage:
            link = f"{storage}{LINK_TAIL}"
            body: dict[str, Any] = {"status": "COMPLETED"}
            if link_from == "retrieve":
                body["url"] = link
            async with _server(_Api(body)) as api, _real_client(api) as client:
                with pytest.raises(aiohttp.ClientResponseError) as excinfo:
                    await _poll_and_save(
                        client,
                        model="m",
                        queue_id="q",
                        output_path=str(tmp_path / "video.mp4"),
                        plain=True,
                        download_url=link if link_from == "queue" else None,
                    )

        assert excinfo.value.status == 410
        out = capsys.readouterr()
        _assert_no_link(str(excinfo.value), repr(excinfo.value), out.out, out.err)
        assert not (tmp_path / "video.mp4").exists()

    async def test_failed_cleanup_is_reported_and_does_not_fail_the_save(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async def delete_refused(request: web.Request) -> web.Response:
            if request.method == "DELETE":
                return web.Response(status=403)
            return web.Response(body=b"VIDEO")

        async with (
            _server(delete_refused) as storage,
            _server(_Api({"status": "COMPLETED"})) as api,
            _real_client(api) as client,
        ):
            saved = await _poll_and_save(
                client,
                model="m",
                queue_id="q",
                output_path=str(tmp_path / "video.mp4"),
                plain=True,
                download_url=f"{storage}{LINK_TAIL}",
            )

        assert saved is True
        out = capsys.readouterr()
        assert "Server cleanup: skipped" in out.out
        _assert_no_link(out.out, out.err)

    async def test_cleanup_answering_false_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_client = AsyncMock()
        mock_client.video.retrieve = AsyncMock(
            return_value=SimpleNamespace(status="COMPLETED", data=None, url="https://h/v.mp4")
        )
        mock_client.fetch_external = AsyncMock(return_value=b"VIDEO")
        mock_client.video.cancel = AsyncMock(return_value=SimpleNamespace(success=False))

        saved = await _poll_and_save(
            mock_client,
            model="m",
            queue_id="q",
            output_path=str(tmp_path / "video.mp4"),
            plain=True,
        )

        assert saved is True
        assert "Server cleanup: skipped" in capsys.readouterr().out


_FAST_RETRIES = RetryOptions(max_attempts=2, base_delay=0.001, max_delay=0.002)


def _generate_kwargs(tmp_path: Path, *, no_poll: bool = False) -> dict[str, Any]:
    return {
        "model": "m",
        "duration": "5s",
        "resolution": None,
        "output": str(tmp_path / "video.mp4"),
        "save_dir": str(tmp_path),
        "no_poll": no_poll,
        "auto_open": False,
    }


async def _queue_with(command: str, tmp_path: Path, *, no_poll: bool = False) -> None:
    ctx = _make_ctx(plain=True)
    if command == "generate":
        await _generate_async(
            ctx,
            prompt="p",
            aspect_ratio="16:9",
            negative_prompt=None,
            **_generate_kwargs(tmp_path, no_poll=no_poll),
        )
    else:
        image = tmp_path / "in.png"
        image.write_bytes(b"PNG")
        await _from_image_async(
            ctx,
            input_file=str(image),
            prompt="p",
            **_generate_kwargs(tmp_path, no_poll=no_poll),
        )


@pytest.mark.parametrize("command", ["generate", "from-image"])
class TestQueueTimeLink:
    """A model that hands its link out only in the queue response."""

    async def test_link_from_the_queue_response_reaches_the_download(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
    ) -> None:
        storage_host = _Storage()
        api_host = _Api({"status": "COMPLETED"})
        async with _stack(api_host, storage_host) as (_api, storage):
            api_host.queue["download_url"] = f"{storage}{LINK_TAIL}"
            await _queue_with(command, tmp_path)

        assert (tmp_path / "video.mp4").read_bytes() == b"VIDEO"
        assert storage_host.requests == ["GET", "DELETE"]
        out = capsys.readouterr()
        _assert_no_link(out.out, out.err)

    async def test_no_poll_warns_that_the_link_cannot_be_recovered(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
    ) -> None:
        api_host = _Api({"status": "COMPLETED"})
        async with _stack(api_host, _Storage()) as (_api, storage):
            api_host.queue["download_url"] = f"{storage}{LINK_TAIL}"
            await _queue_with(command, tmp_path, no_poll=True)

        out = capsys.readouterr()
        assert "cannot be downloaded later" in " ".join(out.out.split())
        _assert_no_link(out.out, out.err)

    async def test_no_poll_on_an_ordinary_job_names_the_model_and_does_not_warn(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
    ) -> None:
        async with _stack(_Api({"status": "COMPLETED"}), _Storage()):
            await _queue_with(command, tmp_path, no_poll=True)

        out = " ".join(capsys.readouterr().out.split())
        assert "video status q --model m" in out
        assert "cannot be downloaded later" not in out


class TestStatusNeverPrintsTheLink:
    """`video status` works from a queue ID and a model alone, so it has no queue-time link."""

    @pytest.mark.parametrize("plain", [True, False], ids=["plain", "rich"])
    async def test_completed_with_a_url_says_how_to_save_it(
        self, capsys: pytest.CaptureFixture[str], plain: bool
    ) -> None:
        async with _server(_Storage()) as storage:
            body = {"status": "COMPLETED", "url": f"{storage}{LINK_TAIL}"}
            async with _stack(_Api(body), _Storage()):
                await _status_async(_make_ctx(plain=plain), job_id="q", model="m")

        out = capsys.readouterr()
        assert "--output" in out.out
        _assert_no_link(out.out, out.err)

    async def test_output_saves_a_url_job_through_the_sdk(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        storage_host = _Storage()
        target = tmp_path / "sub" / "saved.mp4"
        async with _server(storage_host) as storage:
            body = {"status": "COMPLETED", "url": f"{storage}{LINK_TAIL}"}
            async with _stack(_Api(body), _Storage()):
                await _status_async(
                    _make_ctx(plain=True), job_id="q", model="m", output=str(target)
                )

        assert target.read_bytes() == b"VIDEO"
        out = capsys.readouterr()
        assert "Video saved" in out.out
        _assert_no_link(out.out, out.err)

    async def test_output_raises_without_the_link_when_the_download_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async with _server(_Storage(410)) as storage:
            body = {"status": "COMPLETED", "url": f"{storage}{LINK_TAIL}"}
            async with _stack(_Api(body), _Storage()):
                with pytest.raises(aiohttp.ClientResponseError) as excinfo:
                    await _status_async(
                        _make_ctx(plain=True),
                        job_id="q",
                        model="m",
                        output=str(tmp_path / "v.mp4"),
                    )

        out = capsys.readouterr()
        _assert_no_link(str(excinfo.value), repr(excinfo.value), out.out, out.err)

    async def test_output_saves_inline_data(self, tmp_path: Path) -> None:
        async def inline(request: web.Request) -> web.Response:
            if request.path.endswith("/video/retrieve"):
                return web.Response(body=b"INLINE", content_type="video/mp4")
            return web.json_response({"object": "list", "data": []})

        target = tmp_path / "saved.mp4"
        async with _server(inline) as api:
            with patch(
                "venice_ai.cli.config.get_client_kwargs",
                return_value={"api_key": "sk-fake", "base_url": f"{api}/api/v1"},
            ):
                await _status_async(
                    _make_ctx(plain=True), job_id="q", model="m", output=str(target)
                )

        assert target.read_bytes() == b"INLINE"

    @pytest.mark.parametrize("output", [None, "v.mp4"], ids=["no-output", "output"])
    async def test_completed_with_nothing_to_download_says_it_cannot_be_recovered(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], output: str | None
    ) -> None:
        path = str(tmp_path / output) if output else None
        async with _stack(_Api({"status": "COMPLETED"}), _Storage()):
            if output:
                with pytest.raises(SystemExit):
                    await _status_async(_make_ctx(plain=True), job_id="q", model="m", output=path)
            else:
                await _status_async(_make_ctx(plain=True), job_id="q", model="m", output=path)

        out = " ".join(capsys.readouterr().out.split())
        assert "cannot be downloaded from its job ID" in out
        assert "without --no-poll" in out
        assert not (tmp_path / "v.mp4").exists()


def test_no_cli_command_opens_its_own_http_session() -> None:
    """A bare session skips the client's proxy, SSL, retry and error handling."""
    import venice_ai.cli

    cli_root = Path(venice_ai.cli.__file__).parent
    offenders = [
        str(path.relative_to(cli_root))
        for path in cli_root.rglob("*.py")
        if "ClientSession(" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
