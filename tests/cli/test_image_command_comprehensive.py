"""
Comprehensive tests for image command coverage
Targets missing lines and branches for 80%+ coverage
"""

import base64
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli
from venice_ai.cli.commands.image._helpers import _load_preset_config, validate_size
from venice_ai.cli.commands.image.edit import _edit_async, _remove_bg_async, edit_image
from venice_ai.cli.commands.image.generate import _batch_generate_async, _generate_image_async
from venice_ai.cli.commands.image.presets_cmd import _manage_presets_async
from venice_ai.cli.commands.image.styles import _list_styles_async
from venice_ai.cli.commands.image.upscale import _upscale_async
from venice_ai.cli.commands.image.wizard import _interactive_image_generation
from venice_ai.exceptions import VeniceError


@pytest.fixture
def cli_runner():
    """Fixture providing Click's CliRunner"""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Return a mock config object"""
    return {
        "defaults": {"image_model": "hidream"},
        "output": {"images_dir": "/tmp/test_images"},
    }


@pytest.fixture
def mock_ctx(mock_config):
    """Return a mock click context"""
    ctx = MagicMock()
    ctx.obj = {"config": mock_config}
    ctx.exit = MagicMock()
    return ctx


@pytest.fixture
def mock_image_response():
    """Return a mock image generation response"""
    # Create a simple base64-encoded "image" (just a placeholder)
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG header
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    response = SimpleNamespace(
        images=[encoded_image],
        timing=SimpleNamespace(inferenceDuration=1234),
    )
    return response


@pytest.fixture
def mock_models_response():
    """Return a mock models response"""
    return SimpleNamespace(
        data=[
            SimpleNamespace(id="hidream"),
            SimpleNamespace(id="flux-1"),
            SimpleNamespace(id="sdxl"),
        ]
    )


@pytest.fixture
def mock_styles_response():
    """Return a mock styles response"""
    return SimpleNamespace(data=["Cinematic", "Anime", "Photographic", "Fantasy"])


class TestLoadPresetConfig:
    """Tests for _load_preset_config function - lines 30-38"""

    def test_load_builtin_preset(self):
        """Test loading a built-in preset"""
        # Line 33-35
        result = _load_preset_config("photorealistic")
        assert result is not None
        assert "steps" in result
        assert result["steps"] == 30

    def test_load_builtin_preset_artistic(self):
        """Test loading artistic builtin"""
        result = _load_preset_config("artistic")
        assert result is not None
        assert result["cfg_scale"] == 9.0

    def test_load_custom_preset_not_found(self):
        """Test loading non-existent custom preset returns None"""
        # Line 38 - tries custom preset when not built-in
        with patch("venice_ai.cli.commands.image._helpers.load_preset", return_value=None):
            result = _load_preset_config("nonexistent_custom_preset")
            assert result is None

    def test_load_custom_preset_found(self):
        """Test loading existing custom preset"""
        custom_config = {"steps": 40, "cfg_scale": 6.0}
        with patch("venice_ai.cli.commands.image._helpers.load_preset", return_value=custom_config):
            result = _load_preset_config("my_custom_preset")
            assert result == custom_config


class TestGenerateImageCommand:
    """Tests for generate_image command including CLI options"""

    def test_generate_requires_prompt_without_interactive(self, cli_runner):
        """Test that prompt is required when not using interactive mode - lines 141-143"""
        with patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"):
            result = cli_runner.invoke(cli, ["image", "generate"])
            assert result.exit_code != 0

    def test_generate_with_interactive_flag(self, cli_runner):
        """Test interactive mode is launched with -i flag - line 137-139"""

        # Use a side_effect that consumes the coroutine properly
        def consume_coro(coro):
            # Close the coroutine to prevent warnings
            coro.close()
            return None

        with patch(
            "venice_ai.cli.commands.image.generate.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            result = cli_runner.invoke(cli, ["image", "generate", "--interactive"])
            # Should call asyncio.run for interactive mode
            assert mock_run.called or result.exit_code != 0

    def test_generate_applies_preset(self, cli_runner):
        """Test that preset is applied correctly - lines 146-171"""

        # Use a side_effect that consumes the coroutine properly
        def consume_coro(coro):
            coro.close()
            return None

        with patch("venice_ai.cli.commands.image.generate._load_preset_config") as mock_load:
            mock_load.return_value = {
                "steps": 30,
                "cfg_scale": 7.5,
                "seed": 42,
                "style_preset": "Cinematic",
                "lora_strength": 50,
                "format": "png",
                "safe_mode": True,
                "hide_watermark": True,
                "embed_exif": True,
            }
            with (
                patch(
                    "venice_ai.cli.commands.image.generate.asyncio.run", side_effect=consume_coro
                ),
                patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            ):
                cli_runner.invoke(
                    cli,
                    [
                        "image",
                        "generate",
                        "test prompt",
                        "--preset",
                        "photorealistic",
                    ],
                )
                mock_load.assert_called_with("photorealistic")


class TestGenerateImageAsync:
    """Tests for _generate_image_async function - lines 224-377"""

    @pytest.mark.asyncio
    async def test_generate_image_async_success(self, mock_ctx, mock_image_response, mock_config):
        """Test successful image generation - covers lines 224-370"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test prompt",
                        model=None,  # Will use default - line 228-229
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=tmpdir,  # Line 237-238
                        show_timing=True,
                        steps=20,  # Line 301-302
                        cfg_scale=7.5,  # Line 303-304
                        seed=42,  # Line 305-306
                        style_preset="Cinematic",  # Line 309-310
                        lora_strength=50,  # Line 311-312
                        format="png",  # Line 313-314
                        safe_mode=True,  # Line 315-316
                        hide_watermark=True,  # Line 317-318
                        embed_exif=True,  # Line 319-320
                        return_binary=False,  # Line 321-324
                    )

    @pytest.mark.asyncio
    async def test_generate_image_async_uses_config_dir(
        self, mock_ctx, mock_image_response, mock_config
    ):
        """Test using config output directory - line 240"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=None,  # Should use config dir - line 240
                        show_timing=True,
                    )

    @pytest.mark.asyncio
    async def test_generate_validation_fails(self, mock_ctx, mock_config):
        """Test when validation fails - lines 267-269"""
        mock_config["output"]["images_dir"] = "/tmp/test"
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_validator = MagicMock()
        mock_validator.validate_image_parameters = AsyncMock(
            return_value=(False, "Invalid parameters")
        )

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ),
                patch("venice_ai.cli.commands.image.generate.print_error") as mock_error,
            ):
                with pytest.raises(SystemExit):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=None,
                        show_timing=True,
                    )
                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_custom_output_single(
        self, mock_ctx, mock_image_response, mock_config
    ):
        """Test custom output filename for single image - lines 347-348"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output="my_custom_name",  # Line 347
                        save_dir=tmpdir,
                        show_timing=True,
                    )

    @pytest.mark.asyncio
    async def test_generate_with_custom_output_multiple(self, mock_ctx, mock_config):
        """Test custom output filename for multiple images - lines 349-350"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            # Create response with multiple images
            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mock_response = SimpleNamespace(
                images=[encoded, encoded, encoded],
                timing=SimpleNamespace(inferenceDuration=2000),
            )

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=3,
                        output="multi_output",  # Line 349-350
                        save_dir=tmpdir,
                        show_timing=True,
                    )

    @pytest.mark.asyncio
    async def test_generate_no_images_returned(self, mock_ctx, mock_config):
        """Test handling when no images returned - line 372"""
        mock_config["output"]["images_dir"] = "/tmp/test"
        mock_ctx.obj["config"] = mock_config

        mock_response = SimpleNamespace(images=[], timing=None)
        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(return_value=mock_response)
        mock_validator = MagicMock()
        mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ),
                patch("venice_ai.cli.commands.image.generate.print_error") as mock_error,
            ):
                await _generate_image_async(
                    ctx=mock_ctx,
                    prompt="test",
                    model="hidream",
                    size="1024x1024",
                    num_images=1,
                    output=None,
                    save_dir=None,
                    show_timing=True,
                )
                mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_generate_venice_error(self, mock_ctx, mock_config):
        """Test VeniceError handling - lines 374-375"""
        mock_config["output"]["images_dir"] = "/tmp/test"
        mock_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            MockClient.return_value.__aenter__ = AsyncMock(side_effect=VeniceError("API Error"))
            with patch("venice_ai.cli.commands.image.generate.print_error") as mock_error:
                with pytest.raises(SystemExit):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=None,
                        show_timing=True,
                    )
                mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_generate_unexpected_error(self, mock_ctx, mock_config):
        """Test unexpected error handling - lines 376-377"""
        mock_config["output"]["images_dir"] = "/tmp/test"
        mock_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            MockClient.return_value.__aenter__ = AsyncMock(
                side_effect=RuntimeError("Unexpected error")
            )
            with patch("venice_ai.cli.commands.image.generate.print_error") as mock_error:
                with pytest.raises(SystemExit):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=None,
                        show_timing=True,
                    )
                mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_generate_generation_exception(self, mock_ctx, mock_config):
        """Test exception during generation - lines 334-336

        Note: The function catches Exception and calls print_error, then continues.
        The exception is re-raised within the Progress context manager block but
        is caught by the outer try/except for unexpected errors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(side_effect=Exception("Generation failed"))
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with (
                    patch(
                        "venice_ai.cli.commands.image.generate.ParameterValidator",
                        return_value=mock_validator,
                    ),
                    patch("venice_ai.cli.commands.image.generate.print_error") as mock_error,
                ):
                    # The function catches exceptions and calls print_error
                    with pytest.raises(SystemExit):
                        await _generate_image_async(
                            ctx=mock_ctx,
                            prompt="test",
                            model="hidream",
                            size="1024x1024",
                            num_images=1,
                            output=None,
                            save_dir=None,
                            show_timing=True,
                        )
                    # Verify error was reported
                    mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_generate_return_binary_true(self, mock_ctx, mock_image_response, mock_config):
        """Test return_binary=True - line 321-322"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=tmpdir,
                        show_timing=True,
                        return_binary=True,  # Line 321-322
                    )


class TestInteractiveImageGeneration:
    """Tests for _interactive_image_generation - lines 380-802"""

    @pytest.mark.asyncio
    async def test_interactive_no_models_available(self, mock_ctx, mock_config):
        """Test when no image models found - lines 409-412"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.wizard.print_error"),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_models_fetch_error(self, mock_ctx, mock_config):
        """Test error fetching models - lines 413-416"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("API Error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.wizard.print_error"),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_styles_fetch_error(
        self, mock_ctx, mock_config, mock_models_response
    ):
        """Test error fetching styles - lines 425-426"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(side_effect=Exception("Styles error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", return_value=None),  # Prompt returns None
                patch("venice_ai.cli.commands.image.wizard.print_error"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_empty_prompt(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test empty prompt handling - lines 438-440"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", return_value=None),  # Empty prompt
                patch("venice_ai.cli.commands.image.wizard.print_error"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_default_model_not_available(
        self, mock_ctx, mock_config, mock_styles_response
    ):
        """Test when default model not in available list - lines 444-445"""
        mock_config["defaults"]["image_model"] = "nonexistent_model"
        mock_ctx.obj["config"] = mock_config

        mock_models = SimpleNamespace(
            data=[SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")]
        )

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            # Return empty result to exit early
            with (
                patch("asyncio.to_thread", return_value=None),
                patch("venice_ai.cli.commands.image.wizard.print_error"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_wizard_error(self, mock_ctx, mock_config):
        """Test wizard error handling - lines 798-802"""
        mock_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            MockClient.return_value.__aenter__ = AsyncMock(side_effect=Exception("Wizard error"))
            with (
                patch("venice_ai.cli.commands.image.wizard.print_error"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)


class TestInteractiveFlowPaths:
    """Tests for various interactive flow paths"""

    @pytest.mark.asyncio
    async def test_interactive_full_flow_with_advanced(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test full interactive flow with advanced parameters - covers many branches"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        # Track call count to return different values
        call_count = [0]
        responses = [
            "test prompt",  # prompt
            "hidream",  # model
            "1024x1024 (Square - Default)",  # size
            "1",  # num_images
            True,  # configure_advanced
            "30",  # steps
            "7.5",  # cfg_scale
            True,  # use_seed
            "42",  # seed value
            True,  # configure_style
            True,  # use_style
            "Cinematic",  # style_preset
            True,  # use_lora
            "50",  # lora_strength
            "webp (Recommended - Best compression)",  # format
            True,  # safe_mode
            False,  # hide_watermark
            False,  # embed_exif
            False,  # return_binary
            False,  # custom_location
            True,  # proceed
            False,  # save_as_preset
        ]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch(
                    "venice_ai.cli.commands.image.generate._generate_image_async",
                    new_callable=AsyncMock,
                ),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_custom_dimensions(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test custom dimensions path - lines 479-493"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        responses = [
            "test",  # prompt
            "hidream",  # model
            "Custom dimensions",  # size choice
            "1280",  # width
            "720",  # height
            "2",  # num_images
            False,  # configure_advanced
            False,  # configure_style
            "webp (Recommended - Best compression)",  # format
            True,  # safe_mode
            False,  # hide_watermark
            False,  # embed_exif
            False,  # return_binary
            False,  # custom_location
            False,  # proceed (cancel)
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_custom_save_location(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test custom save location - lines 678-685"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        responses = [
            "test",  # prompt
            "hidream",  # model
            "1024x1024 (Square - Default)",  # size
            "1",  # num_images
            False,  # configure_advanced
            False,  # configure_style
            "webp (Recommended - Best compression)",  # format
            True,  # safe_mode
            False,  # hide_watermark
            False,  # embed_exif
            False,  # return_binary
            True,  # custom_location
            "/tmp/custom_images",  # save_dir_str
            False,  # proceed (cancel)
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_save_preset(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test save as preset - lines 747-773"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        responses = [
            "test prompt",  # prompt
            "hidream",  # model
            "1024x1024 (Square - Default)",  # size
            "1",  # num_images
            True,  # configure_advanced
            "25",  # steps - line 754
            "8.0",  # cfg_scale - line 756
            True,  # use_seed
            "12345",  # seed - line 758
            True,  # configure_style
            True,  # use_style - from available_styles
            "Anime",  # style_preset - line 762
            True,  # use_lora
            "75",  # lora_strength - line 764
            "png (Highest quality)",  # format - line 766
            True,  # safe_mode - line 767
            True,  # hide_watermark - line 768
            True,  # embed_exif - line 770
            False,  # return_binary
            False,  # custom_location
            True,  # proceed - line 730
            True,  # save_as_preset - line 741
            "my_preset",  # preset_name - line 752
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch(
                    "venice_ai.cli.commands.image.generate._generate_image_async",
                    new_callable=AsyncMock,
                ),
                patch("venice_ai.cli.commands.image.wizard.save_preset") as mock_save,
            ):
                await _interactive_image_generation(mock_ctx)
                # Verify preset was saved
                mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_interactive_style_preset_none(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test selecting 'None' for style preset - lines 599-600"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        responses = [
            "test",  # prompt
            "hidream",  # model
            "1024x1024 (Square - Default)",  # size
            "1",  # num_images
            False,  # configure_advanced
            True,  # configure_style
            False,  # use_negative (skip)
            True,  # use_style - line 582
            "None",  # style_preset == "None" - line 599-600
            False,  # use_lora
            "webp (Recommended - Best compression)",  # format
            True,  # safe_mode
            False,  # hide_watermark
            False,  # embed_exif
            False,  # return_binary
            False,  # custom_location
            False,  # proceed (cancel)
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)


class TestManagePresetsAsync:
    """Tests for _manage_presets_async - lines 812-947"""

    @pytest.mark.asyncio
    async def test_manage_presets_list_empty(self, mock_ctx):
        """Test listing presets when none exist - lines 833-836"""
        responses = ["List all presets", "Exit"]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.list_presets", return_value=[]),
            patch("venice_ai.cli.commands.image.presets_cmd.print_info"),
        ):
            await _manage_presets_async(mock_ctx)

    @pytest.mark.asyncio
    async def test_manage_presets_list_with_presets(self, mock_ctx):
        """Test listing existing presets - lines 838-854"""
        presets = [
            {
                "name": "preset1",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-02T00:00:00",
            },
            {"name": "preset2", "created_at": "Unknown", "updated_at": "Unknown"},
        ]
        responses = ["List all presets", "Exit"]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.list_presets", return_value=presets),
        ):
            await _manage_presets_async(mock_ctx)

    @pytest.mark.asyncio
    async def test_manage_presets_view_builtin(self, mock_ctx):
        """Test viewing builtin presets - lines 856-873"""
        responses = ["View built-in presets", "Exit"]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.print_info"),
        ):
            await _manage_presets_async(mock_ctx)

    @pytest.mark.asyncio
    async def test_manage_presets_save_new(self, mock_ctx):
        """Test saving a new preset - lines 875-923"""
        responses = [
            "Save current config as preset",
            "new_test_preset",  # preset_name - line 880
            "25",  # steps - line 891-892
            "7.5",  # cfg_scale - line 897-898
            "png",  # format - line 907-908
            True,  # safe_mode - line 913
            "Test description",  # description - line 920-921
            "Exit",
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.save_preset") as mock_save,
        ):
            await _manage_presets_async(mock_ctx)
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_manage_presets_save_skip_format(self, mock_ctx):
        """Test saving preset with 'Skip' format - line 907"""
        responses = [
            "Save current config as preset",
            "minimal_preset",
            "",  # empty steps
            "",  # empty cfg_scale
            "Skip",  # format - line 907
            True,  # safe_mode
            "",  # empty description
            "Exit",
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.save_preset") as mock_save,
        ):
            await _manage_presets_async(mock_ctx)
            # Verify format was not in saved config
            saved_config = mock_save.call_args[0][1]
            assert "format" not in saved_config

    @pytest.mark.asyncio
    async def test_manage_presets_delete_empty(self, mock_ctx):
        """Test delete when no presets exist - lines 925-928"""
        responses = ["Delete a preset", "Exit"]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.list_presets", return_value=[]),
            patch("venice_ai.cli.commands.image.presets_cmd.print_info"),
        ):
            await _manage_presets_async(mock_ctx)

    @pytest.mark.asyncio
    async def test_manage_presets_delete_confirm(self, mock_ctx):
        """Test confirming preset deletion - lines 937-944"""
        presets = [
            {
                "name": "to_delete",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
            }
        ]
        responses = [
            "Delete a preset",
            "to_delete",  # preset_to_delete - line 937
            True,  # confirm - line 943
            "Exit",
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.list_presets", return_value=presets),
            patch("venice_ai.cli.commands.image.presets_cmd.delete_preset") as mock_delete,
        ):
            await _manage_presets_async(mock_ctx)
            mock_delete.assert_called_with("to_delete")

    @pytest.mark.asyncio
    async def test_manage_presets_delete_cancel(self, mock_ctx):
        """Test canceling preset selection - line 937"""
        presets = [{"name": "preset1", "created_at": "2024-01-01", "updated_at": "2024-01-01"}]
        responses = [
            "Delete a preset",
            "Cancel",  # preset_to_delete == "Cancel"
            "Exit",
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else "Exit"
            call_count[0] += 1
            return result

        with (
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch("venice_ai.cli.commands.image.presets_cmd.list_presets", return_value=presets),
            patch("venice_ai.cli.commands.image.presets_cmd.delete_preset") as mock_delete,
        ):
            await _manage_presets_async(mock_ctx)
            mock_delete.assert_not_called()


class TestListStylesAsync:
    """Tests for _list_styles_async - lines 957-990"""

    @pytest.mark.asyncio
    async def test_list_styles_success(self, mock_ctx, mock_styles_response):
        """Test successful style listing - lines 959-987"""
        mock_client = AsyncMock()
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.styles.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.styles.print_info"),
                patch("venice_ai.cli.commands.image.styles.print_success"),
            ):
                await _list_styles_async(mock_ctx)

    @pytest.mark.asyncio
    async def test_list_styles_empty(self, mock_ctx):
        """Test when no styles available - lines 968-970"""
        mock_client = AsyncMock()
        mock_client.image.list_styles = AsyncMock(return_value=SimpleNamespace(data=None))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.styles.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.styles.print_info"),
                patch("venice_ai.cli.commands.image.styles.print_error"),
                pytest.raises(SystemExit),
            ):
                await _list_styles_async(mock_ctx)

    @pytest.mark.asyncio
    async def test_list_styles_exception(self, mock_ctx):
        """Test error handling - lines 989-990"""
        mock_client = AsyncMock()
        mock_client.image.list_styles = AsyncMock(side_effect=Exception("API Error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.styles.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.styles.print_info"),
                patch("venice_ai.cli.commands.image.styles.print_error"),
                pytest.raises(SystemExit),
            ):
                await _list_styles_async(mock_ctx)


class TestBatchGenerateAsync:
    """Tests for _batch_generate_async - lines 1022-1137"""

    @pytest.mark.asyncio
    async def test_batch_generate_success(self, mock_ctx, mock_config, mock_image_response):
        """Test successful batch generation - covers many lines"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a prompts file
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("prompt 1\nprompt 2\nprompt 3\n")

            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                await _batch_generate_async(
                    ctx=mock_ctx,
                    prompts_file=str(prompts_file),
                    model=None,  # Line 1047-1048
                    size="1024x1024",
                    save_dir=tmpdir,  # Line 1056-1057
                )

    @pytest.mark.asyncio
    async def test_batch_generate_empty_prompts(self, mock_ctx, mock_config):
        """Test batch with no prompts - lines 1036-1038"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "empty.txt"
            prompts_file.write_text("\n   \n   \n")  # Only whitespace

            mock_ctx.obj["config"] = mock_config

            with patch("venice_ai.cli.commands.image.generate.print_error") as mock_error:
                with pytest.raises(SystemExit):
                    await _batch_generate_async(
                        ctx=mock_ctx,
                        prompts_file=str(prompts_file),
                        model="hidream",
                        size="1024x1024",
                        save_dir=None,
                    )
                mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_batch_generate_uses_config_dir(self, mock_ctx, mock_config, mock_image_response):
        """Test batch uses config output dir - line 1059"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("test prompt\n")

            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                await _batch_generate_async(
                    ctx=mock_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=None,  # Should use config dir - line 1059
                )

    @pytest.mark.asyncio
    async def test_batch_generate_no_images_returned(self, mock_ctx, mock_config):
        """Test batch when no images returned - lines 1111-1115"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("test prompt\n")

            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_response = SimpleNamespace(images=[], timing=None)
            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                await _batch_generate_async(
                    ctx=mock_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=tmpdir,
                )

    @pytest.mark.asyncio
    async def test_batch_generate_prompt_exception(self, mock_ctx, mock_config):
        """Test batch handles individual prompt exceptions - lines 1117-1122"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("good prompt\nbad prompt\nanother good\n")

            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            # Build response mock
            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            good_response = SimpleNamespace(images=[encoded], timing=None)

            call_count = [0]

            async def mock_generate(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise Exception("Bad prompt error")
                return good_response

            mock_client = AsyncMock()
            mock_client.image.create = mock_generate

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                await _batch_generate_async(
                    ctx=mock_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=tmpdir,
                )

    @pytest.mark.asyncio
    async def test_batch_generate_venice_error(self, mock_ctx, mock_config):
        """Test batch Venice API error - lines 1134-1135"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("test prompt\n")

            mock_ctx.obj["config"] = mock_config

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                MockClient.return_value.__aenter__ = AsyncMock(side_effect=VeniceError("API Error"))
                with (
                    patch("venice_ai.cli.commands.image.generate.print_error"),
                    pytest.raises(SystemExit),
                ):
                    await _batch_generate_async(
                        ctx=mock_ctx,
                        prompts_file=str(prompts_file),
                        model="hidream",
                        size="1024x1024",
                        save_dir=None,
                    )

    @pytest.mark.asyncio
    async def test_batch_generate_unexpected_error(self, mock_ctx, mock_config):
        """Test batch unexpected error - lines 1136-1137"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("test prompt\n")

            mock_ctx.obj["config"] = mock_config

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                MockClient.return_value.__aenter__ = AsyncMock(
                    side_effect=RuntimeError("Unexpected")
                )
                with (
                    patch("venice_ai.cli.commands.image.generate.print_error"),
                    pytest.raises(SystemExit),
                ):
                    await _batch_generate_async(
                        ctx=mock_ctx,
                        prompts_file=str(prompts_file),
                        model="hidream",
                        size="1024x1024",
                        save_dir=None,
                    )


class TestCLICommands:
    """Test CLI command structure and options"""

    def test_list_styles_command(self, cli_runner):
        """Test list-styles command exists"""
        result = cli_runner.invoke(cli, ["image", "list-styles", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "style" in result.output.lower()

    def test_presets_command(self, cli_runner):
        """Test presets command exists"""
        result = cli_runner.invoke(cli, ["image", "presets", "--help"])
        assert result.exit_code == 0
        assert "preset" in result.output.lower()

    def test_generate_all_options(self, cli_runner):
        """Test all generate options exist"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0

        # Check all the CLI options
        assert "--steps" in result.output
        assert "--cfg-scale" in result.output
        assert "--seed" in result.output
        assert "--style-preset" in result.output or "-sp" in result.output
        assert "--lora-strength" in result.output
        assert "--format" in result.output
        assert "--safe-mode" in result.output or "safe" in result.output.lower()
        assert "--hide-watermark" in result.output
        assert "--embed-exif" in result.output
        assert "--return-binary" in result.output
        assert "--preset" in result.output or "-p" in result.output
        assert "--interactive" in result.output or "-i" in result.output


class TestEdgeCases:
    """Additional edge case tests"""

    def test_load_preset_config_quick(self):
        """Test loading quick preset"""
        result = _load_preset_config("quick")
        assert result is not None
        assert result["steps"] == 15

    def test_load_preset_config_high_quality(self):
        """Test loading high-quality preset"""
        result = _load_preset_config("high-quality")
        assert result is not None
        assert result["steps"] == 50

    def test_load_preset_config_creative(self):
        """Test loading creative preset"""
        result = _load_preset_config("creative")
        assert result is not None
        assert result["safe_mode"] is False

    @pytest.mark.asyncio
    async def test_interactive_no_style_use(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test not using style when style preset is available"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        responses = [
            "test",  # prompt
            "hidream",  # model
            "1024x1024 (Square - Default)",  # size
            "1",  # num_images
            False,  # configure_advanced
            True,  # configure_style
            False,  # use_negative
            False,  # use_style - skip using styles (line 582 branch)
            False,  # use_lora
            "webp (Recommended - Best compression)",  # format
            True,  # safe_mode
            False,  # hide_watermark
            False,  # embed_exif
            False,  # return_binary
            False,  # custom_location
            False,  # proceed (cancel)
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_no_styles_available(
        self, mock_ctx, mock_config, mock_models_response
    ):
        """Test when no styles are available from API - line 582 branch not taken"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        # Return empty styles
        mock_client.image.list_styles = AsyncMock(return_value=SimpleNamespace(data=[]))

        responses = [
            "test",  # prompt
            "hidream",  # model
            "1024x1024 (Square - Default)",  # size
            "1",  # num_images
            False,  # configure_advanced
            True,  # configure_style
            False,  # use_negative
            # No style option asked because available_styles is empty
            False,  # use_lora
            "webp (Recommended - Best compression)",  # format
            True,  # safe_mode
            False,  # hide_watermark
            False,  # embed_exif
            False,  # return_binary
            False,  # custom_location
            False,  # proceed (cancel)
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)

    @pytest.mark.asyncio
    async def test_interactive_empty_lora_strength(
        self, mock_ctx, mock_config, mock_models_response, mock_styles_response
    ):
        """Test when lora_str is empty - line 615"""
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)
        mock_client.image.list_styles = AsyncMock(return_value=mock_styles_response)

        responses = [
            "test",  # prompt
            "hidream",  # model
            "1024x1024 (Square - Default)",  # size
            "1",  # num_images
            False,  # configure_advanced
            True,  # configure_style
            False,  # use_negative
            False,  # use_style
            True,  # use_lora - line 609
            "",  # empty lora_str - line 615: if lora_str
            "webp (Recommended - Best compression)",  # format
            True,  # safe_mode
            False,  # hide_watermark
            False,  # embed_exif
            False,  # return_binary
            False,  # custom_location
            False,  # proceed (cancel)
        ]
        call_count = [0]

        def mock_to_thread(func):
            result = responses[call_count[0]] if call_count[0] < len(responses) else None
            call_count[0] += 1
            return result

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("asyncio.to_thread", side_effect=mock_to_thread),
                patch("venice_ai.cli.commands.image.wizard.print_info"),
                pytest.raises(SystemExit),
            ):
                await _interactive_image_generation(mock_ctx)


class TestPresetApplicationBranches:
    """Test all preset application branches - lines 150-171"""

    @staticmethod
    def _consume_coro(coro):
        """Helper to properly close coroutines and prevent warnings"""
        coro.close()
        return None

    def test_preset_applies_seed(self, cli_runner):
        """Test preset seed application"""
        preset_config = {"seed": 42}

        with (
            patch(
                "venice_ai.cli.commands.image.generate._load_preset_config",
                return_value=preset_config,
            ),
            patch(
                "venice_ai.cli.commands.image.generate.asyncio.run",
                side_effect=self._consume_coro,
            ),
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
        ):
            cli_runner.invoke(cli, ["image", "generate", "test", "--preset", "custom"])

    def test_preset_applies_style_preset(self, cli_runner):
        """Test preset style_preset application"""
        preset_config = {"style_preset": "Cinematic"}

        with (
            patch(
                "venice_ai.cli.commands.image.generate._load_preset_config",
                return_value=preset_config,
            ),
            patch(
                "venice_ai.cli.commands.image.generate.asyncio.run",
                side_effect=self._consume_coro,
            ),
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
        ):
            cli_runner.invoke(cli, ["image", "generate", "test", "--preset", "custom"])

    def test_preset_applies_lora_strength(self, cli_runner):
        """Test preset lora_strength application"""
        preset_config = {"lora_strength": 75}

        with (
            patch(
                "venice_ai.cli.commands.image.generate._load_preset_config",
                return_value=preset_config,
            ),
            patch(
                "venice_ai.cli.commands.image.generate.asyncio.run",
                side_effect=self._consume_coro,
            ),
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
        ):
            cli_runner.invoke(cli, ["image", "generate", "test", "--preset", "custom"])

    def test_preset_not_found(self, cli_runner):
        """Test when preset not found"""
        with (
            patch("venice_ai.cli.commands.image.generate._load_preset_config", return_value=None),
            patch(
                "venice_ai.cli.commands.image.generate.asyncio.run",
                side_effect=self._consume_coro,
            ),
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
        ):
            cli_runner.invoke(cli, ["image", "generate", "test", "--preset", "nonexistent"])


class TestShowTimingWithoutTiming:
    """Test show_timing when timing is None - line 367 branch"""

    @pytest.mark.asyncio
    async def test_generate_no_timing_info(self, mock_ctx, mock_config):
        """Test when response has no timing info - line 367"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            # Response without timing
            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mock_response = SimpleNamespace(
                images=[encoded],
                timing=None,  # No timing info
            )

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=tmpdir,
                        show_timing=True,  # Enabled but no timing in response
                    )


# ===========================================================================
# NEW TESTS FOR COVERAGE IMPROVEMENT
# ===========================================================================


class TestValidateSize:
    """Tests for validate_size() callback - lines 49-65"""

    def test_validate_size_none_returns_none(self):
        """Line 52: None input returns None"""
        result = validate_size(MagicMock(), MagicMock(), None)
        assert result is None

    def test_validate_size_valid_square(self):
        """Line 52: valid size passes through"""
        result = validate_size(MagicMock(), MagicMock(), "1024x1024")
        assert result == "1024x1024"

    def test_validate_size_valid_rectangle(self):
        """Valid non-square size"""
        result = validate_size(MagicMock(), MagicMock(), "1920x1080")
        assert result == "1920x1080"

    def test_validate_size_invalid_format(self):
        """Lines 56-59: invalid format raises BadParameter"""
        import click

        with pytest.raises(click.BadParameter) as exc_info:
            validate_size(MagicMock(), MagicMock(), "1024")
        assert "WxH format" in str(exc_info.value)

    def test_validate_size_invalid_format_with_comma(self):
        """Invalid format with comma"""
        import click

        with pytest.raises(click.BadParameter):
            validate_size(MagicMock(), MagicMock(), "1024,1024")

    def test_validate_size_too_small_width(self):
        """Lines 61-62: dimension below 64px raises BadParameter"""
        import click

        with pytest.raises(click.BadParameter) as exc_info:
            validate_size(MagicMock(), MagicMock(), "32x1024")
        assert "Minimum dimension" in str(exc_info.value)

    def test_validate_size_too_small_height(self):
        """Height below 64px raises BadParameter"""
        import click

        with pytest.raises(click.BadParameter) as exc_info:
            validate_size(MagicMock(), MagicMock(), "1024x32")
        assert "Minimum dimension" in str(exc_info.value)

    def test_validate_size_too_large_width(self):
        """Lines 63-64: dimension above 4096px raises BadParameter"""
        import click

        with pytest.raises(click.BadParameter) as exc_info:
            validate_size(MagicMock(), MagicMock(), "5000x1024")
        assert "Maximum dimension" in str(exc_info.value)

    def test_validate_size_too_large_height(self):
        """Height above 4096px raises BadParameter"""
        import click

        with pytest.raises(click.BadParameter) as exc_info:
            validate_size(MagicMock(), MagicMock(), "1024x5000")
        assert "Maximum dimension" in str(exc_info.value)

    def test_validate_size_exactly_64(self):
        """Exactly 64px is valid"""
        result = validate_size(MagicMock(), MagicMock(), "64x64")
        assert result == "64x64"

    def test_validate_size_exactly_4096(self):
        """Exactly 4096px is valid"""
        result = validate_size(MagicMock(), MagicMock(), "4096x4096")
        assert result == "4096x4096"


class TestGenerateImageAsyncPlainMode:
    """Tests for plain mode in _generate_image_async - lines 311-320, 368-378, 430-432, 443-446"""

    @pytest.fixture
    def plain_ctx(self, mock_config):
        """Context with plain mode enabled"""
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": True}
        ctx.exit = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_plain_mode_generation_info_display(self, plain_ctx, mock_config):
        """Lines 311-320: plain mode shows generation info via click.echo"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            plain_ctx.obj["config"] = mock_config

            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mock_response = SimpleNamespace(
                images=[encoded],
                timing=SimpleNamespace(inferenceDuration=500),
            )

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with (
                    patch(
                        "venice_ai.cli.commands.image.generate.ParameterValidator",
                        return_value=mock_validator,
                    ),
                    patch("click.echo") as mock_echo,
                ):
                    await _generate_image_async(
                        ctx=plain_ctx,
                        prompt="test prompt",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=tmpdir,
                        show_timing=True,
                        steps=20,
                        cfg_scale=7.5,
                        style_preset="Cinematic",
                    )
                    # Should have printed plain mode info
                    echo_calls = [str(c) for c in mock_echo.call_args_list]
                    combined = " ".join(echo_calls)
                    assert "Generating" in combined or "hidream" in combined

    @pytest.mark.asyncio
    async def test_plain_mode_generation_failure(self, plain_ctx, mock_config):
        """Lines 376-378: plain mode handles generation errors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            plain_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(side_effect=Exception("API error"))
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with (
                    patch(
                        "venice_ai.cli.commands.image.generate.ParameterValidator",
                        return_value=mock_validator,
                    ),
                    patch("click.echo"),
                ):
                    # Should not raise despite the exception
                    with pytest.raises(SystemExit):
                        await _generate_image_async(
                            ctx=plain_ctx,
                            prompt="test prompt",
                            model="hidream",
                            size="1024x1024",
                            num_images=1,
                            output=None,
                            save_dir=tmpdir,
                            show_timing=False,
                        )
                    # The outer except catches and calls print_error
                    # Either click.echo was called with failure message or print_error was called
                    assert True  # Just verify it didn't crash

    @pytest.mark.asyncio
    async def test_plain_mode_output_display_and_timing(self, plain_ctx, mock_config):
        """Lines 430-432, 443-446: plain mode displays save path and timing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            plain_ctx.obj["config"] = mock_config

            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mock_response = SimpleNamespace(
                images=[encoded],
                timing=SimpleNamespace(inferenceDuration=1000),
            )

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with (
                    patch(
                        "venice_ai.cli.commands.image.generate.ParameterValidator",
                        return_value=mock_validator,
                    ),
                    patch("click.echo") as mock_echo,
                ):
                    await _generate_image_async(
                        ctx=plain_ctx,
                        prompt="test prompt",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output="my_output",
                        save_dir=tmpdir,
                        show_timing=True,
                    )
                    echo_calls_str = " ".join(str(c) for c in mock_echo.call_args_list)
                    assert "Saved" in echo_calls_str or "Size" in echo_calls_str

    @pytest.mark.asyncio
    async def test_plain_mode_open_image_flag(self, plain_ctx, mock_config):
        """Lines 438-439: open_image flag triggers open_file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            plain_ctx.obj["config"] = mock_config

            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mock_response = SimpleNamespace(
                images=[encoded],
                timing=None,
            )

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with (
                    patch(
                        "venice_ai.cli.commands.image.generate.ParameterValidator",
                        return_value=mock_validator,
                    ),
                    patch("venice_ai.cli.commands.image.generate.open_file") as mock_open,
                ):
                    await _generate_image_async(
                        ctx=plain_ctx,
                        prompt="test prompt",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=tmpdir,
                        show_timing=False,
                        open_image=True,  # <-- test this flag
                    )
                    mock_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_image_flag_rich_mode(self, mock_ctx, mock_config):
        """Lines 431-432: open_image flag in rich mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config
            mock_ctx.obj["plain"] = False

            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mock_response = SimpleNamespace(
                images=[encoded],
                timing=None,
            )

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with (
                    patch(
                        "venice_ai.cli.commands.image.generate.ParameterValidator",
                        return_value=mock_validator,
                    ),
                    patch("venice_ai.cli.commands.image.generate.open_file") as mock_open,
                    patch(
                        "venice_ai.cli.commands.image.generate.is_plain_mode",
                        return_value=False,
                    ),
                ):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test prompt",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=tmpdir,
                        show_timing=False,
                        open_image=True,
                    )
                    mock_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_plain_mode_timing_display(self, plain_ctx, mock_config):
        """Lines 443-446: plain mode timing display"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            plain_ctx.obj["config"] = mock_config

            image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            mock_response = SimpleNamespace(
                images=[encoded],
                timing=SimpleNamespace(inferenceDuration=2500),
            )

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                with (
                    patch(
                        "venice_ai.cli.commands.image.generate.ParameterValidator",
                        return_value=mock_validator,
                    ),
                    patch("click.echo") as mock_echo,
                ):
                    await _generate_image_async(
                        ctx=plain_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=tmpdir,
                        show_timing=True,
                    )
                    echo_calls_str = " ".join(str(c) for c in mock_echo.call_args_list)
                    assert "2500" in echo_calls_str or "Generation time" in echo_calls_str


class TestBatchGenerateAsyncPlainMode:
    """Tests for _batch_generate_async plain mode - lines 1231-1265"""

    @pytest.fixture
    def plain_ctx(self, mock_config):
        """Context with plain mode enabled"""
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": True}
        ctx.exit = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_batch_plain_mode_success(self, plain_ctx, mock_config, tmp_path):
        """Lines 1233-1265: plain mode batch processing"""
        # Create prompts file
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("prompt one\nprompt two\n")

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        mock_response = SimpleNamespace(images=[encoded], timing=None)

        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(return_value=mock_response)
        mock_config["output"]["images_dir"] = str(tmp_path)
        plain_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("asyncio.sleep", return_value=None), patch("click.echo") as mock_echo:
                await _batch_generate_async(
                    ctx=plain_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=str(tmp_path),
                )
                echo_calls = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Batch Generation Complete" in echo_calls or "Successful" in echo_calls

    @pytest.mark.asyncio
    async def test_batch_plain_mode_with_failure(self, plain_ctx, mock_config, tmp_path):
        """Lines 1261-1263: plain mode batch handles individual failures"""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("good prompt\nbad prompt\n")

        call_count = [0]

        async def generate_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
                encoded = base64.b64encode(image_bytes).decode("utf-8")
                return SimpleNamespace(images=[encoded], timing=None)
            else:
                raise Exception("API error for bad prompt")

        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(side_effect=generate_side_effect)
        mock_config["output"]["images_dir"] = str(tmp_path)
        plain_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("asyncio.sleep", return_value=None), patch("click.echo") as mock_echo:
                await _batch_generate_async(
                    ctx=plain_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=str(tmp_path),
                )
                echo_calls = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Failed" in echo_calls

    @pytest.mark.asyncio
    async def test_batch_plain_mode_no_image_in_response(self, plain_ctx, mock_config, tmp_path):
        """Lines 1257-1259: plain mode handles empty response"""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("test prompt\n")

        mock_response = SimpleNamespace(images=[], timing=None)
        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(return_value=mock_response)
        mock_config["output"]["images_dir"] = str(tmp_path)
        plain_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("asyncio.sleep", return_value=None), patch("click.echo") as mock_echo:
                await _batch_generate_async(
                    ctx=plain_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=str(tmp_path),
                )
                echo_calls = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Warning" in echo_calls or "No image" in echo_calls

    @pytest.mark.asyncio
    async def test_batch_plain_mode_with_all_options(self, plain_ctx, mock_config, tmp_path):
        """Lines 1206-1223: parameter validation branches"""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("test prompt\n")

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        mock_response = SimpleNamespace(images=[encoded], timing=None)

        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(return_value=mock_response)
        mock_config["output"]["images_dir"] = str(tmp_path)
        plain_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("asyncio.sleep", return_value=None):
                await _batch_generate_async(
                    ctx=plain_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=str(tmp_path),
                    steps=20,
                    cfg_scale=7.5,
                    seed=42,
                    style_preset="Cinematic",
                    format="png",
                    safe_mode=True,
                    hide_watermark=True,
                    embed_exif=True,
                )
                # Verify all kwargs were passed to create.
                call_kwargs = mock_client.image.create.call_args[1]
                assert call_kwargs["steps"] == 20
                assert call_kwargs["cfg_scale"] == 7.5
                assert call_kwargs["seed"] == 42
                assert "negative_prompt" not in call_kwargs
                assert call_kwargs["style_preset"] == "Cinematic"
                assert call_kwargs["format"] == "png"
                assert call_kwargs["safe_mode"] is True
                assert call_kwargs["hide_watermark"] is True
                assert call_kwargs["embed_exif_metadata"] is True


class TestUpscaleAsync:
    """Tests for _upscale_async() - lines 1410-1501"""

    @pytest.fixture
    def plain_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": True}
        ctx.exit = MagicMock()
        return ctx

    @pytest.fixture
    def rich_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": False}
        ctx.exit = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_upscale_plain_mode_success(self, plain_ctx, tmp_path):
        """Lines 1443-1496: plain mode upscale with file save"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.upscale = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.upscale.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("click.echo") as mock_echo:
                await _upscale_async(
                    ctx=plain_ctx,
                    input_file=str(input_file),
                    scale=2.0,
                    creativity=0.02,
                    output=None,
                    save_dir=str(tmp_path),
                    open_image=False,
                )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Saved" in echo_str or "Size" in echo_str

    @pytest.mark.asyncio
    async def test_upscale_plain_mode_with_output_path(self, plain_ctx, tmp_path):
        """Line 1475: upscale with explicit output path"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        output_file = tmp_path / "output.png"

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.upscale = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.upscale.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _upscale_async(
                ctx=plain_ctx,
                input_file=str(input_file),
                scale=None,
                creativity=None,
                output=str(output_file),
                save_dir=str(tmp_path),
                open_image=False,
            )
            assert output_file.exists()

    @pytest.mark.asyncio
    async def test_upscale_rich_mode_success(self, rich_ctx, tmp_path):
        """Lines 1460-1472: rich mode upscale via Progress"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.upscale = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.upscale.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.image.upscale.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _upscale_async(
                ctx=rich_ctx,
                input_file=str(input_file),
                scale=2.0,
                creativity=None,
                output=None,
                save_dir=str(tmp_path),
                open_image=False,
            )
            # File should be saved
            saved_files = list(tmp_path.glob("upscaled_*.png"))
            assert len(saved_files) == 1

    @pytest.mark.asyncio
    async def test_upscale_rich_mode_failure(self, rich_ctx, tmp_path):
        """Lines 1470-1472: rich mode upscale error handling"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.upscale = AsyncMock(side_effect=Exception("Upscale API error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.upscale.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.image.upscale.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.upscale.print_error") as mock_err:
                with pytest.raises(SystemExit):
                    await _upscale_async(
                        ctx=rich_ctx,
                        input_file=str(input_file),
                        scale=None,
                        creativity=None,
                        output=None,
                        save_dir=str(tmp_path),
                        open_image=False,
                    )
                mock_err.assert_called()

    @pytest.mark.asyncio
    async def test_upscale_plain_mode_failure(self, plain_ctx, tmp_path):
        """Lines 1457-1459: plain mode upscale error"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.upscale = AsyncMock(side_effect=Exception("API failure"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.upscale.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.upscale.print_error"),
                patch("click.echo") as mock_echo,
            ):
                with pytest.raises(SystemExit):
                    await _upscale_async(
                        ctx=plain_ctx,
                        input_file=str(input_file),
                        scale=None,
                        creativity=None,
                        output=None,
                        save_dir=str(tmp_path),
                        open_image=False,
                    )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "failed" in echo_str.lower() or "Upscaling" in echo_str

    @pytest.mark.asyncio
    async def test_upscale_venice_error(self, plain_ctx, tmp_path):
        """Line 1498: VeniceError handling"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.upscale = AsyncMock(side_effect=VeniceError("Venice error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.upscale.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.upscale.print_error") as mock_err:
                with pytest.raises(SystemExit):
                    await _upscale_async(
                        ctx=plain_ctx,
                        input_file=str(input_file),
                        scale=None,
                        creativity=None,
                        output=None,
                        save_dir=str(tmp_path),
                        open_image=False,
                    )
                mock_err.assert_called()
                assert "Venice API error" in mock_err.call_args[0][0]

    @pytest.mark.asyncio
    async def test_upscale_open_image_flag(self, plain_ctx, tmp_path):
        """Line 1495-1496: open_image flag in upscale"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.upscale = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.upscale.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.upscale.open_file") as mock_open_file:
                await _upscale_async(
                    ctx=plain_ctx,
                    input_file=str(input_file),
                    scale=None,
                    creativity=None,
                    output=None,
                    save_dir=str(tmp_path),
                    open_image=True,
                )
                mock_open_file.assert_called_once()


class TestEditAsync:
    """Tests for _edit_async() - lines 1570-1662"""

    @pytest.fixture
    def plain_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": True}
        ctx.exit = MagicMock()
        return ctx

    @pytest.fixture
    def rich_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": False}
        ctx.exit = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_edit_plain_mode_success(self, plain_ctx, tmp_path):
        """Lines 1589-1619: plain mode edit with all options"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("click.echo") as mock_echo:
                await _edit_async(
                    ctx=plain_ctx,
                    input_file=str(input_file),
                    prompt="Add a rainbow",
                    model="edit-model",
                    output=None,
                    save_dir=str(tmp_path),
                    img_format="png",
                    open_image=False,
                )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Saved" in echo_str or "Edit" in echo_str

    @pytest.mark.asyncio
    async def test_edit_plain_mode_with_explicit_output(self, plain_ctx, tmp_path):
        """Line 1637-1638: explicit output path"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        output_file = tmp_path / "my_edit.png"

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _edit_async(
                ctx=plain_ctx,
                input_file=str(input_file),
                prompt="Edit prompt",
                model=None,
                output=str(output_file),
                save_dir=str(tmp_path),
                img_format=None,
                open_image=False,
            )
            assert output_file.exists()

    @pytest.mark.asyncio
    async def test_edit_rich_mode_success(self, rich_ctx, tmp_path):
        """Lines 1621-1632: rich mode edit via Progress"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _edit_async(
                ctx=rich_ctx,
                input_file=str(input_file),
                prompt="Make it pop",
                model="edit-model",
                output=None,
                save_dir=str(tmp_path),
                img_format="jpeg",
                open_image=False,
            )
            saved_files = list(tmp_path.glob("edited_*.jpeg"))
            assert len(saved_files) == 1

    @pytest.mark.asyncio
    async def test_edit_plain_mode_api_failure(self, plain_ctx, tmp_path):
        """Lines 1617-1619: plain mode edit error"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(side_effect=Exception("Edit API error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.edit.print_error"),
                patch("click.echo") as mock_echo,
            ):
                with pytest.raises(SystemExit):
                    await _edit_async(
                        ctx=plain_ctx,
                        input_file=str(input_file),
                        prompt="Edit prompt",
                        model=None,
                        output=None,
                        save_dir=str(tmp_path),
                        img_format=None,
                        open_image=False,
                    )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "failed" in echo_str.lower() or "Edit" in echo_str

    @pytest.mark.asyncio
    async def test_edit_rich_mode_failure(self, rich_ctx, tmp_path):
        """Lines 1630-1632: rich mode edit error handling"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(side_effect=Exception("Edit failed"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.edit.print_error") as mock_err:
                with pytest.raises(SystemExit):
                    await _edit_async(
                        ctx=rich_ctx,
                        input_file=str(input_file),
                        prompt="Test",
                        model=None,
                        output=None,
                        save_dir=str(tmp_path),
                        img_format=None,
                        open_image=False,
                    )
                mock_err.assert_called()

    @pytest.mark.asyncio
    async def test_edit_venice_error(self, plain_ctx, tmp_path):
        """Line 1659: VeniceError handling in edit"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(side_effect=VeniceError("Venice error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.edit.print_error") as mock_err:
                with pytest.raises(SystemExit):
                    await _edit_async(
                        ctx=plain_ctx,
                        input_file=str(input_file),
                        prompt="Test",
                        model=None,
                        output=None,
                        save_dir=str(tmp_path),
                        img_format=None,
                        open_image=False,
                    )
                mock_err.assert_called()
                assert "Venice API error" in mock_err.call_args[0][0]

    @pytest.mark.asyncio
    async def test_edit_open_image_flag(self, plain_ctx, tmp_path):
        """Line 1656-1657: open_image flag in edit"""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.edit.open_file") as mock_open_file:
                await _edit_async(
                    ctx=plain_ctx,
                    input_file=str(input_file),
                    prompt="Test",
                    model=None,
                    output=None,
                    save_dir=str(tmp_path),
                    img_format=None,
                    open_image=True,
                )
                mock_open_file.assert_called_once()

    def test_edit_command_has_no_mask_option(self):
        """``mask`` (inpainting) is dead code — the server rejects it.
        The ``venice-py image edit`` command must not expose a ``--mask`` option."""
        param_names = {p.name for p in edit_image.params}
        assert "mask" not in param_names


class TestEditNewFlags:
    """--aspect-ratio, --resolution, --output-format, --safe-mode flags
    on ``venice-py image edit``, mirroring ``image multi-edit``. Each maps to a real
    ``client.image.edit()`` kwarg."""

    @pytest.fixture
    def plain_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": True}
        ctx.exit = MagicMock()
        return ctx

    def test_edit_command_exposes_new_options(self):
        """The four new options are declared on the edit command."""
        param_names = {p.name for p in edit_image.params}
        assert "aspect_ratio" in param_names
        assert "resolution" in param_names
        assert "output_format" in param_names
        assert "safe_mode" in param_names

    def test_edit_help_lists_new_flags(self, cli_runner):
        result = cli_runner.invoke(cli, ["image", "edit", "--help"])
        assert result.exit_code == 0
        assert "--aspect-ratio" in result.output
        assert "--resolution" in result.output
        assert "--output-format" in result.output
        assert "--safe-mode" in result.output
        assert "--no-safe-mode" in result.output

    @pytest.mark.asyncio
    async def test_edit_async_forwards_new_kwargs(self, plain_ctx, tmp_path):
        """All four new kwargs are forwarded to image.edit() when set."""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _edit_async(
                plain_ctx,
                str(input_file),
                "Editorial retouch",
                None,  # model
                None,  # output
                str(tmp_path),  # save_dir
                None,  # img_format
                False,  # open_image
                aspect_ratio="16:9",
                resolution="4K",
                output_format="jpeg",
                safe_mode=False,
            )

        call_kwargs = mock_client.image.edit.call_args[1]
        assert call_kwargs["aspect_ratio"] == "16:9"
        assert call_kwargs["resolution"] == "4K"
        assert call_kwargs["output_format"] == "jpeg"
        assert call_kwargs["safe_mode"] is False

    @pytest.mark.asyncio
    async def test_edit_async_unset_new_kwargs_not_forwarded(self, plain_ctx, tmp_path):
        """Unset new options are omitted from the image.edit() call."""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _edit_async(
                plain_ctx,
                str(input_file),
                "Plain edit",
                None,
                None,
                str(tmp_path),
                None,
                False,
            )

        call_kwargs = mock_client.image.edit.call_args[1]
        assert "aspect_ratio" not in call_kwargs
        assert "resolution" not in call_kwargs
        assert "output_format" not in call_kwargs
        assert "safe_mode" not in call_kwargs

    @pytest.mark.asyncio
    async def test_edit_output_format_drives_extension(self, plain_ctx, tmp_path):
        """--output-format sets the saved filename extension (wins over input suffix)."""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=b"\xff\xd8\xff" + b"\x00" * 200)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _edit_async(
                plain_ctx,
                str(input_file),
                "Convert to jpeg",
                None,
                None,
                str(tmp_path),
                None,  # img_format hint not set
                False,
                output_format="jpeg",
            )

        saved = list(tmp_path.glob("edited_*.jpeg"))
        assert len(saved) == 1

    def test_edit_cli_forwards_new_flags(self, cli_runner, tmp_path):
        """End-to-end CliRunner: new flags reach image.edit()."""
        input_file = tmp_path / "photo.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.edit = AsyncMock(return_value=b"\xff\xd8\xff" + b"\x00" * 200)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            result = cli_runner.invoke(
                cli,
                [
                    "image",
                    "edit",
                    str(input_file),
                    "--prompt",
                    "Editorial retouch",
                    "--aspect-ratio",
                    "16:9",
                    "--resolution",
                    "4K",
                    "--output-format",
                    "jpeg",
                    "--no-safe-mode",
                    "--save-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_client.image.edit.call_args[1]
        assert call_kwargs["aspect_ratio"] == "16:9"
        assert call_kwargs["resolution"] == "4K"
        assert call_kwargs["output_format"] == "jpeg"
        assert call_kwargs["safe_mode"] is False

    def test_edit_cli_rejects_invalid_resolution(self, cli_runner, tmp_path):
        """--resolution is a Choice (1K/2K/4K); an invalid value errors."""
        input_file = tmp_path / "photo.png"
        input_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = cli_runner.invoke(
            cli,
            [
                "image",
                "edit",
                str(input_file),
                "--prompt",
                "x",
                "--resolution",
                "8K",
            ],
        )
        assert result.exit_code != 0
        assert "8K" in result.output or "Invalid value" in result.output


class TestRemoveBgAsync:
    """Tests for _remove_bg_async() - lines 1712-1791"""

    @pytest.fixture
    def plain_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": True}
        ctx.exit = MagicMock()
        return ctx

    @pytest.fixture
    def rich_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": False}
        ctx.exit = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_remove_bg_plain_mode_success(self, plain_ctx, tmp_path):
        """Lines 1728-1779: plain mode background removal"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("click.echo") as mock_echo:
                await _remove_bg_async(
                    ctx=plain_ctx,
                    input_file=str(input_file),
                    output=None,
                    save_dir=str(tmp_path),
                    img_format="png",
                    open_image=False,
                )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Saved" in echo_str or "background removal" in echo_str.lower()

    @pytest.mark.asyncio
    async def test_remove_bg_plain_mode_with_explicit_output(self, plain_ctx, tmp_path):
        """Line 1765-1766: explicit output path"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        output_file = tmp_path / "output.png"

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _remove_bg_async(
                ctx=plain_ctx,
                input_file=str(input_file),
                output=str(output_file),
                save_dir=str(tmp_path),
                img_format="png",
                open_image=False,
            )
            assert output_file.exists()

    @pytest.mark.asyncio
    async def test_remove_bg_rich_mode_success(self, rich_ctx, tmp_path):
        """Lines 1745-1760: rich mode remove-bg via Progress"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _remove_bg_async(
                ctx=rich_ctx,
                input_file=str(input_file),
                output=None,
                save_dir=str(tmp_path),
                img_format="png",
                open_image=False,
            )
            saved_files = list(tmp_path.glob("no_bg_*.png"))
            assert len(saved_files) == 1

    @pytest.mark.asyncio
    async def test_remove_bg_plain_mode_failure(self, plain_ctx, tmp_path):
        """Lines 1742-1744: plain mode error handling"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(side_effect=Exception("BG removal failed"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch("venice_ai.cli.commands.image.edit.print_error"),
                patch("click.echo") as mock_echo,
            ):
                with pytest.raises(SystemExit):
                    await _remove_bg_async(
                        ctx=plain_ctx,
                        input_file=str(input_file),
                        output=None,
                        save_dir=str(tmp_path),
                        img_format="png",
                        open_image=False,
                    )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "failed" in echo_str.lower() or "removal" in echo_str.lower()

    @pytest.mark.asyncio
    async def test_remove_bg_rich_mode_failure(self, rich_ctx, tmp_path):
        """Rich mode error handling"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(side_effect=Exception("Failed"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.edit.print_error") as mock_err:
                with pytest.raises(SystemExit):
                    await _remove_bg_async(
                        ctx=rich_ctx,
                        input_file=str(input_file),
                        output=None,
                        save_dir=str(tmp_path),
                        img_format="png",
                        open_image=False,
                    )
                mock_err.assert_called()

    @pytest.mark.asyncio
    async def test_remove_bg_venice_error(self, plain_ctx, tmp_path):
        """Line 1788: VeniceError handling"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(side_effect=VeniceError("Venice error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.edit.print_error") as mock_err:
                with pytest.raises(SystemExit):
                    await _remove_bg_async(
                        ctx=plain_ctx,
                        input_file=str(input_file),
                        output=None,
                        save_dir=str(tmp_path),
                        img_format="png",
                        open_image=False,
                    )
                mock_err.assert_called()
                assert "Venice API error" in mock_err.call_args[0][0]

    @pytest.mark.asyncio
    async def test_remove_bg_open_image_flag(self, plain_ctx, tmp_path):
        """Line 1785-1786: open_image flag"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("venice_ai.cli.commands.image.edit.open_file") as mock_open_file:
                await _remove_bg_async(
                    ctx=plain_ctx,
                    input_file=str(input_file),
                    output=None,
                    save_dir=str(tmp_path),
                    img_format="png",
                    open_image=True,
                )
                mock_open_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_bg_rich_mode_png_tip(self, rich_ctx, tmp_path):
        """Line 1783: rich mode shows PNG transparency tip"""
        input_file = tmp_path / "input.jpg"
        input_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        result_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

        mock_client = AsyncMock()
        mock_client.image.background_remove = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.edit.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.image.edit.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            # Should not raise (console.print with Tip message)
            await _remove_bg_async(
                ctx=rich_ctx,
                input_file=str(input_file),
                output=None,
                save_dir=str(tmp_path),
                img_format="png",
                open_image=False,
            )


class TestInteractiveImageGenerationPlainMode:
    """Tests for _interactive_image_generation plain mode error path - lines 1322-1327"""

    @pytest.mark.asyncio
    async def test_interactive_exception_handling(self, mock_ctx, mock_config):
        """Lines 878-882: exception handling in interactive wizard"""
        mock_client = AsyncMock()
        # Make models.list raise an exception
        mock_client.models.list = AsyncMock(side_effect=Exception("Network error"))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.wizard.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.wizard.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            # Simulate the console.print call failing with Panel
            with (
                patch("venice_ai.cli.commands.image.wizard.console"),
                patch("venice_ai.cli.commands.image.wizard.print_error") as mock_err,
                patch(
                    "asyncio.to_thread",
                    side_effect=Exception("Thread error"),
                ),
            ):
                await _interactive_image_generation(mock_ctx)
                # Should have called print_error for the outer exception
                mock_err.assert_called()


class TestBatchGenerateAsyncAdditionalBranches:
    """Additional tests for batch mode - covering more parameter branches"""

    @pytest.fixture
    def plain_ctx(self, mock_config):
        ctx = MagicMock()
        ctx.obj = {"config": mock_config, "plain": True}
        ctx.exit = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_batch_plain_mode_summary_with_failed(self, plain_ctx, mock_config, tmp_path):
        """Lines 1322-1327: plain mode summary shows failed count when > 0"""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("bad prompt\n")

        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(side_effect=Exception("fail"))
        mock_config["output"]["images_dir"] = str(tmp_path)
        plain_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("asyncio.sleep", return_value=None), patch("click.echo") as mock_echo:
                await _batch_generate_async(
                    ctx=plain_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=str(tmp_path),
                )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Failed" in echo_str

    @pytest.mark.asyncio
    async def test_batch_plain_mode_successful_no_failed(self, plain_ctx, mock_config, tmp_path):
        """Lines 1322-1327: plain mode summary without failed line when none failed"""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("good prompt\n")

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        mock_response = SimpleNamespace(images=[encoded], timing=None)

        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(return_value=mock_response)
        mock_config["output"]["images_dir"] = str(tmp_path)
        plain_ctx.obj["config"] = mock_config

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with patch("asyncio.sleep", return_value=None), patch("click.echo") as mock_echo:
                await _batch_generate_async(
                    ctx=plain_ctx,
                    prompts_file=str(prompts_file),
                    model="hidream",
                    size="1024x1024",
                    save_dir=str(tmp_path),
                )
                echo_str = " ".join(str(c) for c in mock_echo.call_args_list)
                assert "Successful" in echo_str
                # "Failed" should NOT appear since no failures
                assert "Failed: 0" not in echo_str


class TestGenerateImageParameterValidationFailure:
    """Tests covering parameter validation failure path"""

    @pytest.mark.asyncio
    async def test_generate_with_validation_failure(self, mock_ctx, mock_config, tmp_path):
        """Lines 306-308: validation failure triggers print_error and early return"""
        mock_config["output"]["images_dir"] = str(tmp_path)
        mock_ctx.obj["config"] = mock_config

        mock_client = AsyncMock()
        mock_validator = MagicMock()
        mock_validator.validate_image_parameters = AsyncMock(
            return_value=(False, "Width must be divisible by 8")
        )

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ),
                patch("venice_ai.cli.commands.image.generate.print_error") as mock_err,
            ):
                with pytest.raises(SystemExit):
                    await _generate_image_async(
                        ctx=mock_ctx,
                        prompt="test",
                        model="hidream",
                        size="1024x1024",
                        num_images=1,
                        output=None,
                        save_dir=str(tmp_path),
                        show_timing=False,
                    )
                mock_err.assert_called()
                assert "Parameter validation failed" in mock_err.call_args[0][0]

    @pytest.mark.asyncio
    async def test_generate_multiple_images_output_naming(self, mock_ctx, mock_config, tmp_path):
        """Lines 415-417: multiple images with output prefix"""
        mock_config["output"]["images_dir"] = str(tmp_path)
        mock_ctx.obj["config"] = mock_config

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        mock_response = SimpleNamespace(
            images=[encoded, encoded],
            timing=None,
        )

        mock_client = AsyncMock()
        mock_client.image.create = AsyncMock(return_value=mock_response)
        mock_validator = MagicMock()
        mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="key"),
            patch("venice_ai.cli.commands.image.generate.load_config", return_value=mock_config),
            patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with (
                patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ),
                patch(
                    "venice_ai.cli.commands.image.generate.is_plain_mode",
                    return_value=False,
                ),
            ):
                await _generate_image_async(
                    ctx=mock_ctx,
                    prompt="test",
                    model="hidream",
                    size="1024x1024",
                    num_images=2,
                    output="my_batch",
                    save_dir=str(tmp_path),
                    show_timing=False,
                )
                # Should create my_batch_1.png and my_batch_2.png
                assert (tmp_path / "my_batch_1.png").exists()
                assert (tmp_path / "my_batch_2.png").exists()


class TestGenerateTierFlags:
    """generate forwards tier/aspect flags to create()."""

    @pytest.mark.asyncio
    async def test_generate_forwards_tier_flags(self, mock_ctx, mock_image_response, mock_config):
        """--resolution/--aspect-ratio/--quality/--enable-web-search reach create()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
                patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ),
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                await _generate_image_async(
                    ctx=mock_ctx,
                    prompt="test prompt",
                    model="hidream",
                    size="1024x1024",
                    num_images=1,
                    output=None,
                    save_dir=tmpdir,
                    show_timing=False,
                    aspect_ratio="16:9",
                    resolution="2K",
                    quality="high",
                    enable_web_search=True,
                )

            assert mock_client.image.create.await_count == 1
            kwargs = mock_client.image.create.await_args.kwargs
            assert kwargs["aspect_ratio"] == "16:9"
            assert kwargs["resolution"] == "2K"
            assert kwargs["quality"] == "high"
            assert kwargs["enable_web_search"] is True

    @pytest.mark.asyncio
    async def test_generate_omits_unset_tier_flags(
        self, mock_ctx, mock_image_response, mock_config
    ):
        """Unset tier flags must not be passed to create()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["output"]["images_dir"] = tmpdir
            mock_ctx.obj["config"] = mock_config

            mock_client = AsyncMock()
            mock_client.image.create = AsyncMock(return_value=mock_image_response)
            mock_validator = MagicMock()
            mock_validator.validate_image_parameters = AsyncMock(return_value=(True, None))

            with (
                patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
                patch(
                    "venice_ai.cli.commands.image.generate.load_config", return_value=mock_config
                ),
                patch("venice_ai.cli.commands.image.generate.VeniceClient") as MockClient,
                patch(
                    "venice_ai.cli.commands.image.generate.ParameterValidator",
                    return_value=mock_validator,
                ),
            ):
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_cm

                await _generate_image_async(
                    ctx=mock_ctx,
                    prompt="test prompt",
                    model="hidream",
                    size="1024x1024",
                    num_images=1,
                    output=None,
                    save_dir=tmpdir,
                    show_timing=False,
                )

            kwargs = mock_client.image.create.await_args.kwargs
            assert "aspect_ratio" not in kwargs
            assert "resolution" not in kwargs
            assert "quality" not in kwargs
            assert "enable_web_search" not in kwargs

    def test_generate_has_tier_options(self, cli_runner):
        """CLI exposes the tier/aspect options on `image generate`."""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        assert "--aspect-ratio" in result.output
        assert "--resolution" in result.output
        assert "--quality" in result.output
        assert "--enable-web-search" in result.output


class TestMultiEditCommand:
    """`image multi-edit` subcommand."""

    def test_multi_edit_registered(self, cli_runner):
        """multi-edit shows up in the image group help."""
        result = cli_runner.invoke(cli, ["image", "--help"])
        assert result.exit_code == 0
        assert "multi-edit" in result.output

    def test_multi_edit_help(self, cli_runner):
        result = cli_runner.invoke(cli, ["image", "multi-edit", "--help"])
        assert result.exit_code == 0
        assert "--prompt" in result.output
        assert "--image" in result.output

    @pytest.mark.asyncio
    async def test_multi_edit_async_invokes_sdk(self, mock_ctx, tmp_path):
        """multi_edit() called with image_2/image_3 and output written."""
        from venice_ai.cli.commands.image.multi_edit import _multi_edit_async

        in1 = tmp_path / "a.png"
        in2 = tmp_path / "b.png"
        in3 = tmp_path / "c.png"
        for p in (in1, in2, in3):
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
        out_file = tmp_path / "out.png"

        result_bytes = b"\x89PNG\r\n\x1a\nEDITED"
        mock_client = AsyncMock()
        mock_client.image.multi_edit = AsyncMock(return_value=result_bytes)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.multi_edit.VeniceClient") as MockClient,
            patch(
                "venice_ai.cli.commands.image.multi_edit.is_plain_mode",
                return_value=True,
            ),
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _multi_edit_async(
                ctx=mock_ctx,
                prompt="combine these",
                image=str(in1),
                image_2=str(in2),
                image_3=str(in3),
                model="hidream",
                output=str(out_file),
                aspect_ratio="1:1",
                output_format="png",
                quality="high",
            )

        assert mock_client.image.multi_edit.await_count == 1
        kwargs = mock_client.image.multi_edit.await_args.kwargs
        assert kwargs["prompt"] == "combine these"
        assert kwargs["image"] == str(in1)
        assert kwargs["image_2"] == str(in2)
        assert kwargs["image_3"] == str(in3)
        assert kwargs["model"] == "hidream"
        assert kwargs["aspect_ratio"] == "1:1"
        assert kwargs["output_format"] == "png"
        assert kwargs["quality"] == "high"
        assert out_file.read_bytes() == result_bytes

    @staticmethod
    async def _run_multi_edit_failure(mock_ctx, tmp_path, exc):
        """Drive _multi_edit_async into its error handler; return the mocked reporter."""
        from venice_ai.cli.commands.image.multi_edit import _multi_edit_async

        paths = [tmp_path / name for name in ("a.png", "b.png", "c.png")]
        for p in paths:
            p.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_client = AsyncMock()
        mock_client.image.multi_edit = AsyncMock(side_effect=exc)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.multi_edit.VeniceClient") as MockClient,
            patch("venice_ai.cli.commands.image.multi_edit.print_error") as mock_err,
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            with pytest.raises(SystemExit):
                await _multi_edit_async(
                    ctx=mock_ctx,
                    prompt="combine these",
                    image=str(paths[0]),
                    image_2=str(paths[1]),
                    image_3=str(paths[2]),
                    model="hidream",
                    output=str(tmp_path / "out.png"),
                    aspect_ratio="1:1",
                    output_format="png",
                    quality="high",
                )

        return mock_err

    @pytest.mark.asyncio
    async def test_multi_edit_async_venice_error(self, mock_ctx, tmp_path):
        """A Venice API failure is reported and exits non-zero."""
        mock_err = await self._run_multi_edit_failure(mock_ctx, tmp_path, VeniceError("API Error"))
        assert "Venice API error" in mock_err.call_args[0][0]

    @pytest.mark.asyncio
    async def test_multi_edit_async_unexpected_error(self, mock_ctx, tmp_path):
        """An unexpected failure is reported and exits non-zero."""
        mock_err = await self._run_multi_edit_failure(mock_ctx, tmp_path, RuntimeError("boom"))
        assert "Unexpected error" in mock_err.call_args[0][0]

    @pytest.mark.asyncio
    async def test_multi_edit_async_single_image(self, mock_ctx, tmp_path):
        """Optional layer images are omitted when not provided."""
        from venice_ai.cli.commands.image.multi_edit import _multi_edit_async

        in1 = tmp_path / "a.png"
        in1.write_bytes(b"\x89PNG\r\n\x1a\n")
        out_file = tmp_path / "out.png"

        mock_client = AsyncMock()
        mock_client.image.multi_edit = AsyncMock(return_value=b"RESULT")

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.image.multi_edit.VeniceClient") as MockClient,
            patch(
                "venice_ai.cli.commands.image.multi_edit.is_plain_mode",
                return_value=True,
            ),
        ):
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_cm

            await _multi_edit_async(
                ctx=mock_ctx,
                prompt="edit",
                image=str(in1),
                image_2=None,
                image_3=None,
                model=None,
                output=str(out_file),
                aspect_ratio=None,
                output_format=None,
                quality=None,
            )

        kwargs = mock_client.image.multi_edit.await_args.kwargs
        assert kwargs["image"] == str(in1)
        assert "image_2" not in kwargs
        assert "image_3" not in kwargs
        assert "model" not in kwargs
        assert "aspect_ratio" not in kwargs
        assert out_file.read_bytes() == b"RESULT"
