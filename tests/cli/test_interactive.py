"""
Tests for CLI interactive image generation wizard

This module tests the interactive wizard flow in _interactive_image_generation()
which uses questionary for user prompts and asyncio.to_thread for async execution.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from venice_ai.cli.commands.image.wizard import _interactive_image_generation
from venice_ai.exceptions import VeniceError


@pytest.fixture
def mock_click_context():
    """Fixture providing a mocked Click context"""
    ctx = MagicMock()
    ctx.obj = {
        "config": {
            "defaults": {"image_model": "hidream"},
            "output": {"images_dir": str(Path.home() / "Pictures" / "venice")},
        }
    }
    return ctx


@pytest.fixture
def mock_venice_client():
    """Fixture providing a mocked VeniceClient with models and styles"""
    client = MagicMock()

    # Mock models.list response
    models_response = MagicMock()
    models_response.data = [
        MagicMock(id="hidream"),
        MagicMock(id="fluently-xl"),
        MagicMock(id="stable-diffusion-xl"),
    ]
    client.models.list = AsyncMock(return_value=models_response)

    # Mock image.list_styles response
    styles_response = MagicMock()
    styles_response.data = ["anime", "photorealistic", "oil-painting"]
    client.image.list_styles = AsyncMock(return_value=styles_response)

    # Make it work as async context manager
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    return client


@pytest.fixture
def mock_console_functions():
    """Mock console output functions"""
    with (
        patch("venice_ai.cli.commands.image.wizard.print_error") as mock_error,
        patch("venice_ai.cli.commands.image.wizard.print_info") as mock_info,
        patch("venice_ai.cli.commands.image.wizard.console") as mock_console,
    ):
        yield {
            "error": mock_error,
            "success": MagicMock(),
            "info": mock_info,
            "console": mock_console,
        }


@pytest.fixture
def mock_ensure_api_key():
    """Mock ensure_api_key to return a test API key.

    wizard.py resolves the key via ``config.get_client_kwargs()`` (which calls
    ``config.ensure_api_key``), so patch it at the config module — wizard no
    longer imports ``ensure_api_key`` directly.
    """
    with patch("venice_ai.cli.config.ensure_api_key", return_value="test_api_key"):
        yield


@pytest.fixture
def mock_load_config(mock_click_context):
    """Mock load_config to return test configuration"""
    with patch(
        "venice_ai.cli.commands.image.wizard.load_config",
        return_value=mock_click_context.obj["config"],
    ):
        yield


class TestInteractiveWizardBasic:
    """Test basic interactive wizard workflows"""

    async def test_successful_complete_workflow(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test successful complete workflow through the wizard"""

        # Mock questionary responses in order
        questionary_responses = {
            "text": [
                "a beautiful sunset over mountains",  # prompt
                "1",  # num_images
            ],
            "select": [
                "hidream",  # model
                "1024x1024 (Square - Default)",  # size_choice
                "webp (Recommended - Best compression)",  # format_choice
            ],
            "confirm": [
                False,  # configure_advanced
                False,  # configure_style
                True,  # safe_mode
                False,  # hide_watermark
                False,  # embed_exif
                False,  # return_binary
                False,  # custom_location
                True,  # proceed
                False,  # save_as_preset
            ],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            # Setup questionary mocks with sequential responses
            call_count = {"text": 0, "select": 0, "confirm": 0, "path": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            # Run the interactive wizard
            await _interactive_image_generation(mock_click_context)

            # Verify _generate_image_async was called with correct parameters
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args.kwargs["prompt"] == "a beautiful sunset over mountains"
            assert call_args.kwargs["model"] == "hidream"
            assert call_args.kwargs["size"] == "1024x1024"
            assert call_args.kwargs["num_images"] == 1
            assert call_args.kwargs["steps"] is None
            assert call_args.kwargs["cfg_scale"] is None
            assert call_args.kwargs["seed"] is None
            assert call_args.kwargs["format"] == "webp"
            assert call_args.kwargs["safe_mode"] is True

    async def test_cancellation_at_confirmation(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test that cancelling at confirmation prevents generation"""

        # Mock questionary to proceed through wizard but cancel at confirmation
        questionary_responses = {
            "text": ["test prompt", "1"],
            "select": [
                "hidream",
                "1024x1024 (Square - Default)",
                "webp (Recommended - Best compression)",
            ],
            "confirm": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],  # proceed=False (8th confirm)
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            # Setup questionary mocks with sequential responses
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            # Run the wizard
            await _interactive_image_generation(mock_click_context)

            # Verify generation was NOT called
            mock_generate.assert_not_called()

            # Verify cancellation message was shown
            mock_console_functions["info"].assert_called_with("Image generation cancelled.")


class TestInteractiveWizardAdvancedParams:
    """Test advanced parameter configuration"""

    async def test_with_advanced_parameters_enabled(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard with advanced generation parameters enabled"""

        questionary_responses = {
            "text": [
                "test prompt",  # prompt
                "1",  # num_images
                "30",  # steps
                "7.5",  # cfg_scale
                "12345",  # seed
            ],
            "select": [
                "hidream",  # model
                "1024x1024 (Square - Default)",  # size
                "png (Highest quality)",  # format
            ],
            "confirm": [
                True,  # configure_advanced
                True,  # use_seed
                False,  # configure_style
                True,  # safe_mode
                False,  # hide_watermark
                False,  # embed_exif
                False,  # return_binary
                False,  # custom_location
                True,  # proceed
                False,  # save_as_preset
            ],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify advanced parameters were passed
            call_args = mock_generate.call_args
            assert call_args.kwargs["steps"] == 30
            assert call_args.kwargs["cfg_scale"] == 7.5
            assert call_args.kwargs["seed"] == 12345
            assert call_args.kwargs["format"] == "png"

    async def test_with_advanced_parameters_disabled(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard with advanced parameters disabled"""

        questionary_responses = {
            "text": ["test prompt", "1"],
            "select": [
                "hidream",
                "1024x1024 (Square - Default)",
                "webp (Recommended - Best compression)",
            ],
            "confirm": [False, False, True, False, False, False, False, True, False],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify advanced parameters are None
            call_args = mock_generate.call_args
            assert call_args.kwargs["steps"] is None
            assert call_args.kwargs["cfg_scale"] is None
            assert call_args.kwargs["seed"] is None


class TestInteractiveWizardStyleParams:
    """Test style parameter configuration"""

    async def test_with_style_parameters_enabled(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard with style parameters enabled"""

        questionary_responses = {
            "text": [
                "test prompt",  # prompt
                "1",  # num_images
                "75",  # lora_strength
            ],
            "select": [
                "hidream",  # model
                "1024x1024 (Square - Default)",  # size
                "anime",  # style_preset
                "webp (Recommended - Best compression)",  # format
            ],
            "confirm": [
                False,  # configure_advanced
                True,  # configure_style
                True,  # use_style
                True,  # use_lora
                True,  # safe_mode
                False,  # hide_watermark
                False,  # embed_exif
                False,  # return_binary
                False,  # custom_location
                True,  # proceed
                False,  # save_as_preset
            ],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify style parameters were passed.
            call_args = mock_generate.call_args
            assert "negative_prompt" not in call_args.kwargs
            assert call_args.kwargs["style_preset"] == "anime"
            assert call_args.kwargs["lora_strength"] == 75

    async def test_with_style_parameters_disabled(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard with style parameters disabled"""

        questionary_responses = {
            "text": ["test prompt", "1"],
            "select": [
                "hidream",
                "1024x1024 (Square - Default)",
                "webp (Recommended - Best compression)",
            ],
            "confirm": [False, False, True, False, False, False, False, True, False],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify style parameters are None / absent
            call_args = mock_generate.call_args
            assert "negative_prompt" not in call_args.kwargs
            assert call_args.kwargs["style_preset"] is None
            assert call_args.kwargs["lora_strength"] is None


class TestInteractiveWizardDimensions:
    """Test image dimension configuration"""

    async def test_custom_dimensions(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard with custom dimensions"""

        questionary_responses = {
            "text": [
                "test prompt",  # prompt
                "1920",  # custom width
                "1080",  # custom height
                "1",  # num_images
            ],
            "select": [
                "hidream",  # model
                "Custom dimensions",  # size_choice
                "webp (Recommended - Best compression)",  # format
            ],
            "confirm": [False, False, True, False, False, False, False, True, False],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify custom dimensions were used
            call_args = mock_generate.call_args
            assert call_args.kwargs["size"] == "1920x1080"

    async def test_preset_size_selection(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard with preset size selection"""

        questionary_responses = {
            "text": ["test prompt", "2"],
            "select": [
                "hidream",
                "1280x720 (16:9 Landscape)",
                "jpeg (Most compatible)",
            ],
            "confirm": [False, False, True, False, False, False, False, True, False],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify preset size was used
            call_args = mock_generate.call_args
            assert call_args.kwargs["size"] == "1280x720"
            assert call_args.kwargs["num_images"] == 2
            assert call_args.kwargs["format"] == "jpeg"


class TestInteractiveWizardErrorHandling:
    """Test error handling in the wizard"""

    async def test_api_error_fetching_models(
        self,
        mock_click_context,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard handles API errors when fetching models"""

        # Create a client that raises an exception
        error_client = MagicMock()
        error_client.models.list = AsyncMock(side_effect=VeniceError("API connection failed"))
        error_client.__aenter__ = AsyncMock(return_value=error_client)
        error_client.__aexit__ = AsyncMock(return_value=None)

        with patch("venice_ai.cli.commands.image.wizard.VeniceClient", return_value=error_client):
            with pytest.raises(SystemExit) as exit_info:
                await _interactive_image_generation(mock_click_context)
            assert exit_info.value.code == 1

            # Verify error message was displayed
            mock_console_functions["error"].assert_called()
            error_calls = [str(call) for call in mock_console_functions["error"].call_args_list]
            assert any("Could not fetch models" in str(call) for call in error_calls)

    async def test_no_prompt_provided(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard handles missing prompt"""

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            # Return empty prompt
            mock_to_thread.return_value = ""

            with pytest.raises(SystemExit):
                await _interactive_image_generation(mock_click_context)

            # Verify error was shown
            mock_console_functions["error"].assert_called_with("Prompt is required!")

    async def test_no_models_available(
        self,
        mock_click_context,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test wizard handles no models being returned from API"""

        # Create client that returns empty models list
        empty_client = MagicMock()
        models_response = MagicMock()
        models_response.data = []  # No models
        empty_client.models.list = AsyncMock(return_value=models_response)
        empty_client.__aenter__ = AsyncMock(return_value=empty_client)
        empty_client.__aexit__ = AsyncMock(return_value=None)

        with patch("venice_ai.cli.commands.image.wizard.VeniceClient", return_value=empty_client):
            with pytest.raises(SystemExit) as exit_info:
                await _interactive_image_generation(mock_click_context)
            assert exit_info.value.code == 1

            # Verify error message was displayed
            mock_console_functions["error"].assert_called_with(
                "No image models found in API response."
            )


class TestInteractiveWizardPresets:
    """Test preset saving functionality"""

    async def test_save_as_preset_workflow(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test saving configuration as preset"""

        questionary_responses = {
            "text": [
                "test prompt",  # prompt
                "1",  # num_images
                "30",  # steps
                "7.5",  # cfg_scale
                "my_preset",  # preset name (no seed question when use_seed=False)
            ],
            "select": [
                "hidream",
                "1024x1024 (Square - Default)",
                "png (Highest quality)",
            ],
            "confirm": [
                True,  # configure_advanced
                False,  # use_seed
                False,  # configure_style
                True,  # safe_mode
                False,  # hide_watermark
                True,  # embed_exif
                False,  # return_binary
                False,  # custom_location
                True,  # proceed
                True,  # save_as_preset
            ],
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.save_preset") as mock_save_preset,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify save_preset was called
            mock_save_preset.assert_called_once()
            preset_name, preset_config = mock_save_preset.call_args[0]
            assert preset_name == "my_preset"
            assert preset_config["steps"] == 30
            assert preset_config["cfg_scale"] == 7.5
            assert preset_config["format"] == "png"
            assert preset_config["safe_mode"] is True
            assert preset_config["embed_exif"] is True

            # Verify generation still happened
            mock_generate.assert_called_once()

    async def test_skip_saving_preset(
        self,
        mock_click_context,
        mock_venice_client,
        mock_console_functions,
        mock_ensure_api_key,
        mock_load_config,
    ):
        """Test skipping preset save"""

        questionary_responses = {
            "text": ["test prompt", "1"],
            "select": [
                "hidream",
                "1024x1024 (Square - Default)",
                "webp (Recommended - Best compression)",
            ],
            "confirm": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
            ],  # save_as_preset=False
        }

        with (
            patch(
                "venice_ai.cli.commands.image.wizard.VeniceClient",
                return_value=mock_venice_client,
            ),
            patch(
                "venice_ai.cli.commands.image.generate._generate_image_async",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch("venice_ai.cli.commands.image.wizard.save_preset") as mock_save_preset,
            patch("venice_ai.cli.commands.image.wizard.questionary") as mock_questionary,
            patch("asyncio.to_thread", side_effect=lambda f: f()),
        ):
            call_count = {"text": 0, "select": 0, "confirm": 0}

            def make_mock_with_ask(response_type):
                def mock_method(*args, **kwargs):
                    mock_obj = MagicMock()
                    mock_obj.ask.return_value = questionary_responses[response_type][
                        call_count[response_type]
                    ]
                    call_count[response_type] += 1
                    return mock_obj

                return mock_method

            mock_questionary.text.side_effect = make_mock_with_ask("text")
            mock_questionary.select.side_effect = make_mock_with_ask("select")
            mock_questionary.confirm.side_effect = make_mock_with_ask("confirm")

            await _interactive_image_generation(mock_click_context)

            # Verify save_preset was NOT called
            mock_save_preset.assert_not_called()

            # Verify generation still happened
            mock_generate.assert_called_once()
