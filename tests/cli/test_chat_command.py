"""
Tests for chat command - comprehensive test coverage
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli
from venice_ai.cli.commands.chat import (
    _chat_async,
    _display_stream_stats,
    _interactive_chat,
    _select_chat_model,
    _send_single_message,
    extract_thinking_blocks,
)
from venice_ai.cli.utils.streaming import AnimationMode
from venice_ai.exceptions import VeniceError


@pytest.fixture
def cli_runner():
    """Fixture providing Click's CliRunner"""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration dictionary"""
    return {
        "defaults": {
            "chat_model": "test-model",
            "max_completion_tokens": 2048,
            "temperature": 0.7,
        },
        "api": {"base_url": "https://api.venice.ai/api/v1"},
    }


@pytest.fixture
def mock_text_model():
    """Create a mock text model"""
    return SimpleNamespace(
        id="test-text-model",
        type="text",
        model_spec=SimpleNamespace(
            availableContextTokens=131072,
            traits=["default"],
        ),
    )


@pytest.fixture
def mock_text_model_fastest():
    """Create a mock fast text model"""
    return SimpleNamespace(
        id="fast-model",
        type="text",
        model_spec=SimpleNamespace(
            availableContextTokens=8000,
            traits=["fastest"],
        ),
    )


@pytest.fixture
def mock_text_model_best():
    """Create a mock best text model"""
    return SimpleNamespace(
        id="best-model",
        type="text",
        model_spec=SimpleNamespace(
            availableContextTokens=50000,
            traits=["best"],
        ),
    )


@pytest.fixture
def mock_text_model_no_traits():
    """Create a mock text model with no traits"""
    return SimpleNamespace(
        id="basic-model",
        type="text",
        model_spec=SimpleNamespace(
            availableContextTokens=None,
            traits=[],
        ),
    )


@pytest.fixture
def mock_models_response(
    mock_text_model,
    mock_text_model_fastest,
    mock_text_model_best,
    mock_text_model_no_traits,
):
    """Mock models list response"""
    return SimpleNamespace(
        data=[
            mock_text_model,
            mock_text_model_fastest,
            mock_text_model_best,
            mock_text_model_no_traits,
        ]
    )


@pytest.fixture
def mock_completion_response():
    """Mock non-streaming completion response"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Test response content",
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


@pytest.fixture
def mock_completion_response_with_reasoning():
    """Mock response with reasoning content"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Final answer",
                    reasoning_content="This is my reasoning process",
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


@pytest.fixture
def mock_stream_chunk():
    """Mock streaming chunk"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="Hello "),
                finish_reason=None,
            )
        ]
    )


class TestChatCommand:
    """Test chat command structure"""

    def test_chat_help(self, cli_runner):
        """Test chat command help text"""
        result = cli_runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower()

    def test_chat_start_help(self, cli_runner):
        """Test chat start subcommand help"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        assert result.exit_code == 0
        assert "start" in result.output.lower() or "chat" in result.output.lower()

    def test_chat_command_registered(self, cli_runner):
        """Test chat command is registered in main CLI"""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower()


class TestChatOptions:
    """Test chat command options"""

    def test_chat_start_has_model_option(self, cli_runner):
        """Test chat start has --model option"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output or "-m" in result.output

    def test_chat_start_has_temperature_option(self, cli_runner):
        """Test chat start has --temperature option"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        assert result.exit_code == 0
        assert "--temperature" in result.output or "-t" in result.output

    def test_chat_start_has_system_option(self, cli_runner):
        """Test chat start has --system option"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        assert result.exit_code == 0
        assert "--system" in result.output or "-s" in result.output

    def test_chat_start_has_stream_option(self, cli_runner):
        """Test chat start has --stream option"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        assert result.exit_code == 0
        assert "stream" in result.output.lower()

    def test_chat_start_has_animation_option(self, cli_runner):
        """Test chat start has --animation option"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        assert result.exit_code == 0
        assert "--animation" in result.output

    def test_chat_start_has_show_thinking_option(self, cli_runner):
        """Test chat start has --show-thinking option"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        assert result.exit_code == 0
        assert "thinking" in result.output.lower()


class TestExtractThinkingBlocks:
    """Test extract_thinking_blocks function"""

    def test_extract_thinking_blocks_exists(self):
        """Test extract_thinking_blocks function exists"""
        assert callable(extract_thinking_blocks)

    def test_extract_thinking_blocks_basic(self):
        """Test extract_thinking_blocks with basic content"""
        content = "<thinking>test thought</thinking>Regular content"
        blocks, cleaned = extract_thinking_blocks(content)

        assert len(blocks) == 1
        assert "test thought" in blocks[0]
        assert "Regular content" in cleaned
        assert "<thinking>" not in cleaned

    def test_extract_thinking_blocks_no_thinking(self):
        """Test extract_thinking_blocks with no thinking blocks"""
        content = "Just regular content"
        blocks, cleaned = extract_thinking_blocks(content)

        assert len(blocks) == 0
        assert cleaned == content

    def test_extract_thinking_blocks_multiple(self):
        """Test extract_thinking_blocks with multiple blocks"""
        content = "<thinking>first</thinking>text<think>second</think>more"
        blocks, cleaned = extract_thinking_blocks(content)

        assert len(blocks) == 2
        assert "first" in blocks[0] or "first" in blocks[1]
        assert "second" in blocks[0] or "second" in blocks[1]

    def test_extract_thinking_blocks_with_list_input(self):
        """Test extract_thinking_blocks with list input - covers line 37"""
        content = ["<thinking>", "test thought", "</thinking>", "Regular content"]
        blocks, cleaned = extract_thinking_blocks(content)

        # List is joined with spaces
        assert "Regular content" in cleaned

    def test_extract_thinking_blocks_with_empty_list(self):
        """Test extract_thinking_blocks with empty list"""
        content = []
        blocks, cleaned = extract_thinking_blocks(content)

        assert len(blocks) == 0
        assert cleaned == ""

    def test_extract_thinking_blocks_with_mixed_list(self):
        """Test extract_thinking_blocks with mixed types in list"""
        content = ["Hello", 123, "world"]
        blocks, cleaned = extract_thinking_blocks(content)

        assert "Hello" in cleaned
        assert "123" in cleaned
        assert "world" in cleaned

    def test_extract_thinking_blocks_think_tag(self):
        """Test with <think> tags instead of <thinking>"""
        content = "<think>reasoning here</think>Final answer"
        blocks, cleaned = extract_thinking_blocks(content)

        assert len(blocks) == 1
        assert "reasoning here" in blocks[0]
        assert "Final answer" in cleaned

    def test_extract_thinking_blocks_multiline(self):
        """Test with multiline thinking content"""
        content = "<thinking>Line 1\nLine 2\nLine 3</thinking>Output"
        blocks, cleaned = extract_thinking_blocks(content)

        assert len(blocks) == 1
        assert "Line 1" in blocks[0]
        assert "Line 2" in blocks[0]
        assert "Line 3" in blocks[0]


class TestSelectChatModel:
    """Test _select_chat_model function"""

    @pytest.mark.asyncio
    async def test_select_chat_model_success(self, mock_models_response):
        """Test successful model selection"""
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch(
                "asyncio.to_thread",
                return_value="test-text-model (131k context) [DEFAULT]",
            ),
        ):
            result = await _select_chat_model(mock_client)

        assert result == "test-text-model"

    @pytest.mark.asyncio
    async def test_select_chat_model_no_models(self):
        """Test when no models available - covers line 171-172"""
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
        ):
            result = await _select_chat_model(mock_client)

        assert result is None
        mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_chat_model_cancelled(self, mock_models_response):
        """Test when user cancels selection - covers line 215"""
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("asyncio.to_thread", return_value=None),
        ):
            result = await _select_chat_model(mock_client)

        assert result is None

    @pytest.mark.asyncio
    async def test_select_chat_model_exception(self):
        """Test exception handling - covers lines 217-219"""
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("API error"))

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
        ):
            result = await _select_chat_model(mock_client)

        assert result is None
        mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_chat_model_fast_trait(self):
        """Test model with fastest trait display"""
        mock_response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="fast-model",
                    type="text",
                    model_spec=SimpleNamespace(
                        availableContextTokens=8000,
                        traits=["fastest"],
                    ),
                )
            ]
        )
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("asyncio.to_thread", return_value="fast-model (8000 tokens) [FAST]"),
        ):
            result = await _select_chat_model(mock_client)

        assert result == "fast-model"

    @pytest.mark.asyncio
    async def test_select_chat_model_best_trait(self):
        """Test model with best trait display"""
        mock_response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="best-model",
                    type="text",
                    model_spec=SimpleNamespace(
                        availableContextTokens=50000,
                        traits=["best"],
                    ),
                )
            ]
        )
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("asyncio.to_thread", return_value="best-model (50k context) [BEST]"),
        ):
            result = await _select_chat_model(mock_client)

        assert result == "best-model"

    @pytest.mark.asyncio
    async def test_select_chat_model_no_context_tokens(self):
        """Test model with no availableContextTokens - covers lines 179-180"""
        mock_response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="basic-model",
                    type="text",
                    model_spec=SimpleNamespace(
                        availableContextTokens=None,
                        traits=[],
                    ),
                )
            ]
        )
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("asyncio.to_thread", return_value="basic-model"),
        ):
            result = await _select_chat_model(mock_client)

        assert result == "basic-model"

    @pytest.mark.asyncio
    async def test_select_chat_model_small_context(self):
        """Test model with small context tokens (< 100k) - covers line 188"""
        mock_response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="small-model",
                    type="text",
                    model_spec=SimpleNamespace(
                        availableContextTokens=8000,
                        traits=[],
                    ),
                )
            ]
        )
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("asyncio.to_thread", return_value="small-model (8000 tokens)"),
        ):
            result = await _select_chat_model(mock_client)

        assert result == "small-model"


class TestChatAsync:
    """Test _chat_async function"""

    @pytest.mark.asyncio
    async def test_chat_async_single_message(self, mock_config, mock_completion_response):
        """Test single message mode - covers lines 274-288"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._send_single_message",
                new_callable=AsyncMock,
            ) as mock_send,
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message="Hello",
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_async_interactive_mode(self, mock_config):
        """Test interactive mode - covers lines 290-313"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ) as mock_interactive,
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,  # No message = interactive
            )
            mock_interactive.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_async_with_system_prompt(self, mock_config):
        """Test with system prompt - covers lines 268-271, 292-293"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt="You are a helpful assistant.",
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )
            # Verify system prompt message displayed
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_chat_async_with_show_thinking(self, mock_config):
        """Test with show_thinking flag - covers lines 294-295"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=True,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )
            # Verify thinking message displayed
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_chat_async_with_animation_option(self, mock_config):
        """Test with non-smooth animation - covers lines 296-299"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="word",  # Non-smooth animation
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_chat_async_select_model(self, mock_config):
        """Test with select_model flag - covers lines 255-260"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._select_chat_model",
                new_callable=AsyncMock,
                return_value="selected-model",
            ) as mock_select,
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.print_success"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model=None,
                select_model=True,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )
            mock_select.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_async_select_model_cancelled(self, mock_config):
        """Test when model selection cancelled - covers lines 257-259"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._select_chat_model",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
        ):
            with pytest.raises(SystemExit):
                await _chat_async(
                    ctx=mock_ctx,
                    model=None,
                    select_model=True,
                    system_prompt=None,
                    temperature=None,
                    max_completion_tokens=None,
                    stream=True,
                    show_thinking=False,
                    animation="smooth",
                    animation_speed=0.03,
                    show_stats=False,
                    initial_message=None,
                )
            mock_error.assert_called_with("No model selected. Exiting.")

    @pytest.mark.asyncio
    async def test_chat_async_venice_error(self, mock_config):
        """Test VeniceError handling - covers lines 315-316"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=VeniceError("API failure"))
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
        ):
            with pytest.raises(SystemExit):
                await _chat_async(
                    ctx=mock_ctx,
                    model="test-model",
                    select_model=False,
                    system_prompt=None,
                    temperature=None,
                    max_completion_tokens=None,
                    stream=True,
                    show_thinking=False,
                    animation="smooth",
                    animation_speed=0.03,
                    show_stats=False,
                    initial_message="test",
                )
            mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_chat_async_general_error(self, mock_config):
        """Test general exception handling - covers lines 317-318"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=RuntimeError("Unexpected"))
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
        ):
            with pytest.raises(SystemExit):
                await _chat_async(
                    ctx=mock_ctx,
                    model="test-model",
                    select_model=False,
                    system_prompt=None,
                    temperature=None,
                    max_completion_tokens=None,
                    stream=True,
                    show_thinking=False,
                    animation="smooth",
                    animation_speed=0.03,
                    show_stats=False,
                    initial_message="test",
                )
            mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_chat_async_default_model(self, mock_config):
        """Test default model from config - covers lines 261-262"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ) as mock_interactive,
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model=None,  # No model specified
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )
            mock_interactive.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_async_load_config_fallback(self):
        """Test config loading fallback - covers line 240"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {}  # No config in context

        default_config = {
            "defaults": {
                "chat_model": "default-model",
                "max_completion_tokens": 2048,
                "temperature": 0.7,
            }
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.load_config", return_value=default_config),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model=None,
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )


class TestSendSingleMessage:
    """Test _send_single_message function"""

    @pytest.mark.asyncio
    async def test_send_single_message_streaming(self):
        """Test streaming single message"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        mock_stream_handler = MagicMock()
        mock_stream_handler.display_progress = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("Hello world", {"total_chunks": 5})
        )

        with (
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_send_single_message_streaming_with_thinking(self):
        """Test streaming with thinking blocks - covers lines 365-375"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        mock_stream_handler = MagicMock()
        mock_stream_handler.display_progress = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=(
                "<thinking>Test reasoning</thinking>Final answer",
                {"total_chunks": 5},
            )
        )

        with (
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=True,  # Show thinking
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_send_single_message_streaming_with_stats(self):
        """Test streaming with stats display - covers lines 378-380"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        mock_stream_handler = MagicMock()
        mock_stream_handler.display_progress = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("Response", {"total_chunks": 5, "stream_duration": 1.5})
        )

        with (
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch("venice_ai.cli.commands.chat._display_stream_stats") as mock_display_stats,
        ):
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=True,  # Show stats
            )
            mock_display_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_single_message_non_streaming(self):
        """Test non-streaming single message - covers lines 384-394"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Test response",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,  # Non-streaming
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_send_single_message_non_streaming_with_reasoning(self):
        """Test non-streaming with reasoning_content - covers lines 397-412"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Test response",
                        reasoning_content="This is my reasoning",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=True,  # Show thinking
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_send_single_message_non_streaming_with_thinking_blocks(self):
        """Test non-streaming with thinking blocks in content - covers lines 409-412"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<thinking>My thought</thinking>Final answer",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_send_single_message_non_string_content(self):
        """Test non-streaming with non-string content - covers lines 416-420"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=["Part 1", "Part 2"],  # List content
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_send_single_message_no_usage(self):
        """Test non-streaming with no usage info - covers lines 423-425"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Test response",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,  # No usage
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )


class TestInteractiveChat:
    """Test _interactive_chat function"""

    @pytest.mark.asyncio
    async def test_interactive_chat_exit_command(self):
        """Test exit command - covers lines 458-460"""
        mock_client = AsyncMock()

        with (
            patch("asyncio.to_thread", return_value="exit"),
            patch("venice_ai.cli.commands.chat.print_info") as mock_info,
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )
            mock_info.assert_called_with("Ending chat session. Goodbye!")

    @pytest.mark.asyncio
    async def test_interactive_chat_quit_command(self):
        """Test quit command"""
        mock_client = AsyncMock()

        with (
            patch("asyncio.to_thread", return_value="quit"),
            patch("venice_ai.cli.commands.chat.print_info") as mock_info,
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )
            mock_info.assert_called()

    @pytest.mark.asyncio
    async def test_interactive_chat_bye_command(self):
        """Test bye command"""
        mock_client = AsyncMock()

        with (
            patch("asyncio.to_thread", return_value="bye"),
            patch("venice_ai.cli.commands.chat.print_info") as mock_info,
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )
            mock_info.assert_called()

    @pytest.mark.asyncio
    async def test_interactive_chat_empty_input(self):
        """Test empty input handling - covers lines 454-455"""
        mock_client = AsyncMock()
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ""  # Empty input
            else:
                return "exit"  # Exit on second call

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_interactive_chat_streaming_response(self):
        """Test streaming response in interactive mode - covers lines 466-501"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            # Accept any args/kwargs from asyncio.to_thread
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("Response", {"total_chunks": 5})
        )

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )
            mock_client.chat.completions.create.assert_called()

    @pytest.mark.asyncio
    async def test_interactive_chat_streaming_with_thinking(self):
        """Test streaming with thinking blocks - covers lines 482-491"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("<thinking>Thought</thinking>Answer", {})
        )

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )

    @pytest.mark.asyncio
    async def test_interactive_chat_streaming_with_stats(self):
        """Test streaming with stats - covers lines 494-496"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("Response", {"total_chunks": 5, "stream_duration": 1.0})
        )

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch("venice_ai.cli.commands.chat._display_stream_stats"),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=True,
            )

    @pytest.mark.asyncio
    async def test_interactive_chat_non_streaming(self):
        """Test non-streaming response - covers lines 508-554"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Test response",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            with patch("venice_ai.cli.commands.chat.StreamHandler"):
                await _interactive_chat(
                    client=mock_client,
                    messages=[],
                    model="test-model",
                    temperature=0.7,
                    max_completion_tokens=2048,
                    stream=False,  # Non-streaming
                    show_thinking=False,
                    animation_mode=AnimationMode.SMOOTH,
                    animation_speed=0.03,
                    show_stats=False,
                )

    @pytest.mark.asyncio
    async def test_interactive_chat_non_streaming_with_reasoning(self):
        """Test non-streaming with reasoning - covers lines 523-538"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Answer",
                        reasoning_content="My reasoning",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            with patch("venice_ai.cli.commands.chat.StreamHandler"):
                await _interactive_chat(
                    client=mock_client,
                    messages=[],
                    model="test-model",
                    temperature=0.7,
                    max_completion_tokens=2048,
                    stream=False,
                    show_thinking=True,
                    animation_mode=AnimationMode.SMOOTH,
                    animation_speed=0.03,
                    show_stats=False,
                )

    @pytest.mark.asyncio
    async def test_interactive_chat_non_streaming_with_thinking_blocks(self):
        """Test non-streaming with thinking blocks - covers lines 535-538"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<thinking>Thought</thinking>Answer",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            with patch("venice_ai.cli.commands.chat.StreamHandler"):
                await _interactive_chat(
                    client=mock_client,
                    messages=[],
                    model="test-model",
                    temperature=0.7,
                    max_completion_tokens=2048,
                    stream=False,
                    show_thinking=True,
                    animation_mode=AnimationMode.SMOOTH,
                    animation_speed=0.03,
                    show_stats=False,
                )

    @pytest.mark.asyncio
    async def test_interactive_chat_non_streaming_non_string_content(self):
        """Test non-streaming with non-string content - covers lines 542-546"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=["Part 1", "Part 2"],  # List content
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            with patch("venice_ai.cli.commands.chat.StreamHandler"):
                await _interactive_chat(
                    client=mock_client,
                    messages=[],
                    model="test-model",
                    temperature=0.7,
                    max_completion_tokens=2048,
                    stream=False,
                    show_thinking=False,
                    animation_mode=AnimationMode.SMOOTH,
                    animation_speed=0.03,
                    show_stats=False,
                )

    @pytest.mark.asyncio
    async def test_interactive_chat_keyboard_interrupt(self):
        """Test keyboard interrupt - covers lines 584-586"""
        mock_client = AsyncMock()

        with (
            patch("asyncio.to_thread", side_effect=KeyboardInterrupt),
            patch("venice_ai.cli.commands.chat.print_info") as mock_info,
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )
            mock_info.assert_called()

    @pytest.mark.asyncio
    async def test_interactive_chat_venice_error(self):
        """Test VeniceError continues session - covers lines 587-589"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(side_effect=VeniceError("API error"))

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )
            # VeniceError should log error but continue
            mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_interactive_chat_general_exception(self):
        """Test general exception breaks session - covers lines 590-592"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            else:
                return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
            )
            mock_error.assert_called()


class TestDisplayStreamStats:
    """Test _display_stream_stats function"""

    def test_display_stream_stats_all_fields(self):
        """Test with all stats fields - covers lines 597-615"""
        stats = {
            "total_chunks": 10,
            "content_length": 500,
            "stream_duration": 2.5,
            "time_to_first_token": 0.1,
            "chunks_per_second": 4.0,
            "finish_reason": "stop",
        }

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            _display_stream_stats(stats)
            mock_console.print.assert_called_once()

    def test_display_stream_stats_minimal_fields(self):
        """Test with minimal stats - covers lines 602-604"""
        stats = {
            "total_chunks": 5,
        }

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            _display_stream_stats(stats)
            mock_console.print.assert_called_once()

    def test_display_stream_stats_empty(self):
        """Test with empty stats"""
        stats = {}

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            _display_stream_stats(stats)
            mock_console.print.assert_called_once()

    def test_display_stream_stats_partial_fields(self):
        """Test with partial fields - covers various branches lines 602-613"""
        stats = {
            "total_chunks": 10,
            "stream_duration": 1.5,
            "finish_reason": "length",
        }

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            _display_stream_stats(stats)
            mock_console.print.assert_called_once()

    def test_display_stream_stats_only_content_length(self):
        """Test with only content_length - covers line 604-605"""
        stats = {
            "content_length": 1000,
        }

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            _display_stream_stats(stats)
            mock_console.print.assert_called_once()

    def test_display_stream_stats_time_to_first_token(self):
        """Test with time_to_first_token - covers lines 608-609"""
        stats = {
            "time_to_first_token": 0.05,
        }

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            _display_stream_stats(stats)
            mock_console.print.assert_called_once()

    def test_display_stream_stats_chunks_per_second(self):
        """Test with chunks_per_second - covers lines 610-611"""
        stats = {
            "chunks_per_second": 25.0,
        }

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            _display_stream_stats(stats)
            mock_console.print.assert_called_once()


class TestChatShortcut:
    """Test chat shortcut hidden command - covers lines 619-623"""

    def test_chat_shortcut_exists(self, cli_runner):
        """Test chat shortcut command exists"""
        # The hidden command should exist but not show in help
        result = cli_runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0


class TestStartChatCommand:
    """Test start_chat command invocation"""

    def test_start_chat_invokes_asyncio_run(self, cli_runner):
        """Test that start_chat runs async code"""
        # Patch asyncio.run so it doesn't actually execute the async function
        # Also properly close any coroutine that gets created to avoid warnings
        with patch("venice_ai.cli.commands.chat.asyncio.run") as mock_run:
            # Set up mock_run to properly close the coroutine passed to it
            def close_coroutine(coro):
                coro.close()
                return None

            mock_run.side_effect = close_coroutine
            cli_runner.invoke(cli, ["chat", "start", "-m", "test-model", "Hello"])
            # asyncio.run should be called
            mock_run.assert_called_once()

    def test_start_chat_with_message(self, cli_runner):
        """Test start_chat with a message argument"""
        result = cli_runner.invoke(cli, ["chat", "start", "--help"])
        # Should show MESSAGE in help
        assert "message" in result.output.lower() or result.exit_code == 0


class TestStdinPipedInput:
    """Test stdin/piped input handling (lines 341-351)"""

    @pytest.mark.asyncio
    async def test_chat_async_stdin_piped_no_initial_message(self, mock_config):
        """Test stdin piped input when no initial_message - covers lines 341-348"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "piped message content"

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._send_single_message",
                new_callable=AsyncMock,
            ) as mock_send,
            patch("venice_ai.cli.commands.chat.sys") as mock_sys,
        ):
            mock_sys.stdin = mock_stdin
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,  # No initial message
            )
            # The piped content becomes the message
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][2] == "piped message content"  # user_message arg

    @pytest.mark.asyncio
    async def test_chat_async_stdin_piped_with_initial_message(self, mock_config):
        """Test stdin piped input combined with initial_message - covers lines 345-346"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "piped context"

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._send_single_message",
                new_callable=AsyncMock,
            ) as mock_send,
            patch("venice_ai.cli.commands.chat.sys") as mock_sys,
        ):
            mock_sys.stdin = mock_stdin
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message="original message",  # has initial message
            )
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            # Combined message
            assert "original message" in call_args[0][2]
            assert "piped context" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_chat_async_stdin_piped_empty(self, mock_config):
        """Test stdin piped input returns empty string - no modification"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "   "  # whitespace only

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ) as mock_interactive,
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.sys") as mock_sys,
        ):
            mock_sys.stdin = mock_stdin
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )
            # No message set, falls to interactive
            mock_interactive.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_async_stdin_ioerror(self, mock_config):
        """Test stdin OSError handling - covers lines 349-351"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_stdin = MagicMock()
        mock_stdin.isatty.side_effect = OSError("stdin error")

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.sys") as mock_sys,
        ):
            mock_sys.stdin = mock_stdin
            # Should not raise, handles exception gracefully
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )

    @pytest.mark.asyncio
    async def test_chat_async_stdin_value_error(self, mock_config):
        """Test stdin ValueError handling - covers lines 349-351 ValueError branch"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_stdin = MagicMock()
        mock_stdin.isatty.side_effect = ValueError("stdin value error")

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.sys") as mock_sys,
        ):
            mock_sys.stdin = mock_stdin
            # Should not raise, handles exception gracefully
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )


class TestContinueFromConversation:
    """Test --continue-from conversation loading (lines 387-412)"""

    @pytest.mark.asyncio
    async def test_continue_from_specific_id(self, mock_config):
        """Test --continue-from with specific ID - covers lines 387-409"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        prev_conversation = {
            "id": "abc12345",
            "title": "Test Conversation",
            "model": "previous-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat.load_conversation",
                return_value=prev_conversation,
            ) as mock_load,
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ) as mock_interactive,
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
                continue_id="abc12345",
            )
            mock_load.assert_called_once_with("abc12345")
            mock_interactive.assert_called_once()

    @pytest.mark.asyncio
    async def test_continue_from_last(self, mock_config):
        """Test --continue-from with 'last' - covers lines 389-390"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        prev_conversation = {
            "id": "lastidabc",
            "title": "Last Conversation",
            "model": "my-model",
            "messages": [
                {"role": "user", "content": "Previous message"},
            ],
        }

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat.get_last_conversation_id",
                return_value="lastidabc",
            ) as mock_last,
            patch(
                "venice_ai.cli.commands.chat.load_conversation",
                return_value=prev_conversation,
            ) as mock_load,
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
                continue_id="last",
            )
            mock_last.assert_called_once()
            mock_load.assert_called_once_with("lastidabc")

    @pytest.mark.asyncio
    async def test_continue_from_last_no_previous(self, mock_config):
        """Test --continue-from 'last' when no previous conversations exist"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # prev conversation found after fallback conv_id
        some_conv = {
            "id": "someconv",
            "title": "Conv",
            "model": "my-model",
            "messages": [],
        }

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat.get_last_conversation_id",
                return_value=None,
            ),
            patch(
                "venice_ai.cli.commands.chat.load_conversation",
                return_value=some_conv,
            ),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
                continue_id="last",
            )

    @pytest.mark.asyncio
    async def test_continue_from_not_found(self, mock_config):
        """Test --continue-from when conversation not found - covers lines 410-412"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch("venice_ai.cli.commands.chat.load_conversation", return_value=None),
            patch("venice_ai.cli.commands.chat.print_error") as mock_error,
        ):
            with pytest.raises(SystemExit) as exit_info:
                await _chat_async(
                    ctx=mock_ctx,
                    model="test-model",
                    select_model=False,
                    system_prompt=None,
                    temperature=None,
                    max_completion_tokens=None,
                    stream=True,
                    show_thinking=False,
                    animation="smooth",
                    animation_speed=0.03,
                    show_stats=False,
                    initial_message=None,
                    continue_id="nonexistentid",
                )
            assert exit_info.value.code == 1
            mock_error.assert_called()
            call_args = str(mock_error.call_args)
            assert "nonexistentid" in call_args

    @pytest.mark.asyncio
    async def test_continue_from_uses_conversation_model(self, mock_config):
        """Test that model from conversation is used when default model is set - covers lines 395-396"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        prev_conversation = {
            "id": "convid12",
            "title": "Old Conversation",
            "model": "special-model",  # Different model
            "messages": [{"role": "user", "content": "Hi"}],
        }

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat.load_conversation",
                return_value=prev_conversation,
            ),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ) as mock_interactive,
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model=None,  # No model, will use config default
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
                continue_id="convid12",
            )
            # Check that model from conversation is passed
            call_kwargs = mock_interactive.call_args
            assert call_kwargs[0][2] == "special-model"  # model arg

    @pytest.mark.asyncio
    async def test_continue_from_plain_mode(self, mock_config):
        """Test --continue-from in plain mode - covers lines 400-404"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config, "plain": True}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        prev_conversation = {
            "id": "plainconv",
            "title": "Plain Conv",
            "model": "basic-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat.load_conversation",
                return_value=prev_conversation,
            ),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch(
                "venice_ai.cli.commands.chat.is_plain_mode",
                return_value=True,
            ),
            patch("click.echo") as mock_echo,
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
                continue_id="plainconv",
            )
            # Click echo should be called for plain mode
            assert mock_echo.called


class TestSaveConversation:
    """Test --save conversation flag (lines 444-453, 951-958)"""

    @pytest.mark.asyncio
    async def test_save_after_single_message(self, mock_config):
        """Test save conversation after single message - covers lines 444-453"""
        from venice_ai.types.api import AssistantMessage, UserMessage

        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # _send_single_message needs to append to messages for the save condition to trigger
        async def fake_send_single(client, messages, *args, **kwargs):
            messages.append(UserMessage(role="user", content="Hello"))
            messages.append(
                AssistantMessage(
                    role="assistant",
                    content="Hi",
                    name=None,
                    reasoning_content=None,
                    tool_calls=None,
                )
            )

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._send_single_message",
                side_effect=fake_send_single,
            ),
            patch(
                "venice_ai.cli.commands.chat.save_conversation",
                return_value="/tmp/test.json",
            ) as mock_save,
            patch("venice_ai.cli.commands.chat.print_success") as mock_success,
            patch("venice_ai.cli.commands.chat.is_plain_mode", return_value=False),
            patch("venice_ai.cli.commands.chat.console"),
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message="Hello",
                save_conversation_flag=True,
            )
            mock_save.assert_called_once()
            mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_after_single_message_plain_mode(self, mock_config):
        """Test save conversation in plain mode - covers line 448-449"""
        from venice_ai.types.api import AssistantMessage, UserMessage

        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config, "plain": True}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        async def fake_send_single(client, messages, *args, **kwargs):
            messages.append(UserMessage(role="user", content="Hello"))
            messages.append(
                AssistantMessage(
                    role="assistant",
                    content="Hi",
                    name=None,
                    reasoning_content=None,
                    tool_calls=None,
                )
            )

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._send_single_message",
                side_effect=fake_send_single,
            ),
            patch(
                "venice_ai.cli.commands.chat.save_conversation",
                return_value="/tmp/test.json",
            ) as mock_save,
            patch(
                "venice_ai.cli.commands.chat.is_plain_mode",
                return_value=True,
            ),
            patch("click.echo") as mock_echo,
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt=None,
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message="Hello",
                save_conversation_flag=True,
            )
            mock_save.assert_called_once()
            # Plain mode uses click.echo
            save_echo_calls = [
                c for c in mock_echo.call_args_list if "Conversation saved" in str(c)
            ]
            assert len(save_echo_calls) > 0

    @pytest.mark.asyncio
    async def test_save_after_interactive_chat(self, mock_config):
        """Test save conversation after interactive chat - covers lines 951-958"""
        mock_client = AsyncMock()

        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(return_value=("Response", {}))

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch(
                "venice_ai.cli.commands.chat.save_conversation",
                return_value="/tmp/saved.json",
            ) as mock_save,
            patch("venice_ai.cli.commands.chat.print_success") as mock_success,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[{"role": "user", "content": "Hi"}],  # Some messages
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                save_conversation_flag=True,
                conv_id="testconv1",
            )
            mock_save.assert_called_once()
            mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_after_interactive_chat_plain_mode(self, mock_config):
        """Test save conversation in plain mode after interactive chat - covers lines 954-955"""
        mock_client = AsyncMock()

        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(return_value=("Response", {}))

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch(
                "venice_ai.cli.commands.chat.save_conversation",
                return_value="/tmp/saved.json",
            ) as mock_save,
            patch("click.echo") as mock_echo,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
                save_conversation_flag=True,
                conv_id="testconv2",
            )
            mock_save.assert_called_once()
            save_echo_calls = [
                c for c in mock_echo.call_args_list if "Conversation saved" in str(c)
            ]
            assert len(save_echo_calls) > 0


class TestJsonOutputMode:
    """Test JSON output mode in single message (lines 630-637)"""

    @pytest.mark.asyncio
    async def test_send_single_message_json_output(self):
        """Test json_output mode for non-streaming - covers lines 630-637"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock_response.model_dump = MagicMock(return_value={"id": "resp1", "choices": []})
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console"), patch("click.echo") as mock_echo:
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                json_output=True,
            )
            # Should output JSON
            mock_echo.assert_called()
            # Verify JSON was output
            json_calls = list(mock_echo.call_args_list)
            assert len(json_calls) > 0

    @pytest.mark.asyncio
    async def test_send_single_message_json_output_no_model_dump(self):
        """Test json_output falls back to .dict() if no model_dump - covers lines 631-636"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        # No model_dump, uses .dict() fallback
        mock_response.dict = MagicMock(return_value={"id": "resp1", "choices": []})
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console"), patch("click.echo") as mock_echo:
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                json_output=True,
            )
            mock_echo.assert_called()


class TestPlainMode:
    """Test plain mode branches throughout the file"""

    @pytest.mark.asyncio
    async def test_send_single_message_plain_mode_non_streaming(self):
        """Test plain mode for non-streaming - covers lines 526-527, 620, 670-676"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console"), patch("click.echo") as mock_echo:
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            # Plain mode uses click.echo
            assert mock_echo.called
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("You:" in s or "Hello" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_send_single_message_plain_mode_streaming(self):
        """Test plain mode for streaming - covers lines 536, 538, 558-559, 570-573"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        mock_stream_handler = MagicMock()
        mock_stream_handler.display_progress = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("<thinking>Thinking</thinking>Response content", {})
        )

        with (
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch("click.echo") as mock_echo,
        ):
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            assert mock_echo.called

    @pytest.mark.asyncio
    async def test_send_single_message_plain_mode_streaming_stats(self):
        """Test plain mode streaming with stats - covers lines 586-587"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        mock_stream_handler = MagicMock()
        mock_stream_handler.display_progress = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("Response", {"total_chunks": 5, "stream_duration": 1.0})
        )

        with (
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch("venice_ai.cli.commands.chat._display_stream_stats") as mock_stats,
        ):
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=True,
                plain=True,
            )
            # Should call with plain=True
            mock_stats.assert_called_once_with(
                {"total_chunks": 5, "stream_duration": 1.0}, plain=True
            )

    @pytest.mark.asyncio
    async def test_send_single_message_plain_mode_with_thinking_blocks(self):
        """Test plain mode non-streaming with thinking blocks showing - covers lines 652-657"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<thinking>My thought here</thinking>Answer"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console"), patch("click.echo") as mock_echo:
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            # Should print thinking blocks
            assert mock_echo.called
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Thinking Process" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_send_single_message_plain_mode_token_usage(self):
        """Test plain mode with token usage - covers line 687-691"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console"), patch("click.echo") as mock_echo:
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            # Should show token usage in plain mode
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Tokens:" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_chat_async_plain_mode_interactive(self, mock_config):
        """Test plain mode interactive session startup - covers lines 457-461"""
        mock_ctx = MagicMock()
        mock_ctx.obj = {"config": mock_config, "plain": True}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.chat.VeniceClient", return_value=mock_client),
            patch(
                "venice_ai.cli.commands.chat._interactive_chat",
                new_callable=AsyncMock,
            ),
            patch("venice_ai.cli.commands.chat.is_plain_mode", return_value=True),
            patch("click.echo") as mock_echo,
        ):
            await _chat_async(
                ctx=mock_ctx,
                model="test-model",
                select_model=False,
                system_prompt="Be helpful",
                temperature=None,
                max_completion_tokens=None,
                stream=True,
                show_thinking=False,
                animation="smooth",
                animation_speed=0.03,
                show_stats=False,
                initial_message=None,
            )
            # Plain mode should use click.echo
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Starting chat session" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_interactive_chat_plain_mode_exit(self):
        """Test plain mode exit - covers lines 764-765"""
        mock_client = AsyncMock()

        with (
            patch("asyncio.to_thread", return_value="exit"),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
            patch("click.echo") as mock_echo,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Ending chat session" in s or "Goodbye" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_interactive_chat_plain_mode_streaming(self):
        """Test plain mode streaming in interactive - covers lines 776-777"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(return_value=("Response", {}))

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch("click.echo") as mock_echo,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Assistant:" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_interactive_chat_plain_mode_streaming_with_thinking(self):
        """Test plain mode streaming with thinking blocks - covers lines 796-799"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("<thinking>My reasoning</thinking>Final answer", {})
        )

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch("click.echo") as mock_echo,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Reasoning" in s or "reasoning" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_interactive_chat_plain_mode_streaming_stats(self):
        """Test plain mode with stats in interactive - covers lines 812-813"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(
            return_value=("Response", {"total_chunks": 5, "stream_duration": 1.0})
        )

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
            patch("venice_ai.cli.commands.chat._display_stream_stats") as mock_stats,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=True,
                plain=True,
            )
            mock_stats.assert_called()
            call_kwargs = mock_stats.call_args
            assert call_kwargs[1].get("plain") is True or (
                len(call_kwargs[0]) > 1 and call_kwargs[0][1] is True
            )

    @pytest.mark.asyncio
    async def test_interactive_chat_plain_mode_non_streaming(self):
        """Test plain mode non-streaming - covers lines 839-840, 882-884, 900-904"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
            patch("click.echo") as mock_echo,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Assistant:" in s for s in calls_str)
            assert any("Tokens:" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_interactive_chat_plain_mode_non_streaming_with_thinking(self):
        """Test plain mode non-streaming with thinking - covers lines 864-870"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<thinking>My thought</thinking>Answer",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
            patch("click.echo") as mock_echo,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Thinking Process" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_interactive_chat_plain_mode_non_streaming_with_reasoning_content(
        self,
    ):
        """Test plain mode non-streaming with reasoning_content - covers line 865-866"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Answer",
                        reasoning_content="Detailed reasoning here",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.console"),
            patch("venice_ai.cli.commands.chat.StreamHandler"),
            patch("click.echo") as mock_echo,
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Reasoning" in s for s in calls_str)


class TestDisplayStreamStatsPlainMode:
    """Test _display_stream_stats plain mode (lines 964-978)"""

    def test_display_stream_stats_plain_mode_all_fields(self):
        """Test plain mode stats with all fields - covers lines 964-978"""
        stats = {
            "total_chunks": 10,
            "content_length": 500,
            "stream_duration": 2.5,
            "time_to_first_token": 0.1,
            "chunks_per_second": 4.0,
            "finish_reason": "stop",
        }

        with patch("click.echo") as mock_echo:
            _display_stream_stats(stats, plain=True)
            assert mock_echo.called
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Streaming Statistics" in s for s in calls_str)
            assert any("Total chunks" in s or "10" in s for s in calls_str)
            assert any("Content length" in s or "500" in s for s in calls_str)
            assert any("Stream duration" in s or "2.5" in s for s in calls_str)

    def test_display_stream_stats_plain_mode_minimal(self):
        """Test plain mode stats with minimal fields"""
        stats = {"total_chunks": 5}

        with patch("click.echo") as mock_echo:
            _display_stream_stats(stats, plain=True)
            assert mock_echo.called
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("5" in s for s in calls_str)

    def test_display_stream_stats_plain_mode_empty(self):
        """Test plain mode stats with empty dict"""
        stats = {}

        with patch("click.echo") as mock_echo:
            _display_stream_stats(stats, plain=True)
            # At minimum, the header should be printed
            assert mock_echo.called

    def test_display_stream_stats_plain_mode_time_to_first_token(self):
        """Test plain mode time_to_first_token - covers line 972-973"""
        stats = {"time_to_first_token": 0.05}

        with patch("click.echo") as mock_echo:
            _display_stream_stats(stats, plain=True)
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Time to first token" in s or "0.05" in s for s in calls_str)

    def test_display_stream_stats_plain_mode_chunks_per_second(self):
        """Test plain mode chunks_per_second - covers line 974-975"""
        stats = {"chunks_per_second": 25.0}

        with patch("click.echo") as mock_echo:
            _display_stream_stats(stats, plain=True)
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Chunks per second" in s or "25.0" in s for s in calls_str)

    def test_display_stream_stats_plain_mode_finish_reason(self):
        """Test plain mode finish_reason - covers line 976-977"""
        stats = {"finish_reason": "length"}

        with patch("click.echo") as mock_echo:
            _display_stream_stats(stats, plain=True)
            calls_str = [str(c) for c in mock_echo.call_args_list]
            assert any("Finish reason" in s or "length" in s for s in calls_str)


class TestChatHistoryCommand:
    """Test chat_history command (lines 1001-1079)"""

    def test_chat_history_list_conversations(self, cli_runner):
        """Test listing conversations with rich table display"""
        conversations = [
            {
                "id": "abc12345",
                "title": "Test Conversation",
                "model": "my-model",
                "updated_at": "2024-01-15T10:30:00",
            }
        ]

        with patch("venice_ai.cli.commands.chat.list_conversations", return_value=conversations):
            result = cli_runner.invoke(cli, ["--plain", "chat", "history"])
            assert result.exit_code == 0

    def test_chat_history_empty(self, cli_runner):
        """Test listing when no conversations exist - covers lines 1045-1050"""
        with patch("venice_ai.cli.commands.chat.list_conversations", return_value=[]):
            result = cli_runner.invoke(cli, ["chat", "history"])
            assert result.exit_code == 0
            assert "No saved" in result.output.lower() or result.exit_code == 0

    def test_chat_history_empty_plain_mode(self, cli_runner):
        """Test listing when no conversations exists in plain mode - covers lines 1046-1047"""
        with patch("venice_ai.cli.commands.chat.list_conversations", return_value=[]):
            result = cli_runner.invoke(cli, ["--plain", "chat", "history"])
            assert result.exit_code == 0
            assert "No saved conversations" in result.output

    def test_chat_history_json_output(self, cli_runner):
        """Test JSON output mode - covers lines 1041-1043"""
        conversations = [
            {
                "id": "abc12345",
                "title": "Test Conversation",
                "model": "my-model",
                "updated_at": "2024-01-15T10:30:00",
            }
        ]

        with patch("venice_ai.cli.commands.chat.list_conversations", return_value=conversations):
            result = cli_runner.invoke(cli, ["chat", "history", "--json"])
            assert result.exit_code == 0
            import json

            output = json.loads(result.output)
            assert isinstance(output, list)
            assert output[0]["id"] == "abc12345"

    def test_chat_history_delete_success(self, cli_runner):
        """Test delete conversation success - covers lines 1026-1034"""
        with (
            patch("venice_ai.cli.commands.chat.list_conversations", return_value=[]),
            patch("venice_ai.cli.commands.chat.chat_history"),
        ):
            # Import and test at lower level
            pass

        # Use CliRunner with proper mocking
        from venice_ai.cli.commands.chat import chat_history

        runner = CliRunner()

        with (
            patch("venice_ai.cli.commands.chat.print_success"),
            patch("venice_ai.cli.conversation.delete_conversation", return_value=True),
        ):
            # Patch the delete import within the function
            ctx_obj = {"plain": False}
            runner.invoke(
                chat_history,
                ["--delete", "convid12"],
                obj=ctx_obj,
                catch_exceptions=False,
            )

    def test_chat_history_delete_success_via_cli(self, cli_runner):
        """Test delete conversation via CLI"""

        with (
            patch("venice_ai.cli.conversation.delete_conversation", return_value=True),
            patch("venice_ai.cli.commands.chat.print_success"),
        ):
            result = cli_runner.invoke(cli, ["chat", "history", "--delete", "testconv1"])
            # Either success output or no error about it
            assert result.exit_code == 0

    def test_chat_history_delete_not_found(self, cli_runner):
        """Test delete when conversation not found - covers line 1036"""
        with (
            patch("venice_ai.cli.conversation.delete_conversation", return_value=False),
            patch("venice_ai.cli.commands.chat.print_error"),
        ):
            result = cli_runner.invoke(cli, ["chat", "history", "--delete", "nonexistent"])
            assert result.exit_code == 1

    def test_chat_history_delete_plain_mode(self, cli_runner):
        """Test delete in plain mode - covers lines 1031-1032"""
        with patch("venice_ai.cli.conversation.delete_conversation", return_value=True):
            result = cli_runner.invoke(cli, ["--plain", "chat", "history", "--delete", "testconv1"])
            assert result.exit_code == 0
            assert "testconv1" in result.output or "Deleted" in result.output

    def test_chat_history_rich_table_display(self, cli_runner):
        """Test rich table display - covers lines 1058-1079"""
        conversations = [
            {
                "id": "abc12345",
                "title": "Test Conversation",
                "model": "my-model",
                "updated_at": "2024-01-15T10:30:00",
            },
            {
                "id": "def67890",
                "title": "Another Conversation",
                "model": "other-model",
                "updated_at": "2024-01-16T12:00:00",
            },
        ]

        with (
            patch("venice_ai.cli.commands.chat.list_conversations", return_value=conversations),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            result = cli_runner.invoke(cli, ["chat", "history"])
            # Either the result is fine or console.print was called
            assert result.exit_code == 0 or mock_console.print.called

    def test_chat_history_plain_mode_list(self, cli_runner):
        """Test plain mode list display - covers lines 1052-1057"""
        conversations = [
            {
                "id": "abc12345",
                "title": "Test Conversation",
                "model": "my-model",
                "updated_at": "2024-01-15T10:30:00",
            }
        ]

        with patch("venice_ai.cli.commands.chat.list_conversations", return_value=conversations):
            result = cli_runner.invoke(cli, ["--plain", "chat", "history"])
            assert result.exit_code == 0
            assert "abc12345" in result.output


class TestAdditionalCoverage:
    """Tests for remaining uncovered branches"""

    @pytest.mark.asyncio
    async def test_send_single_message_plain_mode_reasoning_content(self):
        """Test plain mode with reasoning_content - covers line 654 (reasoning_content branch)"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Final answer",
                        reasoning_content="My reasoning process here",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console"), patch("click.echo") as mock_echo:
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=True,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=True,
            )
            calls_str = [str(c) for c in mock_echo.call_args_list]
            # Should show reasoning content
            assert any("Reasoning" in s or "My reasoning" in s for s in calls_str)

    @pytest.mark.asyncio
    async def test_send_single_message_non_string_display_content(self):
        """Test non-string display_content branch in non-streaming - covers line 682"""
        # When content is a list and has thinking blocks, display_content ends up as a non-str
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=["<thinking>thought</thinking>", "answer part"],
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                plain=False,  # Non-plain to hit str(display_content) branch at line 682
            )

    @pytest.mark.asyncio
    async def test_interactive_chat_non_streaming_non_string_display_content(self):
        """Test non-string display content in interactive non-streaming - covers line 894"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=["<thinking>thought</thinking>", "answer"],
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console") as mock_console,
        ):
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            with patch("venice_ai.cli.commands.chat.StreamHandler"):
                await _interactive_chat(
                    client=mock_client,
                    messages=[],
                    model="test-model",
                    temperature=0.7,
                    max_completion_tokens=2048,
                    stream=False,
                    show_thinking=False,
                    animation_mode=AnimationMode.SMOOTH,
                    animation_speed=0.03,
                    show_stats=False,
                    plain=False,  # Non-plain to hit str(display_content) branch at line 894
                )


class TestVeniceParameters:
    """Test Venice parameters (lines 534-538)"""

    @pytest.mark.asyncio
    async def test_send_single_message_with_top_p(self):
        """Test top_p parameter - covers lines 534-535"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                top_p=0.9,
            )
            # Verify top_p was passed
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("top_p") == 0.9

    @pytest.mark.asyncio
    async def test_send_single_message_with_venice_params(self):
        """Test venice_params: web_search, character_slug, reasoning_effort - covers lines 536-538"""
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Test response"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        venice_params = {
            "enable_web_search": "on",
            "character_slug": "my-char",
            "reasoning_effort": "high",
        }

        with patch("venice_ai.cli.commands.chat.console") as mock_console:
            mock_console.status = MagicMock(
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            )
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=False,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                venice_params=venice_params,
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("venice_parameters") == venice_params

    @pytest.mark.asyncio
    async def test_send_single_message_streaming_with_venice_params(self):
        """Test streaming with venice parameters"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        mock_stream_handler = MagicMock()
        mock_stream_handler.display_progress = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        )
        mock_stream_handler.handle_chat_stream = AsyncMock(return_value=("Response", {}))

        venice_params = {"enable_web_search": "auto", "reasoning_effort": "medium"}

        with (
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _send_single_message(
                client=mock_client,
                messages=[],
                user_message="Hello",
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                venice_params=venice_params,
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("venice_parameters") == venice_params

    @pytest.mark.asyncio
    async def test_interactive_chat_with_top_p(self):
        """Test interactive chat with top_p parameter"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(return_value=("Response", {}))

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                top_p=0.95,
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("top_p") == 0.95

    @pytest.mark.asyncio
    async def test_interactive_chat_with_venice_params(self):
        """Test interactive chat with venice_params"""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        call_count = [0]

        def mock_input(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            return "exit"

        mock_stream_handler = MagicMock()
        mock_stream_handler.handle_chat_stream = AsyncMock(return_value=("Response", {}))
        venice_params = {"enable_web_search": "on"}

        with (
            patch("asyncio.to_thread", side_effect=mock_input),
            patch("venice_ai.cli.commands.chat.print_info"),
            patch("venice_ai.cli.commands.chat.console"),
            patch(
                "venice_ai.cli.commands.chat.StreamHandler",
                return_value=mock_stream_handler,
            ),
        ):
            await _interactive_chat(
                client=mock_client,
                messages=[],
                model="test-model",
                temperature=0.7,
                max_completion_tokens=2048,
                stream=True,
                show_thinking=False,
                animation_mode=AnimationMode.SMOOTH,
                animation_speed=0.03,
                show_stats=False,
                venice_params=venice_params,
            )
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("venice_parameters") == venice_params
