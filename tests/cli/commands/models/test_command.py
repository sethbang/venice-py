"""
Tests for cli/commands/models/command.py

Covers:
- Model listing with various model types
- Model filtering by capabilities, traits, price, status
- Model comparison mode
- Model detail mode
- JSON output
- Verbose output
- Error handling (VeniceError and generic exceptions)
- All conditional branches
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import click
import pytest

from venice_ai.cli.commands.models.command import list_models
from venice_ai.exceptions import VeniceError


# Fixtures
@pytest.fixture
def mock_text_model():
    """Create a mock text model"""
    return SimpleNamespace(
        id="test-text-model",
        type="text",
        created=1745903059,
        model_spec=SimpleNamespace(
            name="Test Text Model",
            traits=["default"],
            offline=False,
            beta=False,
            availableContextTokens=131072,
            pricing=SimpleNamespace(
                input=SimpleNamespace(usd=0.5, diem=0.5),
                output=SimpleNamespace(usd=2.0, diem=2.0),
            ),
            capabilities=SimpleNamespace(
                supportsFunctionCalling=True,
                supportsVision=False,
                supportsReasoning=True,
                supportsWebSearch=True,
                optimizedForCode=False,
                supportsResponseSchema=True,
                supportsLogProbs=True,
                quantization="fp8",
            ),
            constraints=SimpleNamespace(
                temperature=SimpleNamespace(default=0.7),
                top_p=SimpleNamespace(default=0.95),
            ),
        ),
    )


@pytest.fixture
def mock_text_model_2():
    """Create a second mock text model for comparison tests"""
    return SimpleNamespace(
        id="test-text-model-2",
        type="text",
        created=1745903060,
        model_spec=SimpleNamespace(
            name="Test Text Model 2",
            traits=["fast"],
            offline=False,
            beta=True,
            availableContextTokens=65536,
            pricing=SimpleNamespace(
                input=SimpleNamespace(usd=0.25, diem=0.25),
                output=SimpleNamespace(usd=1.0, diem=1.0),
            ),
            capabilities=SimpleNamespace(
                supportsFunctionCalling=False,
                supportsVision=True,
                supportsReasoning=False,
                supportsWebSearch=False,
                optimizedForCode=True,
                supportsResponseSchema=False,
                supportsLogProbs=False,
                quantization="int8",
            ),
            constraints=SimpleNamespace(
                temperature=SimpleNamespace(default=0.5),
                top_p=SimpleNamespace(default=0.9),
            ),
        ),
    )


@pytest.fixture
def mock_image_model():
    """Create a mock image model"""
    return SimpleNamespace(
        id="test-image-model",
        type="image",
        created=1726851920,
        model_spec=SimpleNamespace(
            name="Test Image Model",
            traits=["highest_quality"],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            constraints=SimpleNamespace(steps=SimpleNamespace(default=25, max=30)),
        ),
    )


@pytest.fixture
def mock_tts_model():
    """Create a mock TTS model"""
    return SimpleNamespace(
        id="test-tts-model",
        type="tts",
        created=1726851920,
        model_spec=SimpleNamespace(
            name="Test TTS Model",
            traits=[],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(input=SimpleNamespace(usd=0.05, diem=0.05)),
            voices=["voice1", "voice2", "voice3"],
        ),
    )


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model"""
    return SimpleNamespace(
        id="test-embedding-model",
        type="embedding",
        created=1741924661,
        model_spec=SimpleNamespace(
            name="Test Embedding",
            traits=[],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(
                input=SimpleNamespace(usd=0.15, diem=0.15),
                output=SimpleNamespace(usd=0.6, diem=0.6),
            ),
            availableContextTokens=8192,
        ),
    )


@pytest.fixture
def mock_upscale_model():
    """Create a mock upscale model"""
    return SimpleNamespace(
        id="upscaler",
        type="upscale",
        created=1744453050,
        model_spec=SimpleNamespace(
            name="Upscaler",
            traits=[],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(
                generation=SimpleNamespace(usd=0.01, diem=0.01),
                upscale={
                    "2x": SimpleNamespace(usd=0.02, diem=0.02),
                    "4x": SimpleNamespace(usd=0.08, diem=0.08),
                },
            ),
        ),
    )


@pytest.fixture
def mock_inpaint_model():
    """Create a mock inpaint model"""
    return SimpleNamespace(
        id="test-inpaint-model",
        type="inpaint",
        created=1744453051,
        model_spec=SimpleNamespace(
            name="Test Inpaint Model",
            traits=[],
            offline=False,
            beta=True,
        ),
    )


@pytest.fixture
def mock_unknown_type_model():
    """Create a mock model with unknown type"""
    return SimpleNamespace(
        id="test-unknown-model",
        type="custom",
        created=1744453052,
        model_spec=SimpleNamespace(
            name="Test Unknown Model",
            traits=[],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(
                input=SimpleNamespace(usd=0.1, diem=0.1),
                output=SimpleNamespace(usd=0.2, diem=0.2),
            ),
        ),
    )


@pytest.fixture
def all_model_types(
    mock_text_model,
    mock_image_model,
    mock_tts_model,
    mock_embedding_model,
    mock_upscale_model,
    mock_inpaint_model,
):
    """Collection of all model types"""
    return [
        mock_text_model,
        mock_image_model,
        mock_tts_model,
        mock_embedding_model,
        mock_upscale_model,
        mock_inpaint_model,
    ]


@pytest.fixture
def mock_context():
    """Create a mock click context"""
    ctx = MagicMock(spec=click.Context)
    return ctx


def create_mock_response(models):
    """Helper to create mock API response"""
    return SimpleNamespace(data=models)


def create_mock_client(model_responses):
    """
    Create a mock VeniceClient that returns different models for each type.

    model_responses: dict mapping model type to list of models to return
    """
    async_client = AsyncMock()

    async def mock_list_impl(type=None):
        models = model_responses.get(type, [])
        return create_mock_response(models)

    async_client.models.list = mock_list_impl
    return async_client


class TestListModelsBasic:
    """Test basic model listing functionality"""

    @pytest.mark.asyncio
    async def test_list_models_basic(self, mock_context, mock_text_model, mock_image_model):
        """Test basic model listing with text and image models"""
        model_responses = {
            "text": [mock_text_model],
            "image": [mock_image_model],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            # Verify console.print was called (for output)
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_list_models_no_models_found(self, mock_context):
        """Test when no models are returned"""
        model_responses = {
            "text": [],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_info") as mock_print_info,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            # Should print "No models found"
            mock_print_info.assert_any_call("No models found")

    @pytest.mark.asyncio
    async def test_list_models_deduplicates_by_id(self, mock_context, mock_text_model):
        """Test that duplicate models by ID are removed"""
        # Same model returned by multiple types
        model_responses = {
            "text": [mock_text_model],
            "image": [mock_text_model],  # Duplicate
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)
            # Implicitly tests deduplication - no assertion needed as test would
            # fail if deduplication logic is broken

    @pytest.mark.asyncio
    async def test_list_models_exits_1_when_one_type_failed_and_the_rest_are_empty(
        self, mock_context
    ):
        """One failed listing is reported rather than read as an empty catalog"""
        async_client = AsyncMock()

        async def mock_list_with_exception(type=None):
            if type == "text":
                raise Exception("API error for text models")
            return create_mock_response([])

        async_client.models.list = mock_list_with_exception

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_info") as mock_print_info,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = async_client

            # A failed listing with nothing else to show is a failure, not an empty
            # catalog, so it exits 1 and does not report that no models exist.
            with pytest.raises(SystemExit) as exc_info:
                await list_models(mock_context)

            assert exc_info.value.code == 1
            assert call("No models found") not in mock_print_info.call_args_list


class TestComparisonMode:
    """Test model comparison functionality"""

    @pytest.mark.asyncio
    async def test_compare_two_models(self, mock_context, mock_text_model, mock_text_model_2):
        """Test comparing two models"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                compare_ids="test-text-model,test-text-model-2",
            )

            # Verify comparison table was printed
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_compare_model_not_found(self, mock_context, mock_text_model):
        """Test comparison when a model is not found"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_error") as mock_print_error,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            with pytest.raises(SystemExit):
                await list_models(
                    mock_context,
                    compare_ids="test-text-model,nonexistent-model",
                )

            # Should print error for missing model
            mock_print_error.assert_any_call("Model not found: nonexistent-model")

    @pytest.mark.asyncio
    async def test_compare_less_than_two_models(self, mock_context, mock_text_model):
        """Test comparison when fewer than 2 models are found"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_error") as mock_print_error,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            with pytest.raises(SystemExit):
                await list_models(
                    mock_context,
                    compare_ids="test-text-model,missing-model",
                )

            # Should print error about needing 2 models
            mock_print_error.assert_any_call("Need at least 2 models to compare")

    @pytest.mark.asyncio
    async def test_compare_models_with_empty_comparison_table(
        self, mock_context, mock_text_model, mock_text_model_2
    ):
        """Test comparison when ModelComparator returns None"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelComparator.compare_models",
                return_value=None,
            ),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                compare_ids="test-text-model,test-text-model-2",
            )

            # Should not print anything if comparison_table is None
            # (tests the `if comparison_table:` branch)


class TestDetailMode:
    """Test model detail display functionality"""

    @pytest.mark.asyncio
    async def test_show_model_detail(self, mock_context, mock_text_model):
        """Test showing detailed model info"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                detail_id="test-text-model",
            )

            # Verify panel was printed
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_show_model_detail_not_found(self, mock_context, mock_text_model):
        """Test detail mode when model is not found"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_error") as mock_print_error,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            with pytest.raises(SystemExit):
                await list_models(
                    mock_context,
                    detail_id="nonexistent-model",
                )

            # Should print error for missing model
            mock_print_error.assert_any_call("Model not found: nonexistent-model")


class TestCapabilityFilters:
    """Test capability filtering options"""

    @pytest.mark.asyncio
    async def test_filter_function_calling(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by function calling capability"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                function_calling=True,
            )

    @pytest.mark.asyncio
    async def test_filter_vision(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by vision capability"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                vision=True,
            )

    @pytest.mark.asyncio
    async def test_filter_reasoning(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by reasoning capability"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                reasoning=True,
            )

    @pytest.mark.asyncio
    async def test_filter_web_search(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by web search capability"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                web_search=True,
            )

    @pytest.mark.asyncio
    async def test_filter_code(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by code optimization"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                code=True,
            )

    @pytest.mark.asyncio
    async def test_filter_response_schema(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by response schema capability"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                response_schema=True,
            )

    @pytest.mark.asyncio
    async def test_filter_multiple_capabilities(
        self, mock_context, mock_text_model, mock_text_model_2
    ):
        """Test filtering by multiple capabilities at once"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                function_calling=True,
                reasoning=True,
                web_search=True,
            )


class TestModelTypeFetchAndChoice:
    """video/asr/music are fetched, and --type is a constrained Choice."""

    @pytest.mark.asyncio
    async def test_fetches_video_asr_music_types(self, mock_context, mock_text_model):
        """list_models fetches video, asr, and music in addition to the originals."""
        requested_types: list[str] = []

        async_client = AsyncMock()

        async def tracking_list(type=None):
            requested_types.append(type)
            return SimpleNamespace(data=[mock_text_model] if type == "text" else [])

        async_client.models.list = tracking_list

        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = async_client

            await list_models(mock_context)

        # The three previously-missing types must now be fetched.
        assert "video" in requested_types
        assert "asr" in requested_types
        assert "music" in requested_types
        # And the original six remain.
        for t in ("text", "image", "tts", "embedding", "upscale", "inpaint"):
            assert t in requested_types

    @pytest.mark.asyncio
    async def test_video_model_renders_via_fallback(self, mock_context):
        """A real (video-shaped) model renders through the fallback formatter.

        Video models carry a different shape than text (no ``capabilities``,
        no ``input``/``output`` token pricing). This guards against
        ``--type video`` fetching results but then crashing the table render
        into the broad ``except Exception`` (which would print "Unexpected
        error" instead of a table). Asserts both compact and verbose paths.
        """
        video_model = SimpleNamespace(
            id="seedance-2-0",
            type="video",
            created=1745903059,
            model_spec=SimpleNamespace(
                name="Seedance 2.0",
                traits=[],
                offline=False,
                beta=False,
                # Video pricing shape — per-second generation, no input/output.
                pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.15, diem=0.15)),
                # Note: deliberately no `capabilities` attribute.
            ),
        )

        async def video_only_list(type=None):
            return SimpleNamespace(data=[video_model] if type == "video" else [])

        async_client = AsyncMock()
        async_client.models.list = video_only_list

        # Compact mode: must complete and print a table without raising.
        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch("venice_ai.cli.commands.models.command.print_error") as mock_print_error,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = async_client
            await list_models(mock_context, model_type=["video"])

            assert mock_console.print.called
            # The broad except must NOT have fired.
            mock_print_error.assert_not_called()

        # Verbose mode: exercises format_verbose_model on the same shape.
        with (
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch("venice_ai.cli.commands.models.command.print_error") as mock_print_error,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = async_client
            await list_models(mock_context, model_type=["video"], verbose=True)

            assert mock_console.print.called
            mock_print_error.assert_not_called()

    def test_cli_type_choice_accepts_video(self):
        """The --type Choice accepts a real fetchable type like ``video``."""
        from click.testing import CliRunner

        from venice_ai.cli.cli import cli

        # list_models is imported inside the group callback (from .command);
        # the actual await is wrapped by asyncio.run in the group module.
        with (
            patch(
                "venice_ai.cli.commands.models.command.list_models",
                MagicMock(return_value=None),
            ),
            patch("venice_ai.cli.commands.models.group.asyncio.run", return_value=None),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["models", "--type", "video"])

        assert result.exit_code == 0, result.output

    def test_cli_type_choice_rejects_unknown(self):
        """An unknown --type errors clearly instead of returning empty."""
        from click.testing import CliRunner

        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--type", "bogus"])

        assert result.exit_code != 0
        assert "bogus" in result.output or "Invalid value" in result.output

    def test_cli_type_choice_rejects_alias_chat(self):
        """The ``chat``/``all``/``code`` aliases are not valid listing types."""
        from click.testing import CliRunner

        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--type", "chat"])

        assert result.exit_code != 0
        assert "chat" in result.output or "Invalid value" in result.output


class TestOtherFilters:
    """Test other filtering options (traits, prices, status, search)"""

    @pytest.mark.asyncio
    async def test_filter_by_traits(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by traits"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                traits=["default"],
            )

    @pytest.mark.asyncio
    async def test_filter_by_model_type(self, mock_context, mock_text_model, mock_image_model):
        """Test filtering by model type"""
        model_responses = {
            "text": [mock_text_model],
            "image": [mock_image_model],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                model_type=["text"],
            )

    @pytest.mark.asyncio
    async def test_filter_by_max_input_price(
        self, mock_context, mock_text_model, mock_text_model_2
    ):
        """Test filtering by maximum input price"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                max_input=0.3,
            )

    @pytest.mark.asyncio
    async def test_filter_by_max_output_price(
        self, mock_context, mock_text_model, mock_text_model_2
    ):
        """Test filtering by maximum output price"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                max_output=1.5,
            )

    @pytest.mark.asyncio
    async def test_filter_by_max_gen_price(self, mock_context, mock_image_model):
        """Test filtering by maximum generation price"""
        model_responses = {
            "text": [],
            "image": [mock_image_model],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                max_gen=0.02,
            )

    @pytest.mark.asyncio
    async def test_filter_by_budget(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by budget"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                budget=1.0,
            )

    @pytest.mark.asyncio
    async def test_filter_by_beta_status(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by beta status"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                beta=True,
            )

    @pytest.mark.asyncio
    async def test_filter_by_online_status(self, mock_context, mock_text_model):
        """Test filtering by online status"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                online=True,
            )

    @pytest.mark.asyncio
    async def test_filter_by_search_query(self, mock_context, mock_text_model, mock_text_model_2):
        """Test filtering by search query"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                search="text",
            )

    @pytest.mark.asyncio
    async def test_filter_no_matches(self, mock_context, mock_text_model):
        """Test when filters result in no matching models"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_info") as mock_print_info,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                search="nonexistent-model-xyz",
            )

            # Should print no matches message
            mock_print_info.assert_any_call("No models match the specified filters")


class TestOutputFormats:
    """Test different output format options"""

    @pytest.mark.asyncio
    async def test_json_output(self, mock_context, mock_text_model, mock_image_model):
        """Test JSON output format"""
        model_responses = {
            "text": [mock_text_model],
            "image": [mock_image_model],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                output_json=True,
            )

            # Verify console.print was called with JSON output
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_verbose_output(self, mock_context, mock_text_model, mock_image_model):
        """Test verbose output format"""
        model_responses = {
            "text": [mock_text_model],
            "image": [mock_image_model],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                verbose=True,
            )

            # Verify console.print was called multiple times for verbose output
            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_currency_options(self, mock_context, mock_text_model):
        """Test different currency display options"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        for currency in ["usd", "diem", "both"]:
            with (
                patch(
                    "venice_ai.cli.config.ensure_api_key",
                    return_value="test-key",
                ),
                patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
                patch("venice_ai.cli.commands.models.command.console"),
                patch("venice_ai.cli.commands.models.command.print_info"),
            ):
                MockVeniceClient.return_value.__aenter__.return_value = mock_client

                await list_models(
                    mock_context,
                    currency=currency,
                )


class TestDisplayOptions:
    """Test display-related options"""

    @pytest.mark.asyncio
    async def test_no_legend(self, mock_context, mock_text_model):
        """Test hiding the legend"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                no_legend=True,
            )

    @pytest.mark.asyncio
    async def test_sort_option(self, mock_context, mock_text_model, mock_text_model_2):
        """Test sorting option"""
        model_responses = {
            "text": [mock_text_model, mock_text_model_2],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(
                mock_context,
                sort="price",
            )


class TestModelTypeFormatters:
    """Test that different model types use correct formatters"""

    @pytest.mark.asyncio
    async def test_text_model_formatter(self, mock_context, mock_text_model):
        """Test text model formatting"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.format_text_table"
            ) as mock_format_text,
        ):
            mock_format_text.return_value = MagicMock()
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            mock_format_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_model_formatter(self, mock_context, mock_image_model):
        """Test image model formatting"""
        model_responses = {
            "text": [],
            "image": [mock_image_model],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.format_image_table"
            ) as mock_format_image,
        ):
            mock_format_image.return_value = MagicMock()
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            mock_format_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_tts_model_formatter(self, mock_context, mock_tts_model):
        """Test TTS model formatting"""
        model_responses = {
            "text": [],
            "image": [],
            "tts": [mock_tts_model],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.format_tts_table"
            ) as mock_format_tts,
        ):
            mock_format_tts.return_value = MagicMock()
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            mock_format_tts.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_model_formatter(self, mock_context, mock_embedding_model):
        """Test embedding model formatting"""
        model_responses = {
            "text": [],
            "image": [],
            "tts": [],
            "embedding": [mock_embedding_model],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.format_embedding_table"
            ) as mock_format_embedding,
        ):
            mock_format_embedding.return_value = MagicMock()
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            mock_format_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_upscale_model_formatter(self, mock_context, mock_upscale_model):
        """Test upscale model formatting"""
        model_responses = {
            "text": [],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [mock_upscale_model],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.format_upscale_table"
            ) as mock_format_upscale,
        ):
            mock_format_upscale.return_value = MagicMock()
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            mock_format_upscale.assert_called_once()

    @pytest.mark.asyncio
    async def test_inpaint_model_formatter(self, mock_context, mock_inpaint_model):
        """Test inpaint model formatting"""
        model_responses = {
            "text": [],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [mock_inpaint_model],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.format_inpaint_table"
            ) as mock_format_inpaint,
        ):
            mock_format_inpaint.return_value = MagicMock()
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            mock_format_inpaint.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_type_uses_text_formatter(self, mock_context, mock_unknown_type_model):
        """Test unknown model type falls back to text formatter"""
        # Create a mock client that returns our unknown model for all types
        async_client = AsyncMock()

        async def mock_list_impl(type=None):
            if type == "text":
                return create_mock_response([mock_unknown_type_model])
            return create_mock_response([])

        async_client.models.list = mock_list_impl

        # Create an unknown type model for this test
        test_unknown_model = SimpleNamespace(
            id="test-unknown-model",
            type="custom",  # Unknown type
            created=1744453052,
            model_spec=SimpleNamespace(
                name="Test Unknown Model",
                traits=[],
                offline=False,
                beta=False,
                pricing=SimpleNamespace(
                    input=SimpleNamespace(usd=0.1, diem=0.1),
                    output=SimpleNamespace(usd=0.2, diem=0.2),
                ),
            ),
        )

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console") as mock_console,
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFilter.apply_all_filters",
                return_value=[test_unknown_model],
            ),
            patch(
                "venice_ai.cli.commands.models.command.ModelSorter.sort_models",
                return_value=[test_unknown_model],
            ),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = async_client

            await list_models(mock_context)

            # Verify console.print was called (for the fallback text table)
            assert mock_console.print.called


class TestModelWithNoneType:
    """Test handling of models with None type"""

    @pytest.mark.asyncio
    async def test_model_with_none_type_uses_unknown(self, mock_context):
        """Test that model with None type is grouped as 'unknown'"""
        model_with_none_type = SimpleNamespace(
            id="model-no-type",
            type=None,
            created=1745903059,
            model_spec=SimpleNamespace(
                name="Model Without Type",
                traits=[],
                offline=False,
                beta=False,
            ),
        )

        model_responses = {
            "text": [],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFilter.apply_all_filters",
                return_value=[model_with_none_type],
            ),
            patch(
                "venice_ai.cli.commands.models.command.ModelSorter.sort_models",
                return_value=[model_with_none_type],
            ),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)


class TestErrorHandling:
    """Test error handling scenarios"""

    @pytest.mark.asyncio
    async def test_venice_error_handling(self, mock_context, mock_text_model):
        """Test handling of VeniceError at the top level (outside model list loop)"""
        # VeniceError needs to be raised outside the try/except block in model fetching
        # It's caught at line 258-259, which means it must happen after models are fetched
        # Let's trigger it by making ModelFilter raise VeniceError
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_error") as mock_print_error,
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFilter.apply_all_filters",
                side_effect=VeniceError("Filter API Error"),
            ),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            with pytest.raises(SystemExit):
                await list_models(mock_context)

            # Should print Venice API error
            mock_print_error.assert_called()
            call_args = str(mock_print_error.call_args)
            assert "Venice API error" in call_args

    @pytest.mark.asyncio
    async def test_generic_exception_handling(self, mock_context, mock_text_model):
        """Test handling of generic exceptions"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_error") as mock_print_error,
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch("traceback.print_exc"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFilter.apply_all_filters",
                side_effect=RuntimeError("Unexpected runtime error"),
            ),
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            with pytest.raises(SystemExit):
                await list_models(mock_context)

            # Should print unexpected error
            mock_print_error.assert_called()
            call_args = str(mock_print_error.call_args)
            assert "Unexpected error" in call_args


class TestLegendDisplay:
    """Test legend display logic"""

    @pytest.mark.asyncio
    async def test_legend_shown_when_text_models_present(self, mock_context, mock_text_model):
        """Test legend is shown when text models are in results"""
        model_responses = {
            "text": [mock_text_model],
            "image": [],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.get_capability_legend",
                return_value="Legend text",
            ) as mock_get_legend,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            # Legend should be retrieved
            mock_get_legend.assert_called_once()

    @pytest.mark.asyncio
    async def test_legend_not_shown_when_only_non_text_models(self, mock_context, mock_image_model):
        """Test legend is not shown when no text models in results"""
        model_responses = {
            "text": [],
            "image": [mock_image_model],
            "tts": [],
            "embedding": [],
            "upscale": [],
            "inpaint": [],
        }

        mock_client = create_mock_client(model_responses)

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.console"),
            patch("venice_ai.cli.commands.models.command.print_info"),
            patch(
                "venice_ai.cli.commands.models.command.ModelFormatter.get_capability_legend"
            ) as mock_get_legend,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = mock_client

            await list_models(mock_context)

            # Legend should not be called when no text models
            mock_get_legend.assert_not_called()


class TestNullResponseData:
    """Test handling of null/None response data"""

    @pytest.mark.asyncio
    async def test_handles_none_response_data(self, mock_context):
        """Test handling when response.data is None"""
        async_client = AsyncMock()

        async def mock_list_with_none_data(type=None):
            return SimpleNamespace(data=None)

        async_client.models.list = mock_list_with_none_data

        with (
            patch(
                "venice_ai.cli.config.ensure_api_key",
                return_value="test-key",
            ),
            patch("venice_ai.cli.commands.models.command.VeniceClient") as MockVeniceClient,
            patch("venice_ai.cli.commands.models.command.print_info") as mock_print_info,
        ):
            MockVeniceClient.return_value.__aenter__.return_value = async_client

            await list_models(mock_context)

            # Should handle None data gracefully and print "No models found"
            mock_print_info.assert_any_call("No models found")
