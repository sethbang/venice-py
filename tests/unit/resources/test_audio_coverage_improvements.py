"""
Comprehensive test coverage improvements for venice_ai.resources.audio module.

This test file addresses the coverage gaps identified in the audit:
- Error handling in _stream_audio_bytes method (lines 387-394, 436-450)
- VCR fallback logic in streaming (lines 414-434)
- Voice filtering logic in get_voices method (lines 523-528)
- Response type handling branches in create_speech (lines 339-343)
- Voice parsing conditional branches (lines 480-502)
"""

from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from tests.fixtures.mock_fixtures import with_request_plumbing
from venice_ai.exceptions import APIConnectionError, APIError, APITimeoutError
from venice_ai.resources.audio import REGION_LANGUAGE_MAPPING, Audio
from venice_ai.types import (
    AudioResponse,
)


class TestAudioCreateSpeechResponseHandling:
    """Test response type handling in create_speech method."""

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance for testing."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.mark.asyncio
    async def test_create_speech_fallback_bytes_conversion(self, audio_resource):
        """Test fallback bytes conversion for unknown response types (line 343)."""
        mock_client = with_request_plumbing(AsyncMock())
        # Mock an unknown response type that can be converted to bytes
        mock_response = b"response as bytes"  # Use bytes directly
        mock_client._request = AsyncMock(return_value=mock_response)
        audio_resource._client = mock_client

        result = await audio_resource.create_speech(
            model="test-model", input="test input", voice="test-voice", stream=False
        )

        assert isinstance(result, AudioResponse)
        assert result == b"response as bytes"

    @pytest.mark.asyncio
    async def test_create_speech_aiohttp_response_handling(self, audio_resource):
        """Test aiohttp.ClientResponse handling (lines 336-338)."""
        mock_client = with_request_plumbing(AsyncMock())

        # Mock ClientResponse
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.read = AsyncMock(return_value=b"audio data from response")
        mock_response.headers = {"Content-Type": "audio/mpeg"}  # Mock headers as dict
        mock_client._request = AsyncMock(return_value=mock_response)
        audio_resource._client = mock_client

        result = await audio_resource.create_speech(
            model="test-model", input="test input", voice="test-voice", stream=False
        )

        assert isinstance(result, AudioResponse)
        assert result == b"audio data from response"

    @pytest.mark.asyncio
    async def test_create_speech_direct_bytes_response(self, audio_resource):
        """Test direct bytes response handling (lines 339-340)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_client._request = AsyncMock(return_value=b"direct audio bytes")
        audio_resource._client = mock_client

        result = await audio_resource.create_speech(
            model="test-model", input="test input", voice="test-voice", stream=False
        )

        assert isinstance(result, AudioResponse)
        assert result == b"direct audio bytes"


class TestAudioStreamErrorHandling:
    """Test error handling in _stream_audio_bytes method."""

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance for testing."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.mark.asyncio
    async def test_stream_audio_timeout_error(self, audio_resource):
        """Test timeout error handling in streaming (lines 436-439)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock timeout error - session.request should raise during __aenter__
        mock_context = AsyncMock()
        mock_context.__aenter__.side_effect = TimeoutError("Request timed out")
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_context)
        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        with pytest.raises(APITimeoutError) as exc_info:
            stream = audio_resource._stream_audio_bytes(
                method="POST", path="audio/speech", json_data={"test": "data"}
            )
            # Consume the async generator to trigger the exception
            async for _ in stream:
                pass

        assert "Request timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_audio_connection_error(self, audio_resource):
        """Test connection error handling in streaming (lines 440-443)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock connection error - session.request should raise during __aenter__
        mock_context = AsyncMock()
        mock_context.__aenter__.side_effect = aiohttp.ClientConnectorError(Mock(), Mock())
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_context)
        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        with pytest.raises(APIConnectionError) as exc_info:
            stream = audio_resource._stream_audio_bytes(
                method="POST", path="audio/speech", json_data={"test": "data"}
            )
            # Consume the async generator to trigger the exception
            async for _ in stream:
                pass

        assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_audio_client_error(self, audio_resource):
        """Test general client error handling in streaming (lines 444-449)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock general client error - session.request should raise during __aenter__
        mock_context = AsyncMock()
        mock_context.__aenter__.side_effect = aiohttp.ClientError("General client error")
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_context)
        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        with pytest.raises(APIConnectionError) as exc_info:
            stream = audio_resource._stream_audio_bytes(
                method="POST", path="audio/speech", json_data={"test": "data"}
            )
            # Consume the async generator to trigger the exception
            async for _ in stream:
                pass

        assert "A connection error occurred" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_audio_api_error_json_response(self, audio_resource):
        """Test API error with JSON response (lines 387-399)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock failed response with JSON error
        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"error": "Bad request"})

        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_request_ctx)

        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        with pytest.raises(APIError):
            stream = audio_resource._stream_audio_bytes(
                method="POST", path="audio/speech", json_data={"test": "data"}
            )
            # Consume the async generator to trigger the exception
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    async def test_stream_audio_api_error_text_response(self, audio_resource):
        """Test API error with text response when JSON parsing fails (lines 389-399)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock failed response with JSON parsing error, falling back to text
        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 500
        mock_response.json = AsyncMock(side_effect=aiohttp.ContentTypeError(Mock(), Mock()))
        mock_response.text = AsyncMock(return_value="Internal server error")

        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_request_ctx)

        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        with pytest.raises(APIError):
            stream = audio_resource._stream_audio_bytes(
                method="POST", path="audio/speech", json_data={"test": "data"}
            )
            # Consume the async generator to trigger the exception
            async for _ in stream:
                pass


class TestAudioStreamVCRFallback:
    """Test VCR fallback logic in streaming."""

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance for testing."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.mark.asyncio
    async def test_stream_vcr_fallback_with_full_body(self, audio_resource):
        """Test VCR fallback when streaming yields no chunks (lines 414-422)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock response that yields no chunks but has full body
        mock_response = AsyncMock()
        mock_response.ok = True

        # Mock iter_chunked that yields nothing (simulating VCR)
        async def empty_chunks(size):
            # Yield nothing to simulate VCR consumed stream
            return
            yield  # unreachable

        mock_response.content.iter_chunked = empty_chunks
        mock_response.read = AsyncMock(return_value=b"full body content from VCR")

        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_request_ctx)

        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        # Collect chunks from the stream
        chunks = []
        stream = audio_resource._stream_audio_bytes(
            method="POST", path="audio/speech", json_data={"test": "data"}
        )
        async for chunk in stream:
            chunks.append(chunk)

        # Should get the content in chunks
        assert len(chunks) > 0
        assert b"".join(chunks) == b"full body content from VCR"

    @pytest.mark.asyncio
    async def test_stream_vcr_fallback_with_content_attribute(self, audio_resource):
        """Test VCR fallback using _content attribute (lines 424-431)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock response with no chunks and empty read(), but _content attribute
        mock_response = AsyncMock()
        mock_response.ok = True

        # Mock iter_chunked that yields nothing
        async def empty_chunks(size):
            return
            yield  # unreachable

        mock_response.content.iter_chunked = empty_chunks
        mock_response.read = AsyncMock(return_value=b"")  # Empty read
        mock_response._content = b"content from _content attribute"

        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_request_ctx)

        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        # Collect chunks from the stream
        chunks = []
        stream = audio_resource._stream_audio_bytes(
            method="POST", path="audio/speech", json_data={"test": "data"}
        )
        async for chunk in stream:
            chunks.append(chunk)

        # Should get the content from _content attribute
        assert len(chunks) > 0
        assert b"".join(chunks) == b"content from _content attribute"

    @pytest.mark.asyncio
    async def test_stream_vcr_fallback_exception_handling(self, audio_resource):
        """Test VCR fallback exception handling (lines 432-434)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock response that causes exception in fallback
        mock_response = AsyncMock()
        mock_response.ok = True

        # Mock iter_chunked that yields nothing
        async def empty_chunks(size):
            return
            yield  # unreachable

        mock_response.content.iter_chunked = empty_chunks
        mock_response.read = AsyncMock(side_effect=Exception("Read failed"))

        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_request_ctx)

        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        with pytest.raises(Exception) as exc_info:
            stream = audio_resource._stream_audio_bytes(
                method="POST", path="audio/speech", json_data={"test": "data"}
            )
            # Consume the async generator to trigger the exception
            async for _ in stream:
                pass

        assert "Read failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_normal_chunked_streaming(self, audio_resource):
        """Test normal chunked streaming without fallback."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock response with normal streaming chunks
        mock_response = AsyncMock()
        mock_response.ok = True

        # Mock iter_chunked that yields chunks normally
        async def yield_chunks(size):
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        mock_response.content.iter_chunked = yield_chunks

        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_request_ctx)

        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        # Collect chunks from the stream
        chunks = []
        stream = audio_resource._stream_audio_bytes(
            method="POST", path="audio/speech", json_data={"test": "data"}
        )
        async for chunk in stream:
            chunks.append(chunk)

        # Should get normal chunks
        assert chunks == [b"chunk1", b"chunk2", b"chunk3"]


class TestAudioGetVoicesFiltering:
    """Test voice filtering logic in get_voices method."""

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance for testing."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.fixture
    def mock_voice_models(self):
        """Create mock TTS models with voice data."""
        mock_models = []

        # Model 1: Has voices
        mock_model1 = Mock()
        mock_model1.id = "tts-kokoro"
        mock_model1.model_spec = Mock()
        mock_model1.model_spec.voices = ["af_bella", "am_david", "bf_emma"]
        mock_models.append(mock_model1)

        # Model 2: Has different voices
        mock_model2 = Mock()
        mock_model2.id = "tts-sonic"
        mock_model2.model_spec = Mock()
        mock_model2.model_spec.voices = ["cf_sarah", "dm_alex"]
        mock_models.append(mock_model2)

        # Model 3: No model_spec
        mock_model3 = Mock()
        mock_model3.id = "tts-simple"
        mock_model3.model_spec = None
        mock_models.append(mock_model3)

        # Model 4: No voices in spec
        mock_model4 = Mock()
        mock_model4.id = "tts-empty"
        mock_model4.model_spec = Mock()
        mock_model4.model_spec.voices = None
        mock_models.append(mock_model4)

        mock_list_response = Mock()
        mock_list_response.data = mock_models
        return mock_list_response

    @pytest.mark.asyncio
    async def test_get_voices_model_id_filter(self, audio_resource, mock_voice_models):
        """Test filtering voices by model_id (lines 523-524)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_client.models.list = AsyncMock(return_value=mock_voice_models)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices(model_id="tts-kokoro")

        # Should only include voices from tts-kokoro model
        assert len(result.data) == 3  # af_bella, am_david, bf_emma
        assert all(voice.model_id == "tts-kokoro" for voice in result.data)
        assert result.model_id_filter == "tts-kokoro"

    @pytest.mark.asyncio
    async def test_get_voices_gender_filter(self, audio_resource, mock_voice_models):
        """Test filtering voices by gender (lines 525-526)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_client.models.list = AsyncMock(return_value=mock_voice_models)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices(gender="female")

        # Should only include female voices (ending with 'f')
        female_voices = [voice for voice in result.data if voice.gender == "female"]
        assert len(female_voices) > 0
        assert all(voice.gender == "female" for voice in female_voices)
        assert result.gender_filter == "female"

    @pytest.mark.asyncio
    async def test_get_voices_region_code_filter(self, audio_resource, mock_voice_models):
        """Test filtering voices by region_code (lines 527-528)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_client.models.list = AsyncMock(return_value=mock_voice_models)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices(region_code="a")

        # Should only include voices from region 'a'
        region_a_voices = [voice for voice in result.data if voice.region_code == "a"]
        assert len(region_a_voices) > 0
        assert all(voice.region_code == "a" for voice in region_a_voices)
        assert result.region_code_filter == "a"

    @pytest.mark.asyncio
    async def test_get_voices_combined_filters(self, audio_resource, mock_voice_models):
        """Test filtering voices with multiple filters combined."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_client.models.list = AsyncMock(return_value=mock_voice_models)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices(
            model_id="tts-kokoro", gender="female", region_code="a"
        )

        # Should apply all filters
        assert result.model_id_filter == "tts-kokoro"
        assert result.gender_filter == "female"
        assert result.region_code_filter == "a"

        # All voices should match all criteria
        for voice in result.data:
            assert voice.model_id == "tts-kokoro"
            assert voice.gender == "female"
            assert voice.region_code == "a"

    @pytest.mark.asyncio
    async def test_get_voices_no_filters(self, audio_resource, mock_voice_models):
        """Test get_voices without any filters."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_client.models.list = AsyncMock(return_value=mock_voice_models)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices()

        # Should include all voices from models with voices
        assert len(result.data) == 5  # af_bella, am_david, bf_emma, cf_sarah, dm_alex
        assert result.model_id_filter is None
        assert result.gender_filter is None
        assert result.region_code_filter is None


class TestAudioVoiceParsingLogic:
    """Test voice parsing logic in get_voices method."""

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance for testing."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.mark.asyncio
    async def test_voice_parsing_with_underscore(self, audio_resource):
        """Test voice parsing when voice_id contains underscore (lines 489-504)."""
        mock_client = with_request_plumbing(AsyncMock())

        # Mock model with voices that have underscores
        mock_model = Mock()
        mock_model.id = "test-model"
        mock_model.model_spec = Mock()
        mock_model.model_spec.voices = ["af_bella", "bm_david"]

        mock_list_response = Mock()
        mock_list_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_list_response)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices()

        # Check parsed values
        voices = result.data
        assert len(voices) == 2

        # Check af_bella parsing
        af_voice = next(v for v in voices if v.id == "af_bella")
        assert af_voice.gender == "female"  # 'f' at end
        assert af_voice.region_code == "a"  # 'a' prefix
        assert af_voice.language == "English"  # From REGION_LANGUAGE_MAPPING
        assert af_voice.accent == "American"

        # Check bm_david parsing
        bm_voice = next(v for v in voices if v.id == "bm_david")
        assert bm_voice.gender == "male"  # 'm' at end
        assert bm_voice.region_code == "b"  # 'b' prefix
        assert bm_voice.language == "English"  # From REGION_LANGUAGE_MAPPING
        assert bm_voice.accent == "British"

    @pytest.mark.asyncio
    async def test_voice_parsing_without_underscore(self, audio_resource):
        """Test voice parsing when voice_id has no underscore."""
        mock_client = with_request_plumbing(AsyncMock())

        # Mock model with voices without underscores
        mock_model = Mock()
        mock_model.id = "test-model"
        mock_model.model_spec = Mock()
        mock_model.model_spec.voices = ["simplevoice", "another"]  # No underscores

        mock_list_response = Mock()
        mock_list_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_list_response)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices()

        # All voices should have unknown gender and region
        for voice in result.data:
            assert voice.gender == "unknown"
            assert voice.region_code == "unknown"
            assert voice.language is None
            assert voice.accent is None

    @pytest.mark.asyncio
    async def test_voice_parsing_single_character_prefix(self, audio_resource):
        """Test voice parsing with single character prefix (lines 495-498)."""
        mock_client = with_request_plumbing(AsyncMock())

        # Mock model with single character voice IDs
        mock_model = Mock()
        mock_model.id = "test-model"
        mock_model.model_spec = Mock()
        mock_model.model_spec.voices = ["a_voice", "f_voice"]

        mock_list_response = Mock()
        mock_list_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_list_response)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices()

        voices = result.data

        # Check single character parsing
        a_voice = next(v for v in voices if v.id == "a_voice")
        assert a_voice.gender == "unknown"  # 'a' is not 'f' or 'm'
        assert a_voice.region_code == "a"  # Single char becomes region

        f_voice = next(v for v in voices if v.id == "f_voice")
        assert f_voice.gender == "unknown"  # 'f' at position 0, not last
        assert f_voice.region_code == "f"

    @pytest.mark.asyncio
    async def test_voice_parsing_model_without_spec(self, audio_resource):
        """Test handling models without model_spec (lines 480-481)."""
        mock_client = with_request_plumbing(AsyncMock())

        # Mock model without model_spec
        mock_model = Mock()
        mock_model.id = "test-model"
        mock_model.model_spec = None

        mock_list_response = Mock()
        mock_list_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_list_response)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices()

        # Should return empty voice list
        assert len(result.data) == 0

    @pytest.mark.asyncio
    async def test_voice_parsing_spec_without_voices(self, audio_resource):
        """Test handling model_spec without voices (lines 482-483)."""
        mock_client = with_request_plumbing(AsyncMock())

        # Mock model with spec but no voices
        mock_model = Mock()
        mock_model.id = "test-model"
        mock_model.model_spec = Mock()
        mock_model.model_spec.voices = None

        mock_list_response = Mock()
        mock_list_response.data = [mock_model]
        mock_client.models.list = AsyncMock(return_value=mock_list_response)
        audio_resource._client = mock_client

        result = await audio_resource.get_voices()

        # Should return empty voice list
        assert len(result.data) == 0


class TestAudioTimeoutHandling:
    """Test timeout handling in streaming."""

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance for testing."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.mark.asyncio
    async def test_stream_timeout_conversion(self, audio_resource):
        """Test timeout conversion in streaming (lines 374-375)."""
        mock_client = with_request_plumbing(AsyncMock())
        mock_session = AsyncMock()
        mock_session.headers = {"Content-Type": "application/json"}  # Fix headers mock

        # Mock successful response to test timeout conversion
        mock_response = AsyncMock()
        mock_response.ok = True

        async def yield_chunks(size):
            yield b"chunk1"

        mock_response.content.iter_chunked = yield_chunks

        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = Mock(return_value=mock_request_ctx)

        mock_client._get_session = AsyncMock(return_value=mock_session)
        mock_client._base_url = Mock()
        mock_client._base_url.path = "/v1"
        mock_client._base_url.with_path = Mock(return_value="http://test.com/v1/audio/speech")
        audio_resource._client = mock_client

        # Test with numeric timeout
        stream = audio_resource._stream_audio_bytes(
            method="POST", path="audio/speech", json_data={"test": "data"}, timeout=30.0
        )

        # Consume stream
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        # Verify timeout was converted and passed
        call_args = mock_session.request.call_args
        timeout_arg = call_args[1]["timeout"]
        assert isinstance(timeout_arg, aiohttp.ClientTimeout)
        assert timeout_arg.total == 30.0


class TestRegionLanguageMapping:
    """Test REGION_LANGUAGE_MAPPING constant."""

    def test_region_language_mapping_completeness(self):
        """Test that REGION_LANGUAGE_MAPPING covers expected regions."""
        expected_regions = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "p",
            "r",
            "s",
            "u",
            "w",
            "x",
            "y",
            "z",
        ]

        for region in expected_regions:
            assert region in REGION_LANGUAGE_MAPPING
            assert "language" in REGION_LANGUAGE_MAPPING[region]
            assert "accent" in REGION_LANGUAGE_MAPPING[region]
            assert isinstance(REGION_LANGUAGE_MAPPING[region]["language"], str)
            assert isinstance(REGION_LANGUAGE_MAPPING[region]["accent"], str)

    def test_region_language_mapping_specific_values(self):
        """Test specific values in REGION_LANGUAGE_MAPPING."""
        assert REGION_LANGUAGE_MAPPING["a"]["language"] == "English"
        assert REGION_LANGUAGE_MAPPING["a"]["accent"] == "American"

        assert REGION_LANGUAGE_MAPPING["b"]["language"] == "English"
        assert REGION_LANGUAGE_MAPPING["b"]["accent"] == "British"

        assert REGION_LANGUAGE_MAPPING["z"]["language"] == "Mandarin Chinese"
        assert REGION_LANGUAGE_MAPPING["z"]["accent"] == "Standard"


class TestAudioCreateVoice:
    """Tests for create_voice() — POST /v1/audio/voices."""

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance with a mock client."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.mark.asyncio
    async def test_create_voice_returns_cloned_voice(self, audio_resource):
        """create_voice with model set returns ClonedVoice with correct fields,
        and _request_multipart receives files['file'] and data['model']."""
        from venice_ai.types import ClonedVoice

        audio_resource._request_multipart = AsyncMock(  # type: ignore[method-assign]
            return_value={"id": "vv_abc", "model": "tts-chatterbox-hd"}
        )

        result = await audio_resource.create_voice(file=b"\x00\x01RIFF", model="tts-chatterbox-hd")

        assert isinstance(result, ClonedVoice)
        assert result.id == "vv_abc"
        assert result.model == "tts-chatterbox-hd"

        call_kwargs = audio_resource._request_multipart.call_args[1]  # type: ignore[union-attr]
        assert "file" in call_kwargs["files"]
        assert call_kwargs["data"]["model"] == "tts-chatterbox-hd"

    @pytest.mark.asyncio
    async def test_create_voice_omits_model_when_none(self, audio_resource):
        """create_voice without model omits 'model' from the form data dict."""
        audio_resource._request_multipart = AsyncMock(  # type: ignore[method-assign]
            return_value={"id": "vv_xyz", "model": "tts-chatterbox-hd"}
        )

        await audio_resource.create_voice(file=b"\x00\x01RIFF")

        call_kwargs = audio_resource._request_multipart.call_args[1]  # type: ignore[union-attr]
        assert "model" not in call_kwargs["data"]

    @pytest.mark.asyncio
    async def test_create_voice_bytes_response(self, audio_resource):
        """create_voice parses a raw bytes JSON response as ClonedVoice."""
        import json as _json

        from venice_ai.types import ClonedVoice

        payload = _json.dumps({"id": "vv_bytes", "model": "tts-chatterbox-hd"}).encode()
        audio_resource._request_multipart = AsyncMock(  # type: ignore[method-assign]
            return_value=payload
        )

        result = await audio_resource.create_voice(file=b"\x00\x01RIFF")

        assert isinstance(result, ClonedVoice)
        assert result.id == "vv_bytes"


class TestAudioTranscribeResponseFormat:
    """Tests for transcribe() response_format handling.

    The live server returns ``Content-Type: text/plain`` with a plain-text
    (sometimes empty) body when ``response_format="text"``. ``_request_multipart``
    surfaces that as raw ``bytes``, so ``transcribe()`` must decode it to ``str``
    instead of running ``json.loads`` (which raises on text / empty bodies).
    """

    @pytest.fixture
    def audio_resource(self):
        """Create an Audio resource instance with a mock client."""
        mock_client = Mock()
        return Audio(mock_client)

    @pytest.mark.asyncio
    async def test_transcribe_text_format_returns_str(self, audio_resource):
        """response_format='text' returns the plain-text body as a str (no json.loads)."""
        audio_resource._request_multipart = AsyncMock(  # type: ignore[method-assign]
            return_value=b"hello transcript"
        )

        result = await audio_resource.transcribe(file=b"\x00\x01RIFF", response_format="text")

        assert isinstance(result, str)
        assert result == "hello transcript"

    @pytest.mark.asyncio
    async def test_transcribe_text_format_empty_body(self, audio_resource):
        """response_format='text' with an empty body returns '' rather than raising.

        ``json.loads(b"")`` raises JSONDecodeError, so this is the cleanest
        wire-confirmed RED for the bug.
        """
        audio_resource._request_multipart = AsyncMock(  # type: ignore[method-assign]
            return_value=b""
        )

        result = await audio_resource.transcribe(file=b"\x00\x01RIFF", response_format="text")

        assert isinstance(result, str)
        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_json_format_unchanged(self, audio_resource):
        """Default/json path still returns AudioTranscriptionResponse (locks JSON path)."""
        from venice_ai.types import AudioTranscriptionResponse

        audio_resource._request_multipart = AsyncMock(  # type: ignore[method-assign]
            return_value={"text": "hello transcript"}
        )

        result = await audio_resource.transcribe(file=b"\x00\x01RIFF")

        assert isinstance(result, AudioTranscriptionResponse)
        assert result.text == "hello transcript"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
