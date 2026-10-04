"""
End-to-end tests for Venice AI Audio resource covering coverage gaps.

These tests exercise the resources/audio.py module via the live Venice API
with VCR cassette recording/replay.  They target the following coverage gaps:

Gap 1 — ``transcribe()`` method (lines 656-736): 0% → full coverage
    - File path input (str / Path)
    - Raw bytes input
    - BytesIO input
    - File-like object input
    - Content-type detection from extensions (.wav, .mp3, unknown)
    - Response parsing (dict path)
    - Error paths (missing file, bad type)

Gap 2 — ``create_speech`` response branches (line 369)
    - Non-streaming with aiohttp.ClientResponse (lines 362-364)
    - Non-streaming with bytes response (lines 365-366)

Gap 3 — Streaming fallback paths (lines 446-448)
    - Normal chunked streaming
    - Streaming with VCR fallback (empty stream → full-body read)

Gap 4 — ``get_voices`` filter branches (lines 583-588)
    - Model-id filter, gender filter, region-code filter
    - No-filter listing

Gap 5 — Voice parsing logic (lines 549-563)
    - Voices with underscore (parsed gender/region)
    - Voices without underscore (unknown defaults)
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from venice_ai import VeniceClient, create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError
from venice_ai.models.selection import DynamicModelSelector
from venice_ai.types import AudioResponse, ResponseFormat, Voice

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def venice_client():
    """Create VeniceClient for E2E audio testing (requires VENICE_API_KEY)."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for E2E tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def offline_client():
    """Create a lightweight VeniceClient that doesn't need a real API key.

    Used for client-side validation tests that never hit the network.
    """
    client = VeniceClient(api_key="test-offline-key")
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def model_selector(venice_client):
    """Dynamic model selector using VCR-recorded model list."""
    return DynamicModelSelector(venice_client)


# ---------------------------------------------------------------------------
# TTS: create_speech  (non-streaming)
# ---------------------------------------------------------------------------


class TestCreateSpeechNonStreaming:
    """Cover create_speech non-streaming response handling (lines 362-369)."""

    async def test_non_streaming_speech_returns_audio_response(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Non-streaming create_speech returns AudioResponse with bytes content."""
        with vcr_cassette:
            model = await model_selector.select_audio_model()

            result = await venice_client.audio.create_speech(
                model=model,
                input="Hello, this is a non-streaming test.",
                voice="af_sky",
                response_format="mp3",
                speed=1.0,
                stream=False,
            )

            assert isinstance(result, AudioResponse)
            assert result.content is not None
            assert len(result.content) > 0
            # MP3 files start with 0xFF 0xFB or ID3 tag
            assert (
                result.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
                or len(result.content) > 100
            )

    async def test_non_streaming_with_enum_voice(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Non-streaming with Voice enum value and ResponseFormat enum."""
        with vcr_cassette:
            model = await model_selector.select_audio_model()

            result = await venice_client.audio.create_speech(
                model=model,
                input="Enum voice test.",
                voice=Voice.AF_SKY,
                response_format=ResponseFormat.MP3,
                stream=False,
            )

            assert isinstance(result, AudioResponse)
            assert len(result.content) > 0

    async def test_non_streaming_with_custom_speed(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Non-streaming with altered speed parameter."""
        with vcr_cassette:
            model = await model_selector.select_audio_model()

            result = await venice_client.audio.create_speech(
                model=model,
                input="Speed test at 1.5x.",
                voice="af_sky",
                speed=1.5,
                stream=False,
            )

            assert isinstance(result, AudioResponse)
            assert len(result.content) > 0


# ---------------------------------------------------------------------------
# TTS: create_speech  (streaming)
# ---------------------------------------------------------------------------


class TestCreateSpeechStreaming:
    """Cover streaming paths and fallback logic (lines 435-494)."""

    async def test_streaming_speech_yields_bytes(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Streaming create_speech yields bytes chunks."""
        with vcr_cassette:
            model = await model_selector.select_audio_model()

            stream = await venice_client.audio.create_speech(
                model=model,
                input="This is a streaming audio test sentence.",
                voice="af_sky",
                stream=True,
            )

            chunks = []
            async for chunk in stream:
                assert isinstance(chunk, bytes)
                chunks.append(chunk)

            assert len(chunks) > 0
            full_audio = b"".join(chunks)
            assert len(full_audio) > 0

    async def test_streaming_collect_all_chunks(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Streaming: collect ALL chunks to completion (exercises fallback path for VCR)."""
        with vcr_cassette:
            model = await model_selector.select_audio_model()

            stream = await venice_client.audio.create_speech(
                model=model,
                input="Longer text to exercise fallback streaming path for full coverage.",
                voice="af_sky",
                response_format="mp3",
                stream=True,
            )

            total_bytes = 0
            chunk_count = 0
            async for chunk in stream:
                total_bytes += len(chunk)
                chunk_count += 1

            assert chunk_count > 0
            assert total_bytes > 0


# ---------------------------------------------------------------------------
# TTS: create_speech  (validation)
# ---------------------------------------------------------------------------


class TestCreateSpeechValidation:
    """Cover input validation branches (lines 313-317).

    These tests use offline_client — they exercise client-side validation
    and never make network requests.
    """

    async def test_empty_input_raises_value_error(self, offline_client):
        """Empty input text raises ValueError (line 317)."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            await offline_client.audio.create_speech(
                model="tts-kokoro",
                input="",
                voice="af_sky",
                stream=False,
            )

    async def test_invalid_model_raises_error(self, offline_client):
        """Invalid model ID raises validation error (line 313)."""
        with pytest.raises((ValueError, APIError)):
            await offline_client.audio.create_speech(
                model="",
                input="This should fail.",
                voice="af_sky",
                stream=False,
            )


# ---------------------------------------------------------------------------
# Voice discovery: get_voices with filters  (lines 535-590)
# ---------------------------------------------------------------------------


class TestGetVoices:
    """Cover get_voices filtering and voice parsing logic."""

    async def test_get_all_voices(
        self,
        venice_client,
        vcr_cassette,
    ):
        """Listing all voices without filters (line 535-590)."""
        with vcr_cassette:
            result = await venice_client.audio.get_voices()

            assert len(result.data) > 0
            assert result.model_id_filter is None
            assert result.gender_filter is None
            assert result.region_code_filter is None

            # Verify voice detail structure
            voice = result.data[0]
            assert voice.id is not None
            assert voice.model_id is not None

    async def test_filter_by_model_id(
        self,
        venice_client,
        vcr_cassette,
    ):
        """Filter voices by model_id (line 583-584)."""
        with vcr_cassette:
            result = await venice_client.audio.get_voices(model_id="tts-kokoro")

            assert result.model_id_filter == "tts-kokoro"
            for voice in result.data:
                assert voice.model_id == "tts-kokoro"

    async def test_filter_by_gender(
        self,
        venice_client,
        vcr_cassette,
    ):
        """Filter voices by gender (line 585-586)."""
        with vcr_cassette:
            result = await venice_client.audio.get_voices(gender="female")

            assert result.gender_filter == "female"
            for voice in result.data:
                assert voice.gender == "female"

    async def test_filter_by_region_code(
        self,
        venice_client,
        vcr_cassette,
    ):
        """Filter voices by region_code (line 587-588)."""
        with vcr_cassette:
            result = await venice_client.audio.get_voices(region_code="a")

            assert result.region_code_filter == "a"
            for voice in result.data:
                assert voice.region_code == "a"

    async def test_combined_filters(
        self,
        venice_client,
        vcr_cassette,
    ):
        """Multiple filters applied simultaneously."""
        with vcr_cassette:
            result = await venice_client.audio.get_voices(
                model_id="tts-kokoro",
                gender="female",
                region_code="a",
            )

            for voice in result.data:
                assert voice.model_id == "tts-kokoro"
                assert voice.gender == "female"
                assert voice.region_code == "a"

    async def test_voice_parsing_populates_language_accent(
        self,
        venice_client,
        vcr_cassette,
    ):
        """Voice parsing extracts language/accent from REGION_LANGUAGE_MAPPING (lines 549-580)."""
        with vcr_cassette:
            result = await venice_client.audio.get_voices(
                model_id="tts-kokoro",
                region_code="a",
            )

            for voice in result.data:
                assert voice.language == "English"
                assert voice.accent == "American"


# ---------------------------------------------------------------------------
# Transcription: transcribe()  (lines 656-736)
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_ms: int = 200) -> bytes:
    """
    Generate a minimal valid WAV file (silence) for testing.

    Returns a complete WAV file as bytes containing PCM silence.
    """
    import struct

    sample_rate = 16000
    bits_per_sample = 16
    num_channels = 1
    num_samples = int(sample_rate * duration_ms / 1000)
    data_size = num_samples * num_channels * (bits_per_sample // 8)

    # WAV header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # file size - 8
        b"WAVE",
        b"fmt ",
        16,  # PCM format chunk size
        1,  # PCM format tag
        num_channels,
        sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),  # byte rate
        num_channels * (bits_per_sample // 8),  # block align
        bits_per_sample,
        b"data",
        data_size,
    )
    # Silent PCM data
    audio_data = b"\x00\x00" * num_samples
    return header + audio_data


class TestTranscribeFromBytes:
    """Cover transcribe() with raw bytes input (line 676-677).

    The Venice API requires a recognisable audio content-type.  Raw ``bytes``
    carry no filename, so the client names them from their magic bytes
    (``audio.wav`` for RIFF/WAVE) and derives the content type from that name.

    The *branch* is tested offline (mocking ``_request_multipart``) and the
    *happy-path* via a temp file with a ``.wav`` extension.
    """

    async def test_transcribe_bytes_branch_offline(self, offline_client):
        """Verify the bytes branch is entered and builds the correct multipart payload."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        fake_response = {"text": "hello"}

        with patch.object(
            offline_client.audio,
            "_request_multipart",
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_req:
            result = await offline_client.audio.transcribe(
                file=wav_bytes,
                model="nvidia/parakeet-tdt-0.6b-v3",
            )

            assert result.text == "hello"
            # Verify the file tuple sent to _request_multipart
            call_kwargs = mock_req.call_args
            files_dict = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
            _fname, _content, _ctype = files_dict["file"]
            assert _content is wav_bytes
            # _detect_audio_filename recognises WAV magic bytes → "audio.wav"
            assert _fname == "audio.wav"
            assert _ctype == "audio/wav"

    async def test_transcribe_bytes_via_file_path(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """E2E transcribe using a temp .wav file (exercises file-path branch too)."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            try:
                result = await venice_client.audio.transcribe(
                    file=tmp_path,
                    model=model,
                )
                assert result.text is not None
                assert isinstance(result.text, str)
            finally:
                os.unlink(tmp_path)


class TestTranscribeFromBytesIO:
    """Cover transcribe() with BytesIO input (line 678-679).

    A ``BytesIO`` has no filename either, so it is named from its magic bytes
    the same way as raw ``bytes``.  The branch is tested offline.
    """

    async def test_transcribe_bytesio_branch_offline(self, offline_client):
        """Verify the BytesIO branch is entered and reads the buffer."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        buf = io.BytesIO(wav_bytes)
        fake_response = {"text": "world"}

        with patch.object(
            offline_client.audio,
            "_request_multipart",
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_req:
            result = await offline_client.audio.transcribe(
                file=buf,
                model="nvidia/parakeet-tdt-0.6b-v3",
            )

            assert result.text == "world"
            call_kwargs = mock_req.call_args
            files_dict = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
            _fname, _content, _ctype = files_dict["file"]
            assert _content == wav_bytes
            # No name → sniffed from the RIFF/WAVE header
            assert _fname == "audio.wav"
            assert _ctype == "audio/wav"


class TestTranscribeFromFilePath:
    """Cover transcribe() with str/Path input (lines 670-675)."""

    async def test_transcribe_from_string_path(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Transcribe from a string file path."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            try:
                result = await venice_client.audio.transcribe(
                    file=tmp_path,
                    model=model,
                )

                assert result.text is not None
                assert isinstance(result.text, str)
            finally:
                os.unlink(tmp_path)

    async def test_transcribe_from_path_object(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Transcribe from a pathlib.Path object."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = Path(f.name)

            try:
                result = await venice_client.audio.transcribe(
                    file=tmp_path,
                    model=model,
                )

                assert result.text is not None
            finally:
                tmp_path.unlink()

    async def test_transcribe_missing_file_raises(self, offline_client):
        """Transcribe with non-existent file raises ValueError (line 672-673)."""
        with pytest.raises(ValueError, match="not found"):
            await offline_client.audio.transcribe(
                file="/nonexistent/path/audio.wav",
                model="nvidia/parakeet-tdt-0.6b-v3",
            )


class TestTranscribeFromFileLike:
    """Cover transcribe() with file-like object input (lines 680-692)."""

    async def test_transcribe_file_like_sync_read_offline(self, offline_client):
        """Verify file-like branch reads content and extracts filename (offline, lines 680-692)."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        fake_response = {"text": "file-like test"}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            with (
                open(tmp_path, "rb") as fobj,
                patch.object(
                    offline_client.audio,
                    "_request_multipart",
                    new_callable=AsyncMock,
                    return_value=fake_response,
                ) as mock_req,
            ):
                result = await offline_client.audio.transcribe(
                    file=fobj,
                    model="nvidia/parakeet-tdt-0.6b-v3",
                )

                assert result.text == "file-like test"
                call_kwargs = mock_req.call_args
                files_dict = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
                _fname, _content, _ctype = files_dict["file"]
                # File-like with .name → filename extracted from path
                assert _fname.endswith(".wav")
                assert _content == wav_bytes
                assert _ctype == "audio/wav"
        finally:
            os.unlink(tmp_path)

    async def test_transcribe_file_like_no_name_offline(self, offline_client):
        """File-like object without .name is named from its magic bytes (offline)."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        fake_response = {"text": "no-name test"}

        class FakeFile:
            """File-like object without .name attribute."""

            def read(self):
                return wav_bytes

        with patch.object(
            offline_client.audio,
            "_request_multipart",
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_req:
            result = await offline_client.audio.transcribe(
                file=FakeFile(),
                model="nvidia/parakeet-tdt-0.6b-v3",
            )

            assert result.text == "no-name test"
            call_kwargs = mock_req.call_args
            files_dict = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
            _fname, _content, _ctype = files_dict["file"]
            assert _content == wav_bytes
            # No .name → sniffed from the RIFF/WAVE header
            assert _fname == "audio.wav"
            assert _ctype == "audio/wav"

    async def test_transcribe_file_like_no_name_unrecognised_offline(self, offline_client):
        """Nameless file-like with unrecognised content keeps the bare name (offline)."""
        from unittest.mock import AsyncMock, patch

        payload = b"\x00\x01\x02\x03not-audio"

        class FakeFile:
            """File-like object without .name attribute."""

            def read(self):
                return payload

        with patch.object(
            offline_client.audio,
            "_request_multipart",
            new_callable=AsyncMock,
            return_value={"text": ""},
        ) as mock_req:
            await offline_client.audio.transcribe(
                file=FakeFile(),
                model="nvidia/parakeet-tdt-0.6b-v3",
            )

            files_dict = mock_req.call_args.kwargs["files"]
            _fname, _content, _ctype = files_dict["file"]
            assert _content == payload
            assert _fname == "audio"
            assert _ctype == "application/octet-stream"

    async def test_transcribe_file_like_non_bytes_read_raises(self, offline_client):
        """File-like object returning non-bytes from read() raises TypeError (line 686-688)."""

        class BadFile:
            """File-like object that returns a string instead of bytes."""

            def read(self):
                return "not bytes"

        with pytest.raises(TypeError, match="must return bytes"):
            await offline_client.audio.transcribe(
                file=BadFile(),
                model="nvidia/parakeet-tdt-0.6b-v3",
            )

    async def test_transcribe_file_like_sync_read_e2e(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Transcribe from a file-like object with synchronous read() (E2E, line 685)."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            # Open a real file in binary mode (has .read() and .name)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            try:
                with open(tmp_path, "rb") as fobj:
                    result = await venice_client.audio.transcribe(
                        file=fobj,
                        model=model,
                    )

                    assert result.text is not None
                    assert isinstance(result.text, str)
            finally:
                os.unlink(tmp_path)


class TestTranscribeValidation:
    """Cover transcribe() validation and error branches."""

    async def test_transcribe_unsupported_file_type_raises(self, offline_client):
        """Passing unsupported type raises TypeError (line 694)."""
        with pytest.raises(TypeError, match="Unsupported file type"):
            await offline_client.audio.transcribe(
                file=12345,  # type: ignore[arg-type]
                model="nvidia/parakeet-tdt-0.6b-v3",
            )


class TestTranscribeContentTypeDetection:
    """Cover content-type detection branch (lines 697-708)."""

    async def test_transcribe_mp3_content_type_offline(self, offline_client):
        """Content-type detection for .mp3 extension (offline, line 707-708)."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        fake_response = {"text": "mp3 test"}

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            with patch.object(
                offline_client.audio,
                "_request_multipart",
                new_callable=AsyncMock,
                return_value=fake_response,
            ) as mock_req:
                result = await offline_client.audio.transcribe(
                    file=tmp_path,
                    model="nvidia/parakeet-tdt-0.6b-v3",
                )

                assert result.text == "mp3 test"
                call_kwargs = mock_req.call_args
                files_dict = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
                _fname, _content, _ctype = files_dict["file"]
                assert _ctype == "audio/mpeg"
        finally:
            os.unlink(tmp_path)

    async def test_transcribe_unknown_extension_offline(self, offline_client):
        """Content-type falls back to application/octet-stream for unknown ext (offline, line 707 false branch)."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        fake_response = {"text": "unknown ext"}

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            with patch.object(
                offline_client.audio,
                "_request_multipart",
                new_callable=AsyncMock,
                return_value=fake_response,
            ) as mock_req:
                result = await offline_client.audio.transcribe(
                    file=tmp_path,
                    model="nvidia/parakeet-tdt-0.6b-v3",
                )

                assert result.text == "unknown ext"
                call_kwargs = mock_req.call_args
                files_dict = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
                _fname, _content, _ctype = files_dict["file"]
                assert _ctype == "application/octet-stream"
        finally:
            os.unlink(tmp_path)

    async def test_transcribe_wav_content_type_offline(self, offline_client):
        """Content-type detection for .wav extension (offline)."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        fake_response = {"text": "wav test"}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            with patch.object(
                offline_client.audio,
                "_request_multipart",
                new_callable=AsyncMock,
                return_value=fake_response,
            ) as mock_req:
                result = await offline_client.audio.transcribe(
                    file=tmp_path,
                    model="nvidia/parakeet-tdt-0.6b-v3",
                )

                assert result.text == "wav test"
                call_kwargs = mock_req.call_args
                files_dict = call_kwargs.kwargs.get("files") or call_kwargs[1].get("files")
                _fname, _content, _ctype = files_dict["file"]
                assert _ctype == "audio/wav"
        finally:
            os.unlink(tmp_path)

    async def test_transcribe_mp3_extension_e2e(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Content-type detection for .mp3 extension (E2E)."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            # Write with .mp3 extension to exercise content-type mapping
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            try:
                result = await venice_client.audio.transcribe(
                    file=tmp_path,
                    model=model,
                )
                assert result.text is not None
            finally:
                os.unlink(tmp_path)


class TestTranscribeResponseParsing:
    """Cover transcribe() response parsing branches (lines 727-736)."""

    async def test_transcribe_bytes_response_offline(self, offline_client):
        """When _request_multipart returns bytes, parse as JSON (line 729-733)."""
        import json as _json
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)
        fake_response_bytes = _json.dumps({"text": "bytes response"}).encode()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            with patch.object(
                offline_client.audio,
                "_request_multipart",
                new_callable=AsyncMock,
                return_value=fake_response_bytes,
            ):
                result = await offline_client.audio.transcribe(
                    file=tmp_path,
                    model="nvidia/parakeet-tdt-0.6b-v3",
                )

                assert result.text == "bytes response"
        finally:
            os.unlink(tmp_path)

    async def test_transcribe_unexpected_response_type_offline(self, offline_client):
        """When _request_multipart returns unexpected type, raises TypeError (line 735-736)."""
        from unittest.mock import AsyncMock, patch

        wav_bytes = _make_wav_bytes(duration_ms=200)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            with (
                patch.object(
                    offline_client.audio,
                    "_request_multipart",
                    new_callable=AsyncMock,
                    return_value=12345,
                ),
                pytest.raises(TypeError, match="Unexpected response type"),
            ):
                await offline_client.audio.transcribe(
                    file=tmp_path,
                    model="nvidia/parakeet-tdt-0.6b-v3",
                )
        finally:
            os.unlink(tmp_path)


class TestTranscribeWithOptions:
    """Cover transcribe() optional parameters (response_format, timestamps, language).

    Uses temp .wav files so the API receives a valid content-type.
    """

    async def test_transcribe_with_timestamps(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Transcribe with timestamps=True populates word-level data."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            try:
                result = await venice_client.audio.transcribe(
                    file=tmp_path,
                    model=model,
                    timestamps=True,
                )

                assert result.text is not None
                # timestamps may or may not return words for silence,
                # but the parameter was exercised
            finally:
                os.unlink(tmp_path)

    async def test_transcribe_with_response_format(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Transcribe with explicit response_format='json'."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            try:
                result = await venice_client.audio.transcribe(
                    file=tmp_path,
                    model=model,
                    response_format="json",
                )

                assert result.text is not None
            finally:
                os.unlink(tmp_path)

    async def test_transcribe_with_language(
        self,
        venice_client,
        model_selector,
        vcr_cassette,
    ):
        """Transcribe with explicit language='en'."""
        with vcr_cassette:
            wav_bytes = _make_wav_bytes(duration_ms=500)
            model = await model_selector.select_asr_model()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            try:
                result = await venice_client.audio.transcribe(
                    file=tmp_path,
                    model=model,
                    language="en",
                )

                assert result.text is not None
            finally:
                os.unlink(tmp_path)
