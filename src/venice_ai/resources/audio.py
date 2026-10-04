"""
Venice AI Audio API resources.

This module provides classes for interacting with the Venice AI Audio API
for text-to-speech (TTS) and automatic-speech-recognition (ASR / Whisper).

Music generation has its own resource at :mod:`venice_ai.resources.music`
(accessed via ``client.music``); pre-v2.0.0 it lived here alongside TTS,
but the namespaces are now split so each resource covers a single content
domain.

The audio API allows for:
- Converting text to natural-sounding speech (text-to-speech)
- Selecting from multiple voice options for speech synthesis
- Controlling speech speed and output format
- Both full and streaming response modes
- Transcribing and translating audio to text (Whisper models)
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Literal,
    cast,
    overload,
)

import aiohttp

from .._resource import APIResource
from ..types.api import (
    AudioResponse,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    ClonedVoice,
    VoiceDetail,
    VoiceList,
)
from ..types.enums import ResponseFormat, Voice
from ..utils.errors import read_body, resolve_timeout, wrap_aiohttp_errors
from ..validation.validators import validate_model_id
from ._inline_audio import sniff_audio_format

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

logger = logging.getLogger(__name__)

#: Mapping of single-letter region codes to language and accent information.
#:
#: This dictionary provides language and accent metadata for voice model region codes
#: used in TTS model identifiers. The region codes are typically found as prefixes
#: in voice model names (e.g., "tts-kokoro-a" uses region "a" for American English).
#:
#: Each region code maps to a dictionary containing:
#:     - **language**: The primary language spoken by voices in this region
#:     - **accent**: The specific accent or variant within that language
#:
#: Region Codes:
#:     - **a**: American English
#:     - **b**: British English
#:     - **c**: Canadian English
#:     - **d**: Standard German
#:     - **e**: European Spanish
#:     - **f**: Standard French
#:     - **g**: General English
#:     - **h**: Standard Hindi
#:     - **i**: Standard Italian
#:     - **j**: Standard Japanese
#:     - **k**: Standard Korean
#:     - **p**: Standard Portuguese
#:     - **r**: Standard Russian
#:     - **s**: Scottish English
#:     - **u**: US English (alternative encoding)
#:     - **w**: Welsh English
#:     - **x**: Australian English
#:     - **y**: Indian English
#:     - **z**: Mandarin Chinese
#:
#: Note:
#:     This mapping is used internally by the :meth:`Audio.get_voices` method
#:     to provide language and accent information when listing available voices.
REGION_LANGUAGE_MAPPING: dict[str, dict[str, str]] = {
    "a": {"language": "English", "accent": "American"},
    "b": {"language": "English", "accent": "British"},
    "c": {"language": "English", "accent": "Canadian"},
    "d": {"language": "German", "accent": "Standard"},
    "e": {"language": "Spanish", "accent": "European Standard"},
    "f": {"language": "French", "accent": "Standard"},
    "g": {"language": "English", "accent": "General"},
    "h": {"language": "Hindi", "accent": "Standard"},
    "i": {"language": "Italian", "accent": "Standard"},
    "j": {"language": "Japanese", "accent": "Standard"},
    "k": {"language": "Korean", "accent": "Standard"},
    "p": {"language": "Portuguese", "accent": "Standard"},
    "r": {"language": "Russian", "accent": "Standard"},
    "s": {"language": "English", "accent": "Scottish"},
    "u": {"language": "English", "accent": "US"},  # Alternative/Specific US
    "w": {"language": "English", "accent": "Welsh"},
    "x": {"language": "English", "accent": "Australian"},
    "y": {"language": "English", "accent": "Indian"},
    "z": {"language": "Mandarin Chinese", "accent": "Standard"},
}


class Audio(APIResource["VeniceClient"]):
    """
    Asynchronous interface for Venice AI's Audio API.

    The Audio class covers two capabilities served under the ``/audio/*``
    namespace: Text-to-Speech (TTS) synthesis and speech-to-text transcription
    (Whisper). Async music generation was split out to its own resource in
    v2.0.0; use :class:`~venice_ai.resources.music.Music` (accessed as
    ``client.music``) instead.

    **Core Capabilities:**
        - **Text-to-Speech Generation**: Convert text input to high-quality speech audio
        - **Voice Selection**: Choose from multiple voice models with different characteristics
        - **Format Control**: Generate audio in various formats (MP3, WAV, etc.)
        - **Speed Adjustment**: Control speech rate from 0.25x to 4.0x normal speed
        - **Streaming Support**: Real-time audio generation and chunk-based delivery
        - **Voice Discovery**: List and filter available voice models by attributes
        - **Transcription & Translation**: Speech-to-text via Whisper-family models

    **Usage Patterns:**
        The Audio class is designed to be accessed through the Venice AI client's
        :attr:`~venice_ai._client.VeniceClient.audio` property rather than instantiated directly.
        This ensures proper authentication, configuration, and connection management.

    **Performance Considerations:**
        - Streaming mode reduces latency for long text inputs
        - Batch operations are more efficient than individual requests
        - Voice model caching improves subsequent request performance
        - Audio format selection impacts file size and quality trade-offs

    Args:
        client: The Venice AI client instance providing authentication and connection management.
            This client handles all HTTP communication, error handling, and response parsing.

    Example:
        Basic text-to-speech generation::

            async with VeniceClient() as client:
                # Generate speech audio
                audio_bytes = await client.audio.create_speech(
                    model="tts-kokoro",
                    input="Welcome to Venice AI's text-to-speech service!",
                    voice="af_heart",
                    response_format="mp3"
                )

                # Save audio to file
                with open("welcome.mp3", "wb") as f:
                    f.write(audio_bytes)

        Real-time streaming generation::

            async with VeniceClient() as client:
                # Stream audio chunks as they're generated
                stream = await client.audio.create_speech(
                    model="tts-kokoro",
                    input="This is a longer text that will be streamed...",
                    voice="af_heart",
                    stream=True
                )

                # Process chunks in real-time
                with open("streamed_audio.mp3", "wb") as f:
                    async for chunk in stream:
                        f.write(chunk)
                        # Optionally process or play chunk immediately

    Note:
        All methods in this class are asynchronous and must be awaited. The class
        inherits from :class:`~venice_ai._resource.APIResource` which provides
        the underlying HTTP request infrastructure and error handling.

    See Also:
        - :class:`~venice_ai.types.audio.Voice`: Enumeration of available voice options
        - :class:`~venice_ai.types.audio.ResponseFormat`: Supported audio output formats
        - :class:`~venice_ai.types.audio.AudioResponse`: Response wrapper for audio data
    """

    @overload
    async def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: str | Voice,
        response_format: str | ResponseFormat | None = None,
        speed: float | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: Literal[False] = False,
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> AudioResponse: ...

    @overload
    async def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: str | Voice,
        response_format: str | ResponseFormat | None = None,
        speed: float | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: Literal[True],
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> AsyncIterator[bytes]: ...

    @overload
    async def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: str | Voice,
        response_format: str | ResponseFormat | None = None,
        speed: float | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool,
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> AudioResponse | AsyncIterator[bytes]: ...

    async def create_speech(
        self,
        *,
        input: str,
        model: str,
        voice: str | Voice,
        response_format: str | ResponseFormat | None = None,
        speed: float | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> AudioResponse | AsyncIterator[bytes]:
        """
        Generates audio from input text asynchronously.

        Converts the provided text to speech using the specified model and voice
        using asynchronous requests. The audio can be returned either as complete
        binary data or as an async stream of audio chunks for real-time processing.

        :param model: ID of the model to use for speech generation (e.g., "tts-kokoro").
        :type model: str
        :param input: The text to convert to speech. Maximum length varies by model.
        :type input: str
        :param voice: The voice to use for the generated audio. Can be a string literal
            or a :class:`~venice_ai.types.audio.Voice` enum value (e.g., ``Voice.AF_HEART``
            or ``"af_heart"``). Voice IDs are per-model — call
            :meth:`get_voices` (e.g. ``await client.audio.get_voices(model_id="tts-kokoro")``)
            for the live catalog of voices a given model accepts.
        :type voice: Union[str, venice_ai.types.audio.Voice]
        :param response_format: The format to return the audio in. Can be a string literal or a
            :class:`~venice_ai.types.audio.ResponseFormat` enum value. Defaults to "mp3".
        :type response_format: Optional[Union[str, venice_ai.types.audio.ResponseFormat]]
        :param speed: The speed of the generated audio. Select a value from 0.25 to 4.0.
            Defaults to 1.0.
        :type speed: Optional[float]
        :param language: Optional language hint. Accepted values are model-specific:
            Qwen 3 accepts full names (``"English"``, ``"Chinese"``, ...); xAI and
            ElevenLabs accept ISO 639-1 codes (``"en"``, ``"ja"``, ...); MiniMax
            accepts full names. Unsupported values are silently ignored. Omit to let
            the model auto-detect.
        :type language: Optional[str]
        :param prompt: Optional style prompt controlling emotion and delivery (e.g.
            ``"Very happy."``). Supported by models advertising ``supportsPromptParam``
            (currently Qwen 3 TTS); ignored otherwise. Max 500 characters.
        :type prompt: Optional[str]
        :param temperature: Optional sampling temperature (0–2) for speech generation.
            Supported by models advertising ``supportsTemperatureParam`` (Qwen 3,
            Orpheus, Chatterbox HD).
        :type temperature: Optional[float]
        :param top_p: Optional nucleus-sampling parameter (0–1). Supported by models
            advertising ``supportsTopPParam`` (currently Qwen 3 TTS).
        :type top_p: Optional[float]
        :param stream: Whether to stream the audio data. If True, returns an AsyncIterator
            of audio chunks. If False, returns the complete audio data. Defaults to False.
        :type stream: Optional[bool]
        :param timeout: Request timeout in seconds or an aiohttp.ClientTimeout object.
            If not provided, uses the client's default timeout.


        :return: If stream is False, returns the audio data as AudioResponse (awaitable). If stream is True,
            returns an AsyncIterator yielding chunks of audio data as bytes.


        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises ValueError: If the input text is empty or invalid parameters are provided.

        Example:
            Basic non-streaming text-to-speech:

            .. code-block:: python

                import asyncio
                from venice_ai import VeniceClient
                from venice_ai.types.audio import Voice, ResponseFormat

                async def generate_speech():
                    async with VeniceClient() as client:

                        # Generate speech with enum values
                        audio_bytes = await client.audio.create_speech(
                            model="tts-kokoro",
                            input="Hello, this is a test.",
                            voice=Voice.AF_HEART
                        )

                        # Save to file
                        with open("speech.mp3", "wb") as f:
                            f.write(audio_bytes)

                        # Using string literals and different format
                        audio_bytes = await client.audio.create_speech(
                            model="tts-kokoro",
                            input="Hello with different settings.",
                            voice="af_heart",
                            response_format="wav",
                            speed=1.2
                        )

                asyncio.run(generate_speech())

            Streaming text-to-speech:

            .. code-block:: python

                async def stream_speech():
                    async with VeniceClient() as client:

                        # Stream audio data
                        stream = await client.audio.create_speech(
                            model="tts-kokoro",
                            input="This is a streamed audio example.",
                            voice="af_heart",
                            stream=True
                        )

                        # Write streamed chunks to file
                        with open("streamed_speech.mp3", "wb") as f:
                            async for chunk in stream:
                                f.write(chunk)

                asyncio.run(stream_speech())
        """
        # Validate model ID
        validate_model_id(model, "model")

        # Validate input text is not empty
        if not input:
            raise ValueError("Input text cannot be empty")

        # Create Pydantic request model - only pass non-None values to use Field defaults
        request_params: dict[str, object] = {
            "input": input,
            "model": model,
            "voice": voice,
            "streaming": stream,
        }
        if response_format is not None:
            request_params["response_format"] = response_format
        if speed is not None:
            request_params["speed"] = speed
        if language is not None:
            request_params["language"] = language
        if prompt is not None:
            request_params["prompt"] = prompt
        if temperature is not None:
            request_params["temperature"] = temperature
        if top_p is not None:
            request_params["top_p"] = top_p

        # Filter to only valid AudioSpeechRequest fields for mypy type safety
        valid_fields = AudioSpeechRequest.model_fields.keys()
        filtered_params = {k: v for k, v in request_params.items() if k in valid_fields}

        # Use model_validate for proper type handling with Pydantic
        speech_request = AudioSpeechRequest.model_validate(filtered_params)

        # Build request options with validated payload
        headers = {"Accept": "audio/*"}
        body = speech_request.model_dump(exclude_none=True)

        if stream:
            # Use raw byte streaming for audio data
            return self._stream_audio_bytes(
                method="POST",
                path="audio/speech",
                json_data=body,
                headers=headers,
                timeout=timeout,
            )
        else:
            # Use the client's regular request method with raw_response=True
            response = await self._client._request(
                method="POST",
                path="audio/speech",
                json_data=body,
                headers=headers,
                raw_response=True,
                timeout=timeout,
            )
            # Handle aiohttp.ClientResponse properly
            if isinstance(response, aiohttp.ClientResponse):
                content = await read_body(response)
                return AudioResponse(content, response)
            elif isinstance(response, bytes):
                return AudioResponse(response, None)
            else:
                # Fallback: assume response can be converted to bytes
                return AudioResponse(bytes(response), None)

    async def _stream_audio_bytes(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream raw audio bytes from the API.

        This method handles streaming audio responses that return raw bytes
        rather than JSON data, specifically for the audio/speech endpoint.
        """
        session = await self._client._get_session()
        request_headers = self._client._request_headers(session, headers)

        # Properly join the base URL path with the endpoint path
        base_path = self._client._base_url.path.rstrip("/")
        endpoint_path = path.lstrip("/")
        full_path = f"{base_path}/{endpoint_path}"
        url = self._client._base_url.with_path(full_path)

        final_timeout = resolve_timeout(timeout, self._client._timeout)

        # The whole exchange, the streamed body included, maps transport
        # failures the way every other request does.
        # The SIWE re-signer is published only while the request is sent, where
        # the retry middleware runs, and withdrawn as soon as the response
        # arrives: this generator may be finalized in another context after the
        # caller stops iterating, where the withdrawal could not run.
        with contextlib.ExitStack() as resign_scope:
            resign_scope.enter_context(self._client._siwe_resigning(headers))
            async with (
                wrap_aiohttp_errors(),
                session.request(
                    method,
                    url,
                    json=json_data,
                    params=params,
                    headers=request_headers,
                    timeout=final_timeout,
                ) as response,
            ):
                resign_scope.close()
                if not response.ok:
                    try:
                        body = await response.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        body = await response.text()

                    from ..exceptions import _make_status_error

                    raise _make_status_error(
                        message=f"API request failed with status {response.status}",
                        request=None,
                        body=body,
                        response=response,
                    )

                # The status arrived; a stall from here on is reported as a
                # stream that was cut off, like a chat Stream.
                async with wrap_aiohttp_errors(phase="stream"):
                    # Stream the audio content as raw bytes
                    #
                    # PRIMARY PATH: Standard aiohttp streaming via iter_chunked()
                    # FALLBACK PATH: Ensures compatibility with older aiohttp versions,
                    # proxies that buffer responses, or edge cases where streaming fails.
                    # This fallback handles environments where iter_chunked() may not work
                    # as expected due to HTTP middleware, non-standard server implementations,
                    # or library version differences.
                    chunk_count = 0
                    fallback_reason = None
                    try:
                        # First attempt: normal streaming from HTTP response
                        async for chunk in response.content.iter_chunked(8192):
                            if chunk:
                                chunk_count += 1
                                yield chunk
                        logger.debug(f"Audio streaming: Used standard path, {chunk_count} chunks")
                    except (AttributeError, StopAsyncIteration) as e:
                        fallback_reason = type(e).__name__
                        logger.debug(f"Audio streaming: Standard path failed ({e}), using fallback")
                        pass  # Will fall through to fallback logic

                    # Fallback: If streaming failed or no chunks were yielded, read full body
                    # This handles non-streaming responses, proxies, and various HTTP client behaviors
                    if chunk_count == 0:
                        if fallback_reason is None:
                            fallback_reason = "empty_stream"
                        logger.warning(
                            f"Audio streaming: Using fallback read path (reason: {fallback_reason})"
                        )

                        # Track fallback metrics
                        try:
                            from ..observability.metrics import get_enhanced_metrics

                            metrics = get_enhanced_metrics()
                            if metrics._enabled:
                                metrics.streaming_fallback_total.labels(
                                    endpoint="audio/speech", reason=fallback_reason
                                ).inc()
                        except Exception:
                            pass  # nosec B110

                        full_body = await response.read()
                        if full_body:
                            # Chunk the full body for consistent streaming interface
                            for i in range(0, len(full_body), 8192):
                                chunk = full_body[i : i + 8192]
                                if chunk:
                                    yield chunk
                        else:
                            # Additional fallback: try internal content attributes
                            # Some HTTP implementations store content differently
                            content_attr = getattr(response, "_content", None)
                            if content_attr:
                                full_body = content_attr
                                for i in range(0, len(full_body), 8192):
                                    chunk = full_body[i : i + 8192]
                                    if chunk:
                                        yield chunk

    async def stream_long_text(
        self,
        *,
        input: str,
        model: str,
        voice: str | Voice,
        response_format: str | ResponseFormat = "mp3",
        speed: float | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_words_per_segment: int | None = None,
        max_concurrency: int = 4,
        on_segment_complete: Any = None,
        timeout: float | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio for long input by splitting into parallel segments.

        Convenience method that delegates to
        :func:`venice_ai.audio_helpers.stream_long_text`. Use this instead of
        :meth:`create_speech` when:

        * The input would exceed a model's hard output cap (e.g. the qwen3
          family caps at ~15.9 s of audio regardless of input length).
        * You want pseudo-streaming on a model that buffers the full response
          rather than streaming progressively (qwen3, orpheus, chatterbox,
          inworld, gemini — only xai-v1 and kokoro truly stream today).

        Only ``response_format="mp3"`` is supported.

        :return: An :class:`AsyncIterator` of audio bytes in input order.

        Example::

            # voice values are model-specific — call
            # client.audio.get_voices(model_id=...) for the live catalog.
            async for chunk in client.audio.stream_long_text(
                input=long_poem,
                model="tts-kokoro",
                voice="af_heart",
            ):
                buffer.write(chunk)
        """
        from ..audio_helpers import stream_long_text as _impl

        return _impl(
            self._client,
            input=input,
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
            language=language,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_words_per_segment=max_words_per_segment,
            max_concurrency=max_concurrency,
            on_segment_complete=on_segment_complete,
            timeout=timeout,
        )

    async def get_voices(
        self,
        *,
        model_id: str | None = None,
        gender: Literal["male", "female", "unknown"] | None = None,
        region_code: str | None = None,
    ) -> VoiceList:
        """
        Lists available text-to-speech (TTS) voices by filtering all available models asynchronously.
        This is a client-side convenience method.

        :param model_id: Optional. Filter voices by specific model ID.
        :type model_id: Optional[str]
        :param gender: Optional. Filter voices by gender.

        :param region_code: Optional. Filter voices by region code.


        :return: A list of available TTS voices matching the specified filters.


        :raises venice_ai.exceptions.APIError: If the API request fails.
        """
        # Get TTS models specifically
        tts_models = await self._client.models.list(type="tts")

        voice_details = []
        for model in tts_models.data:
            # TTS models carry their voice list on the spec's ``voices`` attribute.
            # Use ``getattr`` rather than an ``isinstance(spec, TtsModelSpec)`` narrowing
            # so duck-typed specs without a strict TtsModelSpec type still resolve.
            spec = getattr(model, "model_spec", None)
            voices = getattr(spec, "voices", None) if spec is not None else None
            if voices:
                # Each voice in the voices list is a voice option
                for voice_id in voices:
                    # Parse voice ID to determine gender and region
                    parsed_gender = "unknown"
                    parsed_region = "unknown"

                    if "_" in voice_id:
                        parts = voice_id.split("_", 1)  # Split only on first underscore
                        if len(parts) >= 2:
                            prefix = parts[0]
                            parsed_region = (
                                prefix[:-1] if len(prefix) > 1 else prefix
                            )  # Remove last char (gender)
                            gender_char = prefix[-1] if len(prefix) > 1 else ""

                            if gender_char == "f":
                                parsed_gender = "female"
                            elif gender_char == "m":
                                parsed_gender = "male"

                    voice_details.append(
                        VoiceDetail(
                            id=voice_id,
                            model_id=model.id,
                            gender=cast(Literal["male", "female", "unknown"], parsed_gender),
                            region_code=parsed_region,
                            language=REGION_LANGUAGE_MAPPING.get(parsed_region, {}).get("language"),
                            accent=REGION_LANGUAGE_MAPPING.get(parsed_region, {}).get("accent"),
                        )
                    )

        # Apply filters
        if model_id:
            voice_details = [v for v in voice_details if v.model_id == model_id]
        if gender:
            voice_details = [v for v in voice_details if v.gender == gender]
        if region_code:
            voice_details = [v for v in voice_details if v.region_code == region_code]

        return VoiceList(
            object="list",
            data=voice_details,
            model_id_filter=model_id,
            gender_filter=gender,
            region_code_filter=region_code,
        )

    @staticmethod
    def _detect_audio_filename(data: bytes) -> str:
        """Return a filename with an appropriate extension based on magic bytes.

        This allows the multipart upload to include a correct content-type
        even when the caller supplies raw ``bytes`` without a file path.
        Containers are identified by the same sniffer that reads inline job
        results (a leading ID3v2 tag is skipped, AAC ADTS is told apart from
        MPEG audio by its layer bits). What it does not recognise falls back
        to looser upload hints: any MPEG frame sync or a bare ID3 tag is named
        ``.mp3``, and an EBML header ``.webm``.
        """
        sniffed = sniff_audio_format(data)
        if sniffed is not None:
            return f"audio.{sniffed}"
        # MPEG audio frame sync whose header the sniffer could not parse
        if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
            return "audio.mp3"
        # ID3 tag in front of audio the sniffer did not recognise
        if data[:3] == b"ID3":
            return "audio.mp3"
        # WebM / Matroska (EBML header)
        if data[:4] == b"\x1a\x45\xdf\xa3":
            return "audio.webm"
        # Fallback: keep no extension so caller sees the original behaviour
        return "audio"

    async def _prepare_audio_file(
        self, file: str | bytes | BinaryIO | Path
    ) -> tuple[bytes, str, str]:
        """Resolve an audio ``file`` input to ``(content, filename, content_type)``.

        Accepts a path (str/Path), raw bytes, ``io.BytesIO``, or any binary
        file-like object. Content type is inferred from the filename extension;
        inputs without a name are named from their magic bytes first. Anything
        unrecognised is sent as ``application/octet-stream``.
        """
        file_content: bytes
        filename: str = "audio"

        if isinstance(file, (str, Path)):
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError(f"Audio file not found: {file}")
            file_content = file_path.read_bytes()
            filename = file_path.name
        elif isinstance(file, bytes):
            file_content = file
        elif isinstance(file, io.BytesIO):
            file_content = file.read()
        elif hasattr(file, "read") and callable(file.read):
            result = cast(Any, file).read()
            if asyncio.iscoroutine(result):
                file_content = await result
            else:
                file_content = result
            if not isinstance(file_content, bytes):
                raise TypeError("File-like object must return bytes from read()")
            # Try to get filename from file-like object
            if hasattr(file, "name"):
                filename = Path(file.name).name
        else:
            raise TypeError(f"Unsupported file type: {type(file)}")

        if filename == "audio":
            # No name to go by (raw bytes, BytesIO, a nameless file-like):
            # detect the format from magic bytes so the API receives a
            # recognisable content type.
            filename = self._detect_audio_filename(file_content)

        # Detect content type from the file extension
        content_type = "application/octet-stream"
        ext = Path(filename).suffix.lower()
        content_type_map = {
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".mp4": "audio/mp4",
            ".ogg": "audio/ogg",
            ".oga": "audio/ogg",
            ".webm": "audio/webm",
        }
        if ext in content_type_map:
            content_type = content_type_map[ext]

        return file_content, filename, content_type

    @overload
    async def transcribe(
        self,
        *,
        file: str | bytes | BinaryIO | Path,
        model: str = ...,
        response_format: Literal["json"] | None = None,
        timestamps: bool | None = None,
        language: str | None = None,
    ) -> AudioTranscriptionResponse: ...

    @overload
    async def transcribe(
        self,
        *,
        file: str | bytes | BinaryIO | Path,
        model: str = ...,
        response_format: Literal["text"],
        timestamps: bool | None = None,
        language: str | None = None,
    ) -> str: ...

    async def transcribe(
        self,
        *,
        file: str | bytes | BinaryIO | Path,
        model: str = "nvidia/parakeet-tdt-0.6b-v3",
        response_format: str | None = None,
        timestamps: bool | None = None,
        language: str | None = None,
    ) -> AudioTranscriptionResponse | str:
        """Transcribe audio to text (POST /audio/transcriptions).

        Converts audio input into text using the specified ASR (Automatic Speech
        Recognition) model. Supports various audio formats including WAV, FLAC,
        MP3, M4A, AAC, and MP4.

        :param file: Audio file to transcribe. Can be a file path (string or Path),
            raw audio bytes, or a file-like object opened in binary mode.
        :type file: Union[str, bytes, BinaryIO, Path]
        :param model: ASR model ID to use for transcription.
            Defaults to ``"nvidia/parakeet-tdt-0.6b-v3"``.
        :type model: str
        :param response_format: ``"json"`` (default, or ``None``) returns an
            :class:`AudioTranscriptionResponse` with the transcribed text and
            optional word-level timestamps; ``"text"`` returns the raw
            transcript as a plain ``str`` (the server responds with
            ``text/plain``, so it is not JSON-decoded).
        :type response_format: Optional[str]
        :param timestamps: If ``True``, include word-level timestamps in the response.
        :type timestamps: Optional[bool]
        :param language: Language of the input audio (e.g., ``"en"``).
        :type language: Optional[str]

        :return: Either the parsed :class:`AudioTranscriptionResponse` (for
            ``"json"`` / default) or the plain transcript ``str`` (for
            ``"text"``).
        :rtype: AudioTranscriptionResponse | str

        :raises ValueError: If the model ID is invalid or the file cannot be read.
        :raises venice_ai.exceptions.APIError: If the API request fails.

        Example:
            Basic audio transcription::

                async with VeniceClient() as client:
                    result = await client.audio.transcribe(
                        file="recording.mp3",
                        model="nvidia/parakeet-tdt-0.6b-v3",
                    )
                    print(result.text)

            Transcription with timestamps::

                async with VeniceClient() as client:
                    result = await client.audio.transcribe(
                        file="recording.wav",
                        timestamps=True,
                    )
                    print(result.text)
                    if result.words:
                        for word in result.words:
                            print(f"{word.word}: {word.start}s - {word.end}s")
        """
        # Validate model ID
        validate_model_id(model, "model")

        # Validate request parameters via Pydantic model
        transcription_request = AudioTranscriptionRequest(
            model=model,
            response_format=response_format,
            timestamps=timestamps,
            language=language,
        )

        # Prepare the audio file content
        file_content, filename, content_type = await self._prepare_audio_file(file)

        # Build multipart files dict
        files_dict: dict[str, Any] = {
            "file": (filename, file_content, content_type),
        }

        # Build form data from non-None request parameters
        form_data = transcription_request.model_dump(exclude_none=True)

        # Make the multipart request
        response = await self._request_multipart(
            method="POST",
            path="audio/transcriptions",
            files=files_dict,
            data=form_data,
        )

        # response_format="text" yields a text/plain body (possibly empty) that
        # _request_multipart surfaces as raw bytes — return it as a str without
        # JSON-decoding, regardless of the Accept header the SDK sent.
        if response_format == "text":
            if isinstance(response, bytes):
                return response.decode("utf-8")
            return cast(str, response)

        # Parse the JSON response into the response model
        if isinstance(response, dict):
            return AudioTranscriptionResponse.model_validate(response)
        elif isinstance(response, bytes):
            import json as _json

            return AudioTranscriptionResponse.model_validate(_json.loads(response))
        else:
            raise TypeError(f"Unexpected response type: {type(response)}")

    async def create_voice(
        self,
        *,
        file: str | bytes | BinaryIO | Path,
        model: str | None = None,
    ) -> ClonedVoice:
        """Clone a voice from an audio sample (POST /v1/audio/voices).

        Returns a :class:`~venice_ai.types.api.audio.ClonedVoice` whose ``id``
        is a ``vv_<id>`` handle. Pass that handle as the ``voice`` parameter to
        :meth:`create_speech`, paired with the **same** ``model``::

            async with VeniceClient() as client:
                voice = await client.audio.create_voice(file="sample.mp3")
                audio = await client.audio.create_speech(
                    input="Hello in my cloned voice.",
                    model=voice.model,
                    voice=voice.id,
                )

        :param file: Voice sample — a file path (str/Path), raw bytes, or a
            binary file-like object. A clean 5–10s speech recording is
            recommended. Accepted containers depend on the model
            (``tts-chatterbox-hd``: MP3/WAV/FLAC/M4A; ``tts-minimax-speech-02-hd``:
            MP3/WAV).
        :type file: Union[str, bytes, BinaryIO, Path]
        :param model: Optional. The Venice TTS model to pair the handle with —
            ``"tts-chatterbox-hd"`` or ``"tts-minimax-speech-02-hd"``. When
            omitted, the API applies its default (``tts-chatterbox-hd``); the
            chosen model is returned on :attr:`ClonedVoice.model`.
        :type model: Optional[str]
        :return: The cloned-voice handle and its paired model.
        :rtype: ClonedVoice
        :raises ValueError: If ``model`` is provided and invalid, or the file cannot be read.
        :raises venice_ai.exceptions.APIError: If the API request fails.
        """
        if model is not None:
            validate_model_id(model, "model")

        file_content, filename, content_type = await self._prepare_audio_file(file)

        files_dict: dict[str, Any] = {"file": (filename, file_content, content_type)}
        form_data: dict[str, Any] = {}
        if model is not None:
            form_data["model"] = model

        response = await self._request_multipart(
            method="POST",
            path="audio/voices",
            files=files_dict,
            data=form_data,
        )
        if isinstance(response, dict):
            return ClonedVoice.model_validate(response)
        elif isinstance(response, bytes):
            import json as _json

            return ClonedVoice.model_validate(_json.loads(response))
        else:
            raise TypeError(f"Unexpected response type: {type(response)}")
