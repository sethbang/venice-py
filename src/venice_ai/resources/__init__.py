"""
Venice AI Resources Package.

This package contains all the resource modules that provide access to Venice AI's
comprehensive suite of AI services and capabilities. Each resource module encapsulates
a specific set of functionalities and provides a clean, asynchronous interface for
interacting with the corresponding Venice AI API endpoints.

The resources package is designed to provide intuitive access to Venice AI's full
ecosystem of AI services, from text generation and image creation to audio processing
and account management. All resources support async/await patterns and include
comprehensive error handling, automatic retries, and intelligent rate limiting.

Available Resources:
    ChatCompletions: Advanced conversational AI with support for multiple models,
        streaming responses, function calling, and character-based interactions.

    Responses: Stateless multi-modal generation via the OpenAI-compatible
        ``/responses`` endpoint (Alpha).

    Models: Model discovery and capability analysis including semantic traits,
        compatibility mappings, and detailed model specifications.

    Image: Comprehensive image generation and manipulation with support for
        various styles, resolutions, and advanced prompting techniques.

    Video: Asynchronous video generation with a queue → poll → download →
        cleanup lifecycle, plus video transcription.

    Music: Asynchronous music generation mirroring the video lifecycle.

    VoiceChanger: Asynchronous voice conversion over the same queue lifecycle,
        re-voicing an existing recording rather than generating one. Voice
        changing is a capability on music-type models, not a model type.

    Decisions: Typed judgments from decision ("System One") models, which
        answer a map of questions about a state instead of generating text.

    Characters: Access to pre-configured AI personalities and specialized
        assistants for enhanced conversational experiences.

    ApiKeys: Complete API key lifecycle management including creation, deletion,
        Web3 authentication, and usage monitoring.

    Audio: Audio processing capabilities including text-to-speech generation
        with multiple voice options and audio format support.

    Billing: Usage analytics and billing information retrieval with support
        for multiple export formats and comprehensive filtering options.

    Embeddings: Vector embedding generation for semantic search, similarity
        analysis, and advanced AI applications.

    Augment: Web scraping, web search, and text parsing to ground generations
        in external content.

    Crypto: Multi-chain JSON-RPC proxy with per-request billing headers.

    X402: Wallet-billing — balance, transactions, and top-up (requires the
        ``[x402]`` extra for SIWE/SIWX signing).

    Tee: Confidential-compute attestation and end-to-end-encrypted sessions
        (requires the ``[e2ee]`` extra).

Usage:
    Resources are typically accessed through the main VeniceClient instance:

    .. code-block:: python

        from venice_ai import VeniceClient

        async with VeniceClient() as client:
            # Text generation. Model IDs change; resolve one from the live
            # catalog rather than hardcoding.
            response = await client.chat.completions.create(
                model=await client.models.resolve_chat(),
                messages=[{"role": "user", "content": "Hello!"}]
            )

            # Image generation
            image = await client.image.create(
                prompt="A beautiful sunset over mountains",
                model=await client.models.resolve_image()
            )

            # Model discovery
            models = await client.models.list()

            # Usage history
            usage = await client.billing.get_usage_history()

Architecture:
    All resource classes inherit from APIResource and provide:
    - Asynchronous operation support
    - Automatic request formatting and response parsing
    - Comprehensive error handling and retry logic
    - Type-safe interfaces with full IDE support
    - Consistent parameter validation and documentation

See Also:
    - venice_ai.VeniceClient: Main client interface
    - venice_ai.types: Type definitions for all API operations
    - venice_ai.exceptions: Exception hierarchy for error handling
"""

from .api_keys import ApiKeys
from .audio import Audio
from .augment import Augment
from .billing import Billing
from .characters import Characters
from .chat.completions import ChatCompletions
from .crypto import Crypto
from .decisions import Decisions
from .embeddings import Embeddings
from .image import Image
from .models import Models
from .music import Music
from .responses import Responses
from .tee import Tee
from .video import Video
from .voice_changer import VoiceChanger
from .x402 import X402

__all__ = [
    "ApiKeys",
    "Audio",
    "Augment",
    "Billing",
    "ChatCompletions",
    "Characters",
    "Crypto",
    "Decisions",
    "Embeddings",
    "Image",
    "Models",
    "Music",
    "Responses",
    "Tee",
    "Video",
    "VoiceChanger",
    "X402",
]
