"""
Venice AI Embeddings API Resources.

This module provides asynchronous client interfaces for Venice AI's Embeddings API,
enabling developers to generate high-quality vector embeddings from text inputs.
Embeddings are dense vector representations that capture the semantic meaning of text,
making them essential for modern natural language processing applications.

Key Features:
    - **Text Embedding Generation**: Convert text into dense vector representations
    - **Batch Processing**: Generate embeddings for multiple texts in a single API call
    - **Flexible Text Input**: Support for a single string or a batch of strings
    - **Configurable Dimensions**: Adjust embedding dimensions for performance optimization
    - **Format Control**: Choose between float arrays and base64-encoded representations
    - **Asynchronous Operations**: Full async/await support for scalable applications

Common Use Cases:
    - **Semantic Search**: Find documents similar in meaning to a query
    - **Text Classification**: Group texts by semantic similarity
    - **Clustering Analysis**: Discover hidden patterns in text collections
    - **Recommendation Systems**: Suggest content based on semantic relationships
    - **Similarity Scoring**: Measure semantic distance between texts
    - **Content Deduplication**: Identify duplicate or near-duplicate content

Vector embeddings enable sophisticated text analysis by transforming human language
into mathematical representations that preserve semantic relationships. Texts with
similar meanings will have similar embedding vectors, allowing for computational
comparison and analysis of semantic content.

Example:
    .. code-block:: python

        import asyncio
        from venice_ai import VeniceClient

        async def analyze_text_similarity():
            async with VeniceClient() as client:
                # Resolve an embedding model ID at runtime (do not hardcode)
                model = await client.models.resolve_embedding()
                # Generate embeddings for comparison
                response = await client.embeddings.create(
                    model=model,
                    input=[
                        "The weather is beautiful today",
                        "It's a lovely sunny day",
                        "I need to buy groceries"
                    ]
                )

                # Extract embeddings for similarity analysis
                embeddings = [item.embedding for item in response.data]

                # The first two texts should have higher similarity
                # than either compared to the third text
                import numpy as np
                similarity = np.dot(embeddings[0], embeddings[1])
                print(f"Similarity between weather texts: {similarity}")

        asyncio.run(analyze_text_similarity())

Note:
    All operations in this module are asynchronous and require proper async/await
    handling. The Embeddings class is accessed through the :attr:`VeniceClient.embeddings`
    property and provides optimized batch processing for multiple text inputs.
"""

from typing import (
    TYPE_CHECKING,
    Literal,
)

from .._resource import APIResource
from ..exceptions import InvalidRequestError
from ..types.api import EmbeddingsRequest, EmbeddingsResponse

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401


class Embeddings(APIResource["VeniceClient"]):
    """
    Provides access to text embedding generation operations (asynchronous).

    This class manages asynchronous embedding operations through the Venice AI API.
    Embeddings are vector representations of text that capture semantic meaning
    and can be used for various natural language processing tasks such as semantic
    search, clustering, classification, and similarity analysis.

    :param client: The Venice AI client instance used to make API requests.
    :type client: venice_ai._client.VeniceClient
    """

    async def create(
        self,
        *,
        model: str,
        input: str | list[str],
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        user: str | None = None,
    ) -> EmbeddingsResponse:
        """
        Generates embeddings for input text(s) asynchronously.

        This method sends an asynchronous request to the Venice AI API to generate
        vector embeddings for the provided text or token inputs using the specified
        model. The embeddings can be used for semantic search, clustering,
        classification, and other NLP tasks.

        :param model: The ID of the embedding model to use. Resolve a valid ID at
            runtime via ``client.models.resolve_embedding()`` rather than hardcoding
            one, since available models change over time.
        :type model: str
        :param input: The input text(s) to generate embeddings for. Can be a single
            string or a list of strings for batch processing. Token-ID arrays are
            not accepted — the API validates embeddings input as text only.
            For batch processing, all inputs will be
            processed together in a single API call.
        :type input: Union[str, List[str]]
        :param dimensions: The number of dimensions for the output embeddings.
            If not specified, uses the model's default dimensionality. Some models
            support reducing dimensions for efficiency.
        :type dimensions: Optional[int]
        :param encoding_format: The format for the returned embeddings. Defaults
            to ``'float'`` for numerical arrays. Use ``'base64'`` for base64-encoded
            string representation.
        :type encoding_format: Optional[Literal["float", "base64"]]
        :param user: A unique identifier representing your end-user. This parameter
            is supported for compatibility with OpenAI clients but is discarded by
            the Venice API and does not affect the response.


        :return: A response object containing the generated embeddings and usage data.
            The response includes an array of embedding objects, each containing the
            vector representation and associated metadata.


        :raises venice_ai.exceptions.InvalidRequestError: If parameter values are invalid
            (e.g., empty model or input, unsupported encoding format).
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid
            or missing.
        :raises venice_ai.exceptions.PermissionDeniedError: If access to the specified
            model is denied.
        :raises venice_ai.exceptions.NotFoundError: If the specified model is not found.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.

        **Examples:**

        Generate an embedding for a single string:

        .. code-block:: python

            import asyncio
            from venice_ai import VeniceClient

            async def create_embedding():
                async with VeniceClient(api_key="your-api-key") as client:
                    model = await client.models.resolve_embedding()
                    response = await client.embeddings.create(
                        model=model,
                        input="The quick brown fox jumps over the lazy dog."
                    )
                    embedding = response.data[0].embedding
                    print(f"Embedding dimensions: {len(embedding)}")
                    print(f"First 5 dimensions: {embedding[:5]}")

            asyncio.run(create_embedding())

        Generate embeddings for multiple strings (batch processing):

        .. code-block:: python

            async def create_batch_embeddings():
                inputs = [
                    "First sentence for embedding.",
                    "Second sentence for embedding.",
                    "Third sentence for embedding."
                ]
                async with VeniceClient(api_key="your-api-key") as client:
                    model = await client.models.resolve_embedding()
                    batch_response = await client.embeddings.create(
                        model=model,
                        input=inputs
                    )
                    for i, data_item in enumerate(batch_response.data):
                        print(f"Embedding for '{inputs[i]}' (first 3 dims): {data_item.embedding[:3]}")
                    print(f"Total tokens used: {batch_response.usage.total_tokens}")

            asyncio.run(create_batch_embeddings())

        Using optional parameters:

        .. code-block:: python

            async def create_custom_embedding():
                async with VeniceClient(api_key="your-api-key") as client:
                    model = await client.models.resolve_embedding()
                    response = await client.embeddings.create(
                        model=model,
                        input="Sample text for embedding",
                        dimensions=512,  # Reduce dimensions if supported
                        encoding_format="base64",  # Get base64-encoded embeddings
                        user="user-123"  # Track usage by user
                    )

            asyncio.run(create_custom_embedding())

        """
        # Validate input parameters
        if not model:
            raise InvalidRequestError(
                "model parameter is required and cannot be empty.",
                request=None,
                response=None,
                body=None,
            )
        if not input:  # Handles empty string and empty list
            raise InvalidRequestError(
                "input cannot be empty.", request=None, response=None, body=None
            )

        if isinstance(input, list) and any(not isinstance(item, str) for item in input):
            raise InvalidRequestError(
                "input must be text. Token-ID arrays are not accepted — "
                "pass the source text instead.",
                request=None,
                response=None,
                body=None,
            )

        # Validate array length constraint
        if isinstance(input, list) and len(input) > 2048:
            raise InvalidRequestError(
                f"input array must have 2048 or fewer items, but got {len(input)} items.",
                request=None,
                response=None,
                body=None,
            )

        # Create Pydantic request model
        embeddings_request = EmbeddingsRequest(
            model=model,
            input=input,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        )

        # Convert to dictionary, excluding None values
        body = embeddings_request.model_dump(exclude_none=True)

        # Make the API request and return the response
        result = await self._client.post("embeddings", json_data=body, cast_to=EmbeddingsResponse)
        return result
