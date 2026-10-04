#!/usr/bin/env python3
"""
Venice AI SDK - Basic Text Embeddings
=====================================

This example demonstrates how to generate text embeddings using the Venice AI SDK.
Learn how to convert text into vector representations for semantic analysis.
"""

import asyncio
import sys

from venice_ai import NoMatchingModelError, VeniceClient, cosine_similarity
from venice_ai.types.api import EmbeddingsResponse

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77


def float_vectors(response: EmbeddingsResponse) -> list[list[float]]:
    """Return the embedding vectors, one per input, in input order.

    ``EmbeddingObject.embedding`` is ``list[float] | str`` because the API can
    also return base64-encoded vectors. This example uses the default float
    format, so anything else is an error rather than something to skip (which
    would silently misalign vectors and texts).
    """
    vectors: list[list[float]] = []
    for item in response.data:
        if not isinstance(item.embedding, list):
            raise TypeError(f"expected a float vector, got {type(item.embedding).__name__}")
        vectors.append(item.embedding)
    return vectors


def preview(text: str, width: int = 50) -> str:
    """Shorten ``text`` to ``width`` characters, marking it only when cut."""
    return text if len(text) <= width else text[: width - 1] + "…"


async def basic_embedding_generation(client: VeniceClient, model: str) -> bool:
    """Generate an embedding for a single text and check it against the catalog.

    The model's catalog entry declares its native vector size
    (``embeddingDimensions``) and per-input limit (``maxInputTokens``).
    ``model_dump()`` keeps these embedding-specific spec fields, so the dumped
    entry is a complete record to log or cache.
    """
    print("🔢 Basic Embedding Generation")
    print("-" * 30)

    entry = await client.models.get(model)
    spec = entry.model_dump()["model_spec"]
    declared_dims = spec.get("embeddingDimensions")
    print(
        f"📚 Catalog: {declared_dims or 'unspecified'} dimensions, "
        f"{spec.get('maxInputTokens') or 'unspecified'} max input tokens per text"
    )

    response = await client.embeddings.create(
        model=model, input="The quick brown fox jumps over the lazy dog."
    )
    (embedding,) = float_vectors(response)

    print("✅ Generated embedding")
    print(f"📏 Embedding dimensions: {len(embedding)}")
    print(f"🔢 First 5 values: {[round(v, 4) for v in embedding[:5]]}")
    print(f"📊 Usage: {response.usage.total_tokens} tokens")

    if not embedding or not any(embedding):
        print("❌ The embedding is empty or all zeros")
        return False
    if declared_dims is None:
        print("ℹ️  The catalog declares no dimension count, so the size was not checked")
    elif len(embedding) != declared_dims:
        print(f"❌ Expected {declared_dims} dimensions (from the catalog), got {len(embedding)}")
        return False
    else:
        print("✅ Vector size matches the catalog's embeddingDimensions")
    return True


async def batch_embedding_generation(client: VeniceClient, model: str) -> bool:
    """Generate embeddings for multiple texts in one request."""
    print("\n📦 Batch Embedding Generation")
    print("-" * 30)

    texts = [
        "I love sunny weather and outdoor activities.",
        "Rainy days are perfect for reading books indoors.",
        "Machine learning is revolutionizing technology.",
        "Artificial intelligence helps solve complex problems.",
        "Pizza is my favorite food for dinner.",
    ]

    response = await client.embeddings.create(model=model, input=texts)
    vectors = float_vectors(response)

    print(f"✅ Generated {len(vectors)} embeddings for {len(texts)} texts")
    print(f"📏 Each embedding has {len(vectors[0])} dimensions")
    print(f"📊 Total tokens used: {response.usage.total_tokens}")

    for i, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
        print(f"\n📝 Text {i + 1}: {preview(text)}")
        print(f"🔢 Embedding preview: {[round(v, 4) for v in vector[:3]]}")

    if len(vectors) != len(texts) or len({len(v) for v in vectors}) != 1:
        print("❌ Expected one equal-length vector per input text")
        return False
    return True


async def semantic_similarity_analysis(client: VeniceClient, model: str) -> bool:
    """Rank every pair of texts by cosine similarity.

    Absolute similarity values depend on the embedding model (some models keep
    even unrelated texts above 0.4), so the pairs are ranked rather than
    labelled with fixed thresholds. The two paraphrase pairs should come out
    on top.
    """
    print("\n🎯 Semantic Similarity Analysis")
    print("-" * 30)

    texts = [
        "The cat sits on the mat.",
        "A feline rests on the carpet.",  # paraphrase of #1
        "Dogs are loyal companions.",
        "Canines make faithful friends.",  # paraphrase of #3
        "I enjoy eating pizza for lunch.",  # unrelated
    ]
    paraphrase_pairs = {(0, 1), (2, 3)}

    response = await client.embeddings.create(model=model, input=texts)
    vectors = float_vectors(response)

    # The SDK ships a pure-Python cosine_similarity helper — no numpy needed.
    pairs = [
        (cosine_similarity(vectors[i], vectors[j]), i, j)
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]
    pairs.sort(reverse=True)

    print("🔍 All pairs, most similar first (cosine similarity):")
    for rank, (similarity, i, j) in enumerate(pairs, 1):
        marker = "  ← paraphrase" if (i, j) in paraphrase_pairs else ""
        print(f"   {rank:2}. {similarity:.3f}  '{texts[i]}' vs '{texts[j]}'{marker}")

    top_two = {(i, j) for _, i, j in pairs[:2]}
    if top_two != paraphrase_pairs:
        print("❌ The paraphrase pairs did not rank as the two most similar pairs")
        return False
    print("\n✅ Both paraphrase pairs rank above every unrelated pair")
    return True


async def embedding_search_example(client: VeniceClient, model: str) -> bool:
    """Demonstrate simple semantic search using embeddings."""
    print("\n🔍 Semantic Search Example")
    print("-" * 30)

    documents = [
        "Python is a popular programming language for data science.",
        "Machine learning algorithms can predict future trends.",
        "The weather today is sunny and warm.",
        "Cooking pasta requires boiling water and salt.",
        "Deep learning models use neural networks with multiple layers.",
        "Exercise is important for maintaining good health.",
        "JavaScript is commonly used for web development.",
    ]
    off_topic = {2, 3, 5}  # weather, cooking, exercise

    query = "artificial intelligence and programming"

    # Embed all documents and the query in one request
    response = await client.embeddings.create(model=model, input=documents + [query])
    vectors = float_vectors(response)
    doc_vectors, query_vector = vectors[:-1], vectors[-1]

    # cosine_similarity returns a float in [-1, 1]; higher means closer
    ranked = sorted(
        ((cosine_similarity(query_vector, doc), i) for i, doc in enumerate(doc_vectors)),
        reverse=True,
    )

    print(f"🔍 Search query: '{query}'")
    print("\n📋 Most relevant documents:")
    for rank, (similarity, doc_idx) in enumerate(ranked[:3], 1):
        print(f"\n{rank}. Similarity: {similarity:.3f}")
        print(f"   📄 {documents[doc_idx]}")

    if any(doc_idx in off_topic for _, doc_idx in ranked[:3]):
        print("\n❌ An off-topic document made the top 3")
        return False
    return True


async def main() -> int:
    """Run all embedding examples.

    Returns ``0`` only if every section passed its checks, ``1`` otherwise,
    and ``77`` (with a ``SKIPPED:`` line) when the catalog has no embedding
    model.
    """
    print("🚀 Venice AI Basic Embeddings Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        # Resolve the embedding model once and share it across sections
        try:
            model = await client.models.resolve_embedding(prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: no embedding model in the catalog ({e})")
            return EXIT_SKIPPED
        print(f"📍 Using embedding model: {model}\n")

        results = [
            ("basic_embedding_generation", await basic_embedding_generation(client, model)),
            ("batch_embedding_generation", await batch_embedding_generation(client, model)),
            ("semantic_similarity_analysis", await semantic_similarity_analysis(client, model)),
            ("embedding_search_example", await embedding_search_example(client, model)),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    print("\n✨ Embedding examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Single and batch embedding generation")
    print("   - Reading a model's embeddingDimensions from the catalog")
    print("   - Ranking text pairs by cosine similarity")
    print("   - Simple semantic search")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
