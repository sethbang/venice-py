#!/usr/bin/env python3
"""
Venice AI SDK - Semantic Similarity Search
==========================================

This example demonstrates how to perform semantic similarity searches using embeddings.
Learn how to build a simple search engine that understands meaning, not just keywords.

The technology corpus is embedded once and reused by every section, the way a
real search index would be; only the queries are embedded per section.
"""

import asyncio
import sys

from venice_ai import NoMatchingModelError, VeniceClient, cosine_similarity
from venice_ai.types.api import EmbeddingsResponse

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77

# Sample document collection for search examples
TECH_ARTICLES = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "JavaScript powers interactive web applications and runs in the browser.",
    "Machine learning algorithms can identify patterns in large datasets.",
    "Deep neural networks are inspired by biological neural networks in the brain.",
    "Cloud computing provides on-demand access to computing resources over the internet.",
    "Blockchain technology enables secure, decentralized transaction records.",
    "Quantum computers use quantum bits (qubits) for exponentially faster calculations.",
    "Cybersecurity protects systems and networks from digital attacks.",
    "Data science combines statistics, programming, and domain knowledge.",
    "Artificial intelligence aims to create machines that can perform human-like tasks.",
]

# Articles about AI/ML, used to check that AI queries find them.
AI_ARTICLES = {2, 3, 9}

# Queries with a known answer, used to calibrate the similarity threshold for
# whichever embedding model is in use. See calibrate_threshold().
CALIBRATION_QUERIES = [
    ("deep learning and artificial neural networks", True),
    ("protecting computer networks from hackers", True),
    ("gardening tips for growing tomatoes", False),
    ("the history of the Roman Empire", False),
]

COOKING_RECIPES = [
    "Spaghetti carbonara is made with eggs, cheese, pancetta, and black pepper.",
    "Chocolate chip cookies require butter, sugar, eggs, flour, and chocolate chips.",
    "A classic Caesar salad includes romaine lettuce, croutons, and parmesan cheese.",
    "Thai green curry combines coconut milk, green curry paste, and vegetables.",
    "Homemade pizza dough needs flour, yeast, water, salt, and olive oil.",
]


def float_vectors(response: EmbeddingsResponse) -> list[list[float]]:
    """Return one float vector per input, raising if the API sent another format."""
    vectors: list[list[float]] = []
    for item in response.data:
        if not isinstance(item.embedding, list):
            raise TypeError(f"expected a float vector, got {type(item.embedding).__name__}")
        vectors.append(item.embedding)
    return vectors


def rank(query_vector: list[float], doc_vectors: list[list[float]]) -> list[tuple[float, int]]:
    """Return ``(similarity, doc_index)`` pairs, most similar first."""
    scored = [(cosine_similarity(query_vector, doc), i) for i, doc in enumerate(doc_vectors)]
    return sorted(scored, reverse=True)


def shorten(text: str, width: int = 60) -> str:
    """Shorten ``text`` to ``width`` characters, marking it only when cut."""
    return text if len(text) <= width else text[: width - 1] + "…"


async def embed(client: VeniceClient, model: str, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in one request."""
    return float_vectors(await client.embeddings.create(model=model, input=texts))


async def basic_similarity_search(
    client: VeniceClient, model: str, doc_vectors: list[list[float]]
) -> bool:
    """Demonstrate basic semantic search across documents."""
    print("🔍 Basic Similarity Search")
    print("-" * 40)

    query = "artificial intelligence and neural networks"
    print(f"🔎 Search query: '{query}'")
    (query_vector,) = await embed(client, model, [query])

    ranked = rank(query_vector, doc_vectors)

    print("\n📊 Top 3 most relevant articles:")
    for position, (similarity, doc_idx) in enumerate(ranked[:3], 1):
        print(f"\n{position}. Similarity: {similarity:.4f}")
        print(f"   📄 {TECH_ARTICLES[doc_idx]}")

    top_three = {doc_idx for _, doc_idx in ranked[:3]}
    if top_three != AI_ARTICLES:
        print("\n❌ The top 3 should be the three AI/ML articles")
        return False
    return True


async def multi_query_search(
    client: VeniceClient, model: str, doc_vectors: list[list[float]]
) -> bool:
    """Embed several queries in one request and find each one's best match."""
    print("\n🔎 Multi-Query Search")
    print("-" * 40)

    # Each query paired with the articles that answer it. "machine learning and
    # AI" names two topics, so either the ML or the AI article is a correct hit.
    queries = {
        "programming languages": {0},
        "machine learning and AI": {2, 9},
        "web development": {1},
    }

    query_vectors = await embed(client, model, list(queries))

    ok = True
    for (query, expected), query_vector in zip(queries.items(), query_vectors, strict=True):
        similarity, doc_idx = rank(query_vector, doc_vectors)[0]
        print(f"\n🔍 Query: '{query}'")
        print(f"   Best match ({similarity:.4f}): {shorten(TECH_ARTICLES[doc_idx])}")
        if doc_idx not in expected:
            for idx in sorted(expected):
                print(f"   ❌ Expected: {shorten(TECH_ARTICLES[idx])}")
            ok = False
    return ok


def best_score(query_vector: list[float], doc_vectors: list[list[float]]) -> float:
    """The similarity of the closest document to the query."""
    return rank(query_vector, doc_vectors)[0][0]


async def calibrate_threshold(
    client: VeniceClient, model: str, doc_vectors: list[list[float]]
) -> float | None:
    """Pick a similarity threshold for this model from queries with known answers.

    Score ranges differ between embedding models (some keep unrelated text well
    above zero), so a fixed threshold only fits the model it was tuned on. This
    places the threshold midway between the weakest on-topic best score and the
    strongest off-topic best score. Returns ``None`` if the model does not
    separate the two groups at all.
    """
    query_vectors = await embed(client, model, [q for q, _ in CALIBRATION_QUERIES])
    on_topic: list[float] = []
    off_topic: list[float] = []
    for (query, relevant), vector in zip(CALIBRATION_QUERIES, query_vectors, strict=True):
        score = best_score(vector, doc_vectors)
        (on_topic if relevant else off_topic).append(score)
        print(f"   calibration: {score:.4f} {'on' if relevant else 'off'}-topic  {query!r}")

    low_on, high_off = min(on_topic), max(off_topic)
    if low_on <= high_off:
        print(
            f"   ❌ On-topic scores (min {low_on:.4f}) do not clear off-topic (max {high_off:.4f})"
        )
        return None
    return (low_on + high_off) / 2


async def threshold_filtering(
    client: VeniceClient, model: str, doc_vectors: list[list[float]]
) -> bool:
    """Drop weak matches with a similarity threshold.

    Ranking always returns *something*, even for a query the corpus cannot
    answer. A threshold turns "nothing relevant" into an empty result. The
    threshold is calibrated for the current model first (see
    :func:`calibrate_threshold`), then applied to new queries it has not seen.
    """
    print("\n🎯 Threshold-Based Filtering")
    print("-" * 40)

    threshold = await calibrate_threshold(client, model, doc_vectors)
    if threshold is None:
        return False
    print(f"📏 Threshold for {model}: {threshold:.4f}")

    queries = [
        ("neural networks and machine learning", True),
        ("cooking recipes", False),  # the corpus is all technology
    ]
    query_vectors = await embed(client, model, [q for q, _ in queries])

    ok = True
    for (query, on_topic), query_vector in zip(queries, query_vectors, strict=True):
        ranked = rank(query_vector, doc_vectors)
        kept = [(sim, idx) for sim, idx in ranked if sim >= threshold]

        label = "on-topic" if on_topic else "off-topic"
        print(f"\n🔍 Query: '{query}' ({label}, best score {ranked[0][0]:.4f})")
        if kept:
            print(f"   Kept {len(kept)} of {len(ranked)} documents:")
            for sim, idx in kept:
                print(f"   - ({sim:.4f}) {shorten(TECH_ARTICLES[idx], 50)}")
        else:
            print(f"   No document reaches {threshold:.4f}: nothing relevant")

        if on_topic and not kept:
            print("   ❌ The on-topic query should keep at least one document")
            ok = False
        if not on_topic and kept:
            print("   ❌ The off-topic query should keep nothing")
            ok = False
    return ok


async def cross_domain_search(
    client: VeniceClient, model: str, tech_vectors: list[list[float]]
) -> bool:
    """Search one index that mixes two unrelated document collections.

    The technology vectors are already embedded, so only the recipes are new.
    """
    print("\n🌐 Cross-Domain Search")
    print("-" * 40)

    all_docs = TECH_ARTICLES + COOKING_RECIPES
    doc_types = ["tech"] * len(TECH_ARTICLES) + ["cooking"] * len(COOKING_RECIPES)

    print("📚 Document collection:")
    print(f"   Tech articles: {len(TECH_ARTICLES)}")
    print(f"   Cooking recipes: {len(COOKING_RECIPES)}")

    doc_vectors = tech_vectors + await embed(client, model, COOKING_RECIPES)

    queries = {"how to make pasta": "cooking", "artificial intelligence": "tech"}
    query_vectors = await embed(client, model, list(queries))

    ok = True
    for (query, expected_type), query_vector in zip(queries.items(), query_vectors, strict=True):
        print(f"\n🔍 Query: '{query}'")
        print("   Top 3 results:")
        top = rank(query_vector, doc_vectors)[:3]
        for position, (similarity, doc_idx) in enumerate(top, 1):
            doc_type = doc_types[doc_idx]
            emoji = "👨‍💻" if doc_type == "tech" else "👨‍🍳"
            print(f"   {position}. {emoji} [{doc_type}] ({similarity:.4f})")
            print(f"      {shorten(all_docs[doc_idx])}")
        if any(doc_types[idx] != expected_type for _, idx in top):
            print(f"   ❌ Expected all top results to be {expected_type} documents")
            ok = False
    return ok


async def ranked_search_with_metadata(
    client: VeniceClient, model: str, doc_vectors: list[list[float]]
) -> bool:
    """Attach metadata to ranked results."""
    print("\n📊 Ranked Search with Metadata")
    print("-" * 40)

    doc_metadata = [
        {"id": 1, "category": "language", "difficulty": "beginner"},
        {"id": 2, "category": "language", "difficulty": "beginner"},
        {"id": 3, "category": "ml", "difficulty": "intermediate"},
        {"id": 4, "category": "ml", "difficulty": "advanced"},
        {"id": 5, "category": "infrastructure", "difficulty": "intermediate"},
        {"id": 6, "category": "infrastructure", "difficulty": "advanced"},
        {"id": 7, "category": "hardware", "difficulty": "advanced"},
        {"id": 8, "category": "security", "difficulty": "intermediate"},
        {"id": 9, "category": "data", "difficulty": "intermediate"},
        {"id": 10, "category": "ml", "difficulty": "intermediate"},
    ]

    query = "learn about AI"
    print(f"🔍 Query: '{query}'")
    (query_vector,) = await embed(client, model, [query])

    ranked = rank(query_vector, doc_vectors)

    print("\n📋 Top 5 results with metadata:")
    for position, (similarity, doc_idx) in enumerate(ranked[:5], 1):
        meta = doc_metadata[doc_idx]
        print(f"\n{position}. Similarity: {similarity:.4f}")
        print(f"   📄 {shorten(TECH_ARTICLES[doc_idx])}")
        print(f"   🏷️  Category: {meta['category']} | Difficulty: {meta['difficulty']}")

    top_category = doc_metadata[ranked[0][1]]["category"]
    if top_category != "ml":
        print(f"\n❌ Expected the top result to be in the ml category, got {top_category}")
        return False
    return True


async def find_similar_documents(doc_vectors: list[list[float]]) -> bool:
    """Find the documents closest to a given document (no query needed)."""
    print("\n📄 Document-to-Document Similarity")
    print("-" * 40)

    reference_idx = 2  # Machine learning article
    print("📌 Reference document:")
    print(f"   {TECH_ARTICLES[reference_idx]}")

    neighbours = [
        (sim, idx)
        for sim, idx in rank(doc_vectors[reference_idx], doc_vectors)
        if idx != reference_idx
    ]

    print("\n🔍 Most similar documents:")
    for position, (similarity, doc_idx) in enumerate(neighbours[:3], 1):
        print(f"\n{position}. Similarity: {similarity:.4f}")
        print(f"   📄 {TECH_ARTICLES[doc_idx]}")

    if neighbours[0][1] not in AI_ARTICLES | {8}:  # AI articles or data science
        print("\n❌ The nearest neighbour should be an AI or data-science article")
        return False
    return True


async def main() -> int:
    """Run all similarity search examples.

    Returns ``0`` only if every section passed its checks, ``1`` otherwise,
    and ``77`` (with a ``SKIPPED:`` line) when the catalog has no embedding
    model.
    """
    print("🚀 Venice AI Semantic Similarity Search Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_embedding(prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: no embedding model in the catalog ({e})")
            return EXIT_SKIPPED
        print(f"📍 Using embedding model: {model}")

        print(f"📚 Embedding {len(TECH_ARTICLES)} technology articles once for all sections\n")
        doc_vectors = await embed(client, model, TECH_ARTICLES)

        results = [
            ("basic_similarity_search", await basic_similarity_search(client, model, doc_vectors)),
            ("multi_query_search", await multi_query_search(client, model, doc_vectors)),
            ("threshold_filtering", await threshold_filtering(client, model, doc_vectors)),
            ("cross_domain_search", await cross_domain_search(client, model, doc_vectors)),
            (
                "ranked_search_with_metadata",
                await ranked_search_with_metadata(client, model, doc_vectors),
            ),
            ("find_similar_documents", await find_similar_documents(doc_vectors)),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    print("\n✨ Similarity search examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Embedding a corpus once and reusing the vectors")
    print("   - Multi-query batch embedding")
    print("   - Calibrating a similarity threshold for the model in use")
    print("   - Cross-domain document search")
    print("   - Ranked results with metadata")
    print("   - Document-to-document similarity")
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
