#!/usr/bin/env python3
"""
Venice AI SDK - Batch Embedding Processing
==========================================

This example demonstrates efficient batch processing of text embeddings.
Learn how to handle large volumes of text efficiently using batching strategies.
"""

import asyncio
import sys
import time

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import InvalidRequestError
from venice_ai.types.api import EmbeddingsResponse

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77


def valid_vectors(responses: list[EmbeddingsResponse], expected_count: int) -> bool:
    """Check the responses hold ``expected_count`` float vectors of one length.

    A count alone would also pass for empty vectors or base64 strings.
    """
    vectors = [item.embedding for response in responses for item in response.data]
    if len(vectors) != expected_count:
        print(f"   ❌ Expected {expected_count} embeddings, got {len(vectors)}")
        return False
    lengths: set[int] = set()
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            print(f"   ❌ Expected a non-empty float vector, got {type(vector).__name__}")
            return False
        if not all(isinstance(x, float) for x in vector):
            print("   ❌ An embedding contains non-float values")
            return False
        lengths.add(len(vector))
    if len(lengths) > 1:
        print(f"   ❌ Vectors have different lengths: {sorted(lengths)}")
        return False
    print(f"   ✓ {len(vectors)} float vectors of {lengths.pop()} dimensions")
    return True


async def simple_batch_embedding(client: VeniceClient, model: str) -> bool:
    """Embed several texts in a single request."""
    print("📦 Simple Batch Embedding")
    print("-" * 40)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Python is a versatile programming language.",
        "Climate change affects global weather patterns.",
        "Quantum computing promises exponential speedups.",
    ]

    print(f"📝 Processing {len(texts)} texts in a single batch...")
    start_time = time.perf_counter()
    response = await client.embeddings.create(model=model, input=texts)
    elapsed = time.perf_counter() - start_time

    print(f"\n✅ Generated {len(response.data)} embeddings")
    print(f"⏱️  Time taken: {elapsed:.3f} seconds")
    print(f"📊 Total tokens: {response.usage.total_tokens}")

    print("\n📏 Embedding details:")
    for i, embedding_data in enumerate(response.data[:3], 1):
        print(f"   Text {i}: {len(embedding_data.embedding)} dimensions")

    return valid_vectors([response], len(texts))


async def chunked_batch_processing(client: VeniceClient, model: str) -> bool:
    """Process a larger dataset in fixed-size chunks."""
    print("\n🔄 Chunked Batch Processing")
    print("-" * 40)

    base_texts = [
        "Artificial intelligence and machine learning",
        "Web development with modern frameworks",
        "Database design and optimization",
        "Cloud computing infrastructure",
        "Cybersecurity best practices",
    ]
    all_texts = [f"{text} - variation {i + 1}" for i in range(5) for text in base_texts]
    print(f"📚 Total texts to process: {len(all_texts)}")

    chunk_size = 10
    total_chunks = (len(all_texts) + chunk_size - 1) // chunk_size
    all_embeddings = []
    responses: list[EmbeddingsResponse] = []
    total_tokens = 0

    start_time = time.perf_counter()
    for i in range(0, len(all_texts), chunk_size):
        chunk = all_texts[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"\n📦 Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} texts)...")

        response = await client.embeddings.create(model=model, input=chunk)
        responses.append(response)
        all_embeddings.extend(item.embedding for item in response.data)
        total_tokens += response.usage.total_tokens
        print(f"   ✓ Chunk {chunk_num} complete ({response.usage.total_tokens} tokens)")
    elapsed = time.perf_counter() - start_time

    print("\n📊 Statistics:")
    print(f"   Total embeddings: {len(all_embeddings)}")
    print(f"   Total tokens: {total_tokens}")
    print(f"   Time taken: {elapsed:.3f} seconds")
    print(f"   Average time per chunk: {elapsed / total_chunks:.3f} seconds")

    return valid_vectors(responses, len(all_texts))


async def concurrent_batch_processing(client: VeniceClient, model: str) -> bool:
    """Send several independent batches at the same time."""
    print("\n⚡ Concurrent Batch Processing")
    print("-" * 40)

    batches = [
        ["Python programming tutorial", "JavaScript web development", "Java enterprise apps"],
        ["React framework basics", "Vue.js components", "Angular modules"],
        ["Machine learning algorithms", "Deep learning networks", "Neural network design"],
    ]
    print(f"🔀 Processing {len(batches)} batches of {len(batches[0])} texts concurrently...")

    start_time = time.perf_counter()
    # gather() raises the first failure, so a failed batch fails this section
    responses = await asyncio.gather(
        *(client.embeddings.create(model=model, input=batch) for batch in batches)
    )
    elapsed = time.perf_counter() - start_time

    total_embeddings = sum(len(r.data) for r in responses)
    total_tokens = sum(r.usage.total_tokens for r in responses)

    print(f"\n⏱️  Total time: {elapsed:.3f} seconds for all {len(batches)} batches")
    print(f"📊 Total embeddings: {total_embeddings}")
    print(f"📊 Total tokens: {total_tokens}")

    return valid_vectors(list(responses), sum(len(b) for b in batches))


async def batch_with_error_handling(
    client: VeniceClient, model: str, max_input_tokens: int | None
) -> bool:
    """Keep going when individual batches are rejected.

    Two batches are invalid on purpose: an empty list, which the SDK rejects
    before sending, and a text longer than the model's input limit, which the
    API rejects. Both raise :class:`InvalidRequestError`, whose message carries
    the server's reason. The oversized text is sized from the model's catalog
    ``maxInputTokens``, so it is over the limit for whichever model is in use.
    The section passes only if exactly the invalid batches fail and every valid
    batch succeeds.
    """
    print("\n🛡️ Batch Processing with Error Handling")
    print("-" * 40)

    batches: list[tuple[list[str], bool]] = [
        (["Valid text 1", "Valid text 2", "Valid text 3"], True),
        (["Another valid text", "More valid content"], True),
        ([], False),  # empty input
    ]
    if max_input_tokens:
        # Each repetition is at least two tokens, so this is about twice the limit.
        oversized_text = "lorem ipsum " * max_input_tokens
        print(f"📏 Model input limit: {max_input_tokens} tokens per text")
        batches.append(([oversized_text], False))
    else:
        print("ℹ️  The catalog lists no input limit for this model; skipping the oversized batch")
    batches.append((["Final batch text 1", "Final batch text 2"], True))

    print(f"📦 Processing {len(batches)} batches ({sum(not v for _, v in batches)} invalid)...")

    all_embeddings = []
    succeeded = rejected = 0
    unexpected: list[int] = []

    for i, (batch, is_valid) in enumerate(batches, 1):
        print(f"\n   Batch {i}/{len(batches)}: {len(batch)} texts")
        try:
            response = await client.embeddings.create(model=model, input=batch)
        except InvalidRequestError as e:
            rejected += 1
            print(f"   🚫 Rejected: {e}")
            if is_valid:
                print("   ❌ This batch was valid and should have succeeded")
                unexpected.append(i)
            else:
                print("   ↪ Expected rejection; skipping this batch and continuing")
            continue

        succeeded += 1
        all_embeddings.extend(item.embedding for item in response.data)
        if not is_valid:
            print(f"   ❌ Accepted ({len(response.data)} embeddings), but this batch was")
            print("      invalid and should have been rejected")
            unexpected.append(i)
            continue
        print(f"   ✅ Success ({len(response.data)} embeddings)")
        if not valid_vectors([response], len(batch)):
            unexpected.append(i)

    print("\n📊 Results:")
    print(f"   Successful batches: {succeeded}")
    print(f"   Rejected batches: {rejected}")
    print(f"   Total embeddings generated: {len(all_embeddings)}")

    if unexpected:
        print(f"   ❌ Unexpected outcome for batch(es): {unexpected}")
        return False
    return True


async def optimized_large_dataset(client: VeniceClient, model: str) -> bool:
    """Process a larger dataset in chunks, a few chunks at a time."""
    print("\n🚀 Optimized Large Dataset Processing")
    print("-" * 40)

    large_dataset = [
        f"Document {i}: This is sample text for document number {i} in our collection."
        for i in range(50)
    ]
    chunk_size = 20
    max_concurrent = 3

    print(f"📚 Dataset size: {len(large_dataset)} documents")
    print(f"⚙️  Chunk size: {chunk_size}, max concurrent batches: {max_concurrent}")

    chunks = [large_dataset[i : i + chunk_size] for i in range(0, len(large_dataset), chunk_size)]
    total_groups = (len(chunks) + max_concurrent - 1) // max_concurrent

    all_embeddings = []
    all_responses: list[EmbeddingsResponse] = []
    total_tokens = 0
    start_time = time.perf_counter()

    for group_start in range(0, len(chunks), max_concurrent):
        group_chunks = chunks[group_start : group_start + max_concurrent]
        group_num = group_start // max_concurrent + 1
        print(f"\n🔄 Processing group {group_num}/{total_groups} ({len(group_chunks)} batches)...")

        responses = await asyncio.gather(
            *(client.embeddings.create(model=model, input=chunk) for chunk in group_chunks)
        )
        for response in responses:
            all_responses.append(response)
            all_embeddings.extend(item.embedding for item in response.data)
            total_tokens += response.usage.total_tokens
        print(f"   ✓ Group {group_num} complete")

    elapsed = time.perf_counter() - start_time

    print("\n📊 Final statistics:")
    print(f"   Total documents: {len(large_dataset)}")
    print(f"   Total embeddings: {len(all_embeddings)}")
    print(f"   Total tokens: {total_tokens}")
    print(f"   Time taken: {elapsed:.3f} seconds")
    print(f"   Average tokens per doc: {total_tokens / len(large_dataset):.1f}")

    return valid_vectors(all_responses, len(large_dataset))


async def batch_deduplication(client: VeniceClient, model: str) -> bool:
    """Skip duplicate texts before embedding them."""
    print("\n🔍 Batch Processing with Deduplication")
    print("-" * 40)

    texts = [
        "Machine learning fundamentals",
        "Python programming basics",
        "Machine learning fundamentals",  # Duplicate
        "Web development with React",
        "Python programming basics",  # Duplicate
        "Database optimization techniques",
        "Cloud computing platforms",
    ]
    print(f"📚 Original dataset: {len(texts)} texts")

    unique_texts = list(dict.fromkeys(texts))  # Preserves order
    duplicates_removed = len(texts) - len(unique_texts)
    print(f"🔍 After deduplication: {len(unique_texts)} unique texts")
    print(f"   Removed {duplicates_removed} duplicates")

    response = await client.embeddings.create(model=model, input=unique_texts)
    tokens_used = response.usage.total_tokens
    tokens_per_text = tokens_used / len(unique_texts)

    # Map every original text to its vector, duplicates included
    by_text = dict(zip(unique_texts, (item.embedding for item in response.data), strict=True))
    embeddings = [by_text[text] for text in texts]

    print(f"\n📊 Embeddings generated: {len(response.data)} (covering all {len(embeddings)} texts)")
    print(f"📊 Tokens used: {tokens_used} ({tokens_per_text:.1f} per text on average)")
    print(f"💰 Tokens saved by deduplication: ~{duplicates_removed * tokens_per_text:.0f}")

    return valid_vectors([response], len(unique_texts))


async def main() -> int:
    """Run all batch processing examples.

    Returns ``0`` only if every section succeeded, ``1`` otherwise,
    and ``77`` (with a ``SKIPPED:`` line) when the catalog has no embedding
    model.
    """
    print("🚀 Venice AI Batch Embedding Processing Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_embedding(prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: no embedding model in the catalog ({e})")
            return EXIT_SKIPPED
        spec = (await client.models.get(model)).model_spec
        max_input_tokens = getattr(spec, "maxInputTokens", None)
        print(f"📍 Using embedding model: {model}\n")

        results = [
            ("simple_batch_embedding", await simple_batch_embedding(client, model)),
            ("chunked_batch_processing", await chunked_batch_processing(client, model)),
            ("concurrent_batch_processing", await concurrent_batch_processing(client, model)),
            (
                "batch_with_error_handling",
                await batch_with_error_handling(client, model, max_input_tokens),
            ),
            ("optimized_large_dataset", await optimized_large_dataset(client, model)),
            ("batch_deduplication", await batch_deduplication(client, model)),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    print("\n✨ Batch processing examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Simple batch embedding")
    print("   - Chunked processing for large datasets")
    print("   - Concurrent batch processing")
    print("   - Error handling in batches")
    print("   - Optimized large-scale processing")
    print("   - Deduplication for efficiency")
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
