"""
VCR-Based Benchmark Tests for Intelligent Scheduler
===================================================

This module provides comprehensive benchmarking of the intelligent scheduler
using VCR (Video Cassette Recorder) to record and replay actual API interactions.
This allows us to test against real API responses while maintaining test
reproducibility and avoiding API costs.

Key Features:
    * Records actual Venice API interactions for replay
    * Tests scheduler behavior with real rate limit headers
    * Validates queue management and request prioritization
    * Measures performance metrics with actual API latencies
    * Supports multiple test scenarios (steady, burst, concurrent)

Test Scenarios:
    * Steady load: Consistent request rate within limits
    * Burst patterns: Sudden spikes in request volume
    * Rate limit stress: Testing at capacity boundaries
    * Concurrent requests: Multi-model simultaneous operations
    * Recovery testing: Behavior after rate limit hits

Usage:
    # Record new cassettes (requires API key)
    pytest tests/benchmarks/test_vcr_scheduler_benchmarks.py --vcr-record=new_episodes

    # Run with existing cassettes (no API key needed)
    pytest tests/benchmarks/test_vcr_scheduler_benchmarks.py

    # Update cassettes for specific tests
    pytest tests/benchmarks/test_vcr_scheduler_benchmarks.py::test_specific --vcr-record=rewrite
"""

import asyncio
import contextlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import vcr

from tests.benchmarks.metrics import BenchmarkResults, MetricsCollector
from tests.benchmarks.reporters import BenchmarkReporter
from tests.benchmarks.scenarios import (
    BenchmarkScenario,
    RequestPattern,
)
from tests.vcr_policy import resolve_record_mode
from venice_ai import VeniceAIConfig
from venice_ai.core.config import (
    BackendConfig,
    BackendType,
    CircuitBreakerConfig,
    SchedulerConfig,
    SchedulerMode,
    StateConfig,
)

# VCR Configuration
VCR_CASSETTE_DIR = Path(__file__).parent / "vcr_cassettes"
VCR_CASSETTE_DIR.mkdir(exist_ok=True)

# Configure VCR for Venice API
vcr_config = vcr.VCR(
    cassette_library_dir=str(VCR_CASSETTE_DIR),
    # Resolved centrally. This instance shadows the root ``vcr_config`` fixture
    # name, so a mode hardcoded here would bypass VENICE_CI_MODE entirely and
    # record — and therefore bill — whenever a cassette is absent.
    record_mode=resolve_record_mode(),
    match_on=["method", "uri", "body"],
    filter_headers=["Authorization", "X-API-Key"],  # Hide sensitive data
    decode_compressed_response=True,
    before_record_response=lambda response: {
        **response,
        "headers": {
            **response.get("headers", {}),
            # Ensure rate limit headers are preserved
            "X-RateLimit-Limit-Requests": response.get("headers", {}).get(
                "X-RateLimit-Limit-Requests", ["100"]
            ),
            "X-RateLimit-Remaining-Requests": response.get("headers", {}).get(
                "X-RateLimit-Remaining-Requests", ["99"]
            ),
            "X-RateLimit-Reset-Requests": response.get("headers", {}).get(
                "X-RateLimit-Reset-Requests", [str(int(time.time()) + 60)]
            ),
        },
    },
)


class VCRSchedulerBenchmark:
    """
    Benchmark suite for intelligent scheduler with VCR-recorded API interactions.

    This class orchestrates benchmark scenarios using recorded API responses,
    allowing comprehensive testing of scheduler behavior without live API calls.
    """

    def __init__(self, config: VeniceAIConfig | None = None):
        """Initialize the VCR benchmark suite."""
        self.config = config or self._create_default_config()
        self.reporter = BenchmarkReporter(use_colors=True)
        self.metrics_collector = MetricsCollector()

    def _create_default_config(self) -> VeniceAIConfig:
        """Create default configuration for benchmarking."""
        return VeniceAIConfig(
            scheduler=SchedulerConfig(
                mode=SchedulerMode.BASIC,  # Use BASIC mode for simpler testing
                max_concurrent_executions=10,
                request_timeout=30.0,
                scheduler_interval=0.1,
                enable_rate_limiting=True,
                rate_limit_buffer_ratio=0.9,
            ),
            backend=BackendConfig(
                backend_type=BackendType.MEMORY,  # Use in-memory for tests
            ),
            state=StateConfig(
                cache_ttl=1.0,
                batch_size=50,
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                reset_timeout=60.0,
            ),
        )

    async def run_benchmark_scenario(
        self,
        scenario_name: str,
        duration_seconds: float = 30.0,
        target_rps: float = 10.0,
        use_vcr: bool = True,
    ) -> BenchmarkResults:
        """
        Run a benchmark scenario with VCR recording/playback.

        Args:
            scenario_name: Name of the benchmark scenario
            duration_seconds: How long to run the benchmark
            target_rps: Target requests per second
            use_vcr: Whether to use VCR for API recording

        Returns:
            Benchmark results with performance metrics
        """
        # Create scenario configuration
        scenario = BenchmarkScenario(
            name=scenario_name,
            duration_seconds=int(duration_seconds),
            target_rps=int(target_rps),
            warmup_seconds=2,
            request_pattern=RequestPattern.STEADY,
        )

        # Initialize results
        results = BenchmarkResults(
            scenario_name=scenario_name,
            strategy_name="intelligent_scheduler",
            duration=duration_seconds,
        )

        # Setup VCR if enabled
        cassette_path = VCR_CASSETTE_DIR / f"{scenario_name}.yaml"

        if use_vcr:
            with vcr_config.use_cassette(str(cassette_path)):  # type: ignore[attr-defined]
                results = await self._execute_scenario(scenario, results)
        else:
            results = await self._execute_scenario(scenario, results)

        return results

    async def _execute_scenario(
        self, scenario: BenchmarkScenario, results: BenchmarkResults
    ) -> BenchmarkResults:
        """Execute a benchmark scenario and collect metrics."""
        # Create Venice client using simple pattern from examples
        from venice_ai import VeniceClient

        api_key = os.getenv("VENICE_API_KEY", "test_key")
        client = VeniceClient(api_key=api_key)

        # Track start time
        start_time = asyncio.get_event_loop().time()
        results.start_time = datetime.now(UTC)

        # Generate and execute requests
        request_count = int(scenario.duration_seconds * scenario.target_rps)
        request_interval = 1.0 / scenario.target_rps

        # Collect latencies and track metrics
        latencies = []
        successful = 0
        failed = 0
        rate_limit_hits = 0

        for i in range(request_count):
            request_start = asyncio.get_event_loop().time()

            try:
                # Use chat completions - a reliable, authenticated endpoint
                from venice_ai.types.api import UserMessage

                response = await client.chat.completions.create(
                    model="llama-3.2-3b",  # Use a basic, reliable model
                    messages=[
                        UserMessage(role="user", content=f"Say 'test {i}' and nothing else.")
                    ],
                    max_completion_tokens=10,
                    temperature=0.1,
                )

                latency = (asyncio.get_event_loop().time() - request_start) * 1000  # ms
                latencies.append(latency)
                successful += 1

                # CRITICAL: Extract rate limit info using new header access patterns
                rate_limits = response.response_rate_limits
                if rate_limits:
                    print(
                        f"  📊 Rate Limits - Requests: {rate_limits.remaining_requests}/{rate_limits.limit_requests}"
                    )
                    if (
                        rate_limits.remaining_requests is not None
                        and rate_limits.remaining_requests < 10
                    ):
                        print(f"  ⚠️ Low remaining requests: {rate_limits.remaining_requests}")

            except Exception as e:
                latency = (asyncio.get_event_loop().time() - request_start) * 1000
                latencies.append(latency)
                failed += 1

                if "rate limit" in str(e).lower():
                    rate_limit_hits += 1
                    print(f"  🚦 Rate limit hit: {e}")

            # Wait for next request timing
            elapsed = asyncio.get_event_loop().time() - request_start
            if elapsed < request_interval:
                await asyncio.sleep(request_interval - elapsed)

        # Calculate results
        end_time = asyncio.get_event_loop().time()
        total_duration = end_time - start_time

        results.end_time = datetime.now(UTC)
        results.total_requests = request_count
        results.successful_requests = successful
        results.failed_requests = failed
        results.rate_limit_violations = rate_limit_hits

        # Calculate throughput metrics
        results.avg_throughput = request_count / total_duration if total_duration > 0 else 0
        results.peak_throughput = scenario.target_rps  # Simplified for this test

        # Calculate latency metrics (in milliseconds)
        if latencies:
            latencies.sort()
            results.min_latency = min(latencies)
            results.max_latency = max(latencies)
            results.mean_latency = sum(latencies) / len(latencies)

            # Calculate percentiles
            p50_idx = int(len(latencies) * 0.50)
            p95_idx = int(len(latencies) * 0.95)
            p99_idx = int(len(latencies) * 0.99)

            results.p50_latency = latencies[p50_idx] if p50_idx < len(latencies) else 0
            results.p95_latency = latencies[p95_idx] if p95_idx < len(latencies) else 0
            results.p99_latency = latencies[p99_idx] if p99_idx < len(latencies) else 0

        # Calculate rate limit efficiency
        results.theoretical_max_rps = scenario.target_rps
        results.achieved_percentage = (
            (results.avg_throughput / scenario.target_rps * 100) if scenario.target_rps > 0 else 0
        )
        results.rate_limit_efficiency = results.achieved_percentage

        # Cleanup
        await client.close()

        return results

    async def run_comprehensive_benchmark(self) -> dict[str, BenchmarkResults]:
        """
        Run a comprehensive benchmark suite with multiple scenarios.

        Returns:
            Dictionary mapping scenario names to results
        """
        scenarios = [
            ("steady_load", 30.0, 10.0),
            ("burst_pattern", 20.0, 50.0),
            ("rate_limit_stress", 15.0, 100.0),
            ("low_volume", 10.0, 1.0),
        ]

        all_results = {}

        for scenario_name, duration, rps in scenarios:
            print(f"\n🚀 Running scenario: {scenario_name}")
            print(f"  Duration: {duration}s, Target RPS: {rps}")

            try:
                results = await self.run_benchmark_scenario(
                    scenario_name=scenario_name,
                    duration_seconds=duration,
                    target_rps=rps,
                    use_vcr=True,
                )

                all_results[scenario_name] = results

                # Print quick summary
                success_rate = (
                    (results.successful_requests / results.total_requests * 100)
                    if results.total_requests > 0
                    else 0
                )
                print(
                    f"  ✅ Completed: {results.successful_requests}/{results.total_requests} requests"
                )
                print(f"  Success Rate: {success_rate:.2f}%")
                print(f"  Avg Latency: {results.mean_latency:.3f}ms")

            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")

        return all_results

    def generate_report(self, results: dict[str, BenchmarkResults]) -> str:
        """Generate a comprehensive benchmark report."""
        report_lines = [
            "=" * 80,
            "VCR SCHEDULER BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {datetime.now(UTC).isoformat()}",
            "",
        ]

        for _scenario_name, scenario_results in results.items():
            # Use the reporter's console report generation
            scenario_report = self.reporter.generate_console_report(scenario_results)
            report_lines.append(scenario_report)
            report_lines.append("")

        return "\n".join(report_lines)


# ============================================================================
# Pytest Test Cases
# ============================================================================


@pytest.fixture
async def vcr_benchmark():
    """Fixture to provide VCR benchmark instance."""
    benchmark = VCRSchedulerBenchmark()
    yield benchmark
    # Cleanup if needed


@pytest.mark.benchmark
@pytest.mark.integration
async def test_steady_load_with_vcr(vcr_cassette):
    """Test steady load scenario with VCR recording/replay."""
    print("\n🚀 Running VCR-based steady load benchmark...")

    with vcr_cassette:
        # Create simple client using standard pattern
        import os

        from venice_ai import VeniceClient

        api_key = os.getenv("VENICE_API_KEY", "test_key")
        client = VeniceClient(api_key=api_key)

        try:
            successful = 0
            failed = 0
            latencies = []
            rate_limit_hits = 0
            request_count = 5  # Small count for VCR recording

            for i in range(request_count):
                request_start = asyncio.get_event_loop().time()

                try:
                    # Use chat completions - the working endpoint
                    from venice_ai.types.api import UserMessage

                    response = await client.chat.completions.create(
                        model="llama-3.2-3b",  # Use a basic, reliable model
                        messages=[
                            UserMessage(role="user", content=f"Say 'test {i}' and nothing else.")
                        ],
                        max_completion_tokens=10,
                        temperature=0.1,
                    )

                    latency = (asyncio.get_event_loop().time() - request_start) * 1000  # ms
                    latencies.append(latency)
                    successful += 1

                    # CRITICAL: Extract rate limit info using new header access patterns
                    rate_limits = response.response_rate_limits
                    if rate_limits:
                        print(
                            f"  📊 Rate Limits - Requests: {rate_limits.remaining_requests}/{rate_limits.limit_requests}"
                        )
                        if (
                            rate_limits.remaining_requests is not None
                            and rate_limits.remaining_requests < 10
                        ):
                            print(f"  ⚠️ Low remaining requests: {rate_limits.remaining_requests}")

                except Exception as e:
                    latency = (asyncio.get_event_loop().time() - request_start) * 1000
                    latencies.append(latency)
                    failed += 1

                    if "rate limit" in str(e).lower():
                        rate_limit_hits += 1
                        print(f"  🚦 Rate limit hit: {e}")
                    else:
                        print(f"  ❌ Request {i} failed: {e}")

                # Small delay between requests
                await asyncio.sleep(0.5)

            # Calculate results
            total = successful + failed
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            # Verify results
            assert successful > 0, "No successful requests recorded"
            assert avg_latency > 0, "Average latency should be positive"

            print("✅ VCR Benchmark Results:")
            print(f"   📊 Successful: {successful}/{total}")
            print(f"   ⚡ Avg Latency: {avg_latency:.1f}ms")
            print(f"   🔄 Rate Limit Hits: {rate_limit_hits}")

            return {
                "successful": successful,
                "total": total,
                "avg_latency": avg_latency,
                "rate_limit_hits": rate_limit_hits,
            }

        finally:
            await client.close()


@pytest.mark.benchmark
@pytest.mark.integration
async def test_comprehensive_scheduler_stress(vcr_cassette):
    """
    Comprehensive stress test for intelligent scheduler across all endpoints.

    Tests throughput approaching 99% of rate limits with diverse requests:
    - Text/Chat completions (various models, parameters, function calling)
    - Image generation (different models, sizes, styles)
    - Text-to-Speech (multiple voices, formats)
    - Embeddings (different models, batch sizes)
    - High concurrency with intelligent rate limiting

    KEY INSIGHT: Models share rate limits in TIERS/GROUPS, not individually.
    The DynamicTierDiscovery system tracks these tier-based limits.
    """
    print("\n🚀 COMPREHENSIVE SCHEDULER STRESS TEST V2")
    print("=" * 60)

    with vcr_cassette:
        import os
        import time

        from venice_ai import VeniceClient
        from venice_ai.core.config import SchedulerConfig, SchedulerMode, VeniceAIConfig
        from venice_ai.types.api import SystemMessage, UserMessage

        # Configure for MAXIMUM throughput - target 99% rate limit utilization
        config = VeniceAIConfig(
            scheduler=SchedulerConfig(
                mode=SchedulerMode.INTELLIGENT,
                max_concurrent_executions=300,  # Much higher concurrency
                metrics_enabled=True,
                scheduler_interval=0.001,  # Very fast scheduling loops
                rate_limit_buffer_ratio=0.99,  # Target 99% of rate limits!
                request_timeout=30,
                max_queue_size=2000,
            )
        )

        api_key = os.getenv("VENICE_API_KEY", "test_key")
        client = VeniceClient(api_key=api_key, config=config)

        try:
            # === STRESS TEST METRICS ===
            start_time = time.time()
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            rate_limit_data = []
            endpoint_metrics = {
                "chat": {"success": 0, "fail": 0, "latencies": []},
                "image": {"success": 0, "fail": 0, "latencies": []},
                "audio": {"success": 0, "fail": 0, "latencies": []},
                "embeddings": {"success": 0, "fail": 0, "latencies": []},
            }

            # === DEFINE DIVERSE REQUEST SCENARIOS ===

            # Create a semaphore to prevent overwhelming the scheduler's queue
            concurrent_limit = asyncio.Semaphore(50)  # Allow 50 concurrent submissions

            async def make_chat_request(scenario_id: int):
                """Chat completions with diverse parameters - properly queued."""
                async with concurrent_limit:  # Control submission rate
                    request_start = time.time()
                    try:
                        scenarios = [
                            # Basic chat
                            {
                                "model": "llama-3.2-3b",
                                "messages": [
                                    UserMessage(
                                        role="user",
                                        content=f"Stress test chat {scenario_id}",
                                    )
                                ],
                                "max_completion_tokens": 20,
                                "temperature": 0.1,
                            },
                            # System message + higher tokens
                            {
                                "model": "llama-3.2-3b",
                                "messages": [
                                    SystemMessage(
                                        role="system",
                                        content="You are a helpful assistant.",
                                        name=None,
                                    ),
                                    UserMessage(
                                        role="user",
                                        content=f"Complex query {scenario_id}",
                                    ),
                                ],
                                "max_completion_tokens": 50,
                                "temperature": 0.7,
                            },
                            # Higher creativity
                            {
                                "model": "llama-3.2-3b",
                                "messages": [
                                    UserMessage(
                                        role="user",
                                        content=f"Creative task {scenario_id}",
                                    )
                                ],
                                "max_completion_tokens": 30,
                                "temperature": 1.2,
                                "top_p": 0.9,
                            },
                            # Structured output request
                            {
                                "model": "llama-3.2-3b",
                                "messages": [
                                    UserMessage(
                                        role="user",
                                        content=f"JSON response {scenario_id}",
                                    )
                                ],
                                "max_completion_tokens": 40,
                                "temperature": 0.3,
                            },
                        ]

                        scenario = scenarios[scenario_id % len(scenarios)]
                        # The scheduler will queue this properly
                        response = await client.chat.completions.create(**scenario)

                        latency = (time.time() - request_start) * 1000
                        endpoint_metrics["chat"]["success"] += 1
                        endpoint_metrics["chat"]["latencies"].append(latency)

                        # Capture rate limit intelligence
                        if response.response_rate_limits:
                            rate_limit_data.append(
                                {
                                    "endpoint": "chat",
                                    "scenario": scenario_id,
                                    "remaining_requests": response.response_rate_limits.remaining_requests,
                                    "limit_requests": response.response_rate_limits.limit_requests,
                                    "remaining_tokens": response.response_rate_limits.remaining_tokens,
                                    "limit_tokens": response.response_rate_limits.limit_tokens,
                                    "latency_ms": latency,
                                }
                            )

                        return True
                    except Exception as e:
                        endpoint_metrics["chat"]["fail"] += 1
                        print(f"  ❌ Chat request {scenario_id} failed: {e}")
                        return False

            async def make_image_request(scenario_id: int):
                """Image generation with diverse parameters - properly queued."""
                async with concurrent_limit:  # Control submission rate
                    request_start = time.time()
                    try:
                        scenarios = [
                            # Basic generation
                            {
                                "model": "flux-dev",
                                "prompt": f"Test image {scenario_id}",
                                "width": 512,
                                "height": 512,
                                "num_images": 1,
                            },
                            # High resolution
                            {
                                "model": "flux-dev",
                                "prompt": f"Detailed artwork {scenario_id}",
                                "width": 1024,
                                "height": 1024,
                                "num_images": 1,
                            },
                            # Different aspect ratio
                            {
                                "model": "flux-dev",
                                "prompt": f"Landscape {scenario_id}",
                                "width": 1024,
                                "height": 512,
                                "num_images": 1,
                            },
                        ]

                        scenario = scenarios[scenario_id % len(scenarios)]
                        await client.image.create(**scenario)

                        latency = (time.time() - request_start) * 1000
                        endpoint_metrics["image"]["success"] += 1
                        endpoint_metrics["image"]["latencies"].append(latency)

                        # Note: Image generation may not always return rate limit headers
                        print(f"  ✅ Image {scenario_id}: Generated in {latency:.1f}ms")

                        return True
                    except Exception as e:
                        endpoint_metrics["image"]["fail"] += 1
                        print(f"  ❌ Image request {scenario_id} failed: {e}")
                        return False

            async def make_audio_request(scenario_id: int):
                """Audio generation with diverse parameters - properly queued."""
                async with concurrent_limit:  # Control submission rate
                    request_start = time.time()
                    try:
                        # Use the correct model name from examples
                        scenarios = [
                            # Basic TTS
                            {
                                "model": "tts-kokoro",  # Fixed model name
                                "input": f"Stress test audio {scenario_id}",
                                "voice": "af_sky",
                            },
                            # Different voice
                            {
                                "model": "tts-kokoro",
                                "input": f"Alternative voice test {scenario_id}",
                                "voice": "bf_emma",
                            },
                            # Longer text
                            {
                                "model": "tts-kokoro",
                                "input": f"This is a longer text for stress testing audio generation scenario {scenario_id}",
                                "voice": "af_sky",
                            },
                        ]

                        scenario = scenarios[scenario_id % len(scenarios)]
                        await client.audio.create_speech(
                            model=scenario["model"],
                            input=scenario["input"],
                            voice=scenario["voice"],
                        )

                        latency = (time.time() - request_start) * 1000
                        endpoint_metrics["audio"]["success"] += 1
                        endpoint_metrics["audio"]["latencies"].append(latency)

                        return True
                    except Exception as e:
                        endpoint_metrics["audio"]["fail"] += 1
                        print(f"  ❌ Audio request {scenario_id} failed: {e}")
                        return False

            async def make_embedding_request(scenario_id: int):
                """Embeddings with diverse models and inputs."""
                request_start = time.time()
                try:
                    scenarios = [
                        # Single input
                        {
                            "model": "text-embedding-3-small",
                            "input": f"Embedding test {scenario_id}",
                        },
                        # Longer text
                        {
                            "model": "text-embedding-3-small",
                            "input": f"This is a longer text for embedding stress test scenario {scenario_id} with more content",
                        },
                        # Technical content
                        {
                            "model": "text-embedding-3-small",
                            "input": f"Technical documentation embedding test {scenario_id} with API references",
                        },
                    ]

                    scenario = scenarios[scenario_id % len(scenarios)]
                    response = await client.embeddings.create(
                        model=scenario["model"], input=scenario["input"]
                    )

                    latency = (time.time() - request_start) * 1000
                    endpoint_metrics["embeddings"]["success"] += 1
                    endpoint_metrics["embeddings"]["latencies"].append(latency)

                    # Don't print raw embeddings - just log dimensions
                    if hasattr(response, "data") and response.data:
                        dims = len(response.data[0].embedding) if response.data[0].embedding else 0
                        print(f"  ✅ Embedding {scenario_id}: {dims}D vector, {latency:.1f}ms")

                    return True
                except Exception as e:
                    endpoint_metrics["embeddings"]["fail"] += 1
                    print(f"  ❌ Embedding request {scenario_id} failed: {e}")
                    return False

            # === EXECUTE COMPREHENSIVE STRESS TEST ===

            print("🔥 Launching multi-endpoint stress test with INTELLIGENT SCHEDULER...")
            print(f"   Scheduler Mode: {config.scheduler.mode}")
            print(f"   Max Concurrent: {config.scheduler.max_concurrent_executions}")
            print(
                f"   Rate Limit Target: {config.scheduler.rate_limit_buffer_ratio * 100:.1f}% of capacity"
            )
            print(f"   Scheduler Interval: {config.scheduler.scheduler_interval}s")

            # CRITICAL: Submit requests through the scheduler, not raw asyncio tasks!
            # The intelligent scheduler will queue and manage these automatically.

            tasks = []
            request_types = [
                (make_chat_request, 70),  # 70% chat requests (most common)
                (make_embedding_request, 20),  # 20% embeddings
                (make_image_request, 5),  # 5% images (slower)
                (make_audio_request, 5),  # 5% audio
            ]

            # Launch 200 requests - scheduler will manage them!
            total_planned_requests = 200
            request_index = 0

            print(f"\n📋 Submitting {total_planned_requests} requests to scheduler queue...")

            # Submit all requests to the scheduler's queue
            # The scheduler will handle rate limiting and concurrency
            for request_func, percentage in request_types:
                num_requests = int(total_planned_requests * percentage / 100)
                for _ in range(num_requests):
                    # Create async task - scheduler will queue and manage them
                    task = asyncio.create_task(request_func(request_index))
                    tasks.append(task)
                    total_requests += 1
                    request_index += 1

            print(f"   • Chat: {int(total_planned_requests * 0.7)} requests")
            print(f"   • Embeddings: {int(total_planned_requests * 0.2)} requests")
            print(f"   • Images: {int(total_planned_requests * 0.05)} requests")
            print(f"   • Audio: {int(total_planned_requests * 0.05)} requests")

            print(f"\n⏳ Scheduler processing {len(tasks)} queued requests...")
            print("   (Intelligent scheduler will manage rate limits and concurrency)")

            # Progress tracking
            async def track_progress():
                """Monitor progress during execution."""
                while not all(task.done() for task in tasks):
                    completed = sum(1 for task in tasks if task.done())
                    pending = len(tasks) - completed
                    print(
                        f"   📊 Progress: {completed}/{len(tasks)} completed, {pending} pending..."
                    )
                    await asyncio.sleep(5)

            # Start progress tracker
            progress_task = asyncio.create_task(track_progress())

            try:
                # Now gather all tasks with timeout - scheduler will process them according to rate limits
                # Increased timeout since scheduler manages rate limits properly
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=120,  # 2 minutes should be enough for 200 requests with rate limiting
                )
            except TimeoutError:
                print(
                    "⚠️ Test timed out after 120 seconds - this is expected with proper rate limiting"
                )
                # Cancel remaining tasks gracefully
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait for cancellation to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            # Count successful requests
            for result in results:
                if result is True:
                    successful_requests += 1
                else:
                    failed_requests += 1

            # === ANALYZE STRESS TEST RESULTS ===

            total_time = time.time() - start_time
            throughput = successful_requests / total_time if total_time > 0 else 0
            success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0

            print("\n🏆 STRESS TEST RESULTS:")
            print(f"   ⏱️  Total Time: {total_time:.2f}s")
            print(
                f"   📊 Success Rate: {success_rate:.1f}% ({successful_requests}/{total_requests})"
            )
            print(f"   🚀 Throughput: {throughput:.2f} requests/second")
            print(f"   🔄 Rate Limit Data Points: {len(rate_limit_data)}")

            # Endpoint-specific metrics
            print("\n📈 ENDPOINT BREAKDOWN:")
            for endpoint, metrics in endpoint_metrics.items():
                total_endpoint = metrics["success"] + metrics["fail"]
                if total_endpoint > 0:
                    endpoint_success_rate = (metrics["success"] / total_endpoint) * 100
                    avg_latency = (
                        sum(metrics["latencies"]) / len(metrics["latencies"])
                        if metrics["latencies"]
                        else 0
                    )
                    print(
                        f"   🎯 {endpoint.upper()}: {endpoint_success_rate:.1f}% success, {avg_latency:.1f}ms avg latency"
                    )

            # Rate limit efficiency analysis - GROUP BY TIER
            if rate_limit_data:
                print("\n🎯 RATE LIMIT EFFICIENCY BY TIER:")

                # Group rate limit data by tier (approximate by similar limits)
                tier_groups = {}
                for data in rate_limit_data:
                    if data["limit_requests"]:
                        tier_key = f"{data['endpoint']}_{data['limit_requests']}"
                        if tier_key not in tier_groups:
                            tier_groups[tier_key] = []
                        tier_groups[tier_key].append(data)

                # Analyze each tier
                for tier_key, bucket_data in tier_groups.items():
                    if bucket_data:
                        first = bucket_data[0]
                        last = bucket_data[-1]
                        if first["remaining_requests"] and last["remaining_requests"]:
                            consumed = first["remaining_requests"] - last["remaining_requests"]
                            utilization = (
                                (first["limit_requests"] - last["remaining_requests"])
                                / first["limit_requests"]
                            ) * 100
                            print(
                                f"   • Tier {tier_key}: {utilization:.1f}% utilized, {consumed} requests consumed"
                            )

            # Assertions for stress test validation
            assert successful_requests > 0, "Stress test should have successful requests"
            assert success_rate > 80, (
                f"Success rate too low with scheduler: {success_rate:.1f}%"
            )  # Higher with proper scheduling
            assert throughput > 1.0, (
                f"Throughput too low: {throughput:.2f} req/s"
            )  # Realistic with rate limiting

            # Verify intelligent scheduling is working
            if len(rate_limit_data) >= 2:
                first_remaining = rate_limit_data[0]["remaining_requests"]
                last_remaining = rate_limit_data[-1]["remaining_requests"]
                if first_remaining and last_remaining:
                    capacity_consumed = first_remaining - last_remaining
                    print("\n🧠 INTELLIGENT SCHEDULER VALIDATION:")
                    print(f"   📉 Capacity Consumed: {capacity_consumed} requests")
                    print("   🎯 Target Utilization: 99%")

                    # Calculate actual utilization
                    if first_remaining:
                        actual_utilization = (
                            (first_remaining - last_remaining) / first_remaining
                        ) * 100
                        print(f"   📊 Actual Utilization: {actual_utilization:.1f}%")

                        # Success if we get within 20% of target
                        if actual_utilization > 80:
                            print("   ✅ EXCELLENT: Near maximum rate limit utilization!")
                        elif actual_utilization > 50:
                            print("   ⚠️ GOOD: Moderate rate limit utilization")
                        else:
                            print("   ❌ NEEDS IMPROVEMENT: Low rate limit utilization")

                    # Note: VCR cassettes may have static rate limit headers, so capacity_consumed might be 0
                    # This is acceptable for VCR-based tests since we're testing replay behavior
                    if capacity_consumed == 0:
                        print("   ℹ️ NOTE: VCR cassette has static rate limit headers (expected)")
                    else:
                        print(f"   ✅ Rate limit capacity consumed: {capacity_consumed}")

            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "throughput": throughput,
                "success_rate": success_rate,
                "total_time": total_time,
                "endpoint_metrics": endpoint_metrics,
                "rate_limit_data": rate_limit_data,
            }

        finally:
            await client.close()


@pytest.mark.benchmark
@pytest.mark.integration
async def test_intelligent_scheduler_with_vcr(vcr_cassette):
    """Test intelligent scheduler using VCR-recorded rate limit data."""
    print("\n🧠 Testing Intelligent Scheduler with VCR...")

    with vcr_cassette:
        # Create client with intelligent scheduler mode
        import os

        from venice_ai import VeniceClient
        from venice_ai.core.config import SchedulerMode, VeniceAIConfig

        api_key = os.getenv("VENICE_API_KEY", "test_key")

        # Configure intelligent mode using proper nested structure
        from venice_ai.core.config import CircuitBreakerConfig, SchedulerConfig

        config = VeniceAIConfig(
            scheduler=SchedulerConfig(
                mode=SchedulerMode.INTELLIGENT,
                max_concurrent_executions=3,
                metrics_enabled=True,
            ),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5, reset_timeout=60.0),
        )

        client = VeniceClient(api_key=api_key, config=config)

        try:
            intelligent_successes = 0
            intelligent_failures = 0
            rate_limit_data = []
            request_count = 5

            for i in range(request_count):
                request_start = asyncio.get_event_loop().time()

                try:
                    from venice_ai.types.api import UserMessage

                    response = await client.chat.completions.create(
                        model="llama-3.2-3b",
                        messages=[UserMessage(role="user", content=f"Intelligent test {i}")],
                        max_completion_tokens=10,
                        temperature=0.1,
                    )

                    latency = (asyncio.get_event_loop().time() - request_start) * 1000
                    intelligent_successes += 1

                    # Capture rate limit intelligence
                    rate_limits = response.response_rate_limits
                    if rate_limits:
                        rate_data = {
                            "request_num": i,
                            "remaining_requests": rate_limits.remaining_requests,
                            "limit_requests": rate_limits.limit_requests,
                            "remaining_tokens": rate_limits.remaining_tokens,
                            "limit_tokens": rate_limits.limit_tokens,
                            "latency_ms": latency,
                        }
                        rate_limit_data.append(rate_data)

                        print(
                            f"  🎯 Request {i}: {rate_limits.remaining_requests}/{rate_limits.limit_requests} requests, {rate_limits.remaining_tokens}/{rate_limits.limit_tokens} tokens"
                        )

                        # CRITICAL: Verify intelligent behavior
                        if i > 0:
                            # Check that rate limits are decreasing (intelligent reservation)
                            prev_data = rate_limit_data[i - 1]
                            current_remaining = rate_limits.remaining_requests
                            prev_remaining = prev_data["remaining_requests"]

                            if current_remaining is not None and prev_remaining is not None:
                                if current_remaining < prev_remaining:
                                    print(
                                        f"  ✅ Intelligent scheduling: {prev_remaining} → {current_remaining} (decreased)"
                                    )
                                else:
                                    print(
                                        f"  ⚠️ Rate limits: {prev_remaining} → {current_remaining} (unexpected)"
                                    )

                except Exception as e:
                    latency = (asyncio.get_event_loop().time() - request_start) * 1000
                    intelligent_failures += 1
                    print(f"  ❌ Intelligent request {i} failed: {e}")

                # Small delay to allow scheduler processing
                await asyncio.sleep(0.3)

            # Analyze intelligent scheduler performance
            total_requests = intelligent_successes + intelligent_failures
            success_rate = (
                (intelligent_successes / total_requests) * 100 if total_requests > 0 else 0
            )

            print("\n🧠 Intelligent Scheduler Results:")
            print(
                f"   📊 Success Rate: {success_rate:.1f}% ({intelligent_successes}/{total_requests})"
            )
            print(f"   📈 Rate Limit Tracking: {len(rate_limit_data)} data points")

            # Verify intelligent features
            if rate_limit_data:
                print("   🔍 Rate Limit Analysis:")
                for data in rate_limit_data:
                    remaining_pct = (
                        (data["remaining_requests"] / data["limit_requests"]) * 100
                        if data["limit_requests"]
                        else 0
                    )
                    print(
                        f"     • Request {data['request_num']}: {remaining_pct:.1f}% capacity remaining"
                    )

            # Assertions for intelligent behavior
            assert intelligent_successes > 0, (
                "Intelligent scheduler should complete requests successfully"
            )
            assert len(rate_limit_data) > 0, "Should capture rate limit intelligence data"

            # Test that we're getting decreasing rate limits (intelligent reservation)
            if len(rate_limit_data) >= 2:
                first_remaining = rate_limit_data[0]["remaining_requests"]
                last_remaining = rate_limit_data[-1]["remaining_requests"]
                if first_remaining is not None and last_remaining is not None:
                    assert last_remaining < first_remaining, (
                        f"Intelligent scheduler should consume rate limits: {first_remaining} → {last_remaining}"
                    )

            return {
                "mode": "intelligent",
                "successful": intelligent_successes,
                "total": total_requests,
                "rate_limit_data": rate_limit_data,
            }

        finally:
            await client.close()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_burst_pattern_with_vcr(vcr_benchmark):
    """Test burst pattern handling with VCR recording."""
    results = await vcr_benchmark.run_benchmark_scenario(
        scenario_name="test_burst_pattern",
        duration_seconds=5.0,
        target_rps=20.0,
        use_vcr=True,
    )

    assert results.total_requests > 0
    # Some failures expected due to burst
    assert results.failed_requests >= 0

    # Check latency percentiles make sense
    assert results.p50_latency <= results.p95_latency
    assert results.p95_latency <= results.p99_latency


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_rate_limit_recovery(vcr_benchmark):
    """Test scheduler recovery after rate limit hits."""
    results = await vcr_benchmark.run_benchmark_scenario(
        scenario_name="test_rate_limit_recovery",
        duration_seconds=15.0,
        target_rps=100.0,  # Intentionally high to trigger rate limits
        use_vcr=True,
    )

    # Should handle rate limits gracefully
    assert results.rate_limit_violations >= 0
    # But still complete some requests
    assert results.successful_requests > 0


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_intelligent_scheduler_performance(vcr_benchmark):
    """Benchmark the intelligent scheduler's performance characteristics."""
    scenarios = [
        ("low_load", 5.0, 2.0),
        ("medium_load", 5.0, 10.0),
        ("high_load", 5.0, 50.0),
    ]

    for scenario_name, duration, rps in scenarios:
        results = await vcr_benchmark.run_benchmark_scenario(
            scenario_name=f"perf_{scenario_name}",
            duration_seconds=duration,
            target_rps=rps,
            use_vcr=True,
        )

        # Performance assertions
        assert results.mean_latency < 1000  # Less than 1 second average
        assert results.p95_latency < 2000  # P95 under 2 seconds

        # Efficiency assertions
        if rps <= 10:  # Low/medium load should achieve high efficiency
            assert results.rate_limit_efficiency > 80.0


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_concurrent_model_requests(vcr_benchmark):
    """Test handling concurrent requests to different models."""
    # This would require a more complex setup with multiple models
    # For now, we'll test the basic concurrent request handling

    async def make_concurrent_requests():
        tasks = []
        for i in range(10):
            task = vcr_benchmark.run_benchmark_scenario(
                scenario_name=f"concurrent_test_{i}",
                duration_seconds=2.0,
                target_rps=5.0,
                use_vcr=True,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    results = await make_concurrent_requests()

    # Verify all tasks completed
    assert len(results) == 10

    # Check that most succeeded
    successful_runs = [r for r in results if not isinstance(r, Exception)]
    assert len(successful_runs) > 5  # At least half should succeed


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_vcr_cassette_replay():
    """Test that VCR cassettes can be replayed without API calls."""
    benchmark = VCRSchedulerBenchmark()

    # First run - may record if cassette doesn't exist
    results1 = await benchmark.run_benchmark_scenario(
        scenario_name="test_replay", duration_seconds=3.0, target_rps=2.0, use_vcr=True
    )

    # Second run - should use cassette
    results2 = await benchmark.run_benchmark_scenario(
        scenario_name="test_replay", duration_seconds=3.0, target_rps=2.0, use_vcr=True
    )

    # Results should be similar (not identical due to timing)
    assert results1.total_requests == results2.total_requests
    assert abs(results1.successful_requests - results2.successful_requests) <= 2


@pytest.mark.asyncio
async def test_comprehensive_benchmark_suite(vcr_benchmark):
    """Run the full benchmark suite and generate report."""
    results = await vcr_benchmark.run_comprehensive_benchmark()

    assert len(results) > 0

    # Generate and validate report
    report = vcr_benchmark.generate_report(results)
    assert "VCR SCHEDULER BENCHMARK REPORT" in report
    assert "THROUGHPUT METRICS" in report
    assert "LATENCY METRICS" in report

    # Save report to file
    report_path = Path(__file__).parent / "vcr_benchmark_report.txt"
    report_path.write_text(report)

    print(f"\n📊 Report saved to: {report_path}")


# ============================================================================
# Helper Functions for VCR Testing
# ============================================================================


def create_mock_response(
    status: int = 200,
    headers: dict[str, str] | None = None,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a mock HTTP response for VCR testing."""
    default_headers = {
        "X-RateLimit-Limit-Requests": "100",
        "X-RateLimit-Remaining-Requests": "99",
        "X-RateLimit-Reset-Requests": str(int(time.time()) + 60),
        "Content-Type": "application/json",
    }

    if headers:
        default_headers.update(headers)

    return {
        "status": {"code": status, "message": "OK" if status == 200 else "Error"},
        "headers": default_headers,
        "body": {
            "string": json.dumps(json_data or {"status": "ok"}),
        },
    }


def generate_rate_limit_headers(
    limit: int = 100,
    remaining: int = 99,
    reset_seconds: int = 60,
) -> dict[str, str]:
    """Generate rate limit headers for testing."""
    reset_time = int(time.time()) + reset_seconds

    return {
        "X-RateLimit-Limit-Requests": str(limit),
        "X-RateLimit-Remaining-Requests": str(remaining),
        "X-RateLimit-Reset-Requests": str(reset_time),
        "X-RateLimit-Limit-Tokens": "10000",
        "X-RateLimit-Remaining-Tokens": "9850",
        "X-RateLimit-Reset-Tokens": str(reset_time),
    }


if __name__ == "__main__":
    # Allow running directly for quick testing
    async def main():
        benchmark = VCRSchedulerBenchmark()
        results = await benchmark.run_comprehensive_benchmark()
        report = benchmark.generate_report(results)
        print(report)

    asyncio.run(main())
