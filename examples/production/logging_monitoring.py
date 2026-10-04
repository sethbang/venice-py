"""
Venice AI SDK - Production Logging and Monitoring

This example demonstrates production-ready logging and monitoring patterns
for the Venice AI SDK, including:

1. Structured logging setup
2. Request/response logging
3. Error tracking and alerting
4. Performance monitoring
5. Custom log formatters
6. Integration patterns with external monitoring systems

Requirements:
    pip install venice-py
    export VENICE_API_KEY="your-api-key"

Optional monitoring tools (shown conceptually):
    pip install sentry-sdk  # For error tracking
    pip install python-json-logger  # For structured JSON logs
"""

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from venice_ai import VeniceClient
from venice_ai.exceptions import NoMatchingModelError, VeniceError
from venice_ai.types.api.requests import UserMessage, VeniceParameters

# Bill only the prompts shown here, not the system prompt Venice adds by default.
OWN_PROMPT_ONLY = VeniceParameters(include_venice_system_prompt=False)

# Resolve results dir relative to this file's location so log files land under
# examples/results/ instead of polluting whatever directory the example is run
# from.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# =============================================================================
# Custom Log Formatters
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """Emit one JSON object per record, including every ``extra=`` field."""

    # Attributes every LogRecord carries. Anything else on the record came from
    # ``extra=`` and belongs in the structured output.
    _STANDARD_ATTRS = frozenset(
        vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None)).keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        """Format a record as a single JSON line."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in vars(record).items():
            if key not in self._STANDARD_ATTRS and not key.startswith("_"):
                log_data[key] = value

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # default=str keeps non-JSON values (Decimal, datetime) from breaking a line
        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for terminal output (development)."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for terminal."""
        original_levelname = record.levelname
        color = self.COLORS.get(original_levelname, self.RESET)
        record.levelname = f"{color}{original_levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


# =============================================================================
# Logging Setup Functions
# =============================================================================


def _reset_handlers(logger: logging.Logger) -> None:
    """Remove and close existing handlers so repeated setup never stacks them."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_development_logging() -> logging.Logger:
    """
    Setup logging for development environment.

    Features:
    - Colored output for terminal
    - DEBUG level logging
    - Detailed formatting
    """
    logger = logging.getLogger("venice_ai")
    _reset_handlers(logger)
    logger.setLevel(logging.DEBUG)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def setup_production_logging(log_file: str | None = None) -> logging.Logger:
    """
    Setup logging for production environment.

    Features:
    - Structured logging to file
    - INFO level (less verbose)
    - JSON-like formatting for log aggregation
    - Separate error log file

    Args:
        log_file: Path to main log file. Defaults to ``examples/results/venice_ai.log``
            so the example never writes into the caller's current directory.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    main_path = Path(log_file) if log_file is not None else RESULTS_DIR / "venice_ai.log"
    error_path = RESULTS_DIR / "venice_ai_errors.log"

    logger = logging.getLogger("venice_ai")
    _reset_handlers(logger)
    logger.setLevel(logging.INFO)

    # Main log file handler (all logs). mode="w" starts a fresh file per run so
    # the log only holds this run's records; production would use a rotating
    # handler instead (see the best-practices section).
    file_handler = logging.FileHandler(main_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(StructuredFormatter())

    # Error log file handler (errors only)
    error_handler = logging.FileHandler(error_path, mode="w")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())

    # Console handler for critical issues
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger


class RecordCollector(logging.Handler):
    """Keep every record it receives, so a run can check what was logged."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# =============================================================================
# Request Logging Wrapper
# =============================================================================


class RequestLogger:
    """Wrapper for logging Venice AI requests with context."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._request_counter = 0

    def _get_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        timestamp = int(time.time() * 1000)
        return f"req_{timestamp}_{self._request_counter}"

    async def log_chat_request(
        self,
        client: VeniceClient,
        model: str,
        messages: list,
        **kwargs: Any,
    ) -> Any:
        """
        Log a chat completion request with timing and response metadata.

        Args:
            client: Venice AI client
            model: Model to use
            messages: Chat messages
            **kwargs: Additional chat completion parameters

        Returns:
            Chat completion response
        """
        request_id = self._get_request_id()
        start_time = time.time()

        # Log request start
        self.logger.info(
            "Starting chat request",
            extra={
                "request_id": request_id,
                "model": model,
                "message_count": len(messages),
            },
        )

        try:
            # Make request
            response = await client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Surface Venice production signals carried on response headers.
            # balance_info is what this API key could still spend before the
            # request (the lesser of the account balance and the key's
            # remaining spend limit), not the account balance; response_rate_limits
            # exposes the request/token budget windows. Both are None when the
            # corresponding headers are absent (e.g. recorded fixtures), so guard.
            balance = response.balance_info
            rate_limits = response.response_rate_limits

            # Log success
            self.logger.info(
                "Chat request completed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "duration_ms": duration_ms,
                    "finish_reason": response.choices[0].finish_reason,
                    "tokens_used": (response.usage.total_tokens if response.usage else None),
                    "balance_diem": balance.diem if balance else None,
                    "balance_usd": balance.usd if balance else None,
                    "remaining_requests": (rate_limits.remaining_requests if rate_limits else None),
                    "remaining_tokens": (rate_limits.remaining_tokens if rate_limits else None),
                },
            )

            return response

        except VeniceError as e:
            # Calculate duration even on error
            duration_ms = int((time.time() - start_time) * 1000)

            # Log error with full context
            self.logger.error(
                f"Chat request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "duration_ms": duration_ms,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise


# =============================================================================
# Performance Monitoring
# =============================================================================


class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: dict[str, Any] = {
            "request_durations": [],
            "token_usage": [],
            "error_counts": {},
        }

    def record_request(
        self,
        duration_ms: int,
        tokens: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Record metrics for a request; errors are counted by exception class."""
        self.metrics["request_durations"].append(duration_ms)

        if tokens:
            self.metrics["token_usage"].append(tokens)

        if error is not None:
            error_type = type(error).__name__
            self.metrics["error_counts"][error_type] = (
                self.metrics["error_counts"].get(error_type, 0) + 1
            )

    def log_summary(self) -> None:
        """Log summary statistics."""
        if not self.metrics["request_durations"]:
            self.logger.info("No requests recorded")
            return

        durations = self.metrics["request_durations"]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        self.logger.info(
            "Performance summary",
            extra={
                "total_requests": len(durations),
                "avg_duration_ms": int(avg_duration),
                "max_duration_ms": max_duration,
                "min_duration_ms": min_duration,
            },
        )

        if self.metrics["token_usage"]:
            tokens = self.metrics["token_usage"]
            self.logger.info(
                "Token usage summary",
                extra={
                    "total_tokens": sum(tokens),
                    "avg_tokens": int(sum(tokens) / len(tokens)),
                },
            )

        if self.metrics["error_counts"]:
            self.logger.warning(
                "Error summary",
                extra={"errors": self.metrics["error_counts"]},
            )


# =============================================================================
# Error Tracking Integration (Conceptual)
# =============================================================================


class ErrorTracker:
    """
    Conceptual integration with error tracking services (e.g., Sentry).

    In production, you would integrate with actual services:
    - Sentry: sentry_sdk.capture_exception()
    - Datadog: statsd.increment('venice.errors')
    - CloudWatch: cloudwatch.put_metric_data()
    """

    def __init__(self, logger: logging.Logger, enable_external: bool = False):
        self.logger = logger
        self.enable_external = enable_external

    async def track_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Track an error with context.

        Args:
            error: The exception that occurred
            context: Additional context (model, request_id, etc.)
        """
        # Log locally
        self.logger.error(
            f"Error tracked: {str(error)}",
            extra=context or {},
            exc_info=True,
        )

        # Forward to an external error tracker (if enabled). This example has
        # none wired up; a real integration (for example Sentry's
        # capture_exception) would send the error and its context here.
        if self.enable_external:
            self.logger.debug("Would forward the error to an external error tracker here")


# =============================================================================
# Example Usage
# =============================================================================


def _format_balance(response: Any) -> str:
    """Render what the API key could spend, in the currencies the response carried.

    This is the key's spendable balance (the lesser of the account balance and
    the key's remaining spend limit), not the account balance, which
    ``client.billing.get_balance()`` returns.
    """
    balance = response.balance_info
    if balance is None:
        return "n/a (no balance headers)"
    parts = []
    if balance.usd is not None:
        parts.append(f"${balance.usd} USD")
    if balance.diem is not None:
        parts.append(f"{balance.diem} DIEM")
    # The header is read before this request was charged.
    return " / ".join(parts) + " (before this request)" if parts else "n/a"


async def example_development_logging() -> bool:
    """Example: Development logging setup."""
    print("=" * 60)
    print("Development Logging Example")
    print("=" * 60)

    # The SDK logs through the standard ``venice_ai`` logger hierarchy, so a
    # DEBUG-level handler on that logger is all it takes to see SDK internals
    # (model selection, request routing, header redaction).
    logger = setup_development_logging()
    # A second handler that keeps the records, to check SDK output arrived.
    collector = RecordCollector()
    logger.addHandler(collector)

    try:
        async with VeniceClient() as client:
            request_logger = RequestLogger(logger)

            model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
            response = await request_logger.log_chat_request(
                client=client,
                model=model,
                messages=[UserMessage(content="Reply with exactly: Hello from Venice AI!")],
                max_completion_tokens=100,
                venice_parameters=OWN_PROMPT_ONLY,
            )
    finally:
        logger.removeHandler(collector)

    finish_reason = response.choices[0].finish_reason if response.choices else None
    print(f"\n   Response: {response.text}")
    print(f"   finish_reason: {finish_reason}")
    if finish_reason == "length" or not response.text:
        print("❌ Response was truncated or empty")
        return False

    # RequestLogger logs at INFO, so every DEBUG record came from the SDK itself.
    sdk_debug = [r for r in collector.records if r.levelno == logging.DEBUG]
    sources = sorted({r.name for r in sdk_debug})
    print(f"   SDK DEBUG records: {len(sdk_debug)} from {', '.join(sources) or 'no logger'}")
    if not sdk_debug:
        print("❌ No SDK DEBUG record reached the venice_ai logger")
        return False
    print("✅ Development logging captured the SDK debug output above")
    return True


async def example_production_logging() -> bool:
    """Example: Production logging with monitoring."""
    print("\n" + "=" * 60)
    print("Production Logging Example")
    print("=" * 60)

    main_log = RESULTS_DIR / "venice_ai.log"
    error_log = RESULTS_DIR / "venice_ai_errors.log"

    # Structured records go to files; only ERROR and above reach the console.
    logger = setup_production_logging(str(main_log))

    attempts = 2
    succeeded = 0
    async with VeniceClient() as client:
        request_logger = RequestLogger(logger)
        perf_monitor = PerformanceMonitor(logger)
        error_tracker = ErrorTracker(logger, enable_external=False)

        model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
        for i in range(attempts):
            start = time.time()
            try:
                response = await request_logger.log_chat_request(
                    client=client,
                    model=model,
                    messages=[
                        UserMessage(content=f"Request #{i + 1}: reply with one short sentence.")
                    ],
                    max_completion_tokens=100,
                    venice_parameters=OWN_PROMPT_ONLY,
                )
            except VeniceError as e:
                await error_tracker.track_error(
                    e,
                    context={"model": model, "request_number": i + 1},
                )
                perf_monitor.record_request(int((time.time() - start) * 1000), error=e)
                print(f"❌ Request {i + 1} failed: {type(e).__name__}: {e}")
                continue

            duration_ms = int((time.time() - start) * 1000)
            tokens = response.usage.total_tokens if response.usage else None
            perf_monitor.record_request(duration_ms, tokens)

            finish_reason = response.choices[0].finish_reason if response.choices else None
            rate_limits = response.response_rate_limits
            rl_str = (
                f"{rate_limits.remaining_requests} req / "
                f"{rate_limits.remaining_tokens} tok remaining"
                if rate_limits
                else "n/a (no rate-limit headers)"
            )
            print(f"Request {i + 1}: {duration_ms}ms, finish_reason={finish_reason}")
            print(f"   key can spend: {_format_balance(response)}")
            print(f"   rate limits: {rl_str}")
            if finish_reason == "length" or not response.text:
                print("   ❌ Response was truncated or empty")
                continue
            succeeded += 1

        perf_monitor.log_summary()

    # Close the file handlers so the log is flushed and later SDK activity in
    # this process stays out of it.
    _reset_handlers(logger)

    print("\n   Logs written to:")
    print(f"   - {main_log} (all logs)")
    print(f"   - {error_log} (errors only)")

    ok = succeeded == attempts and _verify_structured_log(main_log, attempts)
    print(f"{'✅' if ok else '❌'} {succeeded}/{attempts} production requests succeeded")
    return ok


def _verify_structured_log(path: Path, expected_requests: int) -> bool:
    """Read the log back and check it holds the metrics the monitor produced."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    completed = [r for r in records if r["message"] == "Chat request completed"]
    summaries = {r["message"]: r for r in records if r["message"].endswith("summary")}

    problems = []
    if len(completed) != expected_requests:
        problems.append(f"{len(completed)} completion records, expected {expected_requests}")
    for record in completed:
        missing = [k for k in ("tokens_used", "finish_reason", "duration_ms") if k not in record]
        if missing:
            problems.append(f"completion record missing {missing}")
    perf = summaries.get("Performance summary", {})
    if perf.get("total_requests") != expected_requests:
        problems.append("performance summary missing or wrong request count")
    if "total_tokens" not in summaries.get("Token usage summary", {}):
        problems.append("token usage summary missing totals")
    if len(records) != len({json.dumps(r, sort_keys=True) for r in records}):
        problems.append("duplicate records")

    print(f"\n   Read back {len(records)} JSON records from {path.name}")
    if perf:
        print(
            f"   Performance summary: {perf['total_requests']} requests, "
            f"avg {perf['avg_duration_ms']}ms"
        )
    for problem in problems:
        print(f"   ❌ {problem}")
    return not problems


async def example_structured_logging(production_log: Path) -> bool:
    """Example: Structured logging for log aggregation.

    Re-emits the metadata of the last real request from the production log
    through an application logger, to show what one structured line holds.
    """
    print("\n" + "=" * 60)
    print("Structured Logging Example")
    print("=" * 60)

    # A dedicated logger for application events. propagate=False keeps these
    # records out of the ``venice_ai`` handlers (and the production log file).
    logger = logging.getLogger("app.structured")
    _reset_handlers(logger)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

    records = [json.loads(line) for line in production_log.read_text().splitlines() if line]
    completed = [r for r in records if r.get("message") == "Chat request completed"]
    if not completed:
        print("❌ No completed request in the production log to report on")
        return False
    last = completed[-1]

    logger.info(
        "Request completed",
        extra={
            key: last[key]
            for key in ("request_id", "model", "duration_ms", "tokens_used", "finish_reason")
        },
    )

    print("\n💡 Each line is one JSON object, so ELK, Splunk or CloudWatch Logs")
    print("   Insights can index every field without a custom parser.")
    return True


async def example_monitoring_best_practices():
    """Display monitoring best practices."""
    print("\n" + "=" * 60)
    print("Monitoring Best Practices")
    print("=" * 60)

    practices = [
        (
            "Log Levels",
            [
                "DEBUG: Development only (very verbose)",
                "INFO: Normal operations (requests, responses)",
                "WARNING: Unusual but handled situations",
                "ERROR: Errors that need attention",
                "CRITICAL: System-critical failures",
            ],
        ),
        (
            "What to Log",
            [
                "✅ Request/response metadata (model, tokens, duration)",
                "✅ Errors with full context and stack traces",
                "✅ Performance metrics (latency, throughput)",
                "✅ Rate limit status and warnings",
                "🚫 Never log API keys or sensitive data",
                "🚫 Don't log full request/response content in production",
            ],
        ),
        (
            "Log Rotation",
            [
                "Use rotating file handlers for disk space management",
                "Example: RotatingFileHandler(maxBytes=10MB, backupCount=5)",
                "Or use TimedRotatingFileHandler for daily rotation",
            ],
        ),
        (
            "External Services",
            [
                "Sentry/Rollbar: Real-time error tracking with alerting",
                "Datadog/New Relic: Application performance monitoring",
                "ELK Stack: Centralized log aggregation and search",
                "CloudWatch/Stackdriver: Cloud-native monitoring",
            ],
        ),
        (
            "Alerting Rules",
            [
                "Error rate threshold: >1% errors in 5 minutes",
                "Latency threshold: p95 > 2 seconds",
                "Rate limit warnings: >80% capacity used",
                "Circuit breaker trips: Immediate alert",
            ],
        ),
    ]

    for category, items in practices:
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   {item}")


async def main() -> int:
    """Run all logging examples."""
    print("=" * 60)
    print("Venice AI SDK - Production Logging & Monitoring")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    results.append(("Development Logging", await example_development_logging()))
    results.append(("Production Logging", await example_production_logging()))
    results.append(
        ("Structured Logging", await example_structured_logging(RESULTS_DIR / "venice_ai.log"))
    )
    await example_monitoring_best_practices()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    if failed == 0:
        print(f"✅ All {passed}/{len(results)} logging examples succeeded!")
    else:
        print(f"❌ {passed}/{len(results)} examples succeeded; {failed} failed")
        for name, ok in results:
            status = "✓" if ok else "✗"
            print(f"   {status} {name}")
    print("=" * 60)

    print("\n🔑 Key Takeaways:")
    print("   1. Use DEBUG level in development, INFO in production")
    print("   2. Implement structured logging for better searchability")
    print("   3. Monitor performance metrics (latency, tokens, errors)")
    print("   4. Integrate with external monitoring services")
    print("   5. Set up alerts for critical issues")
    print("   6. Never log sensitive data (API keys, PII)")
    print("   7. Use log rotation to manage disk space")
    print("   8. Include request context in all logs")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except NoMatchingModelError as e:
        # The catalog has no model of the kind this example needs.
        print(f"SKIPPED: {e}")
        sys.exit(77)
    except (VeniceError, OSError) as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
