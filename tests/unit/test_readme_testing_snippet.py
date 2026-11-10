"""The README's testing section runs offline, as written, against a loopback stand-in."""

import ast
import inspect
import re
from pathlib import Path

import pytest
from aiohttp import web

README = Path(__file__).parents[2] / "README.md"


def _testing_section() -> str:
    text = README.read_text(encoding="utf-8")
    return text.split("\n## Testing\n", 1)[1].split("\n## ", 1)[0]


def _python_block() -> str:
    match = re.search(r"```python\n(.*?)```", _testing_section(), re.DOTALL)
    assert match is not None
    return match.group(1)


def test_snippet_does_not_use_the_factory_test_client() -> None:
    assert "create_test_venice_client" not in _python_block()


def test_section_points_at_the_offline_example() -> None:
    section = _testing_section()
    assert "examples/basic/offline_testing.py" in section
    assert (Path(__file__).parents[2] / "examples/basic/offline_testing.py").exists()


def test_section_shows_the_explicit_client() -> None:
    block = re.sub(r"\s+", " ", _python_block())
    assert 'VeniceClient( config=create_testing_config(), api_key="test-key", base_url=' in block


CATALOG = {
    "object": "list",
    "type": "text",
    "data": [
        {
            "id": "stand-in-model",
            "object": "model",
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {
                "name": "stand-in-model",
                "capabilities": {
                    "supportsReasoning": False,
                    "optimizedForCode": False,
                    "quantization": "not-applicable",
                    "supportsFunctionCalling": False,
                    "supportsResponseSchema": False,
                    "supportsVision": False,
                    "supportsWebSearch": False,
                    "supportsLogProbs": False,
                },
                "pricing": {"input": {"usd": 0.1}, "output": {"usd": 0.2}},
            },
        }
    ],
}

COMPLETION = {
    "id": "chatcmpl-stand-in",
    "object": "chat.completion",
    "created": 0,
    "model": "stand-in-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "A very old bridge."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


async def test_snippet_runs_against_a_loopback_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    # A dead port: a client that fell back to the environment would fail here.
    monkeypatch.setenv("VENICE_API_BASE_URL", "http://127.0.0.1:9/api/v1")
    received: list[str | None] = []

    async def models(request: web.Request) -> web.Response:
        return web.json_response(CATALOG)

    async def chat(request: web.Request) -> web.Response:
        received.append(request.headers.get("Authorization"))
        return web.json_response(COMPLETION)

    app = web.Application()
    app.router.add_get("/api/v1/models", models)
    app.router.add_post("/api/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]

    source = _python_block().replace("127.0.0.1:8080", f"127.0.0.1:{port}")
    assert f"127.0.0.1:{port}" in source
    code = compile(source, "README.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    namespace: dict[str, object] = {}
    try:
        result = eval(code, namespace)  # noqa: S307
        assert inspect.isawaitable(result)
        await result
    finally:
        await runner.cleanup()

    assert received == ["Bearer test-key"]
    assert getattr(namespace["response"], "text", None) == "A very old bridge."
