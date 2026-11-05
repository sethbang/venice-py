"""
Chat command for Venice AI CLI
"""

import asyncio
import json
import sys
import uuid
from typing import TYPE_CHECKING, Any, cast

import click
import questionary
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from venice_ai import VeniceClient, extract_thinking_blocks
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import AssistantMessage, SystemMessage, UserMessage

from .._model_defaults import resolve_default_model
from ..config import get_client_kwargs, load_config
from ..conversation import (
    get_last_conversation_id,
    list_conversations,
    load_conversation,
    save_conversation,
)
from ..utils import console, is_plain_mode, print_error, print_info, print_success
from ..utils.streaming import AnimationMode, StreamHandler

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Shared helpers – extracted to DRY up streaming / non-streaming / single /
# interactive code-paths that previously duplicated thinking-block display,
# content display, history-append and API-call logic.
# ---------------------------------------------------------------------------


def _display_thinking(
    thinking_blocks: list[str],
    show_thinking: bool,
    plain: bool,
    reasoning_content: str | None = None,
    is_streaming: bool = False,
) -> None:
    """Display thinking/reasoning blocks.

    For streaming responses, blocks are joined into a single panel.
    For non-streaming, *reasoning_content* (a Venice-specific field) takes
    priority over inline ``<think>``/``<thinking>`` blocks.
    """
    if not show_thinking:
        return

    # Non-streaming: reasoning_content takes priority
    if not is_streaming and reasoning_content:
        if plain:
            click.echo(f"\nReasoning:\n{reasoning_content}")
        else:
            console.print("\n[bold dim]Reasoning:[/bold dim]")
            console.print(Panel(reasoning_content, border_style="dim"))
        return

    if not thinking_blocks:
        return

    if is_streaming:
        # Streaming: show all blocks in a single panel
        if plain:
            click.echo("\n--- Reasoning ---")
            click.echo("\n".join(thinking_blocks))
            click.echo("--- End Reasoning ---")
        else:
            console.print()
            console.print(
                Panel(
                    "\n".join(thinking_blocks),
                    title="[dim]Reasoning[/dim]",
                    border_style="dim",
                )
            )
    else:
        # Non-streaming: show each block individually
        if plain:
            click.echo("\nThinking Process:")
            for block in thinking_blocks:
                click.echo(block.strip())
        else:
            console.print("\n[bold dim]Thinking Process:[/bold dim]")
            for block in thinking_blocks:
                console.print(Panel(block.strip(), border_style="dim"))


def _display_nonstreaming_content(display_content: str, plain: bool) -> None:
    """Display assistant response content for non-streaming responses."""
    if plain:
        click.echo("Assistant:")
        click.echo(display_content if isinstance(display_content, str) else str(display_content))
    else:
        console.print("\n[bold green]Assistant:[/bold green]")
        if isinstance(display_content, str):
            console.print(Markdown(display_content))
        else:
            console.print(str(display_content))


def _append_to_history(
    messages: list[Any],
    content: str,
    thinking_blocks: list[str],
    clean_content: str,
    reasoning_content: str | None = None,
) -> None:
    """Append an assistant response to the message history.

    Uses *clean_content* (with thinking blocks stripped) when blocks are
    present, otherwise keeps the original *content* verbatim.
    """
    content_for_history = clean_content if thinking_blocks else content
    content_str = (
        content_for_history if isinstance(content_for_history, str) else str(content_for_history)
    )
    messages.append(
        AssistantMessage(
            role="assistant",
            content=content_str,
            name=None,
            reasoning_content=reasoning_content,
            tool_calls=None,
        )
    )


def _get_reasoning_content(message: Any) -> str | None:
    """Extract reasoning_content from a response message if available."""
    if hasattr(message, "reasoning_content"):
        return getattr(message, "reasoning_content", None)
    return None


async def _make_nonstreaming_request(
    client: VeniceClient,
    model: str,
    messages: list[Any],
    temperature: float,
    max_completion_tokens: int,
    extra_kwargs: dict[str, Any],
    show_spinner: bool = False,
) -> Any:
    """Make a non-streaming chat completion request.

    When *show_spinner* is ``True``, displays a "Thinking..." spinner during
    the request (rich console required).
    """
    if show_spinner:
        with console.status("Thinking...", spinner="dots"):
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                stream=False,
                **extra_kwargs,
            )
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        stream=False,
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
def chat(ctx: click.Context):
    """Chat with AI models.

    Use 'venice-py chat start' to begin a chat session or send a single message.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@chat.command(name="start")
@click.option("--model", "-m", help="AI model to use", default=None)
@click.option(
    "--select-model",
    "-S",
    is_flag=True,
    help="Interactively select model from available options",
)
@click.option("--system", "-s", help="System prompt to set the AI personality", default=None)
@click.option(
    "--temperature",
    "-t",
    type=float,
    help="Temperature for response generation",
    default=None,
)
@click.option("--max-completion-tokens", type=int, help="Maximum tokens in response", default=None)
@click.option("--stream/--no-stream", default=True, help="Enable/disable streaming responses")
@click.option(
    "--show-thinking/--hide-thinking",
    default=False,
    help="Show/hide thinking process for reasoning models",
)
@click.option(
    "--animation",
    "-a",
    type=click.Choice(["none", "smooth", "word", "char", "line", "typewriter"]),
    default="smooth",
    help="Animation style for streaming (default: smooth)",
)
@click.option(
    "--animation-speed",
    type=float,
    default=0.03,
    help="Animation speed in seconds (lower is faster, default: 0.03)",
)
@click.option("--show-stats/--hide-stats", default=False, help="Show streaming statistics")
@click.option(
    "--top-p",
    type=float,
    default=None,
    help="Top-p (nucleus) sampling parameter (0.0-1.0)",
)
@click.option(
    "--web-search",
    type=click.Choice(["auto", "on", "off"]),
    default=None,
    help="Enable web search (auto/on/off)",
)
@click.option(
    "--character",
    "character_slug",
    default=None,
    help="Character slug for character-driven chat",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh", "max"]),
    default=None,
    help="Reasoning effort level for supported models",
)
@click.option(
    "--strip-thinking/--no-strip-thinking",
    default=None,
    help="Strip thinking/reasoning from response",
)
@click.option(
    "--disable-thinking",
    is_flag=True,
    default=False,
    help="Disable thinking/reasoning entirely",
)
@click.option(
    "--json-output",
    "json_output",
    is_flag=True,
    default=False,
    help="Output response as JSON",
)
@click.option(
    "--save",
    "save_conversation_flag",
    is_flag=True,
    default=False,
    help="Save conversation after session ends",
)
@click.option(
    "--continue-from",
    "continue_id",
    default=None,
    help="Continue a previous conversation by ID (use 'last' for most recent)",
)
@click.argument("message", required=False)
@click.pass_context
def start_chat(
    ctx: click.Context,
    model: str | None,
    select_model: bool,
    system: str | None,
    temperature: float | None,
    max_completion_tokens: int | None,
    stream: bool,
    show_thinking: bool,
    animation: str,
    animation_speed: float,
    show_stats: bool,
    top_p: float | None,
    web_search: str | None,
    character_slug: str | None,
    reasoning_effort: str | None,
    strip_thinking: bool | None,
    disable_thinking: bool,
    json_output: bool,
    save_conversation_flag: bool,
    continue_id: str | None,
    message: str | None,
) -> None:
    """Start a chat session or send a single message

    Animation modes:

    • smooth: Live markdown rendering (default)

    • word: Word-by-word animation

    • char: Character-by-character animation

    • line: Line-buffered animation

    • typewriter: Classic typewriter effect

    • none: No animation, instant display

    Examples:

        # Quick message with word animation
        venice-py chat start -a word "Tell me a joke"

        # Interactive session with typewriter effect
        venice-py chat start --animation typewriter --animation-speed 0.05

        # Fast streaming with stats
        venice-py chat start --animation none --show-stats
    """
    # Build venice_parameters dict from Venice-specific options.
    # NB: reasoning_effort is a top-level API field, not a member of
    # venice_parameters (which has extra="forbid"). Pass it separately.
    venice_params: dict[str, Any] = {}
    if web_search is not None:
        venice_params["enable_web_search"] = web_search
    if character_slug is not None:
        venice_params["character_slug"] = character_slug
    if strip_thinking is not None:
        venice_params["strip_thinking_response"] = strip_thinking
    if disable_thinking:
        venice_params["disable_thinking"] = True

    asyncio.run(
        _chat_async(
            ctx,
            model,
            select_model,
            system,
            temperature,
            max_completion_tokens,
            stream,
            show_thinking,
            animation,
            animation_speed,
            show_stats,
            message,
            top_p=top_p,
            venice_params=venice_params if venice_params else None,
            reasoning_effort=reasoning_effort,
            json_output=json_output,
            save_conversation_flag=save_conversation_flag,
            continue_id=continue_id,
        )
    )


async def _select_chat_model(client: VeniceClient) -> str | None:
    """Interactive model selection"""
    try:
        print_info("Fetching available text models...")
        response = await client.models.list(type="text")

        if not response.data:
            print_error("No text models available")
            return None

        # Build list of model choices
        models = sorted(response.data, key=lambda x: x.id)
        choices = []
        # ``type='text'`` dispatches to ``TextModelSpec`` which carries
        # ``availableContextTokens``; narrow explicitly so the type checker
        # sees the field.
        from venice_ai.types.api import TextModelSpec

        for model in models:
            # Get context info
            context = ""
            spec = model.model_spec
            if isinstance(spec, TextModelSpec) and spec.availableContextTokens:
                tokens = int(spec.availableContextTokens)
                if tokens >= 100000:
                    context = f" ({tokens // 1000}k context)"
                else:
                    context = f" ({tokens} tokens)"

            # Check if it's default/recommended
            traits = model.model_spec.traits if hasattr(model.model_spec, "traits") else []
            suffix = ""
            if "default" in traits:
                suffix = " [DEFAULT]"
            elif "fastest" in traits:
                suffix = " [FAST]"
            elif "best" in traits:
                suffix = " [BEST]"

            choice_text = f"{model.id}{context}{suffix}"
            choices.append(choice_text)

        # Let user select
        selected = await asyncio.to_thread(
            lambda: questionary.select("Select a model:", choices=choices).ask()
        )

        if selected:
            # Extract model ID from selection
            model_id: str = selected.split(" (")[0].split(" [")[0]
            return model_id

        return None

    except Exception as e:
        print_error(f"Failed to fetch models: {e}")
        return None


async def _chat_async(
    ctx: click.Context,
    model: str | None,
    select_model: bool,
    system_prompt: str | None,
    temperature: float | None,
    max_completion_tokens: int | None,
    stream: bool,
    show_thinking: bool,
    animation: str,
    animation_speed: float,
    show_stats: bool,
    initial_message: str | None,
    *,
    top_p: float | None = None,
    venice_params: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    json_output: bool = False,
    save_conversation_flag: bool = False,
    continue_id: str | None = None,
) -> None:
    """Async implementation of chat command"""

    # Check if there's piped input (stdin support)
    try:
        if not sys.stdin.isatty():
            piped_input = sys.stdin.read().strip()
            if piped_input:
                if initial_message:
                    initial_message = f"{initial_message}\n\n{piped_input}"
                else:
                    initial_message = piped_input
    except (OSError, ValueError):
        # stdin may not be readable in some environments (e.g., pytest capture)
        pass

    # Get config
    config = ctx.obj.get("config", load_config())
    plain = ctx.obj.get("plain", False) or is_plain_mode()

    # Set defaults from config
    if temperature is None:
        temperature = config["defaults"]["temperature"]
    if max_completion_tokens is None:
        max_completion_tokens = config["defaults"]["max_completion_tokens"]

    # Convert animation string to enum
    animation_mode = AnimationMode(animation)

    # Initialize Venice client
    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            # Handle model selection. ``model_was_explicit`` records whether the
            # user chose a model (via --model or the picker) so --continue-from
            # only inherits a saved conversation's model when they did not.
            model_was_explicit = model is not None
            if select_model:
                model = await _select_chat_model(client)
                if not model:
                    print_error("No model selected. Exiting.")
                    raise SystemExit(1)
                model_was_explicit = True
                print_success(f"Selected model: {model}")
            else:
                model = await resolve_default_model(client, config, "chat", explicit=model)

            # Create message history
            messages: list[Any] = []

            # Generate a conversation ID for this session
            conv_id = uuid.uuid4().hex[:8]

            # Handle --continue-from: load a previous conversation
            if continue_id:
                resolved_id = continue_id
                if continue_id == "last":
                    resolved_id = get_last_conversation_id() or conv_id
                prev = load_conversation(resolved_id)
                if prev:
                    conv_id = resolved_id
                    # Re-use the model from the saved conversation if not overridden
                    if not model_was_explicit and prev.get("model"):
                        # ``prev`` is dict[str, Any]; the truthy guard above means
                        # ``prev["model"]`` is a saved (non-None) model id. cast keeps
                        # ``model`` typed as ``str`` so the resolved value isn't
                        # re-widened to ``str | None`` at the call sites below.
                        model = cast(str, prev["model"])
                    # Restore messages as plain dicts (API accepts them)
                    for msg in prev.get("messages", []):
                        messages.append(msg)
                    if plain:
                        click.echo(
                            f"Continuing conversation '{prev.get('title', resolved_id)}' "
                            f"({len(messages)} messages loaded)"
                        )
                    else:
                        print_info(
                            f"Continuing conversation '[bold]{prev.get('title', resolved_id)}[/bold]' "
                            f"({len(messages)} messages loaded)"
                        )
                else:
                    print_error(f"Conversation '{continue_id}' not found.")
                    raise SystemExit(1)

            # Add system message if provided (after restoring history)
            if system_prompt and not any(
                (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "system"
                for m in messages
            ):
                messages.append(SystemMessage(role="system", content=system_prompt, name=None))

            # Single message mode
            if initial_message:
                await _send_single_message(
                    client,
                    messages,
                    initial_message,
                    model,
                    temperature or config["defaults"]["temperature"],
                    max_completion_tokens or config["defaults"]["max_completion_tokens"],
                    stream,
                    show_thinking,
                    animation_mode,
                    animation_speed,
                    show_stats,
                    plain,
                    top_p=top_p,
                    venice_params=venice_params,
                    reasoning_effort=reasoning_effort,
                    json_output=json_output,
                )
                # Save conversation if requested (mirrors interactive path at end of _interactive_chat)
                if save_conversation_flag and conv_id and messages:
                    filepath = save_conversation(conv_id, model, messages)
                    if plain:
                        click.echo(f"Conversation saved (ID: {conv_id})")
                    else:
                        print_success(
                            f"Conversation saved (ID: [bold]{conv_id}[/bold]) → {filepath}"
                        )
                return

            # Interactive mode
            if plain:
                click.echo(f"Starting chat session with {model}")
                if system_prompt:
                    click.echo(f"System prompt: {system_prompt[:50]}...")
                click.echo("Type 'exit' or 'quit' to end the session\n")
            else:
                print_info(f"Starting chat session with {model}")
                if system_prompt:
                    console.print(f"[dim]System prompt: {system_prompt[:50]}...[/dim]")
                if show_thinking:
                    console.print("[dim]Thinking/reasoning blocks will be shown[/dim]")
                if animation != "smooth":
                    console.print(f"[dim]Animation: {animation} (speed: {animation_speed}s)[/dim]")
                console.print("[dim]Type 'exit' or 'quit' to end the session[/dim]\n")

            await _interactive_chat(
                client,
                messages,
                model,
                temperature or config["defaults"]["temperature"],
                max_completion_tokens or config["defaults"]["max_completion_tokens"],
                stream,
                show_thinking,
                animation_mode,
                animation_speed,
                show_stats,
                plain,
                top_p=top_p,
                venice_params=venice_params,
                reasoning_effort=reasoning_effort,
                save_conversation_flag=save_conversation_flag,
                conv_id=conv_id,
            )

    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


async def _send_single_message(
    client: VeniceClient,
    messages: list[Any],
    user_message: str,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    stream: bool,
    show_thinking: bool,
    animation_mode: AnimationMode,
    animation_speed: float,
    show_stats: bool,
    plain: bool = False,
    *,
    top_p: float | None = None,
    venice_params: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    json_output: bool = False,
) -> None:
    """Send a single message and display the response.

    Appends both the user message and assistant response to ``messages``
    so that callers can inspect or save the full exchange.
    """

    # Add user message
    messages.append(UserMessage(role="user", content=user_message))

    # Display user message
    if plain:
        click.echo(f"You: {user_message}")
    else:
        console.print(Panel(user_message, title="[bold cyan]You[/bold cyan]", border_style="cyan"))

    # Build optional API kwargs for venice_parameters and top_p
    extra_kwargs: dict[str, Any] = {}
    if top_p is not None:
        extra_kwargs["top_p"] = top_p
    if venice_params is not None:
        extra_kwargs["venice_parameters"] = venice_params
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort

    # Get response
    if stream:
        # Stream the response with animation
        stream_handler = StreamHandler(console, animation_mode, animation_speed, plain=plain)

        with stream_handler.display_progress("Thinking...") as progress:
            response_stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                stream=True,
                **extra_kwargs,
            )
            progress.update("")  # Clear progress message

        if plain:
            click.echo("Assistant:")
        else:
            console.print("\n[bold green]Assistant:[/bold green]")
        content, stats = await stream_handler.handle_chat_stream(
            response_stream.__aiter__(), show_stats
        )

        # Extract and display thinking blocks
        thinking_blocks, clean_content = extract_thinking_blocks(content)
        _display_thinking(thinking_blocks, show_thinking, plain, is_streaming=True)

        # Show streaming statistics if requested
        if show_stats and stats:
            if plain:
                _display_stream_stats(stats, plain=True)
            else:
                console.print()
                _display_stream_stats(stats)

        # Append assistant response to messages for callers (e.g. --save)
        _append_to_history(messages, content, thinking_blocks, clean_content)

    else:
        # Non-streaming response
        response = await _make_nonstreaming_request(
            client,
            model,
            messages,
            temperature,
            max_completion_tokens,
            extra_kwargs,
            show_spinner=not plain and not json_output,
        )

        # JSON output mode: dump the full response and return early
        if json_output:
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response.dict()
            )
            click.echo(json.dumps(response_dict, indent=2, default=str))
            return

        raw_content = response.choices[0].message.content or ""
        content = raw_content if isinstance(raw_content, str) else str(raw_content)
        reasoning_content = _get_reasoning_content(response.choices[0].message)

        # Extract thinking blocks from content
        thinking_blocks, clean_content = extract_thinking_blocks(content)

        # Show thinking/reasoning if requested
        _display_thinking(
            thinking_blocks,
            show_thinking,
            plain,
            reasoning_content=reasoning_content,
        )

        # Display assistant response
        display_content = clean_content if thinking_blocks else content
        _display_nonstreaming_content(display_content, plain)

        # Show token usage (only available for non-streaming)
        usage = response.usage
        if usage is not None:
            if plain:
                click.echo(
                    f"Tokens: {usage.prompt_tokens} input | "
                    f"{usage.completion_tokens} output | {usage.total_tokens} total"
                )
            else:
                console.print(
                    f"\n[dim]Tokens: {usage.prompt_tokens} input | "
                    f"{usage.completion_tokens} output | {usage.total_tokens} total[/dim]"
                )

        # Append assistant response to messages for callers (e.g. --save)
        _append_to_history(
            messages,
            content,
            thinking_blocks,
            clean_content,
            reasoning_content=reasoning_content,
        )


async def _interactive_chat(
    client: VeniceClient,
    messages: list[Any],
    model: str,
    temperature: float,
    max_completion_tokens: int,
    stream: bool,
    show_thinking: bool,
    animation_mode: AnimationMode,
    animation_speed: float,
    show_stats: bool,
    plain: bool = False,
    *,
    top_p: float | None = None,
    venice_params: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    save_conversation_flag: bool = False,
    conv_id: str | None = None,
) -> None:
    """Run an interactive chat session"""

    # Build optional API kwargs for venice_parameters and top_p
    extra_kwargs: dict[str, Any] = {}
    if top_p is not None:
        extra_kwargs["top_p"] = top_p
    if venice_params is not None:
        extra_kwargs["venice_parameters"] = venice_params
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort

    stream_handler = StreamHandler(console, animation_mode, animation_speed, plain=plain)

    while True:
        # Get user input
        try:
            if plain:
                user_input = await asyncio.to_thread(lambda: input("> "))
            else:
                user_input = await asyncio.to_thread(
                    lambda: questionary.text("You:", qmark="💬").ask()
                )

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                if plain:
                    click.echo("Ending chat session. Goodbye!")
                else:
                    print_info("Ending chat session. Goodbye!")
                break

            # Add user message to history
            messages.append(UserMessage(role="user", content=user_input))

            # Get and display response
            if stream:
                # Stream the response with animation
                if plain:
                    click.echo("Assistant:")
                else:
                    console.print("\n[bold green]Assistant:[/bold green]")
                response_stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    stream=True,
                    **extra_kwargs,
                )

                content, stats = await stream_handler.handle_chat_stream(
                    response_stream.__aiter__(), show_stats
                )

                # Extract and display thinking blocks
                thinking_blocks, clean_content = extract_thinking_blocks(content)
                _display_thinking(thinking_blocks, show_thinking, plain, is_streaming=True)

                # Show streaming statistics if requested
                if show_stats and stats:
                    if plain:
                        _display_stream_stats(stats, plain=True)
                    else:
                        console.print()
                        _display_stream_stats(stats)

                if not plain:
                    console.print()  # Add spacing

                # Streaming has no reasoning_content
                reasoning_to_store = None

            else:
                # Non-streaming response
                response = await _make_nonstreaming_request(
                    client,
                    model,
                    messages,
                    temperature,
                    max_completion_tokens,
                    extra_kwargs,
                    show_spinner=not plain,
                )

                raw_content = response.choices[0].message.content or ""
                content = raw_content if isinstance(raw_content, str) else str(raw_content)
                reasoning_content = _get_reasoning_content(response.choices[0].message)

                # Extract thinking blocks from content
                thinking_blocks, clean_content = extract_thinking_blocks(content)

                # Show thinking/reasoning if requested
                _display_thinking(
                    thinking_blocks,
                    show_thinking,
                    plain,
                    reasoning_content=reasoning_content,
                )

                # Display assistant response
                display_content = clean_content if thinking_blocks else content
                _display_nonstreaming_content(display_content, plain)
                if not plain:
                    console.print()  # Add spacing

                # Show token usage for non-streaming
                usage = response.usage
                if usage:
                    if plain:
                        click.echo(
                            f"Tokens: {usage.prompt_tokens} input | "
                            f"{usage.completion_tokens} output | "
                            f"{usage.total_tokens} total"
                        )
                    else:
                        console.print(
                            f"[dim]Tokens: {usage.prompt_tokens} input | "
                            f"{usage.completion_tokens} output | "
                            f"{usage.total_tokens} total[/dim]\n"
                        )

                # Store reasoning content for history (only for non-streaming)
                reasoning_to_store = reasoning_content

            # Add assistant response to history
            _append_to_history(
                messages,
                content,
                thinking_blocks,
                clean_content,
                reasoning_content=reasoning_to_store,
            )

        except KeyboardInterrupt:
            print_info("\nChat session interrupted. Goodbye!")
            break
        except VeniceError as e:
            print_error(f"API error: {e}")
            # Continue the session despite API errors
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            break

    # Save conversation if requested
    if save_conversation_flag and conv_id and messages:
        filepath = save_conversation(conv_id, model, messages)
        if plain:
            click.echo(f"Conversation saved (ID: {conv_id})")
        else:
            print_success(f"Conversation saved (ID: [bold]{conv_id}[/bold]) → {filepath}")


def _display_stream_stats(stats: dict[str, Any], plain: bool = False) -> None:
    """Display streaming statistics in a formatted table"""
    if plain:
        click.echo("--- Streaming Statistics ---")
        if "total_chunks" in stats:
            click.echo(f"  Total chunks: {stats['total_chunks']}")
        if "content_length" in stats:
            click.echo(f"  Content length: {stats['content_length']} chars")
        if "stream_duration" in stats:
            click.echo(f"  Stream duration: {stats['stream_duration']}s")
        if "time_to_first_token" in stats:
            click.echo(f"  Time to first token: {stats['time_to_first_token']}s")
        if "chunks_per_second" in stats:
            click.echo(f"  Chunks per second: {stats['chunks_per_second']}")
        if "finish_reason" in stats:
            click.echo(f"  Finish reason: {stats['finish_reason']}")
        return

    table = Table(title="[dim]Streaming Statistics[/dim]", show_header=False, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="yellow")

    # Add rows for available stats
    if "total_chunks" in stats:
        table.add_row("Total chunks", str(stats["total_chunks"]))
    if "content_length" in stats:
        table.add_row("Content length", f"{stats['content_length']} chars")
    if "stream_duration" in stats:
        table.add_row("Stream duration", f"{stats['stream_duration']}s")
    if "time_to_first_token" in stats:
        table.add_row("Time to first token", f"{stats['time_to_first_token']}s")
    if "chunks_per_second" in stats:
        table.add_row("Chunks per second", str(stats["chunks_per_second"]))
    if "finish_reason" in stats:
        table.add_row("Finish reason", stats["finish_reason"])

    console.print(table)


@chat.command("history")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output as JSON")
@click.option("--delete", "delete_id", default=None, help="Delete a conversation by ID")
@click.pass_context
def chat_history(ctx: click.Context, json_output: bool, delete_id: str | None) -> None:
    """View and manage saved conversations.

    Examples:

        # List all saved conversations
        venice-py chat history

        # Output as JSON
        venice-py chat history --json

        # Delete a saved conversation
        venice-py chat history --delete <id>
    """
    plain = ctx.obj.get("plain", False) if ctx.obj else False

    # Handle deletion
    if delete_id:
        from ..conversation import delete_conversation as _delete

        deleted = _delete(delete_id)
        if deleted:
            if plain:
                click.echo(f"Deleted conversation: {delete_id}")
            else:
                print_success(f"Deleted conversation: [bold]{delete_id}[/bold]")
        else:
            print_error(f"Conversation '{delete_id}' not found.")
            raise SystemExit(1)
        return

    conversations = list_conversations()

    if json_output:
        click.echo(json.dumps(conversations, indent=2))
        return

    if not conversations:
        if plain:
            click.echo("No saved conversations.")
        else:
            print_info("No saved conversations found.")
        return

    if plain:
        for conv in conversations:
            updated = conv.get("updated_at", "")[:19].replace("T", " ")
            click.echo(
                f"{conv['id']}  {updated}  {conv.get('model', '')}  {conv.get('title', 'Untitled')}"
            )
    else:
        table = Table(title="Saved Conversations", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="yellow", no_wrap=True)
        table.add_column("Updated", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Title")

        for conv in conversations:
            updated = conv.get("updated_at", "")[:19].replace("T", " ")
            table.add_row(
                conv.get("id", ""),
                updated,
                conv.get("model", ""),
                conv.get("title", "Untitled"),
            )

        console.print(table)
        console.print(
            "\n[dim]Use [bold]venice-py chat start --continue-from <id>[/bold] to resume a conversation.[/dim]"
        )
