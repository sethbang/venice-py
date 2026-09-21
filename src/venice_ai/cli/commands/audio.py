"""Audio commands for Venice AI CLI — Text-to-Speech and Speech-to-Text."""

import asyncio
import json
import os
from datetime import datetime

import click

from venice_ai.cli.utils.console import console


@click.group()
def audio():
    """Audio generation and transcription.

    Text-to-speech and speech-to-text capabilities.
    """
    pass


@audio.command("speak")
@click.argument("text", required=False)
@click.option("--model", "-m", default=None, help="TTS model to use (defaults to API-recommended)")
@click.option("--voice", "-v", default="af_heart", help="Voice to use")
@click.option(
    "--format",
    "audio_format",
    type=click.Choice(["mp3", "wav", "opus", "flac", "aac", "pcm"]),
    default="mp3",
    help="Output audio format",
)
@click.option("--speed", type=float, default=1.0, help="Speech speed (0.25-4.0)")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (default: speech_<timestamp>.<format>)",
)
@click.option("--save-dir", default=".", help="Directory to save audio file")
@click.pass_context
def speak(ctx, text, model, voice, audio_format, speed, output, save_dir):
    """Convert text to speech.

    Examples:
        venice-py audio speak "Hello, world!"
        venice-py audio speak "Welcome to Venice" --voice af_heart --format wav
        venice-py audio speak --model <tts-model> --output greeting.mp3 "Hi there"
        echo "Some text" | venice-py audio speak
    """
    asyncio.run(_speak_async(ctx, text, model, voice, audio_format, speed, output, save_dir))


async def _speak_async(ctx, text, model, voice, audio_format, speed, output, save_dir):
    import sys

    from venice_ai import VeniceClient
    from venice_ai.cli._model_defaults import resolve_default_model
    from venice_ai.cli.config import get_client_kwargs, load_config

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    config = ctx.obj.get("config", load_config()) if ctx.obj else load_config()

    # Support piped input
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        if not text:
            raise click.ClickException("Text is required. Provide as argument or pipe via stdin.")

    # Determine output path
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"speech_{timestamp}.{audio_format}"
    elif not output.endswith(f".{audio_format}"):
        output = f"{output}.{audio_format}"

    save_path = os.path.join(save_dir, output)
    os.makedirs(save_dir, exist_ok=True)

    async with VeniceClient(**get_client_kwargs()) as client:
        model = await resolve_default_model(client, config, "tts", explicit=model)

        if not plain:
            console.print("[bold blue]🔊 Generating speech...[/bold blue]")
            console.print(f"  Model: {model}")
            console.print(f"  Voice: {voice}")
            console.print(f"  Format: {audio_format}")
        else:
            click.echo(f"Generating speech with {model} (voice: {voice})...")

        response = await client.audio.create_speech(
            model=model,
            input=text,
            voice=voice,
            response_format=audio_format,
            speed=speed,
        )

        # AudioResponse has a .content attribute containing the raw bytes
        with open(save_path, "wb") as f:
            f.write(response.content)

    if not plain:
        console.print(f"\n[bold green]✅ Audio saved to:[/bold green] {save_path}")
        file_size = os.path.getsize(save_path)
        console.print(f"  Size: {file_size / 1024:.1f} KB")
    else:
        click.echo(f"Saved: {save_path}")


@audio.command("transcribe")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    default=None,
    help="STT model to use (defaults to API-recommended)",
)
@click.option("--language", "-l", default=None, help="Language code (e.g., en, es, fr)")
@click.option("--timestamps", is_flag=True, default=False, help="Include word-level timestamps")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("--output", "-o", default=None, help="Save transcription to file")
@click.pass_context
def transcribe(ctx, file, model, language, timestamps, output_format, output):
    """Transcribe audio to text.

    Examples:
        venice-py audio transcribe recording.mp3
        venice-py audio transcribe meeting.wav --model <stt-model>
        venice-py audio transcribe audio.mp3 --language en --timestamps
        venice-py audio transcribe audio.mp3 --output transcript.txt
    """
    asyncio.run(_transcribe_async(ctx, file, model, language, timestamps, output_format, output))


async def _transcribe_async(ctx, file, model, language, timestamps, output_format, output):
    from venice_ai import VeniceClient
    from venice_ai.cli._model_defaults import resolve_default_model
    from venice_ai.cli.config import get_client_kwargs, load_config

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    config = ctx.obj.get("config", load_config()) if ctx.obj else load_config()

    async with VeniceClient(**get_client_kwargs()) as client:
        model = await resolve_default_model(client, config, "stt", explicit=model)

        if not plain:
            console.print(f"[bold blue]🎤 Transcribing:[/bold blue] {file}")
            console.print(f"  Model: {model}")
        else:
            click.echo(f"Transcribing {file} with {model}...")

        kwargs = {
            "model": model,
            "file": file,
        }
        if language:
            kwargs["language"] = language
        if timestamps:
            kwargs["timestamps"] = True
        if output_format != "text":
            kwargs["response_format"] = output_format

        response = await client.audio.transcribe(**kwargs)

    # AudioTranscriptionResponse has a .text attribute
    result = response.text

    # Output result
    if output:
        with open(output, "w") as f:
            f.write(result)
        if not plain:
            console.print(f"\n[bold green]✅ Transcription saved to:[/bold green] {output}")
        else:
            click.echo(f"Saved: {output}")
    else:
        if not plain:
            console.print("\n[bold green]📝 Transcription:[/bold green]")
            console.print(result)
        else:
            click.echo(result)

    # Show word-level timestamps if available
    if timestamps and hasattr(response, "words") and response.words:
        if not plain:
            console.print("\n[bold blue]⏱️ Word Timestamps:[/bold blue]")
            for word_info in response.words:
                start = f"{word_info.start:.2f}s" if word_info.start is not None else "N/A"
                end = f"{word_info.end:.2f}s" if word_info.end is not None else "N/A"
                console.print(f"  {word_info.word:<20} {start} → {end}")
        else:
            click.echo("\nWord Timestamps:")
            for word_info in response.words:
                start = f"{word_info.start:.2f}s" if word_info.start is not None else "N/A"
                end = f"{word_info.end:.2f}s" if word_info.end is not None else "N/A"
                click.echo(f"  {word_info.word:<20} {start} -> {end}")


@audio.command("voices")
@click.option(
    "--model",
    "-m",
    "model_id",
    default=None,
    help="Filter voices by TTS model ID (e.g. tts-kokoro). Default: all TTS models.",
)
@click.option(
    "--gender",
    type=click.Choice(["male", "female", "unknown"], case_sensitive=False),
    default=None,
    help="Filter voices by gender.",
)
@click.option(
    "--region",
    "region_code",
    default=None,
    help="Filter voices by region code (e.g. af, bf, zm).",
)
@click.option("--json", "output_json", is_flag=True, help="Output JSON for scripting.")
@click.pass_context
def voices(ctx, model_id, gender, region_code, output_json):
    """List available text-to-speech voices.

    Shows voice IDs, associated TTS models, gender, region/language, and
    accent information. Supports optional filtering by model, gender, or
    region code.

    Examples:

      venice-py audio voices
      venice-py audio voices --gender female
      venice-py audio voices --region af --json
      venice-py audio voices --model tts-kokoro
    """
    asyncio.run(_voices_async(ctx, model_id, gender, region_code, output_json))


async def _voices_async(ctx, model_id, gender, region_code, output_json):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    async with VeniceClient(**get_client_kwargs()) as client:
        voice_list = await client.audio.get_voices(
            model_id=model_id,
            gender=gender.lower() if gender else None,
            region_code=region_code,
        )

    voices_data = getattr(voice_list, "data", []) or []

    if output_json:
        if hasattr(voice_list, "model_dump"):
            click.echo(json.dumps(voice_list.model_dump(), default=str))
        else:
            payload = [v.model_dump() if hasattr(v, "model_dump") else dict(v) for v in voices_data]
            click.echo(json.dumps({"data": payload}, default=str))
        return

    if not voices_data:
        msg = "No voices found."
        if plain:
            click.echo(msg)
        else:
            console.print(f"[yellow]{msg}[/yellow]")
        return

    if plain:
        click.echo(f"{'ID':<20} {'MODEL':<18} {'GENDER':<8} {'REGION':<8} {'LANGUAGE':<25}")
        click.echo("-" * 85)
        for v in voices_data:
            click.echo(
                f"{getattr(v, 'id', ''):<20} "
                f"{getattr(v, 'model_id', '') or '':<18} "
                f"{(getattr(v, 'gender', '') or ''):<8} "
                f"{(getattr(v, 'region_code', '') or ''):<8} "
                f"{(getattr(v, 'language', '') or ''):<25}"
            )
        click.echo(f"\nTotal: {len(voices_data)} voice(s)")
        return

    from rich.table import Table

    table = Table(title="Venice AI Voices", show_lines=False)
    table.add_column("Voice ID", style="bold cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Gender", style="magenta")
    table.add_column("Region", style="dim")
    table.add_column("Language")
    table.add_column("Accent", style="dim")

    for v in voices_data:
        table.add_row(
            getattr(v, "id", "") or "",
            getattr(v, "model_id", "") or "",
            getattr(v, "gender", "") or "",
            getattr(v, "region_code", "") or "",
            getattr(v, "language", "") or "",
            getattr(v, "accent", "") or "",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(voices_data)} voice(s)[/dim]")


@audio.command("voice-change")
@click.argument("source", required=False, type=click.Path(exists=True))
@click.option(
    "--url",
    "audio_url",
    default=None,
    help="Public http(s) URL of the source recording, instead of a local file.",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Voice-changer model (defaults to models.resolve_voice_changer()).",
)
@click.option("--voice", "-v", default=None, help="Target voice (defaults to the model default).")
@click.option(
    "--remove-background-noise",
    is_flag=True,
    default=False,
    help="Strip background noise before converting.",
)
@click.option("--seed", type=int, default=None, help="Seed for reproducible output.")
@click.option("--output", "-o", default=None, help="Where to write the converted audio.")
@click.option(
    "--quote-only",
    is_flag=True,
    default=False,
    help="Price a conversion of --duration seconds and exit without converting.",
)
@click.option(
    "--duration",
    type=int,
    default=None,
    help="Source length in whole seconds, for --quote-only.",
)
@click.option("--poll-interval", type=float, default=3.0, help="Seconds between status polls.")
@click.pass_context
def voice_change(
    ctx,
    source,
    audio_url,
    model,
    voice,
    remove_background_noise,
    seed,
    output,
    quote_only,
    duration,
    poll_interval,
):
    """Convert a recording into a different voice.

    Timing is preserved, so the source length is what you are billed for.

    Examples:
        venice-py audio voice-change recording.mp3 -o converted.mp3
        venice-py audio voice-change recording.mp3 --voice Aria
        venice-py audio voice-change --url https://example.com/clip.mp3 -o out.mp3
        venice-py audio voice-change --quote-only --duration 90
    """
    asyncio.run(
        _voice_change_async(
            ctx,
            source,
            audio_url,
            model,
            voice,
            remove_background_noise,
            seed,
            output,
            quote_only,
            duration,
            poll_interval,
        )
    )


async def _voice_change_async(
    ctx,
    source,
    audio_url,
    model,
    voice,
    remove_background_noise,
    seed,
    output,
    quote_only,
    duration,
    poll_interval,
):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    if quote_only:
        if duration is None:
            raise click.UsageError("--quote-only requires --duration <seconds>.")
    elif bool(source) == bool(audio_url):
        raise click.UsageError("Provide exactly one of SOURCE or --url.")

    async with VeniceClient(**get_client_kwargs()) as client:
        if model is None:
            try:
                model = await client.models.resolve_voice_changer()
            except ValueError as e:
                # Not an error in the CLI's own logic — the capability simply
                # is not enabled for this account.
                raise click.ClickException(str(e)) from e

        if quote_only:
            quote = await client.voice_changer.quote(model=model, duration_seconds=duration)
            if plain:
                click.echo(f"{quote.quote}")
            else:
                console.print(f"[bold blue]💱 Quote:[/bold blue] ${quote.quote:.4f} USD")
                console.print(f"  Model: {model}")
                console.print(f"  Source length: {quote.duration_seconds:.0f}s")
                console.print(
                    "  [dim]Estimate — you are billed on the length Venice measures.[/dim]"
                )
            return

        out_path = output or "converted.mp3"
        if not plain:
            console.print(f"[bold blue]🎚️  Converting:[/bold blue] {source or audio_url}")
            console.print(f"  Model: {model}")

        async with await client.voice_changer.run(
            model=model,
            file=source,
            audio_url=audio_url,
            voice=voice,
            remove_background_noise=remove_background_noise or None,
            seed=seed,
        ) as job:
            if not plain:
                console.print(f"  Queued: {job.queue_id} ({job.duration_seconds:.0f}s billed)")
            status = await job.wait(poll_interval=poll_interval)
            saved = await job.download(out_path, status)

    if plain:
        click.echo(f"Saved: {saved}")
    else:
        console.print(f"\n[bold green]✅ Saved:[/bold green] {saved}")
