"""Video commands for Venice AI CLI — Text-to-Video and Image-to-Video."""

import asyncio
import base64
import os
from datetime import datetime
from pathlib import Path

import click

from venice_ai.cli.utils.console import open_file
from venice_ai.cli.utils.output import OutputManager

# Default poll settings
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_MAX_POLLS = 120  # 120 × 5 s ≈ 10 min


@click.group()
def video():
    """Video generation from text or images.

    Generate videos using text prompts or animate existing images.
    Video generation is asynchronous — jobs are queued and polled until complete.
    """
    pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _image_file_to_data_uri(path: str) -> str:
    """Read a local image file and return it as a base64 data URI."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")
    data = file_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


async def _poll_and_save(
    client,
    *,
    model: str,
    queue_id: str,
    output_path: str,
    plain: bool,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    max_polls: int = _DEFAULT_MAX_POLLS,
) -> bool:
    """Poll for video completion, download the result, and save to disk.

    Returns True on success, False on failure or timeout.
    """
    import aiohttp

    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    if plain and not is_plain_mode():
        enable_plain_mode()

    for _poll_num in range(max_polls):
        status = await client.video.retrieve(model=model, queue_id=queue_id)

        if status.status == "COMPLETED":
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            if status.data:
                with open(output_path, "wb") as f:
                    f.write(status.data)
                file_size = len(status.data)
                OutputManager.success(f"Video saved: {output_path} ({file_size / 1024:.1f} KB)")
            elif status.url:
                OutputManager.echo("Downloading from URL...")
                async with aiohttp.ClientSession() as session, session.get(status.url) as resp:
                    resp.raise_for_status()
                    data = await resp.read()

                with open(output_path, "wb") as f:
                    f.write(data)
                file_size = len(data)
                OutputManager.success(f"Video saved: {output_path} ({file_size / 1024:.1f} KB)")

                try:
                    complete_resp = await client.video.cancel(model=model, queue_id=queue_id)
                    OutputManager.echo(
                        f"Server cleanup: {'OK' if complete_resp.success else 'skipped'}"
                    )
                except Exception as e:
                    # Server-side cleanup is best-effort; log and continue.
                    import logging

                    logging.getLogger(__name__).debug(
                        "video.cancel(queue_id=%s) failed: %s", queue_id, e
                    )
            else:
                OutputManager.warning("Video completed but no data or URL returned.")
                return False

            return True

        elif status.status == "FAILED":
            error_msg = getattr(status, "error", None) or "unknown error"
            OutputManager.error(f"Generation failed: {error_msg}")
            return False

        else:
            progress = getattr(status, "progress_percent", 0) or 0
            remaining_ms = getattr(status, "estimated_remaining_ms", None)
            remaining_s = remaining_ms / 1000 if remaining_ms else None

            if remaining_s is not None:
                OutputManager.progress(f"complete (~{remaining_s:.0f}s remaining)", pct=progress)
            else:
                OutputManager.progress("complete...", pct=progress)

            await asyncio.sleep(poll_interval)

    OutputManager.warning("Timed out waiting for video generation.")
    return False


def _determine_output_path(output: str | None, save_dir: str, ext: str = "mp4") -> str:
    """Resolve the final output file path."""
    if output:
        # If output doesn't have extension, add it
        if not Path(output).suffix:
            output = f"{output}.{ext}"
        # If it's a bare filename (no dir separators), put it in save_dir
        if os.sep not in output and "/" not in output:
            return os.path.join(save_dir, output)
        return output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}.{ext}"
        return os.path.join(save_dir, filename)


# ---------------------------------------------------------------------------
# venice-py video generate
# ---------------------------------------------------------------------------


@video.command("generate")
@click.argument("prompt")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Video model to use (defaults to API-recommended)",
)
@click.option(
    "--duration",
    default="5s",
    show_default=True,
    help="Duration of the generated video (e.g. 5s, 10s)",
)
@click.option(
    "--resolution",
    default=None,
    help="Output resolution (e.g. 720p, 1080p)",
)
@click.option(
    "--aspect-ratio",
    default="16:9",
    show_default=True,
    help="Aspect ratio (e.g. 16:9, 9:16, 1:1)",
)
@click.option(
    "--negative-prompt",
    default=None,
    help="Negative prompt — content to avoid in the video",
)
@click.option(
    "--audio/--no-audio",
    "audio",
    default=None,
    help="Generate audio if the model supports it (omitted unless set)",
)
@click.option(
    "--reference-image-urls",
    multiple=True,
    help="Reference image URL for character/style consistency (repeatable, up to 9)",
)
@click.option(
    "--reference-video-urls",
    multiple=True,
    help="Reference video URL for R2V models (repeatable, up to 3)",
)
@click.option(
    "--reference-audio-urls",
    multiple=True,
    help="Reference audio donor URL for R2V models (repeatable, up to 3)",
)
@click.option(
    "--end-image-url",
    default=None,
    help="End-frame image URL for models that support transitions",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (default: video_<timestamp>.mp4)",
)
@click.option(
    "--save-dir",
    default=".",
    show_default=True,
    help="Directory to save the video file",
)
@click.option(
    "--no-poll",
    is_flag=True,
    default=False,
    help="Queue the job and return the job ID without waiting for completion",
)
@click.option(
    "--open",
    "auto_open",
    is_flag=True,
    default=False,
    help="Open the video after saving",
)
@click.pass_context
def generate(
    ctx,
    prompt,
    model,
    duration,
    resolution,
    aspect_ratio,
    negative_prompt,
    audio,
    reference_image_urls,
    reference_video_urls,
    reference_audio_urls,
    end_image_url,
    output,
    save_dir,
    no_poll,
    auto_open,
):
    """Generate a video from a text prompt.

    Examples:

      venice-py video generate "A sunset over the ocean with gentle waves"

      venice-py video generate "A cat playing piano" --duration 10s --model <video-model>

      venice-py video generate "Cinematic drone shot" --aspect-ratio 16:9 --resolution 1080p

      venice-py video generate "Quick preview" --no-poll
    """
    asyncio.run(
        _generate_async(
            ctx,
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            negative_prompt=negative_prompt,
            output=output,
            save_dir=save_dir,
            no_poll=no_poll,
            auto_open=auto_open,
            audio=audio,
            reference_image_urls=reference_image_urls,
            reference_video_urls=reference_video_urls,
            reference_audio_urls=reference_audio_urls,
            end_image_url=end_image_url,
        )
    )


async def _generate_async(
    ctx,
    *,
    prompt,
    model,
    duration,
    resolution,
    aspect_ratio,
    negative_prompt,
    output,
    save_dir,
    no_poll,
    auto_open,
    audio=None,
    reference_image_urls=None,
    reference_video_urls=None,
    reference_audio_urls=None,
    end_image_url=None,
):
    from venice_ai import VeniceClient
    from venice_ai.cli._model_defaults import resolve_default_model
    from venice_ai.cli.config import get_client_kwargs, load_config
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    config = ctx.obj.get("config", load_config()) if ctx.obj else load_config()
    if plain and not is_plain_mode():
        enable_plain_mode()

    async with VeniceClient(**get_client_kwargs()) as client:
        model = await resolve_default_model(client, config, "video_t2v", explicit=model)

        OutputManager.info(f"Queuing video generation with {model}...")
        OutputManager.echo(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        OutputManager.echo(f"Duration: {duration}, Aspect Ratio: {aspect_ratio}")

        # Build queue kwargs
        queue_kwargs: dict = {
            "model": model,
            "prompt": prompt,
            "duration_seconds": duration,
        }
        if aspect_ratio:
            queue_kwargs["aspect_ratio"] = aspect_ratio
        if resolution:
            queue_kwargs["resolution"] = resolution
        if negative_prompt:
            queue_kwargs["negative_prompt"] = negative_prompt
        if audio is not None:
            queue_kwargs["audio"] = audio
        if reference_image_urls:
            queue_kwargs["reference_image_urls"] = list(reference_image_urls)
        if reference_video_urls:
            queue_kwargs["reference_video_urls"] = list(reference_video_urls)
        if reference_audio_urls:
            queue_kwargs["reference_audio_urls"] = list(reference_audio_urls)
        if end_image_url:
            queue_kwargs["end_image_url"] = end_image_url

        queue_resp = await client.video.submit(**queue_kwargs)
        queue_id = queue_resp.queue_id
        actual_model = queue_resp.model

        OutputManager.success(f"Job queued! Job ID: {queue_id}")

        if no_poll:
            OutputManager.echo(f"Use 'venice-py video status {queue_id}' to check progress.")
            return

        output_path = _determine_output_path(output, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        OutputManager.echo("Polling for completion...")

        success = await _poll_and_save(
            client,
            model=actual_model,
            queue_id=queue_id,
            output_path=output_path,
            plain=plain,
        )

        if success and auto_open:
            open_file(output_path)


# ---------------------------------------------------------------------------
# venice-py video from-image
# ---------------------------------------------------------------------------


@video.command("from-image")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--prompt",
    "-p",
    default="",
    help="Motion prompt to guide the video animation",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Image-to-video model to use (defaults to API-recommended)",
)
@click.option(
    "--duration",
    default="5s",
    show_default=True,
    help="Duration of the generated video (e.g. 5s, 10s)",
)
@click.option(
    "--resolution",
    default=None,
    help="Output resolution (e.g. 720p, 1080p)",
)
@click.option(
    "--audio/--no-audio",
    "audio",
    default=None,
    help="Generate audio if the model supports it (omitted unless set)",
)
@click.option(
    "--reference-image-urls",
    multiple=True,
    help="Reference image URL for character/style consistency (repeatable, up to 9)",
)
@click.option(
    "--reference-video-urls",
    multiple=True,
    help="Reference video URL for R2V models (repeatable, up to 3)",
)
@click.option(
    "--reference-audio-urls",
    multiple=True,
    help="Reference audio donor URL for R2V models (repeatable, up to 3)",
)
@click.option(
    "--end-image-url",
    default=None,
    help="End-frame image URL for models that support transitions",
)
@click.option(
    "--image-strength",
    type=click.FloatRange(0, 1),
    default=None,
    help=(
        "How strongly the input image guides the video, from 0 to 1, on models "
        "that declare image strength (omitted unless set; provider default 1)"
    ),
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (default: video_<timestamp>.mp4)",
)
@click.option(
    "--save-dir",
    default=".",
    show_default=True,
    help="Directory to save the video file",
)
@click.option(
    "--no-poll",
    is_flag=True,
    default=False,
    help="Queue the job and return the job ID without waiting for completion",
)
@click.option(
    "--open",
    "auto_open",
    is_flag=True,
    default=False,
    help="Open the video after saving",
)
@click.pass_context
def from_image(
    ctx,
    input_file,
    prompt,
    model,
    duration,
    resolution,
    audio,
    reference_image_urls,
    reference_video_urls,
    reference_audio_urls,
    end_image_url,
    image_strength,
    output,
    save_dir,
    no_poll,
    auto_open,
):
    """Animate an image into a video.

    Reads the image from INPUT_FILE (local path), encodes it as a base64
    data URI, and submits an image-to-video generation job.

    The prompt describes the desired **motion**, not the image content.

    Examples:

      venice-py video from-image photo.jpg

      venice-py video from-image photo.png --prompt "Gentle breeze through the trees"

      venice-py video from-image portrait.jpg --model <i2v-model> --duration 10s

      venice-py video from-image landscape.jpg --no-poll

      venice-py video from-image sketch.png --image-strength 0.6
    """
    asyncio.run(
        _from_image_async(
            ctx,
            input_file=input_file,
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            output=output,
            save_dir=save_dir,
            no_poll=no_poll,
            auto_open=auto_open,
            audio=audio,
            reference_image_urls=reference_image_urls,
            reference_video_urls=reference_video_urls,
            reference_audio_urls=reference_audio_urls,
            end_image_url=end_image_url,
            image_strength=image_strength,
        )
    )


async def _from_image_async(
    ctx,
    *,
    input_file,
    prompt,
    model,
    duration,
    resolution,
    output,
    save_dir,
    no_poll,
    auto_open,
    audio=None,
    reference_image_urls=None,
    reference_video_urls=None,
    reference_audio_urls=None,
    end_image_url=None,
    image_strength=None,
):
    from venice_ai import VeniceClient
    from venice_ai.cli._model_defaults import resolve_default_model
    from venice_ai.cli.config import get_client_kwargs, load_config
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    config = ctx.obj.get("config", load_config()) if ctx.obj else load_config()
    if plain and not is_plain_mode():
        enable_plain_mode()

    OutputManager.echo(f"Input: {input_file}")
    if prompt:
        OutputManager.echo(f"Motion prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    OutputManager.echo(f"Duration: {duration}")
    OutputManager.echo("Encoding image...")

    image_url = await _image_file_to_data_uri(input_file)

    async with VeniceClient(**get_client_kwargs()) as client:
        model = await resolve_default_model(client, config, "video_i2v", explicit=model)

        OutputManager.info(f"Preparing image-to-video with {model}...")
        effective_prompt = prompt if prompt else "Animate this image with natural motion"

        queue_kwargs: dict = {
            "model": model,
            "prompt": effective_prompt,
            "image_url": image_url,
            "duration_seconds": duration,
        }
        if resolution:
            queue_kwargs["resolution"] = resolution
        if audio is not None:
            queue_kwargs["audio"] = audio
        if reference_image_urls:
            queue_kwargs["reference_image_urls"] = list(reference_image_urls)
        if reference_video_urls:
            queue_kwargs["reference_video_urls"] = list(reference_video_urls)
        if reference_audio_urls:
            queue_kwargs["reference_audio_urls"] = list(reference_audio_urls)
        if end_image_url:
            queue_kwargs["end_image_url"] = end_image_url
        if image_strength is not None:
            queue_kwargs["image_strength"] = image_strength

        OutputManager.echo("Queuing job...")

        queue_resp = await client.video.submit(**queue_kwargs)
        queue_id = queue_resp.queue_id
        actual_model = queue_resp.model

        OutputManager.success(f"Job queued! Job ID: {queue_id}")

        if no_poll:
            OutputManager.echo(f"Use 'venice-py video status {queue_id}' to check progress.")
            return

        output_path = _determine_output_path(output, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        OutputManager.echo("Polling for completion...")

        success = await _poll_and_save(
            client,
            model=actual_model,
            queue_id=queue_id,
            output_path=output_path,
            plain=plain,
        )

        if success and auto_open:
            open_file(output_path)


# ---------------------------------------------------------------------------
# venice-py video status
# ---------------------------------------------------------------------------


@video.command("status")
@click.argument("job_id")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model used when the job was queued",
)
@click.pass_context
def status(ctx, job_id, model):
    """Check the status of a video generation job.

    JOB_ID is the queue ID returned when the job was submitted (e.g. with --no-poll).

    Examples:

      venice-py video status abc123

      venice-py video status abc123 --model <i2v-model>
    """
    asyncio.run(_status_async(ctx, job_id=job_id, model=model))


async def _status_async(ctx, *, job_id, model):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    if plain and not is_plain_mode():
        enable_plain_mode()

    async with VeniceClient(**get_client_kwargs()) as client:
        result = await client.video.retrieve(model=model, queue_id=job_id)

    status_val = result.status

    if status_val == "COMPLETED":
        url = getattr(result, "url", None)
        OutputManager.success(f"COMPLETED  Job: {job_id}")
        if url:
            OutputManager.echo(f"  Video URL: {url}")
        else:
            inline_data = getattr(result, "data", None)
            if inline_data:
                OutputManager.echo(f"  (Inline binary data - {len(inline_data)} bytes)")

    elif status_val == "FAILED":
        error_msg = getattr(result, "error", None) or "unknown error"
        OutputManager.error(f"FAILED  Job: {job_id}")
        OutputManager.echo(f"  Error: {error_msg}")

    else:
        progress = getattr(result, "progress_percent", 0) or 0
        remaining_ms = getattr(result, "estimated_remaining_ms", None)
        remaining_s = remaining_ms / 1000 if remaining_ms else None

        OutputManager.echo(f"PROCESSING  Job: {job_id}")
        if remaining_s is not None:
            OutputManager.progress(f"complete (~{remaining_s:.0f}s remaining)", pct=progress)
        else:
            OutputManager.progress("complete...", pct=progress)
