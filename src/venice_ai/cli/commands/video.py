"""Video commands for Venice AI CLI — Text-to-Video and Image-to-Video."""

import asyncio
import base64
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

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
    download_url: str | None = None,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    max_polls: int = _DEFAULT_MAX_POLLS,
) -> bool:
    """Poll for video completion, download the result, and save to disk.

    The download goes through the client, so its proxy, SSL, timeout and retry
    settings apply, and a failure does not carry the link. Nothing printed here
    contains the link either.

    Args:
        download_url: The link from the queue response, for a model that hands it
            out only there.

    Returns True on success. On failure, on timeout, or when a finished job has
    nothing to download, it reports why and returns False, and the command exits 1.

    Raises:
        aiohttp.ClientResponseError: If the link answers with an error status.
        APITimeoutError: If the download times out.
        APIConnectionError: If the connection fails during the download.
    """
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode
    from venice_ai.resources.video import VideoJob
    from venice_ai.types.api.video import VideoQueueResponse

    if plain and not is_plain_mode():
        enable_plain_mode()

    job = VideoJob(
        client, VideoQueueResponse(model=model, queue_id=queue_id, download_url=download_url)
    )

    for _poll_num in range(max_polls):
        status: Any = await job.poll()

        if status.status == "COMPLETED":
            if not (status.data or status.url or download_url):
                OutputManager.warning("Video completed but no data or URL returned.")
                return False

            if status.data:
                saved = await job.download(output_path, status)
                _report_saved(saved)
                return True

            OutputManager.echo("Downloading video...")
            saved = await job.download(output_path, status)
            _report_saved(saved)

            try:
                complete_resp = await job.cancel()
                OutputManager.echo(
                    f"Server cleanup: {'OK' if complete_resp.success else 'skipped'}"
                )
            except Exception as e:
                # The video is saved, so a failed cleanup does not fail the command.
                OutputManager.echo("Server cleanup: skipped")
                logging.getLogger(__name__).debug("job.cancel(queue_id=%s) failed: %s", queue_id, e)

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


def _report_saved(path: Path) -> None:
    size = path.stat().st_size
    OutputManager.success(f"Video saved: {path} ({size / 1024:.1f} KB)")


def _report_no_poll(queue_resp) -> None:
    """Tell the user how to follow a job queued with ``--no-poll``."""
    if getattr(queue_resp, "download_url", None):
        OutputManager.warning(
            "This model hands out its download link only in the queue response, and the CLI "
            "neither prints nor stores the link, so this video cannot be downloaded later. "
            "Run the command without --no-poll to save the video when it finishes."
        )
    OutputManager.echo(
        f"Use 'venice-py video status {queue_resp.queue_id} --model {queue_resp.model}' "
        "to check progress."
    )


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
            _report_no_poll(queue_resp)
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
            download_url=getattr(queue_resp, "download_url", None),
        )

        if not success:
            raise SystemExit(1)
        if auto_open:
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
            _report_no_poll(queue_resp)
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
            download_url=getattr(queue_resp, "download_url", None),
        )

        if not success:
            raise SystemExit(1)
        if auto_open:
            open_file(output_path)


# ---------------------------------------------------------------------------
# venice-py video status
# ---------------------------------------------------------------------------


@video.command("status")
@click.argument("job_id")
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model used when the job was queued",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Save the video to this path if the job has finished",
)
@click.pass_context
def status(ctx, job_id, model, output):
    """Check the status of a video generation job, and save it once it has finished.

    JOB_ID is the queue ID returned when the job was submitted (e.g. with --no-poll).

    A download link grants access on its own, so it is never printed. A model
    that hands out its link only when the job is queued cannot be downloaded from
    here: run the command that queued it without --no-poll.

    Exits 0 when the status was read, whatever it is. With --output, exits 1 if the
    job failed or finished with nothing to save; a job still running exits 0.

    Examples:

      venice-py video status abc123 --model <video-model>

      venice-py video status abc123 --model <video-model> --output video.mp4
    """
    asyncio.run(_status_async(ctx, job_id=job_id, model=model, output=output))


async def _status_async(ctx, *, job_id, model, output=None):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode
    from venice_ai.resources.video import VideoJob
    from venice_ai.types.api.video import VideoQueueResponse

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    if plain and not is_plain_mode():
        enable_plain_mode()

    saved = None
    async with VeniceClient(**get_client_kwargs()) as client:
        result: Any = await client.video.retrieve(model=model, queue_id=job_id)
        url = getattr(result, "url", None)
        inline_data = getattr(result, "data", None)
        if output and result.status == "COMPLETED" and (url or inline_data):
            job = VideoJob(client, VideoQueueResponse(model=model, queue_id=job_id))
            saved = await job.download(_determine_output_path(output, "."), result)

    status_val = result.status

    if status_val == "COMPLETED":
        OutputManager.success(f"COMPLETED  Job: {job_id}")
        if saved is not None:
            _report_saved(saved)
        elif url or inline_data:
            if inline_data:
                OutputManager.echo(f"  (Inline binary data - {len(inline_data)} bytes)")
            OutputManager.echo(
                f"  Save it with: venice-py video status {job_id} --model {model} --output <path>"
            )
        else:
            OutputManager.warning(
                "Venice returned no video data or link for this job, so it cannot be "
                "downloaded from its job ID. A model that hands out a download link does so "
                "only in the queue response, and the CLI neither prints nor stores the link. "
                "To get such a video, run the command that queued it without --no-poll."
            )
            if output:
                raise SystemExit(1)

    elif status_val == "FAILED":
        error_msg = getattr(result, "error", None) or "unknown error"
        OutputManager.error(f"FAILED  Job: {job_id}")
        OutputManager.echo(f"  Error: {error_msg}")
        if output:
            raise SystemExit(1)

    else:
        progress = getattr(result, "progress_percent", 0) or 0
        remaining_ms = getattr(result, "estimated_remaining_ms", None)
        remaining_s = remaining_ms / 1000 if remaining_ms else None

        OutputManager.echo(f"PROCESSING  Job: {job_id}")
        if remaining_s is not None:
            OutputManager.progress(f"complete (~{remaining_s:.0f}s remaining)", pct=progress)
        else:
            OutputManager.progress("complete...", pct=progress)
