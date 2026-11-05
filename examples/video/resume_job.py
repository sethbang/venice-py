#!/usr/bin/env python3
"""
Venice AI SDK - Resume a Video Job
==================================

This example demonstrates the low-level video lifecycle, and how to collect a
job that was already paid for instead of paying for it again. A job is billed
when it is queued, so running a script a second time buys a second video. Keep
what the queue call returned and resume with that:

    1. ``client.video.retrieve(model=..., queue_id=...)`` reads the job's status.
       It is free to call as often as you like.
    2. Save the finished video and check the file.
    3. ``client.video.cancel(model=..., queue_id=...)`` releases the stored copy.

**What you have to persist.** At queue time, save the ``model``, the
``queue_id`` and, if the response has one, the ``download_url``. Capture them
from ``client.video.submit()``, whose response has all three, or from the same
attributes on the ``VideoJob`` that ``client.video.run()`` returns.

**``retrieve()`` does not rebuild a** ``VideoJob``. It returns a status object,
with no ``wait()``, ``download()`` or ``cancel()``, and it knows nothing about
the job's queue response.

**``download_url`` models need the link persisted too.** They hand out the link
once, in the queue response. ``/video/retrieve`` then reports ``COMPLETED`` with
no video and no URL, and ``/video/complete`` does not know the job. If the link
was not saved, a billed video cannot be collected, and this example says so.
With the link, it builds a ``VideoJob`` by hand from the three saved values and
uses that to fetch the file and delete it. The link grants access on its own,
so treat it like a password; this example never prints it.

Other things worth knowing:

- ``cancel()`` does not stop a running job. It deletes nothing until the job has
  finished, and the job is billed either way, so this example never calls it on
  a job still processing.
- ``retrieve()`` is called without ``delete_media_on_completion=True``. That flag
  deletes the stored copy on delivery, before the file is known to be good.
- ``VideoJob.wait()`` turns a 400 or 422 from ``/video/retrieve`` into
  ``VideoGenerationError``. ``retrieve()`` raises them as they are.

Set these to resume a job::

    VENICE_VIDEO_RESUME_MODEL          the model it was queued on
    VENICE_VIDEO_RESUME_QUEUE_ID       its queue_id
    VENICE_VIDEO_RESUME_DOWNLOAD_URL   its download_url, for models that return one

``text_to_video.py`` and the other video examples print the ``queue_id`` of each
job they queue, but never its link, so a job they queue on a ``download_url`` model
cannot be collected here. To resume such a job, queue it from your own code and save
``job.download_url``.

**Cost: nothing new.** This example never submits a job, so every call only reads
or releases one you already paid for. Without a queue ID it makes no request at
all: it explains the flow and exits 77. A job still processing also exits 77, so
run it again later. A job that failed, an unknown or already released queue ID, a
video that cannot be collected or that fails its check, and a failed release all
exit 1.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

import aiohttp

from venice_ai import VeniceClient, VideoJob
from venice_ai.exceptions import InvalidRequestError, VeniceError
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import SKIPPED, check_video, download_video, print_progress  # noqa: E402

# All example scripts live one level below examples/ (e.g., examples/video/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL_ENV = "VENICE_VIDEO_RESUME_MODEL"
QUEUE_ID_ENV = "VENICE_VIDEO_RESUME_QUEUE_ID"
DOWNLOAD_URL_ENV = "VENICE_VIDEO_RESUME_DOWNLOAD_URL"

# How /video/complete and /video/retrieve say a queue ID is unknown or already released.
UNKNOWN_QUEUE_ID = "request id is invalid"


def explain_flow() -> None:
    """Print the lifecycle this example runs once it is given a job."""
    print("🧭 The flow, once you have a job to resume:")
    print("   1. queue:    keep model, queue_id and any download_url from the queue response")
    print(
        "   2. retrieve: client.video.retrieve(model=..., queue_id=...), free, as often as needed"
    )
    print("   3. save:     write the finished video, then check the file")
    print("   4. release:  client.video.cancel(model=..., queue_id=...) deletes the stored copy")
    print("   retrieve() does not rebuild a VideoJob: it returns a status object.")
    print("   download_url models give their link only at queue time, so persist it too.")


def _is_unknown_queue_id(e: InvalidRequestError) -> bool:
    return UNKNOWN_QUEUE_ID in str(e).lower() or UNKNOWN_QUEUE_ID in str(e.body).lower()


async def _release(client: VeniceClient, job: VideoJob, *, has_link: bool) -> bool:
    """Delete the job's stored copy. Returns ``False`` if it may still be there."""
    try:
        if has_link:
            # /video/complete does not know these jobs: the file lives behind the
            # link, and VideoJob.cancel() sends the DELETE there first.
            result = await job.cancel()
        else:
            result = await client.video.cancel(model=job.model, queue_id=job.queue_id)
    except InvalidRequestError as e:
        if _is_unknown_queue_id(e):
            print("✅ The server no longer knows this queue ID, so nothing is left to release")
            return True
        print(f"❌ Release failed: {e}")
        return False
    except VeniceError as e:
        print(f"❌ Release failed ({type(e).__name__}): {e}")
        return False
    if not result.success:
        print("❌ The server did not confirm the release")
        return False
    print("🧹 Stored copy released")
    return True


async def _save_and_check(job: VideoJob, status: VideoCompletedStatus) -> bool:
    """Save a finished video and check the file. Nothing is released on a failure."""
    safe_name = re.sub(r"[^\w.-]", "_", job.queue_id)
    path = RESULTS_DIR / f"resumed_{safe_name}.mp4"
    try:
        saved = await download_video(job, path, status)
    except aiohttp.ClientResponseError as e:
        # The error shows the storage host at most, never the link.
        print(f"❌ The download answered HTTP {e.status}; the link may have expired")
        print("   Nothing was released.")
        return False
    except VeniceError as e:
        print(f"❌ Could not save the video ({type(e).__name__}): {e}")
        print("   Nothing was released.")
        return False

    data = saved.read_bytes()
    print(f"💾 Saved to {saved} ({len(data):,} bytes)")
    # The request itself is not known here, so this checks for an MP4 with a video track.
    if not check_video(data, {}, expect_audio=None):
        print("❌ The file is not a playable clip. Nothing was released, so you can try again.")
        return False
    return True


async def resume(client: VeniceClient, model: str, queue_id: str, link: str | None) -> int:
    """Read one job's status and act on it. Spends nothing."""
    print(f"📮 Job {queue_id} on {model}")
    try:
        status = await client.video.retrieve(model=model, queue_id=queue_id)
    except InvalidRequestError as e:
        if _is_unknown_queue_id(e):
            print("❌ That queue ID is unknown or was already released.")
        else:
            print(f"❌ The server rejected this job: {e}")
        return 1
    except (VeniceError, ValueError) as e:
        print(f"❌ retrieve failed ({type(e).__name__}): {e}")
        return 1

    # retrieve() gave a status, not a job. The saved values are enough to build one.
    job = VideoJob(client, VideoQueueResponse(model=model, queue_id=queue_id, download_url=link))

    if isinstance(status, VideoFailedStatus):
        code = f" (code={status.error_code})" if status.error_code else ""
        print(f"❌ Generation failed{code}: {status.error}")
        await _release(client, job, has_link=bool(link))
        return 1

    if isinstance(status, VideoProcessingStatus):
        print_progress(status)
        print(f"\nSKIPPED: job {queue_id} is still processing; run this again later (it is free)")
        return SKIPPED

    print("✅ The job has finished")
    if not (status.data or status.url or link):
        print("❌ The server has no video to give: this is a download_url model, and that")
        print(f"   link is only in the queue response. Set {DOWNLOAD_URL_ENV} if you saved it;")
        print("   otherwise this billed video cannot be collected. Nothing was released.")
        return 1

    if not await _save_and_check(job, status):
        return 1
    if not await _release(client, job, has_link=bool(link)):
        print("   The video is saved, but its stored copy may remain.")
        return 1

    print("\n✨ Job collected without paying for it again.")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.video.retrieve() resumes from a saved model and queue_id")
    print("   - Saving and checking the file before releasing anything")
    print("   - client.video.cancel(), or VideoJob.cancel() for a job with a download_url")
    return 0


async def main() -> int:
    print("🚀 Venice AI Video Job Resume Example")
    print("=" * 60)

    model = os.environ.get(MODEL_ENV)
    queue_id = os.environ.get(QUEUE_ID_ENV)
    link = os.environ.get(DOWNLOAD_URL_ENV) or None

    if not queue_id:
        explain_flow()
        print(f"\nSKIPPED: set {QUEUE_ID_ENV} and {MODEL_ENV} to resume a job; nothing was sent")
        return SKIPPED
    if not model:
        # Every call in the lifecycle is model-scoped.
        print(f"SKIPPED: {QUEUE_ID_ENV} is set, but resuming also needs {MODEL_ENV}")
        return SKIPPED

    async with VeniceClient() as client:
        return await resume(client, model, queue_id, link)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
