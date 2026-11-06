"""Saved download links for video jobs queued with ``--no-poll``.

Some video models hand out their download link once, in the queue response. A
job queued without waiting would lose that link at process exit, so the CLI
saves it here until ``video status --output`` has downloaded the video and
released it. The link grants access to the video on its own, so it is treated
like a password:

- each job is one owner-only (``0600``) file in an owner-only (``0700``) directory
  under ``~/.venice-py/``, written atomically and removed after release. Permission
  bits mean nothing on Windows, where these files are only as private as the user's
  home directory, and :func:`files_are_owner_only` says so;
- an entry that is a symbolic link is refused, on every platform;
- a file name is a hash of the queue ID, so a server-supplied ID cannot point
  outside the directory;
- nothing here puts a link in an error message or a ``repr``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from venice_ai.cli import _paths

_OPEN_NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)
_POSIX = os.name == "posix"
_GROUP_OR_OTHER = stat.S_IRWXG | stat.S_IRWXO


class VideoJobStoreError(Exception):
    """The store cannot be used safely, or an entry in it cannot be read."""


@dataclass(frozen=True)
class SavedVideoJob:
    """A job's saved details. ``was_exposed`` means others could read its file."""

    queue_id: str
    model: str
    download_url: str = field(repr=False)
    was_exposed: bool = False


def jobs_dir() -> Path:
    """Return the directory holding one file per saved job."""
    return _paths.app_dir() / "video_jobs"


def files_are_owner_only() -> bool:
    """Whether this platform enforces the owner-only permissions set on the store."""
    return _POSIX


def entry_path(queue_id: str) -> Path:
    """Return the path a job's saved entry has. Its name does not show the queue ID."""
    digest = hashlib.sha256(queue_id.encode("utf-8", "surrogatepass")).hexdigest()
    return jobs_dir() / f"{digest}.json"


def _remove_temporaries(entry: Path) -> None:
    """Delete what a killed run left of an entry: such a file holds the link too."""
    for temporary in entry.parent.glob(f"{entry.stem}.*.tmp"):
        temporary.unlink(missing_ok=True)


def _prepare_jobs_dir() -> Path:
    """Create the store's directories owner-only, narrowing any that are wider."""
    _paths.ensure_migrated()
    directory = jobs_dir()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise VideoJobStoreError(f"{directory} is a symbolic link; refusing to use it.")
    directory.mkdir(mode=0o700, exist_ok=True)
    for path in (directory.parent, directory):
        os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~_GROUP_OR_OTHER)
    return directory


def save_job(queue_id: str, model: str, download_url: str) -> Path:
    """Save a job's link, replacing any entry for the same queue ID.

    The file is created at ``0600`` rather than narrowed afterwards, so the link
    is never readable by others, even for a moment.

    :return: The path of the saved entry.
    :raises VideoJobStoreError: If the store's directory is a symbolic link, or
        the created file is wider than owner-only.
    :raises OSError: If the file cannot be written.
    """
    directory = _prepare_jobs_dir()
    entry = entry_path(queue_id)
    _remove_temporaries(entry)
    body = json.dumps({"queue_id": queue_id, "model": model, "download_url": download_url})
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=f"{entry.stem}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if _POSIX and stat.S_IMODE(os.fstat(stream.fileno()).st_mode) & _GROUP_OR_OTHER:
                raise VideoJobStoreError(f"{temporary} was not created owner-only.")
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, entry)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return entry


def load_job(queue_id: str) -> SavedVideoJob | None:
    """Return a job's saved details, or ``None`` when none are saved.

    An entry others could read is narrowed to ``0600`` and flagged with
    ``was_exposed``, since its link may have been read.

    :raises VideoJobStoreError: If the entry is a symbolic link, belongs to
        another user, or is not a valid entry for this queue ID.
    """
    entry = entry_path(queue_id)
    if entry.is_symlink():
        raise VideoJobStoreError(f"{entry} is a symbolic link; refusing to read it.")
    if not entry.exists():
        return None
    try:
        descriptor = os.open(entry, os.O_RDONLY | _OPEN_NO_FOLLOW)
    except OSError as exc:
        raise VideoJobStoreError(f"{entry} could not be opened: {exc.strerror}.") from None
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        was_exposed = False
        if _POSIX:
            details = os.fstat(stream.fileno())
            if details.st_uid != os.geteuid():
                raise VideoJobStoreError(f"{entry} belongs to another user.")
            mode = stat.S_IMODE(details.st_mode)
            if mode & _GROUP_OR_OTHER:
                os.fchmod(stream.fileno(), 0o600)
                was_exposed = True
        try:
            saved = json.load(stream)
        except (ValueError, RecursionError):
            raise VideoJobStoreError(f"{entry} is not a valid entry.") from None
    if not isinstance(saved, dict):
        raise VideoJobStoreError(f"{entry} is not a valid entry.")
    model, download_url = saved.get("model"), saved.get("download_url")
    if not (isinstance(model, str) and model and isinstance(download_url, str) and download_url):
        raise VideoJobStoreError(f"{entry} is not a valid entry.")
    if saved.get("queue_id") != queue_id:
        raise VideoJobStoreError(f"{entry} belongs to a different job.")
    return SavedVideoJob(queue_id, model, download_url, was_exposed=was_exposed)


def remove_job(queue_id: str) -> None:
    """Delete a job's saved entry, if it has one, and any temporary left beside it."""
    entry = entry_path(queue_id)
    entry.unlink(missing_ok=True)
    _remove_temporaries(entry)
