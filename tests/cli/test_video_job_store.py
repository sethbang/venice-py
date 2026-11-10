"""The local store of link-based video jobs queued with ``--no-poll``."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from venice_ai.cli import _paths
from venice_ai.cli import _video_jobs as store

LINK = "https://storage.example/SECRETPATH/video.mp4?sig=SECRETQUERY"

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


@pytest.fixture(autouse=True)
def home(tmp_path: Path) -> Iterator[Path]:
    with patch.object(_paths, "_home", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def open_umask() -> Iterator[None]:
    previous = os.umask(0)
    yield
    os.umask(previous)


def _entry(queue_id: str = "q") -> Path:
    entries = list(store.jobs_dir().iterdir())
    assert len(entries) == 1
    return entries[0]


def test_saved_job_round_trips() -> None:
    store.save_job("q", "m", LINK)

    job = store.load_job("q")

    assert job is not None
    assert (job.queue_id, job.model, job.download_url) == ("q", "m", LINK)
    assert job.was_exposed is False


def test_unknown_job_is_none_and_creates_nothing(home: Path) -> None:
    assert store.load_job("absent") is None
    assert not store.jobs_dir().exists()


def test_store_lives_under_the_cli_directory(home: Path) -> None:
    assert store.jobs_dir() == home / ".venice-py" / "video_jobs"


@posix_only
@pytest.mark.usefixtures("open_umask")
def test_entry_is_owner_only_whatever_the_umask() -> None:
    store.save_job("q", "m", LINK)

    assert _mode(_entry()) == 0o600
    assert _mode(store.jobs_dir()) == 0o700
    assert _mode(_paths.app_dir()) == 0o700


@posix_only
def test_wider_directories_are_narrowed_on_create() -> None:
    store.jobs_dir().mkdir(parents=True)
    os.chmod(_paths.app_dir(), 0o755)
    os.chmod(store.jobs_dir(), 0o777)

    store.save_job("q", "m", LINK)

    assert _mode(store.jobs_dir()) == 0o700
    assert _mode(_paths.app_dir()) == 0o700


@posix_only
def test_replacing_a_wide_entry_leaves_it_owner_only() -> None:
    store.save_job("q", "m", LINK)
    os.chmod(_entry(), 0o644)

    store.save_job("q", "m", LINK)

    assert _mode(_entry()) == 0o600


@posix_only
def test_symlinked_store_directory_is_refused(home: Path) -> None:
    elsewhere = home / "elsewhere"
    elsewhere.mkdir()
    _paths.app_dir().mkdir()
    store.jobs_dir().symlink_to(elsewhere)

    with pytest.raises(store.VideoJobStoreError):
        store.save_job("q", "m", LINK)

    assert list(elsewhere.iterdir()) == []


@posix_only
def test_symlinked_entry_is_refused_on_read(home: Path) -> None:
    store.save_job("q", "m", LINK)
    entry = _entry()
    target = home / "elsewhere.json"
    target.write_text(entry.read_text())
    entry.unlink()
    entry.symlink_to(target)

    with pytest.raises(store.VideoJobStoreError) as excinfo:
        store.load_job("q")

    assert "SECRET" not in str(excinfo.value)


@posix_only
def test_wide_entry_is_narrowed_and_flagged_on_read() -> None:
    store.save_job("q", "m", LINK)
    os.chmod(_entry(), 0o644)

    job = store.load_job("q")

    assert job is not None
    assert job.was_exposed is True
    assert job.download_url == LINK
    assert _mode(_entry()) == 0o600


@posix_only
def test_entry_owned_by_someone_else_is_refused() -> None:
    store.save_job("q", "m", LINK)

    with (
        patch.object(store.os, "geteuid", return_value=os.geteuid() + 1),
        pytest.raises(store.VideoJobStoreError),
    ):
        store.load_job("q")


@pytest.mark.parametrize(
    "queue_id", ["../../escape", "/etc/passwd", "a/b", "..", "nul\x00byte", "é" * 400]
)
def test_any_queue_id_stays_inside_the_store(home: Path, queue_id: str) -> None:
    store.save_job(queue_id, "m", LINK)

    assert _entry().parent == store.jobs_dir()
    job = store.load_job(queue_id)
    assert job is not None
    assert job.queue_id == queue_id
    assert not (home.parent / "escape").exists()


def test_entry_name_does_not_carry_the_queue_id() -> None:
    store.save_job("visible-queue-id", "m", LINK)

    assert "visible-queue-id" not in _entry().name


def test_entries_are_keyed_by_queue_id() -> None:
    store.save_job("one", "m1", LINK)
    store.save_job("two", "m2", LINK + "2")

    store.remove_job("one")

    assert store.load_job("one") is None
    two = store.load_job("two")
    assert two is not None
    assert two.model == "m2"


def test_remove_deletes_the_file_and_is_safe_when_absent() -> None:
    store.save_job("q", "m", LINK)

    store.remove_job("q")
    store.remove_job("q")

    assert list(store.jobs_dir().iterdir()) == []


def test_job_repr_does_not_show_the_link() -> None:
    assert "SECRET" not in repr(store.SavedVideoJob("q", "m", LINK))
    assert "SECRET" not in str(store.SavedVideoJob("q", "m", LINK))


def test_failed_write_leaves_no_partial_entry() -> None:
    with patch.object(store.os, "replace", side_effect=OSError("disk")), pytest.raises(OSError):
        store.save_job("q", "m", LINK)

    assert list(store.jobs_dir().iterdir()) == []


@pytest.mark.parametrize(
    "content",
    [
        "not json " + LINK,
        json.dumps(["list", LINK]),
        json.dumps({"queue_id": "q", "model": "m"}),
        json.dumps({"queue_id": "other", "model": "m", "download_url": LINK}),
        json.dumps({"queue_id": "q", "model": 3, "download_url": LINK}),
    ],
    ids=["not-json", "not-an-object", "no-link", "other-job", "wrong-type"],
)
def test_unreadable_entry_raises_without_its_content(content: str) -> None:
    store.save_job("q", "m", LINK)
    _entry().write_text(content)

    with pytest.raises(store.VideoJobStoreError) as excinfo:
        store.load_job("q")

    assert "SECRET" not in str(excinfo.value)


def test_deeply_nested_entry_is_reported_as_corrupt() -> None:
    store.save_job("q", "m", LINK)
    _entry().write_text("[" * 200_000)

    with pytest.raises(store.VideoJobStoreError):
        store.load_job("q")


def _stale_temporary(queue_id: str, name: str) -> Path:
    directory = store.jobs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{store.entry_path(queue_id).stem}.{name}.tmp"
    temporary.write_text(json.dumps({"download_url": LINK}))
    return temporary


def test_saving_over_a_temporary_a_killed_run_left_behind_succeeds() -> None:
    stale = _stale_temporary("q", str(os.getpid()))

    store.save_job("q", "m", LINK)

    assert not stale.exists()
    assert [path.suffix for path in store.jobs_dir().iterdir()] == [".json"]


def test_removing_a_job_also_removes_the_link_in_its_stale_temporaries() -> None:
    store.save_job("q", "m", LINK)
    stale = _stale_temporary("q", "12345")
    other_job = _stale_temporary("other", "12345")

    store.remove_job("q")

    assert not stale.exists()
    assert other_job.exists()
    assert not store.entry_path("q").exists()


def test_removing_a_job_is_safe_with_no_store_directory() -> None:
    store.remove_job("q")


@posix_only
def test_file_not_created_owner_only_is_refused_and_removed() -> None:
    real_fstat = os.fstat

    def wide(descriptor: int) -> os.stat_result:
        details = list(real_fstat(descriptor))
        details[0] = stat.S_IFREG | 0o644
        return os.stat_result(details)

    with patch.object(store.os, "fstat", side_effect=wide), pytest.raises(store.VideoJobStoreError):
        store.save_job("q", "m", LINK)

    assert list(store.jobs_dir().iterdir()) == []


def test_symlinked_entry_is_refused_where_the_platform_has_no_o_nofollow(home: Path) -> None:
    store.save_job("q", "m", LINK)
    entry = store.entry_path("q")
    planted = home / "planted.json"
    planted.write_text(entry.read_text().replace("storage.example", "attacker.example"))
    entry.unlink()
    entry.symlink_to(planted)

    with (
        patch.object(store, "_POSIX", False),
        patch.object(store, "_OPEN_NO_FOLLOW", 0),
        pytest.raises(store.VideoJobStoreError, match="symbolic link"),
    ):
        store.load_job("q")


def test_owner_only_is_reported_per_platform() -> None:
    assert store.files_are_owner_only() is (os.name == "posix")
    with patch.object(store, "_POSIX", False):
        assert store.files_are_owner_only() is False
