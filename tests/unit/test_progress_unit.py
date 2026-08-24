import asyncio

import pytest


def test_progress_snapshot_marks_completed_active_and_pending_steps():
    from src.progress import progress_snapshot

    snapshot = progress_snapshot("voice", "running")

    assert [(step.key, step.state) for step in snapshot] == [
        ("prepare", "complete"),
        ("load", "complete"),
        ("voice", "active"),
        ("finish", "pending"),
    ]


def test_progress_snapshot_marks_every_step_complete_after_success():
    from src.progress import progress_snapshot

    assert [step.state for step in progress_snapshot("finish", "success")] == [
        "complete",
        "complete",
        "complete",
        "complete",
    ]


def test_progress_snapshot_marks_the_active_step_when_generation_fails():
    from src.progress import progress_snapshot

    assert [step.state for step in progress_snapshot("load", "error")] == [
        "complete",
        "error",
        "pending",
        "pending",
    ]


def test_run_with_progress_bridges_worker_thread_updates_to_async_caller():
    from src.progress import run_with_progress

    received: list[str] = []

    def work(report):
        report("prepare")
        report("load")
        report("voice")
        report("finish")
        return "audio-ready"

    result = asyncio.run(run_with_progress(work, received.append))

    assert result == "audio-ready"
    assert received == ["prepare", "load", "voice", "finish"]


def test_run_with_progress_preserves_worker_errors():
    from src.progress import run_with_progress

    def work(report):
        report("prepare")
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(run_with_progress(work, lambda _stage: None))
