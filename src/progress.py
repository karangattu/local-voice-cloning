import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True)
class ProgressStep:
    key: str
    label: str
    detail: str
    state: str


_STAGES = (
    ("prepare", "Prep", "Preparing reference"),
    ("load", "Load", "Loading local models"),
    ("voice", "Voice", "Synthesizing speech"),
    ("finish", "Finish", "Finalizing audio"),
)


def progress_snapshot(active_stage: str | None, status: str) -> tuple[ProgressStep, ...]:
    active_index = next(
        (index for index, (key, _label, _detail) in enumerate(_STAGES) if key == active_stage),
        0,
    )
    steps: list[ProgressStep] = []
    for index, (key, label, detail) in enumerate(_STAGES):
        if status == "success":
            state = "complete"
        elif status == "error" and index == active_index:
            state = "error"
        elif index < active_index:
            state = "complete"
        elif status == "running" and index == active_index:
            state = "active"
        else:
            state = "pending"
        steps.append(ProgressStep(key, label, detail, state))
    return tuple(steps)


T = TypeVar("T")


async def run_with_progress(
    work: Callable[[Callable[[str], None]], T],
    on_stage: Callable[[str], None],
) -> T:
    """Run blocking synthesis in a thread while forwarding its stage updates."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    finished = object()

    def report(stage: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, stage)

    def run_worker() -> T:
        try:
            return work(report)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, finished)

    task = asyncio.create_task(asyncio.to_thread(run_worker))
    while True:
        event = await queue.get()
        if event is finished:
            break
        on_stage(str(event))
    return await task
