"""Fast tests for batched generation: the request batcher and the cloner's batched path."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import src.cloner as cloner_module
from src.qwen_batch import GenerationBatcher, supports_batched_icl


class RecordingRunner:
    """Batch runner that records each batch and echoes "text:seed" per row."""

    def __init__(self, gate: threading.Event | None = None):
        self.batches: list[tuple[object, list[str], list[int]]] = []
        self.gate = gate

    def __call__(self, payload, texts, seeds):
        if self.gate is not None:
            self.gate.wait(5)
        self.batches.append((payload, list(texts), list(seeds)))
        return [f"{text}:{seed}" for text, seed in zip(texts, seeds, strict=True)]


def _submit_in_threads(batcher, jobs):
    results: dict[int, list] = {}

    def run(index, group, texts, seeds):
        results[index] = batcher.submit(group, f"payload-{group}", texts, seeds)

    threads = [
        threading.Thread(target=run, args=(index, *job)) for index, job in enumerate(jobs)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    return results


def test_batcher_returns_rows_in_order():
    runner = RecordingRunner()
    batcher = GenerationBatcher(runner, max_batch_size=8, window_seconds=0)
    assert batcher.submit("voice", "p", ["a", "a", "a"], [10, 11, 12]) == ["a:10", "a:11", "a:12"]
    assert runner.batches == [("p", ["a", "a", "a"], [10, 11, 12])]


def test_batcher_merges_concurrent_jobs_with_the_same_group():
    gate = threading.Event()
    runner = RecordingRunner(gate)
    batcher = GenerationBatcher(runner, max_batch_size=8, window_seconds=0.2)
    gate.set()
    results = _submit_in_threads(
        batcher, [("voice", ["one"], [1]), ("voice", ["two"], [2]), ("voice", ["three"], [3])]
    )
    assert sorted(results.values()) == [["one:1"], ["three:3"], ["two:2"]]
    assert len(runner.batches) == 1
    assert sorted(runner.batches[0][1]) == ["one", "three", "two"]


def test_batcher_keeps_different_groups_apart_and_caps_batch_size():
    runner = RecordingRunner()
    batcher = GenerationBatcher(runner, max_batch_size=2, window_seconds=0.2)
    results = _submit_in_threads(
        batcher, [("a", ["x", "x", "x"], [1, 2, 3]), ("b", ["y"], [9])]
    )
    assert results[0] == ["x:1", "x:2", "x:3"]
    assert results[1] == ["y:9"]
    for payload, texts, _seeds in runner.batches:
        assert len(texts) <= 2
        assert len({payload}) == 1
    assert {payload for payload, _texts, _seeds in runner.batches} == {"payload-a", "payload-b"}
    assert batcher.stats()["queued_rows"] == 0


def test_batcher_reports_errors_to_every_job_in_the_batch():
    def failing(_payload, _texts, _seeds):
        raise RuntimeError("out of memory")

    batcher = GenerationBatcher(failing, window_seconds=0)
    with pytest.raises(RuntimeError, match="out of memory"):
        batcher.submit("voice", "p", ["a"], [1])
    # The worker survives the failure and serves the next job.
    batcher._run_batch = RecordingRunner()
    assert batcher.submit("voice", "p", ["b"], [2]) == ["b:2"]


def test_batcher_stats_report_queue_depth_while_a_batch_runs():
    started = threading.Event()
    release = threading.Event()

    def slow(_payload, texts, seeds):
        started.set()
        release.wait(5)
        return [f"{t}:{s}" for t, s in zip(texts, seeds, strict=True)]

    batcher = GenerationBatcher(slow, max_batch_size=1, window_seconds=0)
    threads = [
        threading.Thread(target=batcher.submit, args=("v", "p", ["a"], [seed])) for seed in (1, 2)
    ]
    for thread in threads:
        thread.start()
    assert started.wait(5)
    stats = batcher.stats()
    assert stats["running_rows"] == 1
    assert stats["queued_rows"] + stats["running_rows"] == 2
    release.set()
    for thread in threads:
        thread.join(5)
    assert batcher.stats()["batches_run"] == 2


def test_supports_batched_icl_requires_encoder_and_internals():
    assert not supports_batched_icl(SimpleNamespace())
    model = SimpleNamespace(
        _prepare_batch_inputs=None,
        _decode_icl_generated_codes=None,
        _codec_embeds_for_tokens=None,
        _suppress_codec_tokens=None,
        talker=None,
        config=None,
        speech_tokenizer=SimpleNamespace(has_encoder=True),
    )
    assert supports_batched_icl(model)
    model.speech_tokenizer = SimpleNamespace(has_encoder=False)
    assert not supports_batched_icl(model)


# --- cloner batched path -----------------------------------------------------


@pytest.fixture
def sample_voice_file(tmp_path):
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    path = tmp_path / "voice.wav"
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    return path


class ICLModel:
    sample_rate = 24000
    speech_tokenizer = SimpleNamespace(has_encoder=True)

    def __init__(self):
        self.sequential_calls = 0

    def _generate_icl(self, **_kwargs):
        self.sequential_calls += 1
        yield SimpleNamespace(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000)


class SequenceSTTModel:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)

    def generate(self, _audio_path, **_kwargs):
        return SimpleNamespace(text=self.transcripts.pop(0))


@pytest.fixture
def fake_batch(monkeypatch):
    """Batched generation that fills each take with its seed / 1000."""
    calls: list[dict] = []

    def generate(model, texts, seeds, ref_audio, ref_text, language, repetition_penalty):
        calls.append(
            {"texts": list(texts), "seeds": list(seeds), "ref_text": ref_text,
             "language": language, "repetition_penalty": repetition_penalty}
        )
        return [np.full(2400, seed / 1000, dtype=np.float32) for seed in seeds]

    monkeypatch.setattr(cloner_module, "supports_batched_icl", lambda _model: True)
    monkeypatch.setattr(cloner_module, "generate_icl_batch", generate)
    return calls


def _capture_pieces(monkeypatch):
    captured = []
    monkeypatch.setattr(
        cloner_module,
        "join_with_room_tone",
        lambda pieces, _sr: captured.extend(pieces) or np.concatenate([p for p, _ in pieces]),
    )
    return captured


def test_take_one_runs_alone_and_a_perfect_take_one_skips_the_rest(
    sample_voice_file, fake_batch, monkeypatch
):
    pieces = _capture_pieces(monkeypatch)
    model = ICLModel()
    stt = SequenceSTTModel(["Hello there."])
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model, stt_loader=lambda _: stt)

    cloner.clone_voice(
        sample_voice_file, "Hello there", reference_text="Hi", candidates=3, seed=10
    )

    assert model.sequential_calls == 0
    assert [call["seeds"] for call in fake_batch] == [[10]]
    assert pieces[0][0][0] == pytest.approx(0.010)


def test_remaining_takes_run_as_one_batch_when_take_one_misses(
    sample_voice_file, fake_batch, monkeypatch
):
    pieces = _capture_pieces(monkeypatch)
    model = ICLModel()
    stt = SequenceSTTModel(["Hello their friend", "Hello there.", "Hello."])
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model, stt_loader=lambda _: stt)

    cloner.clone_voice(
        sample_voice_file, "Hello there", reference_text="Hi", candidates=3, seed=10
    )

    assert model.sequential_calls == 0
    assert [call["seeds"] for call in fake_batch] == [[10], [11, 12]]
    assert fake_batch[1]["texts"] == ["Hello there."] * 2
    assert fake_batch[1]["repetition_penalty"] == cloner_module.QWEN_REPETITION_PENALTY
    # Take 2 (seed 11) matches its line exactly, so it is kept and take 3 is
    # not transcribed.
    assert pieces[0][0][0] == pytest.approx(0.011)
    assert stt.transcripts == ["Hello."]


def test_each_line_reuses_the_job_seeds(sample_voice_file, fake_batch):
    stt = SequenceSTTModel(["no"] * 4)
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _: ICLModel(), stt_loader=lambda _: stt
    )
    cloner.clone_voice(sample_voice_file, "One\nTwo", reference_text="Hi", candidates=2, seed=5)
    assert [call["seeds"] for call in fake_batch] == [[5], [6], [5], [6]]


def test_concurrent_requests_for_one_voice_share_a_batch(sample_voice_file, fake_batch, monkeypatch):
    monkeypatch.setattr(cloner_module, "BATCH_WINDOW_SECONDS", 0.3)
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: ICLModel())
    cloner._ensure_tts_model()
    errors: list[BaseException] = []

    def request(seed):
        try:
            cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi", seed=seed)
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=request, args=(seed,)) for seed in (1, 2, 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors
    assert len(fake_batch) == 1
    assert sorted(fake_batch[0]["seeds"]) == [1, 2, 3]
    stats = cloner.queue_stats()
    assert stats["batching"] is True
    assert stats["requests_in_flight"] == 0
    assert stats["queued_rows"] == 0


def test_batching_can_be_turned_off(sample_voice_file, fake_batch, monkeypatch):
    monkeypatch.setattr(cloner_module, "BATCHING_ENABLED", False)
    model = ICLModel()
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model)
    cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi")
    assert fake_batch == []
    assert model.sequential_calls == 1


def test_closing_the_batcher_stops_its_worker():
    batcher = GenerationBatcher(RecordingRunner(), window_seconds=0)
    assert batcher.submit("v", "p", ["a"], [1]) == ["a:1"]
    worker = batcher._thread
    batcher.close()
    worker.join(5)
    assert not worker.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        batcher.submit("v", "p", ["a"], [1])


def test_unloading_shared_cloners_stops_batch_workers(sample_voice_file, fake_batch, monkeypatch):
    monkeypatch.setattr(cloner_module, "_shared_cloners", {})
    cloner = cloner_module.get_shared_cloner("high")
    cloner._tts_loader = lambda _: ICLModel()
    cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi")
    worker = cloner._batcher._thread
    assert cloner_module.unload_shared_cloners() == 1
    worker.join(5)
    assert not worker.is_alive()
