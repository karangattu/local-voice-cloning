"""Batched Qwen3-TTS voice cloning with one random seed per row.

mlx-audio's ``batch_generate`` can clone a voice for many texts at once, but it
draws every row's samples from MLX's single global random state, so a row's
output depends on what else shares the batch. It also raises the repetition
penalty to at least 1.5. This module runs the same batched forward pass with a
separate random key per row. Each key is split exactly the way ``mx.random.seed``
splits the global state, so a row seeded with ``s`` draws the same numbers as a
sequential ``_generate_icl`` call after ``seed_generators(s)``.

All rows share one reference voice (audio and transcript) and one language.
Rows may have different texts; shorter prompts are left-padded and masked.

The module imports MLX lazily so the rest of the package still imports
without it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

QWEN_TEMPERATURE = 0.9
QWEN_TOP_K = 50
QWEN_TOP_P = 1.0
QWEN_MAX_TOKENS = 4096


def supports_batched_icl(model: Any) -> bool:
    """Whether a loaded model exposes the mlx-audio internals this module uses."""
    required = (
        "_prepare_batch_inputs",
        "_decode_icl_generated_codes",
        "_codec_embeds_for_tokens",
        "_suppress_codec_tokens",
        "talker",
        "config",
    )
    if not all(hasattr(model, name) for name in required):
        return False
    tokenizer = getattr(model, "speech_tokenizer", None)
    return bool(getattr(tokenizer, "has_encoder", False))


class _RowKeys:
    """One random key per row, split like MLX's global key sequence."""

    def __init__(self, seeds: Sequence[int]) -> None:
        import mlx.core as mx

        self._mx = mx
        self._keys = [mx.random.key(int(seed) & 0xFFFFFFFFFFFFFFFF) for seed in seeds]

    def next(self, row: int):
        key, sub = self._mx.random.split(self._keys[row])
        self._keys[row] = key
        return sub


def _filter_logits(
    logits,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_tokens: Sequence[Sequence[int]] | None = None,
    suppress_tokens: Sequence[int] | None = None,
    eos_token_id: int | None = None,
):
    """Row-wise copy of ``Model._sample_token`` up to the final draw."""
    import mlx.core as mx
    from mlx_audio.lm.sample_utils import apply_top_k
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import _apply_probability_filters

    batch = logits.shape[0]
    if suppress_tokens:
        suppress_idx = mx.array(list(suppress_tokens), dtype=mx.int32)
        logits = mx.put_along_axis(
            logits,
            mx.broadcast_to(suppress_idx[None, :], (batch, len(suppress_tokens))),
            mx.array(float("-inf"), logits.dtype),
            axis=-1,
        )

    if generated_tokens and repetition_penalty != 1.0:
        rows = []
        for row, tokens in enumerate(generated_tokens):
            row_logits = logits[row : row + 1]
            valid = [t for t in set(tokens) if t < logits.shape[-1]]
            if valid:
                token_ids = mx.array(valid, dtype=mx.int32)
                selected = mx.take(row_logits, token_ids, axis=-1)
                penalized = mx.where(
                    selected < 0, selected * repetition_penalty, selected / repetition_penalty
                )
                row_logits = mx.put_along_axis(row_logits, token_ids[None, :], penalized, axis=-1)
            rows.append(row_logits)
        logits = mx.concatenate(rows, axis=0)

    if temperature <= 0:
        return logits
    if temperature != 1.0:
        logits = logits / temperature

    eos_logit = None
    if eos_token_id is not None and eos_token_id < logits.shape[-1]:
        eos_logit = logits[:, eos_token_id : eos_token_id + 1]
    if 0 < top_k < logits.shape[-1]:
        logits = apply_top_k(logits, top_k)
    logits = _apply_probability_filters(logits, top_p, 0.0)
    if eos_logit is not None:
        eos_idx = mx.full((batch, 1), eos_token_id, dtype=mx.int32)
        logits = mx.put_along_axis(logits, eos_idx, eos_logit, axis=-1)
    return logits


def _draw(logits, keys: _RowKeys, rows: Sequence[int], temperature: float):
    """One token per row ([batch, 1]), each from its own key."""
    import mlx.core as mx

    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)
    picks = [
        mx.random.categorical(logits[index : index + 1], key=keys.next(row))
        for index, row in enumerate(rows)
    ]
    return mx.concatenate(picks, axis=0)[:, None]


def generate_icl_batch(
    model: Any,
    texts: Sequence[str],
    seeds: Sequence[int],
    ref_audio: Any,
    ref_text: str,
    language: str = "auto",
    repetition_penalty: float = 1.2,
    temperature: float = QWEN_TEMPERATURE,
    top_k: int = QWEN_TOP_K,
    top_p: float = QWEN_TOP_P,
    max_tokens: int = QWEN_MAX_TOKENS,
) -> list[Any]:
    """Clone one voice for several (text, seed) rows in a single batched pass.

    Returns one MLX audio array per row (empty when a row produced no codes),
    decoded the same way as ``Model._generate_icl``.
    """
    import mlx.core as mx

    if len(texts) != len(seeds):
        raise ValueError("texts and seeds must have the same length.")
    batch = len(texts)
    if batch == 0:
        return []

    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = model._suppress_codec_tokens(eos_token_id)
    inputs = model._prepare_batch_inputs(
        list(texts),
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        return_metadata=True,
    )
    input_embeds = inputs.input_embeds
    trailing = inputs.trailing_text_hidden
    pad_embed = inputs.tts_pad_embed
    trailing_lens = mx.array(inputs.trailing_lens, dtype=mx.int32)
    # Rows with equal prompts need no mask; the unmasked path matches a
    # single-row run exactly.
    attention_mask = inputs.attention_mask if any(inputs.left_padding) else None

    keys = _RowKeys(seeds)
    cache = model.talker.make_cache()
    code_cache = model.talker.code_predictor.make_cache()
    codes: list[list[Any]] = [[] for _ in range(batch)]
    token_ids: list[list[int]] = [[] for _ in range(batch)]
    finished = [False] * batch
    trailing_idx = 0
    hidden_size = input_embeds.shape[-1]

    for _step in range(max_tokens):
        logits, hidden = model.talker(input_embeds, cache=cache, attention_mask=attention_mask)
        rows = list(range(batch))
        main_logits = _filter_logits(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=token_ids,
            suppress_tokens=suppress_tokens,
            eos_token_id=eos_token_id,
        )
        next_token = _draw(main_logits, keys, rows, temperature)

        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        for cache_layer in code_cache:
            cache_layer.keys = None
            cache_layer.values = None
            cache_layer.offset = 0
        for code_idx in range(config.num_code_groups - 1):
            if code_idx == 0:
                code_input = mx.concatenate(
                    [code_hidden, model.talker.get_input_embeddings()(next_token)], axis=1
                )
            else:
                code_input = model.talker.code_predictor.codec_embedding[code_idx - 1](
                    code_tokens[-1]
                )
            code_logits, code_cache, _ = model.talker.code_predictor(
                code_input, cache=code_cache, generation_step=code_idx
            )
            filtered = _filter_logits(
                code_logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            code_tokens.append(_draw(filtered, keys, rows, temperature))
        all_codes = mx.concatenate(code_tokens, axis=1)

        # Same text schedule as _generate_icl: the next trailing text embedding
        # while a row has any left, then the pad embedding.
        index = mx.minimum(mx.array(trailing_idx), trailing.shape[1] - 1)
        text_embed = trailing[:, index, :][:, None, :]
        exhausted = (trailing_lens <= trailing_idx)[:, None, None]
        text_embed = mx.where(
            exhausted, mx.broadcast_to(pad_embed, (batch, 1, hidden_size)), text_embed
        )
        trailing_idx += 1
        input_embeds = text_embed + model._codec_embeds_for_tokens(code_tokens)
        mx.eval(input_embeds, all_codes)

        token_list = next_token[:, 0].tolist()
        for row, token in enumerate(token_list):
            if finished[row]:
                continue
            if token == eos_token_id:
                finished[row] = True
                continue
            token_ids[row].append(int(token))
            codes[row].append(all_codes[row : row + 1])
        if all(finished):
            break
        if attention_mask is not None:
            attention_mask = mx.concatenate([attention_mask, mx.ones((batch, 1))], axis=1)

    del cache, code_cache, input_embeds
    # Rows decode one at a time: a batched vocoder pass measured slower here
    # (7.4 s vs 5.1 s for three rows) and used more memory.
    audios = []
    for row in range(batch):
        if codes[row]:
            audio = model._decode_icl_generated_codes(codes[row], inputs.ref_codes)
            mx.eval(audio)
        else:
            audio = mx.zeros((0,), dtype=mx.float32)
        codes[row] = []
        audios.append(audio)
    mx.clear_cache()
    return audios


@dataclass(eq=False)
class _Job:
    payload: Any
    size: int
    results: list[Any] = field(default_factory=list)
    remaining: int = 0
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.results = [None] * self.size
        self.remaining = self.size


@dataclass(eq=False)
class _Row:
    job: _Job
    index: int
    group: Hashable
    text: str
    seed: int


BatchRunner = Callable[[Any, list[str], list[int]], list[Any]]


class GenerationBatcher:
    """Merges concurrent generation jobs into shared batches.

    Each job is a list of (text, seed) rows plus a payload (the reference voice
    and settings). Rows whose group keys match may run in the same batch; the
    group key must change whenever the payload would. One background thread
    runs one batch at a time, so the model is never used concurrently.

    Rows join at batch boundaries: a row that arrives while a batch runs waits
    for the next batch. ``window_seconds`` is how long the worker waits after
    the first row arrives so that rows from other requests can join it.
    """

    def __init__(
        self,
        run_batch: BatchRunner,
        max_batch_size: int = 8,
        window_seconds: float = 0.05,
        name: str = "qwen-batcher",
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1.")
        self._run_batch = run_batch
        self.max_batch_size = max_batch_size
        self.window_seconds = max(0.0, window_seconds)
        self._name = name
        self._pending: list[_Row] = []
        self._running_rows = 0
        self._batches_run = 0
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(
        self, group: Hashable, payload: Any, texts: Sequence[str], seeds: Sequence[int]
    ) -> list[Any]:
        """Queue rows and block until all of them are generated, in row order."""
        if len(texts) != len(seeds):
            raise ValueError("texts and seeds must have the same length.")
        if not texts:
            return []
        job = _Job(payload=payload, size=len(texts))
        rows = [
            _Row(job=job, index=i, group=group, text=text, seed=int(seed))
            for i, (text, seed) in enumerate(zip(texts, seeds, strict=True))
        ]
        with self._condition:
            if self._closed:
                raise RuntimeError("The generation batcher is closed.")
            self._pending.extend(rows)
            self._ensure_worker()
            self._condition.notify_all()
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.results

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "queued_rows": len(self._pending),
                "queued_jobs": len({id(row.job) for row in self._pending}),
                "running_rows": self._running_rows,
                "batches_run": self._batches_run,
                "max_batch_size": self.max_batch_size,
            }

    def close(self) -> None:
        """Stop the worker once queued rows finish, so it releases the model."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name=self._name, daemon=True)
            self._thread.start()

    def _take_batch(self) -> list[_Row]:
        """Oldest row's group, FIFO, up to max_batch_size rows (lock held)."""
        group = self._pending[0].group
        batch: list[_Row] = []
        rest: list[_Row] = []
        for row in self._pending:
            if row.group == group and len(batch) < self.max_batch_size:
                batch.append(row)
            else:
                rest.append(row)
        self._pending = rest
        self._running_rows = len(batch)
        return batch

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending:
                    return
            if self.window_seconds:
                time.sleep(self.window_seconds)
            with self._condition:
                batch = self._take_batch()
            try:
                outputs = self._run_batch(
                    batch[0].job.payload, [row.text for row in batch], [row.seed for row in batch]
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"batch returned {len(outputs)} results for {len(batch)} rows"
                    )
            except BaseException as exc:  # noqa: BLE001 - delivered to every waiting job
                self._fail(batch, exc)
                continue
            finally:
                with self._condition:
                    self._running_rows = 0
                    self._batches_run += 1
            for row, output in zip(batch, outputs, strict=True):
                job = row.job
                if job.done.is_set():
                    continue
                job.results[row.index] = output
                job.remaining -= 1
                if job.remaining == 0:
                    job.done.set()

    def _fail(self, batch: list[_Row], exc: BaseException) -> None:
        failed = {id(row.job): row.job for row in batch}
        with self._condition:
            # Drop the failed jobs' other rows so they do not run for nothing.
            self._pending = [row for row in self._pending if id(row.job) not in failed]
        for job in failed.values():
            job.error = exc
            job.done.set()
