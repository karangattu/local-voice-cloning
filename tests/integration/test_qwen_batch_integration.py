"""Real-model tests for src/qwen_batch.py. Slow. Run with: pytest -m integration"""

import numpy as np
import pytest

from src.cloner import LocalVoiceCloner, seed_generators
from src.qwen_batch import generate_icl_batch

pytestmark = pytest.mark.integration

TEXT = "The chart updates as soon as you pick a date."
REF_TEXT = "A steady synthetic tone."


@pytest.fixture(scope="module")
def qwen():
    cloner = LocalVoiceCloner("fast")
    model = cloner._ensure_tts_model()
    sr = 24000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    import mlx.core as mx

    ref_audio = mx.array((0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
    return model, ref_audio


def _audio(array):
    return np.array(array, dtype=np.float32)


def test_one_row_batch_matches_a_seeded_sequential_take(qwen):
    model, ref_audio = qwen
    seed_generators(21)
    sequential = next(
        model._generate_icl(
            text=TEXT, ref_audio=ref_audio, ref_text=REF_TEXT, language="English",
            repetition_penalty=1.2,
        )
    ).audio
    batched = generate_icl_batch(model, [TEXT], [21], ref_audio, REF_TEXT, "English", 1.2)[0]
    assert np.array_equal(_audio(batched), _audio(sequential))


def test_batched_takes_are_reproducible_and_distinct(qwen):
    model, ref_audio = qwen
    first = generate_icl_batch(model, [TEXT] * 3, [5, 6, 7], ref_audio, REF_TEXT, "English", 1.2)
    again = generate_icl_batch(model, [TEXT] * 3, [5, 6, 7], ref_audio, REF_TEXT, "English", 1.2)
    for left, right in zip(first, again, strict=True):
        assert np.array_equal(_audio(left), _audio(right))
    assert not np.array_equal(_audio(first[0])[:24000], _audio(first[1])[:24000])
