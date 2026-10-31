import numpy as np
import pytest
import soundfile as sf

import src.cloner as cloner_module


@pytest.fixture(autouse=True)
def load_icl_reference_without_mlx(monkeypatch):
    """Read the cloning reference with soundfile instead of MLX.

    Linux CI has no MLX runtime, so a fake model that takes Qwen's
    voice-cloning path must not import mlx_audio."""

    def load(path, sample_rate):
        audio, actual_sample_rate = sf.read(str(path), dtype="float32")
        assert actual_sample_rate == sample_rate
        return np.asarray(audio)

    monkeypatch.setattr(cloner_module, "_load_icl_reference", load)
