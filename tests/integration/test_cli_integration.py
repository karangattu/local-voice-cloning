"""Real-model test for src/cli.py. Slow. Run with: pytest -m integration"""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from src.cli import main

pytestmark = pytest.mark.integration


def test_main_successful_run(tmp_path):
    ref_file = tmp_path / "ref.wav"
    out_file = tmp_path / "out.wav"
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sf.write(str(ref_file), (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)

    with patch("sys.argv", ["cli.py", "-r", str(ref_file), "-t", "Hello test", "-o", str(out_file), "--steps", "8"]):
        main()

    assert out_file.exists()
