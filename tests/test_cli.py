from unittest.mock import patch

import pytest

from src.cli import main, parse_args


def test_parse_args_defaults():
    args = parse_args(["--reference", "sample.wav", "--text", "Test speech"])
    assert args.reference == "sample.wav"
    assert args.text == "Test speech"
    assert args.output == "output.wav"
    assert args.speed == 1.0
    assert args.steps is None
    assert args.format is None


def test_parse_args_mp3_format():
    args = parse_args(["-r", "sample.wav", "-t", "Test", "-o", "out.mp3", "-f", "mp3"])
    assert args.output == "out.mp3"
    assert args.format == "mp3"


def test_main_file_not_found():
    with patch("sys.argv", ["cli.py", "--reference", "non_existent.wav", "--text", "Test"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1


def test_main_successful_run(tmp_path):
    ref_file = tmp_path / "ref.wav"
    out_file = tmp_path / "out.wav"
    import numpy as np
    import soundfile as sf
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sf.write(str(ref_file), (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)

    with patch("sys.argv", ["cli.py", "-r", str(ref_file), "-t", "Hello test", "-o", str(out_file), "--steps", "8"]):
        main()

    assert out_file.exists()
