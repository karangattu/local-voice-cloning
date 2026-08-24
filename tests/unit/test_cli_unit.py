from unittest.mock import patch

import pytest

from src.cli import main, parse_args


def test_parse_args_defaults():
    args = parse_args(["--reference", "sample.wav", "--text", "Test speech"])
    assert args.reference == "sample.wav"
    assert args.text == "Test speech"
    assert args.output == "output.wav"
    assert args.speed == 1.0
    assert args.quality == "high"
    assert args.language == "auto"
    assert args.steps is None
    assert args.format is None


def test_parse_args_mp3_format():
    args = parse_args(["-r", "sample.wav", "-t", "Test", "-o", "out.mp3", "-f", "mp3"])
    assert args.output == "out.mp3"
    assert args.format == "mp3"


def test_parse_args_fast_quality():
    args = parse_args(["-r", "sample.wav", "-t", "Test", "--quality", "fast"])
    assert args.quality == "fast"


def test_parse_args_english_language():
    args = parse_args(["-r", "sample.wav", "-t", "Test", "--language", "English"])
    assert args.language == "English"


def test_main_file_not_found():
    with patch("sys.argv", ["cli.py", "--reference", "non_existent.wav", "--text", "Test"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
