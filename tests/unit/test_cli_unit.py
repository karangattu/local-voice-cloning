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


def test_parse_args_warmup():
    args = parse_args(["--warmup", "--engine", "qwen", "--quality", "fast"])
    assert args.warmup is True
    assert args.engine == "qwen"
    assert args.quality == "fast"
    assert args.reference is None
    assert args.text is None


def test_parse_args_requires_reference_and_text_unless_warmup():
    with pytest.raises(SystemExit) as exc:
        parse_args([])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("engine", "include_transcriber"),
    [("qwen", True), ("omnivoice", True), ("chatterbox", False)],
)
def test_main_warmup_loads_models(mocker, engine, include_transcriber):
    cloner = mocker.patch("src.cli.LocalVoiceCloner")
    cloner.return_value.warmup.return_value = {"tts": 0.5}
    cloner.return_value.model_id = "model-id"

    with patch("sys.argv", ["cli.py", "--warmup", "--engine", engine]):
        main()

    cloner.return_value.warmup.assert_called_once_with(include_transcriber=include_transcriber)


def test_main_warmup_failure_exits_one(mocker, capsys):
    cloner = mocker.patch("src.cli.LocalVoiceCloner")
    cloner.return_value.warmup.side_effect = RuntimeError(
        "model download failed for model-id; check network/HF access and rerun warmup"
    )

    with patch("sys.argv", ["cli.py", "--warmup"]), pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert "model download failed" in capsys.readouterr().err


def test_main_file_not_found():
    with patch("sys.argv", ["cli.py", "--reference", "non_existent.wav", "--text", "Test"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
