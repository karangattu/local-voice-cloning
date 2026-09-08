from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import src.cloner as module
from src.api import app
from src.cli import parse_args


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / 'reference.wav'
    sf.write(path, np.full(24000, 0.1), 24000)
    return path


@pytest.mark.parametrize('quality,steps', [('high', 32), ('fast', 16)])
def test_omni_generation_contract(reference, quality, steps):
    calls = []

    class Model:
        sampling_rate = 24000

        def generate(self, **kwargs):
            calls.append(kwargs)
            assert Path(kwargs['ref_audio']).exists()
            return [np.full(2400, 0.1, dtype=np.float32)]

    cloner = module.LocalVoiceCloner(
        engine='omnivoice', quality=quality, tts_loader=lambda _: Model(),
    )
    assert not cloner.model_loaded
    result = cloner.clone_voice(reference, 'First\nSecond', reference_text='Reference',
                                language='Hindi', speed=1.2)
    assert result.sample_rate == 24000
    assert len(result.audio) == 4800 + 9600
    assert result.matched_voice.startswith('OmniVoice')
    assert calls[0]['language'] == 'Hindi'
    assert calls[0]['num_step'] == steps
    assert calls[0]['speed'] == 1.2
    assert not Path(calls[0]['ref_audio']).exists()


def test_omni_auto_transcription_and_empty_output(reference):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return []

    cloner = module.LocalVoiceCloner(
        engine='omnivoice',
        tts_loader=lambda _: SimpleNamespace(generate=generate, sampling_rate=24000),
        stt_loader=lambda _: SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(text='Words')),
    )
    with pytest.raises(RuntimeError, match='no audio'):
        cloner.clone_voice(reference, 'Hello')
    assert calls[0]['ref_text'] == 'Words'
    assert calls[0]['language'] is None
    assert not Path(calls[0]['ref_audio']).exists()


def test_engine_caches_are_separate(monkeypatch):
    monkeypatch.setattr(module, '_shared_cloners', {})
    qwen = module.get_shared_cloner()
    omni = module.get_shared_cloner(engine='omnivoice')
    assert qwen is not omni
    assert module.get_shared_cloner(engine='omnivoice') is omni
    with pytest.raises(ValueError, match='Unknown engine'):
        module.get_shared_cloner(engine='missing')


def test_api_routes_omni_and_accepts_extra_languages(reference, monkeypatch):
    from src import api
    received = []

    def factory(quality, engine):
        received.append((quality, engine))
        return SimpleNamespace(clone_voice=lambda **kwargs: module.SynthesisResult(
            np.zeros(2400, dtype=np.float32), 24000, 0.1, 'OmniVoice'))

    monkeypatch.setattr(api, 'get_shared_cloner', factory)
    client = TestClient(app)
    response = client.post('/synthesize',
                           files={'reference_audio': ('ref.wav', reference.read_bytes())},
                           data={'text': 'Hello', 'engine': 'omnivoice', 'language': 'Hindi'})
    assert response.status_code == 200
    assert response.content[:4] == b'RIFF'
    assert received == [('high', 'omnivoice')]
    response = client.post('/synthesize',
                           files={'reference_audio': ('ref.wav', reference.read_bytes())},
                           data={'text': 'Hello', 'engine': 'bad'})
    assert response.status_code == 422


def test_cli_omni_option():
    args = parse_args(['-r', 'ref.wav', '-t', 'Hi', '--engine', 'omnivoice', '--language', 'Hindi'])
    assert args.engine == 'omnivoice'
    assert args.language == 'Hindi'


def test_missing_optional_package_has_setup_message(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == 'omnivoice':
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(RuntimeError, match='uv sync --extra omnivoice'):
        module._load_omnivoice_model(module.OMNIVOICE_MODEL_ID)


def test_ui_has_engine_and_extended_language_controls():
    from app import app_ui
    rendered = str(app_ui)
    assert 'Voice engine' in rendered
    assert 'OmniVoice' in rendered
    assert 'omni_language' in rendered


@pytest.mark.parametrize('device', ['mps', 'cpu', 'cuda:0'])
def test_loader_disables_async_weight_loading_only_on_mps(monkeypatch, device):
    import os
    import sys

    import torch

    calls = []
    model = object()

    def from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs, os.environ.get('HF_DEACTIVATE_ASYNC_LOAD')))
        return model

    monkeypatch.setitem(sys.modules, 'omnivoice', SimpleNamespace(
        OmniVoice=SimpleNamespace(from_pretrained=from_pretrained)))
    monkeypatch.setattr(module, 'omnivoice_device', lambda: device)
    monkeypatch.delenv('HF_DEACTIVATE_ASYNC_LOAD', raising=False)
    assert module._load_omnivoice_model(module.OMNIVOICE_MODEL_ID) is model
    model_id, kwargs, async_setting = calls[0]
    assert model_id == module.OMNIVOICE_MODEL_ID
    assert kwargs['device_map'] == device
    assert kwargs['dtype'] == (torch.float32 if device == 'cpu' else torch.float16)
    assert async_setting == ('1' if device == 'mps' else None)
