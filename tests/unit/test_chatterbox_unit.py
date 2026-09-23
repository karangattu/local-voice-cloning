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


def test_chatterbox_generation_contract(reference):
    calls = []

    class Model:
        sr = 24000

        def generate(self, text, audio_prompt_path=None, **kwargs):
            calls.append({'text': text, 'audio_prompt_path': audio_prompt_path, **kwargs})
            assert Path(audio_prompt_path).exists()
            return np.full(2400, 0.1, dtype=np.float32)

    cloner = module.LocalVoiceCloner(
        engine='chatterbox', quality='high', tts_loader=lambda _: Model(),
    )
    assert not cloner.model_loaded
    result = cloner.clone_voice(reference, 'First\nSecond', reference_text='Reference')
    assert result.sample_rate == 24000
    assert len(result.audio) == 4800 + 9600
    assert result.matched_voice.startswith('Chatterbox')
    assert calls[0]['text'] == 'First.'
    assert not Path(calls[0]['audio_prompt_path']).exists()


def test_chatterbox_handles_torch_tensor_output(reference):
    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.full(2400, 0.2, dtype=np.float32)

    model = SimpleNamespace(sr=24000, generate=lambda *a, **k: FakeTensor())
    cloner = module.LocalVoiceCloner(
        engine='chatterbox', tts_loader=lambda _: model,
    )
    result = cloner.clone_voice(reference, 'Hello', reference_text='Reference')
    assert result.sample_rate == 24000
    assert len(result.audio) == 2400


def test_chatterbox_sample_rate_comes_from_model_sr(reference):
    model = SimpleNamespace(sr=16000, generate=lambda *a, **k: np.zeros(1600, dtype=np.float32))
    cloner = module.LocalVoiceCloner(engine='chatterbox', tts_loader=lambda _: model)
    assert cloner._ensure_tts_model() is model
    assert cloner.sample_rate == 16000


def test_chatterbox_rejects_non_english_language(reference):
    cloner = module.LocalVoiceCloner(
        engine='chatterbox',
        tts_loader=lambda _: SimpleNamespace(sr=24000),
    )
    with pytest.raises(ValueError, match='English only'):
        cloner.clone_voice(reference, 'Hello', reference_text='Hi', language='Hindi')


def test_chatterbox_model_ids_follow_quality():
    assert module.CHATTERBOX_MODEL_IDS['high'] == 'ResembleAI/chatterbox'
    assert module.CHATTERBOX_MODEL_IDS['fast'] == 'ResembleAI/chatterbox-turbo'
    high = module.LocalVoiceCloner(engine='chatterbox', quality='high')
    fast = module.LocalVoiceCloner(engine='chatterbox', quality='fast')
    assert high.model_id == module.CHATTERBOX_MODEL_IDS['high']
    assert fast.model_id == module.CHATTERBOX_MODEL_IDS['fast']
    assert high.engine_name == 'Chatterbox'


def test_chatterbox_engine_cache_is_separate(monkeypatch):
    monkeypatch.setattr(module, '_shared_cloners', {})
    qwen = module.get_shared_cloner()
    chat = module.get_shared_cloner(engine='chatterbox')
    assert qwen is not chat
    assert module.get_shared_cloner(engine='chatterbox') is chat


def test_loader_picks_full_vs_turbo_class(monkeypatch):
    import sys

    seen = {}

    class Full:
        sr = 24000

        @classmethod
        def from_pretrained(cls, device=None):
            seen['full'] = device
            return 'full-model'

    class Turbo:
        sr = 24000

        @classmethod
        def from_pretrained(cls, device=None):
            seen['turbo'] = device
            return 'turbo-model'

    package = SimpleNamespace(__path__=[])
    monkeypatch.setitem(sys.modules, 'chatterbox', package)
    monkeypatch.setitem(sys.modules, 'chatterbox.tts', SimpleNamespace(ChatterboxTTS=Full))
    monkeypatch.setitem(
        sys.modules, 'chatterbox.tts_turbo', SimpleNamespace(ChatterboxTurboTTS=Turbo)
    )
    monkeypatch.setattr(module, 'chatterbox_device', lambda: 'cpu')
    assert module._load_chatterbox_model(module.CHATTERBOX_MODEL_IDS['high']) == 'full-model'
    assert module._load_chatterbox_model(module.CHATTERBOX_MODEL_IDS['fast']) == 'turbo-model'
    assert seen == {'full': 'cpu', 'turbo': 'cpu'}


def test_missing_chatterbox_package_has_setup_message(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == 'chatterbox.tts' or name.startswith('chatterbox.'):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(RuntimeError, match='pip install chatterbox-tts'):
        module._load_chatterbox_model(module.CHATTERBOX_MODEL_IDS['high'])


def test_api_routes_chatterbox_and_rejects_non_english(reference, monkeypatch):
    from src import api
    received = []

    def factory(quality, engine):
        received.append((quality, engine))
        return SimpleNamespace(clone_voice=lambda **kwargs: module.SynthesisResult(
            np.zeros(2400, dtype=np.float32), 24000, 0.1, 'Chatterbox'))

    monkeypatch.setattr(api, 'get_shared_cloner', factory)
    client = TestClient(app)
    response = client.post('/synthesize',
                           files={'reference_audio': ('ref.wav', reference.read_bytes())},
                           data={'text': 'Hello', 'engine': 'chatterbox'})
    assert response.status_code == 200
    assert response.content[:4] == b'RIFF'
    assert received == [('high', 'chatterbox')]
    response = client.post('/synthesize',
                           files={'reference_audio': ('ref.wav', reference.read_bytes())},
                           data={'text': 'Hello', 'engine': 'chatterbox', 'language': 'Hindi'})
    assert response.status_code == 422


def test_cli_chatterbox_option():
    args = parse_args(['-r', 'ref.wav', '-t', 'Hi', '--engine', 'chatterbox'])
    assert args.engine == 'chatterbox'
    assert args.language == 'auto'


def test_cli_chatterbox_rejects_non_english():
    with pytest.raises(SystemExit):
        parse_args(['-r', 'ref.wav', '-t', 'Hi', '--engine', 'chatterbox',
                    '--language', 'Hindi'])


def test_ui_has_chatterbox_engine_option():
    from app import app_ui
    rendered = str(app_ui)
    assert 'Chatterbox' in rendered
    assert 'chatterbox' in rendered
