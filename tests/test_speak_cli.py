# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm speak``: text to a WAV file, with progress, voices and clear errors.
The synthesis itself is replaced at ``localm.inference.speech.synthesize``."""

import json

import pytest
from click.testing import CliRunner

from localm.cli import main
from localm.inference import speech
from localm.inference.backends.llamacpp import mtmd_gen as g


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    m = tmp_path / "q.gguf"
    m.write_bytes(b"GGUF")
    mm = tmp_path / "mmproj-q.gguf"
    mm.write_bytes(b"GGUF")
    (tmp_path / "registry.json").write_text(json.dumps({"q": {
        "path": str(m), "source": "local", "model_type": "tts", "mmproj": str(mm)}}),
        encoding="utf-8")
    return tmp_path


@pytest.fixture
def synth(monkeypatch):
    calls = []

    def fake(model, text, *, language=None, reference_wav=None, seed=None,
             on_progress=None, should_cancel=None):
        calls.append({"model": model.name, "text": text, "language": language,
                      "reference": reference_wav, "seed": seed})
        on_progress({"stage": "loading"})
        on_progress({"stage": "speaking", "frames": 25, "seconds": 2.0})
        if fake.error is not None:
            raise fake.error
        return speech.SpeechOutput(wav=b"RIFFdata", sample_rate=24000, n_samples=48000,
                                   frames=25, seed=seed if seed is not None else 99)

    fake.error = None
    fake.calls = calls
    monkeypatch.setattr(speech, "synthesize", fake)
    return fake


def run(*args):
    return CliRunner().invoke(main, ["speak", *args])


def test_writes_the_wav_and_reports_the_seed(home, synth, tmp_path):
    out = tmp_path / "o.wav"
    r = run("Hello there.", "-o", str(out))
    assert r.exit_code == 0, r.output
    assert out.read_bytes() == b"RIFFdata"
    assert "2.00 s of audio, seed 99" in r.output
    assert synth.calls[0]["model"] == "q" and synth.calls[0]["reference"] is None


def test_progress_lines_are_shown_off_a_terminal(home, synth, tmp_path):
    r = run("hi", "-o", str(tmp_path / "o.wav"))
    assert "Loading the speech model" in r.stderr and "Speaking: 2.0 s of audio" in r.stderr


def test_text_from_a_file_with_voice_language_and_seed(home, synth, tmp_path):
    (tmp_path / "t.txt").write_text("From a file.", encoding="utf-8")
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFref")
    r = run("--file", str(tmp_path / "t.txt"), "--voice-file", str(ref), "--language", "en",
            "--seed", "5", "-o", str(tmp_path / "o.wav"))
    assert r.exit_code == 0, r.output
    assert synth.calls[0] == {"model": "q", "text": "From a file.", "language": "en",
                              "reference": b"RIFFref", "seed": 5}


def test_a_named_voice(home, synth, tmp_path):
    (home / "voices").mkdir()
    (home / "voices" / "narrator.wav").write_bytes(b"RIFFn")
    assert run("hi", "--voice", "narrator", "-o", str(tmp_path / "o.wav")).exit_code == 0
    assert synth.calls[0]["reference"] == b"RIFFn"


def test_list_voices(home):
    r = run("--list-voices")
    assert r.exit_code == 0 and r.output.splitlines()[0] == "default"


@pytest.mark.parametrize("args,needle", [
    (["-o", "x.wav"], "give the text"),
    (["hi"], "-o/--output"),
    (["hi", "-o", "x.wav", "--voice", "a", "--voice-file", __file__], "not both"),
])
def test_usage_errors(home, synth, args, needle):
    r = run(*args)
    assert r.exit_code == 2 and needle in r.output and synth.calls == []


def test_an_unknown_voice_or_model_exits_1(home, synth, tmp_path):
    r = run("hi", "--voice", "alloy", "-o", str(tmp_path / "o.wav"))
    assert r.exit_code == 1 and "not available" in r.output
    r = run("hi", "--model", "nope", "-o", str(tmp_path / "o.wav"))
    assert r.exit_code == 1 and "not registered" in r.output
    assert synth.calls == []


def test_a_synthesis_error_exits_1_and_writes_nothing(home, synth, tmp_path):
    synth.error = g.SpeechBudgetExceeded("did not finish speaking")
    out = tmp_path / "o.wav"
    r = run("hi", "-o", str(out))
    assert r.exit_code == 1 and "did not finish speaking" in r.output
    assert not out.exists() and not (tmp_path / "o.wav.part").exists()
