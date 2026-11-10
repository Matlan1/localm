# SPDX-License-Identifier: AGPL-3.0-or-later
"""A projector (mmproj) already on disk can be attached to a registered model
offline: `localm add --mmproj`, the local-path branch of `localm pull --mmproj`
and `localm attach-mmproj`. An --mmproj that is not applied is an error, and
`localm rm` / `localm bug-report` report failure with a non-zero exit."""

from __future__ import annotations

import json
import struct

import pytest
from click.testing import CliRunner

import localm.config as config
import localm.model_manager as model_manager
from localm.cli.maintenance import bug_report_cmd as bug_report
from localm.cli.models import add, attach_mmproj_cmd, rm

_STRING, _BOOL = 8, 7


def _s(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path, kv):
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(kv))]
    for key, vtype, val in kv:
        out.append(_s(key))
        out.append(struct.pack("<I", vtype))
        out.append(_s(val) if vtype == _STRING else struct.pack("<?", val))
    path.write_bytes(b"".join(out) + b"\0" * 64)
    return path


def _projector(path, audio=True):
    return _gguf(path, [("general.architecture", _STRING, "clip"),
                        ("clip.has_vision_encoder", _BOOL, not audio),
                        ("clip.has_audio_encoder", _BOOL, audio)])


def _model(path):
    return _gguf(path, [("general.architecture", _STRING, "llama")])


@pytest.fixture
def home(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(model_manager, "MODELS_DIR", models)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _registry(home):
    return json.loads((home / "registry.json").read_text(encoding="utf-8"))


def _register(home, name, path, model_type="tts"):
    (home / "registry.json").write_text(json.dumps(
        {name: {"path": str(path), "source": "local", "model_type": model_type}}),
        encoding="utf-8")


def test_attach_command_records_the_projector_without_loading(home, tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    model = _model(ext / "tts.gguf")
    _projector(ext / "mmproj-a.gguf")
    proj = _projector(ext / "mmproj-b.gguf")
    _register(home, "tts", model)

    result = CliRunner().invoke(attach_mmproj_cmd, ["tts", str(proj)])

    assert result.exit_code == 0, result.output
    assert "audio projector" in result.output
    assert _registry(home)["tts"]["mmproj"] == str(proj.resolve())


def test_attach_command_replaces_a_recorded_projector(home, tmp_path):
    model = _model(tmp_path / "tts.gguf")
    old = _projector(tmp_path / "old.gguf")
    new = _projector(tmp_path / "new.gguf")
    _register(home, "tts", model)
    assert model_manager.attach_mmproj("tts", str(old))[0]
    assert model_manager.attach_mmproj("tts", str(new))[0]
    assert _registry(home)["tts"]["mmproj"] == str(new.resolve())


@pytest.mark.parametrize("target, problem", [
    ("missing-model", "not a registered model"),
    ("tts", "not found"),
    ("tts", "not a projector"),
])
def test_attach_refusals_exit_1_and_write_nothing(home, tmp_path, target, problem):
    model = _model(tmp_path / "tts.gguf")
    notproj = _model(tmp_path / "plain.gguf")
    _register(home, "tts", model)
    arg = {"not found": str(tmp_path / "nope.gguf"),
           "not a projector": str(notproj)}.get(problem, str(notproj))

    result = CliRunner().invoke(attach_mmproj_cmd, [target, arg])

    assert result.exit_code == 1
    assert problem in result.output
    assert "mmproj" not in _registry(home)["tts"]


def test_add_with_mmproj_registers_and_attaches(home, tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    model = _model(ext / "voice.gguf")
    proj = _projector(ext / "mmproj-voice.gguf")

    result = CliRunner().invoke(
        add, [str(model), "--name", "voice", "--mmproj", str(proj), "--type", "tts",
              "--no-hash"])

    assert result.exit_code == 0, result.output
    assert _registry(home)["voice"]["mmproj"] == str(proj.resolve())


def test_add_with_an_unusable_mmproj_is_an_error(home, tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    model = _model(ext / "voice.gguf")

    result = CliRunner().invoke(
        add, [str(model), "--name", "voice", "--mmproj", str(ext / "gone.gguf"),
              "--type", "tts", "--no-hash"])

    assert result.exit_code == 1
    assert "--mmproj was not applied" in result.output
    assert "mmproj" not in _registry(home)["voice"]


def test_pull_of_a_local_file_applies_mmproj(home, tmp_path):
    from localm.model_manager.pull import pull_model
    ext = tmp_path / "ext"
    ext.mkdir()
    model = _model(ext / "voice.gguf")
    proj = _projector(ext / "mmproj-voice.gguf")

    ok = pull_model(str(model), "voice", mmproj_spec=str(proj), model_type="tts")

    assert ok is True
    assert _registry(home)["voice"]["mmproj"] == str(proj.resolve())


def test_pull_of_a_local_file_with_a_remote_mmproj_spec_fails(home, tmp_path):
    from localm.model_manager.pull import pull_model
    ext = tmp_path / "ext"
    ext.mkdir()
    model = _model(ext / "voice.gguf")

    ok = pull_model(str(model), "voice", mmproj_spec="owner/repo:mmproj-x.gguf",
                    model_type="tts")

    assert ok is False


def test_speech_model_resolves_once_a_projector_is_attached(home, tmp_path):
    from localm.inference import speech
    model = _model(tmp_path / "tts.gguf")
    proj = _projector(tmp_path / "mmproj.gguf")
    _register(home, "tts", model)
    with pytest.raises(speech.SpeechModelError) as e:
        speech.resolve_speech_model("tts")
    assert "localm attach-mmproj tts FILE" in str(e.value)

    model_manager.attach_mmproj("tts", str(proj))

    assert speech.resolve_speech_model("tts").mmproj == str(proj.resolve())


def test_run_mmproj_note_names_an_audio_projector(home, tmp_path):
    model = _model(tmp_path / "asr.gguf")
    proj = _projector(tmp_path / "mmproj-asr.gguf", audio=True)
    _register(home, "asr", model, model_type="llm")

    note = model_manager.persist_cli_mmproj("asr", str(proj))

    assert "audio projector" in note and "vision projector" not in note


def test_rm_of_an_unknown_name_exits_1_without_prompting(home):
    result = CliRunner().invoke(rm, ["nope"])
    assert result.exit_code == 1
    assert "Not found" in result.output
    assert "Remove" not in result.output.replace("Not found", "")


def test_rm_of_an_unknown_name_with_yes_exits_1(home):
    assert CliRunner().invoke(rm, ["nope", "-y"]).exit_code == 1


def test_bug_report_without_text_exits_1(home):
    result = CliRunner().invoke(bug_report, [], input="\n\n\n")
    assert result.exit_code == 1
    assert "Describe the problem" in result.output
