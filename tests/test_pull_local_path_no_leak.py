# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registering a local model file must not phone home.

Auto model-type detection must not run before pull_model's is_local_path branch
and the net_mode kill switch. Placed above them, it reaches
_hf_pipeline_tag_to_type() for any spec containing "/" that is not an http(s)
URL - every POSIX absolute path (/home/u/model.gguf) and every forward-slash
relative path (models/secret.gguf) - which issues a real HTTPS GET to
https://huggingface.co/api/models/<the local path>, leaking the private local
path and model filename to a third party.
"""

import localm.model_manager.pull as pull


def test_local_path_pull_does_not_call_hf_detect(tmp_path, monkeypatch):
    # A real local file with forward slashes in its path (the leak trigger).
    f = tmp_path / "private-finetune.gguf"
    f.write_bytes(b"GGUF fake")

    hf_calls = []
    monkeypatch.setattr(pull, "_hf_pipeline_tag_to_type",
                        lambda repo_id: hf_calls.append(repo_id) or "llm")

    add_local_calls = []

    def fake_add_local(path_str, **kw):
        add_local_calls.append((path_str, kw))
        return True

    monkeypatch.setattr(pull._mm, "add_local", fake_add_local)

    # Use a forward-slash absolute path so `"/" in spec` is true (the trigger).
    spec = f.as_posix()
    assert "/" in spec
    ok = pull.pull_model(spec, model_type="auto")

    assert ok is True
    assert hf_calls == [], (
        "registering a local file must NOT call huggingface.co "
        f"(leaked: {hf_calls})")
    assert add_local_calls, "local file should be registered via add_local"
    # Local files carry NO remote type probe: an 'auto' type passes None to
    # add_local so it detects the type OFFLINE (GGUF -> llm, HF dir -> config.json,
    # else the 'unknown' sentinel), never a huggingface.co lookup.
    assert add_local_calls[0][1].get("model_type") is None


def test_local_path_pull_offline_when_net_off(tmp_path, monkeypatch):
    """net_mode=off must not stop registering a LOCAL file (it needs no network),
    and still must not probe HF."""
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF")

    hf_calls = []
    monkeypatch.setattr(pull, "_hf_pipeline_tag_to_type",
                        lambda repo_id: hf_calls.append(repo_id) or "llm")
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "off")
    monkeypatch.setattr(pull._mm, "add_local", lambda path_str, **kw: True)

    assert pull.pull_model(f.as_posix(), model_type="auto") is True
    assert hf_calls == []


def test_remote_auto_detect_still_runs(monkeypatch):
    """Guard: a genuine remote HF spec must still auto-detect its type."""
    hf_calls = []
    monkeypatch.setattr(pull, "_hf_pipeline_tag_to_type",
                        lambda repo_id: hf_calls.append(repo_id) or "vae")
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")
    # resolve_spec passes a bare owner/repo through unchanged; a no-filename
    # owner/repo dispatches to _pull_hf_snapshot.
    monkeypatch.setattr(pull._mm, "resolve_spec", lambda s: s)
    monkeypatch.setattr(pull._mm, "_pull_hf_snapshot",
                        lambda *a, **k: True)

    pull.pull_model("someowner/some-vae-model", model_type="auto")
    assert hf_calls == ["someowner/some-vae-model"], "remote spec must still probe HF"


def test_remote_gguf_file_spec_types_llm_without_probing(monkeypatch):
    """A remote spec that names a .gguf file (owner/repo:file.gguf OR
    owner/repo/file.gguf) is a HARD LLM signal via the filename - it must type
    'llm' WITHOUT probing the repo's HF pipeline_tag, which a GGUF-quant repo often
    lacks - a probe would label it 'unknown', which HIDES the pulled model from
    the launcher and blocks auto-chat."""
    hf_calls = []
    monkeypatch.setattr(pull, "_hf_pipeline_tag_to_type",
                        lambda repo_id: hf_calls.append(repo_id) or "unknown")
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")
    monkeypatch.setattr(pull._mm, "resolve_spec", lambda s: s)
    got = {}
    monkeypatch.setattr(pull._mm, "_pull_gguf_file",
                        lambda spec, name, **kw: got.update(mt=kw.get("model_type")) or True)
    for spec in ("owner/SmolLM2-GGUF:SmolLM2-Q8_0.gguf", "owner/repo/model.gguf"):
        got.clear()
        assert pull.pull_model(spec, model_type="auto") is True
        assert got.get("mt") == "llm", (spec, got)
    assert hf_calls == [], f"a named .gguf must not probe HF (probed: {hf_calls})"
