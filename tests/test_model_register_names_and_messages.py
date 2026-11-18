# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a model is registered as, and what the registering commands say about it:
a folder name with a dot, an ACE-Step checkpoint, a renamed model named by a
project's coder config, the empty result of a typed list, a duplicate answered
for a GUI click, a gated repo, and the Activity label of a local path."""

import json
import struct

import pytest

from localm import model_manager as mm
from tests.test_gui import gui_app  # noqa: F401
from tests.test_model_dedup import _file, fake_registry  # noqa: F401


def _safetensors(path, names):
    header = {n: {"dtype": "F16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
              for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(2 * len(names)))
    return path


def test_a_model_folder_with_a_dot_keeps_its_whole_name(fake_registry, tmp_path):  # noqa: F811
    store, _ = fake_registry
    folder = tmp_path / "Ouro-1.4B"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    _safetensors(folder / "model.safetensors", ["w"])
    assert mm.add_local(str(folder))
    assert "Ouro-1.4B" in store, sorted(store)
    assert "Ouro-1" not in store


def test_an_ace_step_checkpoint_registers_as_a_diffusion_model(tmp_path):
    from localm.model_manager.registry import _media_component_type
    ckpt = _safetensors(tmp_path / "ace_step_v1_3.5b.safetensors", [
        "model.diffusion_model.genre_embedder.w", "model.diffusion_model.lyric_embs.w",
        "text_encoders.umt5base.x"])
    assert _media_component_type(ckpt, None) == "diffusion-unet"


def test_a_typed_list_with_models_of_other_types_does_not_say_the_library_is_empty(
        fake_registry, tmp_path, capsys):  # noqa: F811
    store, _ = fake_registry
    store["chat"] = {"path": str(_file(tmp_path, "c.gguf")), "source": "local",
                     "model_type": "llm"}
    mm.list_models("lora")
    out = capsys.readouterr().out
    assert "No models yet" not in out
    assert "lora" in out and "localm list" in out


def test_an_empty_library_still_says_so(fake_registry, capsys):  # noqa: F811
    mm.list_models("lora")
    assert "No models yet" in capsys.readouterr().out


def test_the_duplicate_question_has_a_gui_answer_and_a_script_answer(monkeypatch):
    from localm.model_manager import registry
    monkeypatch.setattr(registry, "can_prompt", lambda: False)
    monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
    assert registry._prompt_predownload_dup(["x"], "new") == "skip"
    assert registry._prompt_duplicate_action(["x"], "same file") == "skip"
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
    assert registry._prompt_predownload_dup(["x"], "new") == "alias"
    assert registry._prompt_duplicate_action(["x"], "same file") == "alias"


def test_adding_a_registered_file_under_a_new_name_from_the_gui_adds_the_name(
        fake_registry, tmp_path, monkeypatch):  # noqa: F811
    from localm.model_manager import registry
    store, _ = fake_registry
    monkeypatch.setattr(registry, "can_prompt", lambda: False)
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
    f = _file(tmp_path, "m.gguf")
    assert mm.add_local(str(f), "first", on_duplicate="register")
    assert mm.add_local(str(f), "second")
    assert store["second"]["path"] == store["first"]["path"]


def _gated_error():
    import httpx
    from huggingface_hub.errors import GatedRepoError
    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/x/y"))
    return GatedRepoError("401 Client Error. Cannot access gated repo", response=response)


class _Gated:
    def auth_check(self, repo_id, **kwargs):
        raise _gated_error()

    def model_info(self, *a, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(siblings=[], sha=None, gated="auto")


@pytest.mark.parametrize("spec", ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B:m.gguf"])
def test_a_gated_repo_is_refused_in_localms_words_and_leaves_nothing_behind(
        spec, fake_registry, monkeypatch, capsys):  # noqa: F811
    import huggingface_hub
    from localm.model_manager import pull

    def refuse(*a, **kw):
        raise _gated_error()

    _store, models_dir = fake_registry
    monkeypatch.setattr(pull, "_hf_api", lambda: _Gated())
    monkeypatch.setattr(pull, "_hf_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    assert pull.pull_model(spec) is False
    out = capsys.readouterr().out
    assert "is gated" in out and "Hugging Face token" in out
    assert "401" not in out and "Request ID" not in out
    assert [p for p in models_dir.iterdir() if p.name != ".cache"] == []


def test_a_local_path_job_is_labelled_as_a_registration(monkeypatch):
    from localm.plugins.gui.routes.models.acquisition import _pull_label
    assert _pull_label("D:/models/m.gguf") == "Register D:/models/m.gguf"
    assert _pull_label("/srv/m.gguf") == "Register /srv/m.gguf"
    assert _pull_label("owner/repo:m.gguf") == "Model pull owner/repo:m.gguf"


def test_the_pull_route_labels_a_local_path_job_as_a_registration(gui_app, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient
    from tests.test_pull_duplicate_job_409 import _no_real_downloads
    app, _ = gui_app
    seen = _no_real_downloads(monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/models/pull", json={"spec": "D:/models/m.gguf"})
    assert r.status_code == 200, r.text
    assert seen["host_label"] == "Register D:/models/m.gguf", seen
