# SPDX-License-Identifier: AGPL-3.0-or-later
"""The type a user gives ``localm pull`` or ``localm add`` survives when the file is
already registered under another name: an alias takes it, and a run with no one to
ask (a pull job the GUI started) takes that alias rather than skipping."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import huggingface_hub
import pytest

import localm
from localm import model_manager as mm
from localm.config import load_registry
from localm.model_manager import pull as pull_mod
from localm.model_manager import registry as registry_mod

REPO = Path(localm.__file__).resolve().parent.parent
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
BODY = b"the same bytes"
DIGEST = hashlib.sha256(BODY).hexdigest()


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    return h


@pytest.fixture
def have(home, tmp_path):
    """The file, registered as a lora named ``have``."""
    f = tmp_path / "file.safetensors"
    f.write_bytes(BODY)
    mm._register("have", f, "local", sha256=DIGEST, model_type="lora")
    return f


def _no_terminal(monkeypatch):
    monkeypatch.setattr(registry_mod, "can_prompt", lambda: False)
    monkeypatch.setattr(mm, "can_prompt", lambda: False)


def _terminal(monkeypatch, answer: str):
    import click
    monkeypatch.setattr(registry_mod, "can_prompt", lambda: True)
    monkeypatch.setattr(mm, "can_prompt", lambda: True)
    monkeypatch.setattr(click, "prompt", lambda *a, **k: answer)


def _assert_typed_alias(name: str = "new", model_type: str = "vae"):
    reg = load_registry()
    assert reg[name]["model_type"] == model_type
    assert reg[name]["type_user_set"] is True
    assert reg[name]["path"] == reg["have"]["path"]
    assert reg["have"]["model_type"] == "lora"
    assert "type_user_set" not in reg["have"]


def _no_download(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: calls.append(kw))
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, filename: DIGEST)
    return calls


class TestRegisterWithDedup:
    def test_an_alias_answer_takes_the_type(self, have, monkeypatch, capsys):
        _terminal(monkeypatch, "a")
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST, model_type="vae",
                                       type_user_set=True) is True
        _assert_typed_alias()
        assert "typed as vae" in capsys.readouterr().out

    def test_no_terminal_takes_a_typed_alias(self, have, monkeypatch):
        _no_terminal(monkeypatch)
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST, model_type="vae",
                                       type_user_set=True) is True
        _assert_typed_alias()

    def test_an_explicit_on_duplicate_alias_takes_the_type(self, have, monkeypatch):
        _no_terminal(monkeypatch)
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST, model_type="vae",
                                       on_duplicate="alias", type_user_set=True) is True
        _assert_typed_alias()

    def test_an_explicit_on_duplicate_skip_still_skips(self, have, monkeypatch):
        _no_terminal(monkeypatch)
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST, model_type="vae",
                                       on_duplicate="skip", type_user_set=True) is False
        assert "new" not in load_registry()

    def test_a_skip_answer_is_kept(self, have, monkeypatch):
        _terminal(monkeypatch, "s")
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST, model_type="vae",
                                       type_user_set=True) is False
        assert "new" not in load_registry()

    def test_no_type_and_no_terminal_still_skips(self, have, monkeypatch):
        _no_terminal(monkeypatch)
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST,
                                       model_type="lora") is False
        assert "new" not in load_registry()

    def test_a_plain_alias_copies_the_type_and_sets_no_flag(self, have, monkeypatch):
        _terminal(monkeypatch, "a")
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST,
                                       model_type="lora") is True
        reg = load_registry()
        assert reg["new"]["model_type"] == "lora"
        assert "type_user_set" not in reg["new"]

    def test_a_taken_alias_name_is_reported_as_not_registered(self, have, monkeypatch):
        _terminal(monkeypatch, "a")
        monkeypatch.setattr(registry_mod, "_sanitize_name", lambda name: "have")
        assert mm._register_with_dedup("new", have, "local", digest=DIGEST, model_type="vae",
                                       type_user_set=True) is False


class TestAddLocal:
    def test_a_type_equal_to_the_detected_one_still_beats_a_hand_typed_entry(
            self, have, monkeypatch):
        _no_terminal(monkeypatch)
        header = {"decoder.conv_in.weight": {"dtype": "F16", "shape": [8, 4, 3, 3],
                                             "data_offsets": [0, 0]},
                  "encoder.conv_in.weight": {"dtype": "F16", "shape": [8, 3, 3, 3],
                                             "data_offsets": [0, 0]}}
        raw = json.dumps(header).encode("utf-8")
        have.write_bytes(struct.pack("<Q", len(raw)) + raw)
        digest = hashlib.sha256(have.read_bytes()).hexdigest()
        mm._register("have", have, "local", sha256=digest, model_type="lora")
        assert mm.add_local(str(have), "new", model_type="vae") is True
        _assert_typed_alias()


class TestPullBeforeDownload:
    @pytest.mark.parametrize("arrange", [_no_terminal,
                                         lambda mp: _terminal(mp, "a")],
                             ids=["no_terminal", "alias_answer"])
    def test_a_hugging_face_file_takes_the_type(self, have, monkeypatch, arrange):
        downloads = _no_download(monkeypatch)
        arrange(monkeypatch)
        assert mm._pull_gguf_file("o/r:new.safetensors", "new", model_type="vae",
                                  type_user_set=True) is True
        assert downloads == []
        _assert_typed_alias()

    def test_a_hint_is_not_a_choice(self, have, monkeypatch):
        downloads = _no_download(monkeypatch)
        _no_terminal(monkeypatch)
        assert mm._pull_gguf_file("o/r:new.safetensors", "new", model_type="vae",
                                  type_user_set=False) is True
        assert downloads == []
        assert "new" not in load_registry()

    def test_the_same_name_takes_the_type_without_an_alias(self, home, monkeypatch):
        f = mm.MODELS_DIR / "sub" / "file.safetensors"
        f.parent.mkdir()
        f.write_bytes(BODY)
        mm._register("have", f, "local", sha256=DIGEST, model_type="lora")
        _no_download(monkeypatch)
        _no_terminal(monkeypatch)
        assert mm._pull_gguf_file("o/r:elsewhere.safetensors", "have", model_type="vae",
                                  type_user_set=True) is True
        reg = load_registry()
        assert list(reg) == ["have"]
        assert reg["have"]["model_type"] == "vae"
        assert reg["have"]["type_user_set"] is True

    @pytest.mark.parametrize("arrange", [_no_terminal,
                                         lambda mp: _terminal(mp, "a")],
                             ids=["no_terminal", "alias_answer"])
    def test_a_url_with_a_known_digest_takes_the_type(self, have, monkeypatch, arrange):
        arrange(monkeypatch)
        assert pull_mod._pull_url_locked(
            "https://x.test/new.safetensors", "new", "new.safetensors",
            expected_sha256=DIGEST, model_type="vae", type_user_set=True) is True
        _assert_typed_alias()

    def test_a_downloaded_url_that_is_a_duplicate_takes_the_type(self, have, monkeypatch):
        class Response:
            status_code = 200
            headers = {"content-length": str(len(BODY))}
            def raise_for_status(self):
                pass
            def iter_content(self, n):
                yield BODY

        monkeypatch.setattr("socket.getaddrinfo",
                            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        monkeypatch.setattr("localm.netpolicy.pinned_request", lambda method, url, **k: Response())
        monkeypatch.setattr(mm, "_check_disk_space", lambda d, b: True)
        _terminal(monkeypatch, "a")
        assert pull_mod._pull_url_locked("https://x.test/new.safetensors", "new",
                                         "new.safetensors", model_type="vae",
                                         type_user_set=True) is True
        _assert_typed_alias()
        assert not (mm.MODELS_DIR / "new.safetensors").exists()


class TestPullSnapshot:
    @pytest.fixture
    def snapshot(self, home):
        dest = home / "models" / "repo"
        dest.mkdir()
        (dest / "config.json").write_text("{}")
        mm._register("have", dest, "hf:o/r", model_type="lora")
        return dest

    @pytest.mark.parametrize("arrange", [_no_terminal,
                                         lambda mp: _terminal(mp, "a")],
                             ids=["no_terminal", "alias_answer"])
    def test_a_repo_already_pulled_takes_the_type(self, snapshot, monkeypatch, arrange):
        arrange(monkeypatch)
        assert pull_mod._pull_hf_snapshot_locked(
            "o/r", "new", False, "vae", None, None, True) is True
        _assert_typed_alias()


class TestPullAlreadyOnDisk:
    def _here(self, home) -> Path:
        dest = home / "models" / "new.safetensors"
        dest.write_bytes(BODY)
        return dest

    @pytest.mark.parametrize("type_user_set, expected", [(True, False), (False, True)],
                             ids=["typed", "untyped"])
    def test_a_name_held_by_another_file_fails_a_typed_pull_with_no_terminal(
            self, home, tmp_path, monkeypatch, type_user_set, expected):
        dest = mm.MODELS_DIR / "new.safetensors"
        dest.write_bytes(BODY)
        other = tmp_path / "other.safetensors"
        other.write_bytes(b"other bytes")
        mm._register("new", other, "local", model_type="lora")
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, filename: None)
        monkeypatch.setattr(pull_mod, "_refuse_name_held_by_external_file",
                            lambda name, dest: False)
        _no_terminal(monkeypatch)
        assert mm._pull_gguf_file("o/r:new.safetensors", "new", model_type="vae",
                                  type_user_set=type_user_set) is expected
        assert load_registry()["new"]["model_type"] == "lora"

    def test_a_person_who_declines_at_a_terminal_is_not_a_failure(
            self, home, tmp_path, monkeypatch):
        dest = mm.MODELS_DIR / "new.safetensors"
        dest.write_bytes(BODY)
        other = tmp_path / "other.safetensors"
        other.write_bytes(b"other bytes")
        mm._register("new", other, "local", model_type="lora")
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, filename: None)
        monkeypatch.setattr(pull_mod, "_refuse_name_held_by_external_file",
                            lambda name, dest: False)
        _terminal(monkeypatch, "s")
        monkeypatch.setattr("click.confirm", lambda *a, **k: False)
        assert mm._pull_gguf_file("o/r:new.safetensors", "new", model_type="vae",
                                  type_user_set=True) is True

    def test_a_file_already_here_with_no_terminal_takes_a_typed_alias(
            self, home, have, monkeypatch):
        dest = self._here(home)
        _no_terminal(monkeypatch)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo, filename: DIGEST)
        assert mm._pull_gguf_file("o/r:new.safetensors", "new", model_type="vae",
                                  type_user_set=True) is True
        reg = load_registry()
        assert reg["new"]["model_type"] == "vae"
        assert reg["new"]["type_user_set"] is True
        assert dest.exists()


class TestCivitai:
    def _pull(self, monkeypatch, dest_dir: Path, **kw):
        from localm.model_manager.sources import ResolvedDownload

        class Response:
            status_code = 200
            headers = {"content-length": str(len(BODY))}
            def raise_for_status(self):
                pass
            def iter_content(self, n):
                yield BODY

        resolved = ResolvedDownload(
            url="https://civitai.com/api/download/models/1?fileId=2", filename="new.safetensors",
            source_tag="civitai:1", model_type="lora", size_bytes=len(BODY), sha256=DIGEST,
            comfy_subfolder="loras", file_id="2")
        monkeypatch.setattr("localm.model_manager.sources.CivitAISource.resolve_download",
                            lambda self, ref, file, **k: resolved)
        monkeypatch.setattr("localm.media.managed_comfy.comfy_models_dest_dir",
                            lambda subfolder, cfg=None, plugin=None: dest_dir)
        monkeypatch.setattr("localm.model_manager.pull._ssrf_resolve_final_url", lambda url: url)
        monkeypatch.setattr("localm.netpolicy.pinned_request", lambda method, url, **k: Response())
        return pull_mod._pull_civitai_file("1", None, **kw)

    def test_a_chosen_type_is_recorded(self, home, tmp_path, monkeypatch):
        assert self._pull(monkeypatch, tmp_path / "loras", model_type="vae",
                          type_user_set=True) is True
        entry = load_registry()["new"]
        assert entry["model_type"] == "vae"
        assert entry["type_user_set"] is True

    def test_a_file_already_here_takes_a_typed_alias(self, have, tmp_path, monkeypatch):
        (tmp_path / "loras").mkdir()
        (tmp_path / "loras" / "new.safetensors").write_bytes(BODY)
        _no_terminal(monkeypatch)
        assert self._pull(monkeypatch, tmp_path / "loras", model_type="vae",
                          type_user_set=True) is True
        _assert_typed_alias()

    def test_the_type_reaches_the_civitai_pull(self, home, monkeypatch):
        seen = {}
        monkeypatch.setattr(mm, "_pull_civitai_file",
                            lambda *a, **kw: seen.update(kw) or True)
        assert mm.pull_model("civitai:1", None, model_type="vae") is True
        assert seen["type_user_set"] is True
        assert mm.pull_model("civitai:1", None, model_type="vae", type_is_hint=True) is True
        assert seen["type_user_set"] is False


class TestRealCommand:
    """A real ``localm pull`` of a registered file with stdin at the null device, as a
    GUI pull job starts it."""

    def _run(self, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
        home = tmp_path / "home"
        (home / "models").mkdir(parents=True, exist_ok=True)
        (home / "config.json").write_text(
            json.dumps({"update_url": "", "bugreport_upload_url": ""}), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(REPO), "LOCALM_HOME": str(home),
               "PYTHONIOENCODING": "utf-8", "COLUMNS": "200"}
        return subprocess.run([sys.executable, "-m", "localm", *args],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=120, env=env, cwd=str(REPO), stdin=subprocess.DEVNULL,
                              creationflags=NO_WINDOW)

    def _registry(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "home" / "registry.json").read_text(encoding="utf-8"))

    def test_a_typed_pull_of_a_registered_file_takes_a_typed_alias(self, tmp_path):
        f = tmp_path / "file.safetensors"
        f.write_bytes(BODY)
        assert self._run(tmp_path, "add", str(f), "-n", "first", "--type", "lora").returncode == 0
        run = self._run(tmp_path, "pull", str(f), "--name", "second", "--type", "vae")
        assert run.returncode == 0, run.stdout + run.stderr
        assert "typed as vae" in " ".join(run.stdout.split())
        reg = self._registry(tmp_path)
        assert reg["second"]["model_type"] == "vae"
        assert reg["second"]["type_user_set"] is True
        assert reg["first"]["model_type"] == "lora"

    def test_an_untyped_pull_of_a_registered_file_still_skips(self, tmp_path):
        f = tmp_path / "file.safetensors"
        f.write_bytes(BODY)
        assert self._run(tmp_path, "add", str(f), "-n", "first", "--type", "lora").returncode == 0
        self._run(tmp_path, "pull", str(f), "--name", "second")
        assert "second" not in self._registry(tmp_path)
