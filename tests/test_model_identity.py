# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model's ID is what stored references point at; its name is what people see.

Every test runs the real registry, config, job store, RAG store, workflow store
and conversation routes against real files under a throwaway home.
"""

import json
import threading

import pytest

from localm import model_ids
from localm import model_manager as mm


@pytest.fixture()
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def _add(tmp_path, name, content=b"model-bytes"):
    f = tmp_path / f"{name}.gguf"
    f.write_bytes(content + name.encode())
    assert mm.add_local(str(f), name, on_duplicate="register")
    return f


def _raw(home, name="registry.json"):
    return json.loads((home / name).read_text(encoding="utf-8"))


def _rid(home, name):
    return _raw(home)[name]["id"]


class TestIdLife:
    def test_registration_assigns_an_id_unique_to_the_entry(self, home, tmp_path):
        _add(tmp_path, "a")
        _add(tmp_path, "b")
        assert _rid(home, "a") and _rid(home, "b")
        assert _rid(home, "a") != _rid(home, "b")

    def test_entries_without_an_id_get_one_on_first_load_and_keep_it(self, home):
        (home / "registry.json").write_text(json.dumps({
            "a": {"path": "x/a.gguf", "source": "local"},
            "b": {"path": "x/b.gguf", "source": "local"}}), encoding="utf-8")
        first = mm.load_registry()
        assert first["a"]["id"] and first["b"]["id"]
        assert first["a"]["id"] != first["b"]["id"]
        assert _rid(home, "a") == first["a"]["id"], "the ID was not written back"
        assert mm.load_registry() == first

    def test_a_copy_of_an_entry_with_the_same_id_gets_its_own(self, home):
        (home / "registry.json").write_text(json.dumps({
            "a": {"path": "x/a.gguf", "id": "same", "former_names": ["z"]},
            "b": {"path": "x/a.gguf", "id": "same", "former_names": ["z"]}}),
            encoding="utf-8")
        reg = mm.load_registry()
        assert reg["a"]["id"] == "same"
        assert reg["b"]["id"] != "same"
        assert "former_names" not in reg["b"]

    def test_registering_a_name_again_keeps_its_id(self, home, tmp_path):
        f = _add(tmp_path, "a")
        before = _rid(home, "a")
        mm._register("a", f, "local")
        assert _rid(home, "a") == before

    def test_an_alias_is_a_second_entry_with_its_own_id_for_the_same_file(self, home, tmp_path):
        _add(tmp_path, "a")
        assert mm.alias_model("a", "b")
        reg = _raw(home)
        assert reg["a"]["path"] == reg["b"]["path"]
        assert reg["a"]["id"] != reg["b"]["id"]

    def test_renaming_a_name_back_leaves_no_stale_former_name(self, home, tmp_path):
        _add(tmp_path, "a")
        before = _rid(home, "a")
        assert mm.rename_model("a", "b")
        assert mm.rename_model("b", "a")
        entry = _raw(home)["a"]
        assert entry["id"] == before
        assert entry["former_names"] == ["b"]


class TestRename:
    def test_rename_changes_the_key_and_nothing_else_about_the_entry(self, home, tmp_path):
        f = _add(tmp_path, "a")
        before = dict(_raw(home)["a"])
        content = f.read_bytes()
        assert mm.rename_model("a", "b")
        after = _raw(home)
        assert "a" not in after
        entry = dict(after["b"])
        assert entry.pop("former_names") == ["a"]
        assert entry == before
        assert f.read_bytes() == content

    @pytest.mark.parametrize("bad", ["", "   ", "...", "///"])
    def test_a_name_with_nothing_usable_is_refused(self, home, tmp_path, bad):
        _add(tmp_path, "a")
        assert mm.rename_model("a", bad) is False
        assert "a" in _raw(home)
        assert "model" not in _raw(home), "the fallback name must not be used for a rename"

    def test_a_name_another_entry_has_is_refused(self, home, tmp_path):
        _add(tmp_path, "a")
        _add(tmp_path, "b")
        assert mm.rename_model("a", "b") is False
        assert sorted(_raw(home)) == ["a", "b"]

    def test_a_failed_first_time_reference_recording_does_not_stop_the_rename(
            self, home, tmp_path, monkeypatch):
        _add(tmp_path, "a")
        monkeypatch.setattr("localm.rag.store.record_embedding_model_refs",
                            lambda: (_ for _ in ()).throw(RuntimeError("no")))
        ok, notes = mm.rename_model_with_notes("a", "b")
        assert ok is True
        assert any("knowledge" in n for n in notes)
        assert not (home / "model_refs.migrated").exists()


class TestStoredReferences:
    """Each place holds a name a person chose. After a rename it reads as the new
    name, and a new entry taking the old name does not capture it."""

    def _rename_and_reuse(self, tmp_path):
        assert mm.rename_model("orig", "renamed")
        _add(tmp_path, "orig", b"a different model")

    def test_config(self, home, tmp_path):
        from localm.config import load_config, update_config
        _add(tmp_path, "orig")
        _add(tmp_path, "other")

        def _set(cfg):
            cfg["pinned_models"] = ["orig", "other", "not-registered"]
            cfg["embedding_model"] = "orig"
            cfg["rag_rerank_model"] = "orig"
            cfg["coder_reviewer_model"] = "orig"
            cfg["spec_draft_model"] = "orig"
            cfg["default_model"] = "orig"
            cfg["model_avatar_overrides"] = {"orig": "icon"}
            cfg["plugins"] = {"image": {"native": {
                "model": "orig", "vae": "orig", "steps": "orig", "clip_l": "C:/a/file.bin"}}}
        update_config(_set)
        self._rename_and_reuse(tmp_path)

        cfg = load_config()
        assert cfg["pinned_models"] == ["renamed", "other", "not-registered"]
        for key in ("embedding_model", "rag_rerank_model", "coder_reviewer_model",
                    "spec_draft_model", "default_model"):
            assert cfg[key] == "renamed", key
        assert cfg["model_avatar_overrides"] == {"renamed": "icon"}
        native = cfg["plugins"]["image"]["native"]
        assert native["model"] == native["vae"] == "renamed"
        assert native["steps"] == "orig", "a key that is not a model slot was rewritten"
        assert native["clip_l"] == "C:/a/file.bin"
        assert model_ids.REFS_KEY not in cfg, "the sidecar must not reach config readers"

    def test_a_config_write_after_the_rename_keeps_the_reference(self, home, tmp_path):
        from localm.config import load_config, save_config, update_config
        _add(tmp_path, "orig")
        update_config(lambda c: c.__setitem__("embedding_model", "orig"))
        self._rename_and_reuse(tmp_path)
        save_config(load_config())
        stored = _raw(home, "config.json")
        assert stored["embedding_model"] == "renamed"
        assert stored[model_ids.REFS_KEY] == {"renamed": _rid(home, "renamed")}
        assert load_config()["embedding_model"] == "renamed"

    def test_the_default_embedder_follows_a_rename_with_nothing_stored(self, home, tmp_path):
        from localm.config import DEFAULT_CONFIG, load_config
        name = DEFAULT_CONFIG["embedding_model"]
        _add(tmp_path, name)
        assert mm.rename_model(name, "my-embedder")
        assert load_config()["embedding_model"] == "my-embedder"

    def test_a_name_an_older_build_or_a_person_wrote_wins_over_the_old_id(self, home, tmp_path):
        from localm.config import load_config, update_config
        _add(tmp_path, "orig")
        _add(tmp_path, "other")
        update_config(lambda c: c.__setitem__("embedding_model", "orig"))
        stored = _raw(home, "config.json")
        stored["embedding_model"] = "other"
        (home / "config.json").write_text(json.dumps(stored), encoding="utf-8")
        assert load_config()["embedding_model"] == "other"

    def test_a_reference_to_a_removed_model_keeps_its_name(self, home, tmp_path):
        from localm.config import load_config, update_config
        _add(tmp_path, "orig")
        update_config(lambda c: c.__setitem__("spec_draft_model", "orig"))
        mm.remove_model("orig")
        assert load_config()["spec_draft_model"] == "orig"

    def test_jobs(self, home, tmp_path):
        from localm.plugins.builtin.jobs.store import Job, JobStore
        _add(tmp_path, "orig")
        store = JobStore()
        job = store.add(Job(name="j", task_kind="chat", prompt="hi", model="orig"))
        other = store.add(Job(name="k", task_kind="chat", prompt="hi", model="gone"))
        self._rename_and_reuse(tmp_path)
        assert store.get(job.id).model == "renamed"
        assert store.get(other.id).model == "gone"
        store.update(job.id, prompt="later")
        rows = {r["id"]: r for r in _raw(home, "jobs/jobs.json")["jobs"]}
        assert rows[job.id]["model"] == "renamed"
        assert rows[job.id][model_ids.REFS_KEY] == {"renamed": _rid(home, "renamed")}
        assert model_ids.REFS_KEY not in rows[other.id]

    def test_a_rag_collection_and_its_provenance_report(self, home, tmp_path):
        from localm.rag.store import Collection, collection_provenance_report
        _add(tmp_path, "orig")
        coll = Collection("kb").create()
        coll.add_uploads([{"filename": "x.txt", "data": b"apple"}],
                         embed_fn=lambda ts: [[1.0, 0.0] for _ in ts],
                         model_name="orig")
        self._rename_and_reuse(tmp_path)
        assert Collection("kb").embedding_model() == "renamed"
        assert coll.embedding_model() == "renamed", "a cached snapshot read the old name"
        assert [r["built_with"] for r in collection_provenance_report()] == ["renamed"]

    def test_an_adapter_base(self, home, tmp_path):
        reg = {"orig": {"path": "x/a.gguf"}, "adp": {"path": "x/b.gguf", "base": "orig"}}
        (home / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
        mm.load_registry()
        assert mm.rename_model("orig", "renamed")
        assert mm.load_registry()["adp"]["base"] == "renamed"
        assert _raw(home)["adp"]["base"] == "renamed"
        mm.update_registry(lambda r: r.__setitem__(
            "orig", {"path": "x/c.gguf"}))
        assert mm.load_registry()["adp"]["base"] == "renamed"

    def test_a_native_workflow(self, home, tmp_path):
        from localm.media import native_workflows as nw
        _add(tmp_path, "orig")
        wid = nw.save_workflow("image", {
            "format": nw.FORMAT, "version": nw.VERSION, "name": "w",
            "models": {"vae": {"name": "orig"}},
            "loras": [{"name": "orig", "strength": 0.5}]})
        self._rename_and_reuse(tmp_path)
        wf = nw.get_workflow("image", wid)
        assert wf["models"]["vae"]["name"] == "renamed"
        assert wf["loras"][0]["name"] == "renamed"
        assert nw.resolve(wf, "image")["native_models"]["vae"] == "renamed"
        assert nw.export_data(wf)["models"]["vae"]["name"] == "renamed"
        nw.delete_workflow("image", wid)
        assert not list(nw.native_dir("image").iterdir())

    def test_a_workflow_file_stays_in_the_shape_an_older_build_parses(self, home, tmp_path):
        from localm.media import native_workflows as nw
        _add(tmp_path, "orig")
        wid = nw.save_workflow("image", {
            "format": nw.FORMAT, "version": nw.VERSION, "name": "w",
            "models": {"vae": {"name": "orig"}}, "loras": []})
        stored = json.loads((nw.native_dir("image") / f"{wid}.json").read_text(encoding="utf-8"))
        assert set(stored["models"]["vae"]) <= {"name", "file", "source", "comfy"}
        assert model_ids.REFS_KEY not in stored

    def test_a_workflow_written_before_ids_existed(self, home, tmp_path):
        from localm.media import native_workflows as nw
        _add(tmp_path, "orig")
        wid = nw.save_workflow("image", {
            "format": nw.FORMAT, "version": nw.VERSION, "name": "w",
            "models": {"vae": {"name": "orig"}}, "loras": []})
        (nw.native_dir("image") / f"{wid}.model_refs").unlink()
        assert mm.rename_model("orig", "renamed")
        assert nw.get_workflow("image", wid)["models"]["vae"]["name"] == "renamed"

    def test_first_rename_records_references_a_later_name_reuse_cannot_capture(
            self, home, tmp_path, monkeypatch):
        from localm.media import native_workflows as nw
        from localm.model_manager import registry
        from localm.plugins.builtin.chat.plug import rename_notes as chat_notes
        from localm.plugins.builtin.jobs.plug import rename_notes as jobs_notes
        from localm.plugins.builtin.jobs.store import Job, JobStore
        from localm.rag.store import Collection
        monkeypatch.setitem(registry._RENAME_HOOKS, "saved chats", chat_notes)
        monkeypatch.setitem(registry._RENAME_HOOKS, "scheduled jobs", jobs_notes)
        _add(tmp_path, "orig")
        # Written the way a build without IDs wrote them.
        store = JobStore()
        job = store.add(Job(name="j", task_kind="chat", prompt="hi", model="orig"))
        defs = _raw(home, "jobs/jobs.json")
        defs["jobs"][0].pop(model_ids.REFS_KEY)
        (home / "jobs" / "jobs.json").write_text(json.dumps(defs), encoding="utf-8")
        (home / "config.json").write_text(
            json.dumps({"embedding_model": "orig"}), encoding="utf-8")
        (home / "chats").mkdir()
        (home / "chats" / "c1.json").write_text(
            json.dumps({"id": "c1", "pinned_model": "orig", "messages": []}), encoding="utf-8")
        coll_dir = home / "rag" / "kb"
        coll_dir.mkdir(parents=True)
        (coll_dir / "meta.json").write_text(
            json.dumps({"name": "kb", "docs": {}, "embedding_model": "orig"}), encoding="utf-8")
        wid = nw.save_workflow("image", {
            "format": nw.FORMAT, "version": nw.VERSION, "name": "w",
            "models": {"vae": {"name": "orig"}}, "loras": []})
        (nw.native_dir("image") / f"{wid}.model_refs").unlink()

        self._rename_and_reuse(tmp_path)

        from localm.config import load_config
        from localm.plugins.builtin.chat.plug import _conv_with_current_pin
        assert (home / "model_refs.migrated").is_file()
        assert load_config()["embedding_model"] == "renamed"
        assert store.get(job.id).model == "renamed"
        assert Collection("kb").embedding_model() == "renamed"
        assert nw.get_workflow("image", wid)["models"]["vae"]["name"] == "renamed"
        conv = _conv_with_current_pin(_raw(home, "chats/c1.json"))
        assert conv["pinned_model"] == "renamed"
        assert model_ids.REFS_KEY not in conv

    def test_second_rename_does_not_rewrite_records(self, home, tmp_path):
        _add(tmp_path, "orig")
        (home / "config.json").write_text(json.dumps({"spec_draft_model": "orig"}), encoding="utf-8")
        assert mm.rename_model("orig", "second")
        before = (home / "config.json").read_bytes()
        assert mm.rename_model("second", "third")
        assert (home / "config.json").read_bytes() == before
        assert _raw(home)["third"]["former_names"] == ["orig", "second"]


def test_native_setting_keys_cover_every_media_slot():
    from localm.media.native_models import media_keys
    for media in ("image", "video", "music"):
        missing = set(media_keys(media)) - set(model_ids.NATIVE_NAME_KEYS)
        assert not missing, (media, missing)


class TestEdges:
    def test_a_former_name_is_not_a_name_anyone_can_ask_for(self, home, tmp_path):
        from localm.model_manager import unregistered_model_error
        _add(tmp_path, "orig")
        assert mm.rename_model("orig", "renamed")
        assert unregistered_model_error("renamed") is None
        assert unregistered_model_error("orig") is not None
        assert "orig" not in mm.load_registry()


class TestNameBasedDetection:
    def test_a_rename_does_not_change_how_a_model_is_treated(self, home, tmp_path):
        from localm.inference.model_family import is_thinking_model
        from localm.plugins.coder.prompts import detect_model_family
        _add(tmp_path, "gemma-4-small")
        before = (is_thinking_model("gemma-4-small"), detect_model_family("gemma-4-small"))
        assert before == (True, "gemma")
        assert mm.rename_model("gemma-4-small", "daily")
        assert (is_thinking_model("daily"), detect_model_family("daily")) == before
        assert detect_model_family("daily local") == "gemma"
        assert mm.rename_model("daily", "other")
        assert detect_model_family("other") == "gemma"
        assert detect_model_family("not-registered") == "default"


class TestProjectConfigName:
    def _project(self, tmp_path, name):
        proj = tmp_path / "proj"
        (proj / ".localcoder").mkdir(parents=True)
        (proj / ".localcoder" / "config.toml").write_text(f'model = "{name}"\n', encoding="utf-8")
        return proj

    def test_a_name_that_resolves_to_nothing_says_so_before_any_server_starts(
            self, home, tmp_path, monkeypatch):
        from click.testing import CliRunner

        import localm.plugins.coder.cli as coder_cli
        from localm.plugins.coder.cli import _main
        from localm.plugins.engine import PluginManager
        PluginManager(None).set_installed_state("coder", True)
        _add(tmp_path, "orig")
        assert mm.rename_model("orig", "renamed")
        started = []
        monkeypatch.setattr(coder_cli, "make_localm_backend",
                            lambda *a, **k: started.append(a) or pytest.fail("a server was started"))
        result = CliRunner().invoke(
            _main.main, ["--cwd", str(self._project(tmp_path, "orig")), "--trust-hooks",
                         "-y", "say ok"])
        assert result.exit_code == 1, result.output
        text = " ".join(result.output.split())
        assert "'orig' is not registered" in text
        assert ".localcoder/config.toml" in text
        assert started == []

    def test_the_current_name_passes_and_a_flag_name_does_not_blame_the_project(
            self, home, tmp_path):
        from localm.plugins.coder.cli._main import _unregistered_model_message
        _add(tmp_path, "orig")
        assert mm.rename_model("orig", "renamed")
        proj = self._project(tmp_path, "renamed")
        assert _unregistered_model_message("renamed", proj) is None
        message = _unregistered_model_message("gone", proj)
        assert "'gone' is not registered" in message
        assert "config.toml" not in message


class TestLiveCoderSession:
    def _session(self, backend):
        from localm.plugins.coder.sessions import CoderSession

        class _Agent:
            _model_name = "orig"

        session = object.__new__(CoderSession)
        session.agent = _Agent()
        session.agent.backend = backend
        session.model = "orig"
        session.backend_info = {"model": "orig"}
        return session

    def test_the_next_request_names_the_new_name(self):
        from localm.plugins.coder.backends.http import HTTPBackend
        backend = HTTPBackend("http://127.0.0.1:1/v1", "orig")
        session = self._session(backend)
        session.follow_model_rename("orig", "renamed")
        assert backend.model_id == "renamed"
        assert session.agent._model_name == "renamed"
        assert session.model == "renamed"
        assert session.backend_info["model"] == "renamed"

    def test_another_models_session_is_left_alone(self):
        from localm.plugins.coder.backends.http import HTTPBackend
        backend = HTTPBackend("http://127.0.0.1:1/v1", "orig")
        session = self._session(backend)
        session.follow_model_rename("other", "renamed")
        assert backend.model_id == "orig"
        assert session.model == "orig"

    def test_a_shared_engine_session_follows_too(self):
        from localm.plugins.coder.backends.shared_engine import SharedEngineBackend

        class _Engine:
            supports_grammar = False
        backend = SharedEngineBackend(_Engine(), "orig", lock=threading.Lock())
        session = self._session(backend)
        session.follow_model_rename("orig", "renamed")
        assert backend.model_id == "renamed"


def test_a_failing_rename_listener_does_not_stop_the_others():
    seen = []

    def bad(_old, _new):
        raise RuntimeError("no")

    def good(old, new):
        seen.append((old, new))
    model_ids.add_rename_listener(bad)
    model_ids.add_rename_listener(good)
    try:
        model_ids.notify_renamed("a", "b")
    finally:
        model_ids._rename_listeners.remove(bad)
        model_ids._rename_listeners.remove(good)
    assert seen == [("a", "b")]
