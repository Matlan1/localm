# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model NAME from an untrusted caller must never resolve to an arbitrary path.

registry.get_model_info() falls through to ``Path(name)`` on a registry miss and
accepts any GGUF / Ollama blob / HF directory anywhere on disk. That fallthrough
is a documented CLI feature (``localm run D:/models/foo.gguf``), gated behind an
opt-in ``allow_direct_path`` plus a registry-membership check at each untrusted
entry point - the jobs plugin and the MCP server, neither of which is behind a
privileged scope. For an HF *directory* the chosen backend is HFBackend, and
``trust_remote_code=True`` there makes transformers import and execute the model
directory's own .py via ``auto_map``.

Section (e) asserts on the FLAG VALUE reaching transformers rather than on
whether custom code executed: hf.load() calls _require_torch() FIRST, so on a
torch-less GGUF-only build the branch raises before from_pretrained is reached
and an "it did not execute" observation would prove nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated LOCALM_HOME with a NON-EMPTY registry.

    Non-empty matters: the membership check mirrors http_server's existing
    "only enforce registration when the registry is not empty" convention, so an
    empty registry would skip layer 2 and test nothing. Layer 1
    (allow_direct_path) is unconditional and is covered by (d) either way.
    """
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")

    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    good = models / "good-model.gguf"
    good.write_bytes(b"GGUF\x00fake weights")
    cfg.save_registry({"good-model": {"path": str(good), "source": "test",
                                      "model_type": "chat"}})
    return tmp_path


@pytest.fixture
def evil_gguf(tmp_path):
    """A real, loadable-looking GGUF OUTSIDE the registry and outside models/."""
    p = tmp_path / "evil.gguf"
    p.write_bytes(b"GGUF\x00pwned")
    return p


@pytest.fixture
def evil_hf_dir(tmp_path):
    """An HF model directory that declares custom code via auto_map.

    This is the RCE shape: _is_hf_dir() accepts it (config.json + a tokenizer
    file), create_backend picks HFBackend, and auto_map is what transformers
    would import and execute when trust_remote_code is on.
    """
    d = tmp_path / "evil_hf"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "llama",
        # Deliberately content that is NOT a plausible model name: markup, a
        # newline, and a filesystem path.
        "_name_or_path": "<script>alert(1)</script>\nD:/some/host/path",
        "auto_map": {"AutoModelForCausalLM": "modeling_evil.EvilModel"},
    }))
    (d / "tokenizer_config.json").write_text("{}")
    (d / "modeling_evil.py").write_text(
        "raise AssertionError('custom model code must never be imported')\n")
    return d


def _jobs_client(home, monkeypatch):
    """A bare app with the bundled jobs plugin installed, open mode, no key."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    mgr = PluginManager(app, store_root=store_root, installed_root=home / "plugins")
    mgr.install("jobs")
    return TestClient(app)


# --------------------------------------------------------------------------- #
#  (d) the gate itself: get_model_info                                         #
# --------------------------------------------------------------------------- #

def test_get_model_info_refuses_direct_path_by_default(home, evil_gguf):
    """The default is REFUSE. This is the one unconditional security boundary."""
    from localm.model_manager.registry import get_model_info
    assert get_model_info(str(evil_gguf)) is None


def test_get_model_info_allows_direct_path_when_opted_in(home, evil_gguf):
    """...and the documented CLI feature still works when a CLI asks for it."""
    from localm.model_manager.registry import get_model_info
    info = get_model_info(str(evil_gguf), allow_direct_path=True)
    assert info is not None, "localm run <path> must keep working"
    assert Path(info[0]) == evil_gguf


def test_get_model_info_refuses_direct_hf_dir_by_default(home, evil_hf_dir):
    """The directory shape is the RCE one, so it must be refused by default too."""
    from localm.model_manager.registry import get_model_info
    assert get_model_info(str(evil_hf_dir)) is None
    assert get_model_info(str(evil_hf_dir), allow_direct_path=True) is not None


def test_registered_name_still_resolves(home):
    """The gate must not break the normal registry lookup."""
    from localm.model_manager.registry import get_model_info
    info = get_model_info("good-model")
    assert info is not None and Path(info[0]).name == "good-model.gguf"


def test_get_model_path_defaults_to_refusing(home, evil_gguf):
    """get_model_path is a thin wrapper and must inherit the same default."""
    from localm.model_manager.registry import get_model_path
    assert get_model_path(str(evil_gguf)) is None
    assert get_model_path(str(evil_gguf), allow_direct_path=True) == evil_gguf


# --------------------------------------------------------------------------- #
#  (a) POST /api/jobs                                                          #
# --------------------------------------------------------------------------- #

def test_post_jobs_rejects_absolute_model_path(home, evil_gguf, monkeypatch):
    """400 naming the unknown model, and NOTHING persisted."""
    from localm.plugins.builtin.jobs.store import JobStore

    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "evil", "prompt": "hi",
                                      "schedule_kind": "interval", "schedule": 120,
                                      "model": str(evil_gguf)})
        assert r.status_code == 400, r.text
        assert "evil.gguf" in r.text, "the error must name the rejected model"

    assert JobStore().list() == [], "a rejected job must never reach disk"


def test_post_jobs_accepts_registered_model(home, monkeypatch):
    """The gate must not break the legitimate case."""
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "ok", "prompt": "hi",
                                      "schedule_kind": "interval", "schedule": 120,
                                      "model": "good-model"})
        assert r.status_code == 200, r.text


def test_post_jobs_accepts_null_model(home, monkeypatch):
    """model=None means 'use the server default' and stays legal."""
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "ok2", "prompt": "hi",
                                      "schedule_kind": "interval", "schedule": 120})
        assert r.status_code == 200, r.text


def test_put_jobs_rejects_absolute_model_path(home, evil_gguf, monkeypatch):
    """PUT is a second write path into the same field and needs the same gate."""
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "ok3", "prompt": "hi",
                                      "schedule_kind": "interval", "schedule": 120})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]

        r = c.put(f"/api/jobs/{jid}", json={"model": str(evil_gguf)})
        assert r.status_code == 400, r.text

    from localm.plugins.builtin.jobs.store import JobStore
    assert JobStore().get(jid).model in (None, ""), \
        "a rejected update must not be persisted"


# --------------------------------------------------------------------------- #
#  (b) an ALREADY-PERSISTED row (predates the API check)                       #
# --------------------------------------------------------------------------- #

def test_persisted_job_with_absolute_path_never_reaches_engine(
        home, evil_gguf, monkeypatch):
    """Defence in depth: rows written by an older build must be refused at RUN
    time, and Engine() must never be constructed for one."""
    from localm.plugins.builtin.jobs import runner as runner_mod
    from localm.plugins.builtin.jobs.store import Job, JobStore

    # Persist the row the way an older build would have: straight to the store,
    # bypassing the API check entirely.
    store = JobStore()
    job = Job(name="legacy", task_kind="chat", prompt="hi",
              schedule_kind="interval", schedule=60)
    job.model = str(evil_gguf)
    store.add(job)
    assert store.get(job.id).model == str(evil_gguf), "precondition: row is on disk"

    built = []
    import localm.inference.engine as engine_mod
    monkeypatch.setattr(engine_mod, "Engine",
                        lambda *a, **k: built.append((a, k)))
    # No live engine to reuse, which is exactly the --no-model / post-eviction case.
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_engine", None)

    with pytest.raises(RuntimeError) as exc:
        runner_mod._load_engine(job.model)
    assert "evil.gguf" in str(exc.value)
    assert built == [], "Engine() must never be constructed for an unregistered name"


def test_scheduler_interval_path_refuses_too(home, evil_gguf, monkeypatch):
    """Jobs also fire UNATTENDED via the scheduler, so /run is not the only path.

    run_job must return a clean error result for a poisoned row rather than
    loading it (and must not crash the scheduler loop).
    """
    from localm.plugins.builtin.jobs import runner as runner_mod
    from localm.plugins.builtin.jobs.store import Job

    built = []
    import localm.inference.engine as engine_mod
    monkeypatch.setattr(engine_mod, "Engine",
                        lambda *a, **k: built.append((a, k)))
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_engine", None)

    job = Job(name="legacy2", task_kind="chat", prompt="hi",
              schedule_kind="interval", schedule=60)
    job.model = str(evil_gguf)

    result = runner_mod.run_job(job)
    assert result["status"] == "error", result
    assert built == [], "the unattended path must not construct an Engine either"


# --------------------------------------------------------------------------- #
#  (c) MCP                                                                     #
# --------------------------------------------------------------------------- #

def test_mcp_resolve_model_rejects_absolute_path(home, evil_gguf):
    """A client-supplied name must be a registered one."""
    from localm.plugins.mcpserver.server import EngineCache
    engines = EngineCache()
    with pytest.raises(ValueError) as exc:
        engines.resolve_model(str(evil_gguf))
    assert "evil.gguf" in str(exc.value)


def test_mcp_resolve_model_accepts_registered(home):
    from localm.plugins.mcpserver.server import EngineCache
    assert EngineCache().resolve_model("good-model") == "good-model"


def test_mcp_operator_default_model_may_be_a_path(home, evil_gguf):
    """--model / LOCALM_MODEL is OPERATOR-typed, so a direct path stays legal
    there. Gating it would break `localm mcp --model D:/models/foo.gguf`."""
    from localm.plugins.mcpserver.server import EngineCache
    engines = EngineCache(default_model=str(evil_gguf))
    assert engines.resolve_model(None) == str(evil_gguf)


def test_mcp_build_engine_rejects_absolute_path(home, evil_gguf, monkeypatch):
    """_build_engine is reachable independently of resolve_model."""
    from localm.plugins.mcpserver.server import EngineCache

    built = []
    import localm.inference.engine as engine_mod
    monkeypatch.setattr(engine_mod, "Engine", lambda *a, **k: built.append((a, k)))

    engines = EngineCache()
    with pytest.raises(ValueError):
        engines._build_engine(str(evil_gguf))
    assert built == []


# --------------------------------------------------------------------------- #
#  (e) trust_remote_code: assert on the FLAG                                   #
# --------------------------------------------------------------------------- #

def test_trust_remote_code_literal_is_gone_from_the_tree():
    """The grep gate, as a test so it runs in CI too.

    Only a config-driven variable may remain. A bare ``trust_remote_code=True``
    literal is the defect itself.
    """
    root = Path(__file__).resolve().parents[1] / "localm"
    offenders = [
        f"{p.relative_to(root.parent)}:{i}"
        for p in root.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "trust_remote_code=True" in line.replace(" ", "")
        or "trust_remote_code = True" in line
    ]
    assert offenders == [], (
        "trust_remote_code must be config-driven and default OFF; "
        f"hard-coded True at: {offenders}")


def test_trust_remote_code_defaults_off(home):
    from localm.inference.backends.hf import _trust_remote_code_enabled
    assert _trust_remote_code_enabled() is False


def test_trust_remote_code_follows_config(home):
    from localm.config import update_config
    from localm.inference.backends.hf import _trust_remote_code_enabled
    update_config(lambda c: c.update({"hf_trust_remote_code": True}))
    assert _trust_remote_code_enabled() is True


def test_custom_code_model_is_refused_with_an_actionable_error(home, evil_hf_dir):
    """A model dir declaring auto_map must raise a CLEAR error naming the flag,
    not silently execute and not fail with an opaque transformers traceback.

    This runs on a torch-less build too: the check is deliberately placed BEFORE
    _require_torch() so the refusal is the same on every build.
    """
    from localm.inference.backends.hf import _check_custom_code_allowed
    with pytest.raises(RuntimeError) as exc:
        _check_custom_code_allowed(str(evil_hf_dir))
    msg = str(exc.value)
    assert "hf_trust_remote_code" in msg, "the error must name the flag to enable"
    assert "custom code" in msg.lower()


def test_custom_code_model_loads_when_flag_enabled(home, evil_hf_dir):
    from localm.config import update_config
    from localm.inference.backends.hf import _check_custom_code_allowed
    update_config(lambda c: c.update({"hf_trust_remote_code": True}))
    _check_custom_code_allowed(str(evil_hf_dir))     # must not raise


def test_ordinary_hf_dir_is_not_refused(home, tmp_path):
    """No auto_map = no custom code = never refused, flag off or not."""
    from localm.inference.backends.hf import _check_custom_code_allowed
    d = tmp_path / "plain_hf"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "tokenizer_config.json").write_text("{}")
    _check_custom_code_allowed(str(d))               # must not raise


# --------------------------------------------------------------------------- #
#  Fix 5: the arbitrary-config.json sinks                                      #
# --------------------------------------------------------------------------- #

def test_display_name_does_not_echo_arbitrary_config_text(evil_hf_dir):
    """model_display_name reads a directory's config.json, so it must not echo
    _name_or_path straight back to the caller."""
    from localm.inference.engine import model_display_name
    hostile = json.loads((evil_hf_dir / "config.json").read_text())["_name_or_path"]
    assert model_display_name(str(evil_hf_dir)) != hostile


def test_display_name_keeps_a_sane_value(tmp_path):
    """The sanitiser must not throw away a legitimate _name_or_path."""
    from localm.inference.engine import model_display_name
    d = tmp_path / "real_hf"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"_name_or_path": "meta-llama/Llama-3-8B"}))
    assert model_display_name(str(d)) == "meta-llama/Llama-3-8B"


def test_export_is_pinned(home):
    """A dropped line in the export list fails HERE, not as an ImportError in an
    unrelated module."""
    import localm.model_manager as mm
    assert hasattr(mm, "unregistered_model_error")
    assert "unregistered_model_error" in mm.__all__


def test_refusal_does_not_enumerate_the_registry(home, evil_gguf):
    """Listing models needs models:read; this message goes to jobs/mcp callers."""
    from localm.model_manager import unregistered_model_error
    msg = unregistered_model_error(str(evil_gguf))
    assert msg and "evil.gguf" in msg, "it must still name what was rejected"
    assert "good-model" not in msg, (
        "the refusal must not hand the model inventory to a caller that cannot "
        "list models")


# --------------------------------------------------------------------------- #
#  MCP: the two paths that laundered a client string past the gate             #
# --------------------------------------------------------------------------- #

def _mcp_handler(name, monkeypatch):
    """build_tools returns {tool_name: {description, inputSchema, handler}}."""
    import localm.plugins.mcpserver.server as srv
    # run_coder_task is only registered when the coder plugin is available, which
    # it need not be in a bare test home. Force it so the gate is exercised.
    monkeypatch.setattr(srv, "_coder_available", lambda: True, raising=False)
    tools = srv.build_tools(srv.EngineCache(), enable_images=False,
                            enable_coder=True)
    assert name in tools, f"tool {name!r} not exposed; got {sorted(tools)}"
    return tools[name]["handler"]


def test_run_coder_task_refuses_unregistered_model(home, evil_gguf, monkeypatch):
    """run_coder_task builds a command line, so 'it is argv' does not mean an
    operator typed it: the coder CLI spawns `localm gui <model>`, whose startup
    resolver opts into allow_direct_path."""
    import subprocess
    spawned = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: spawned.append(a) or None)

    handler = _mcp_handler("run_coder_task", monkeypatch)
    out = handler({"task": "x", "cwd": str(home), "model": str(evil_gguf)})
    assert out.get("isError"), out
    assert "not registered" in json.dumps(out)
    assert spawned == [], "the coder subprocess must never be launched"


def test_run_coder_task_allows_a_registered_model(home, monkeypatch):
    """The gate must not break the legitimate call: a registered name proceeds
    to the subprocess."""
    import subprocess
    spawned = []

    class _Proc:
        returncode = 0
        # Must be the REAL shape the handler parses: it scans for a line that is
        # exactly "{" (indent=2 output) and raw_decodes from there. A bare "{}"
        # never matches.
        stdout = ('{\n  "response": "done",\n  "turns": 1,\n'
                  '  "total_tokens": 2,\n  "success": true\n}')
        stderr = ""

    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: spawned.append(a) or _Proc())
    handler = _mcp_handler("run_coder_task", monkeypatch)
    handler({"task": "x", "cwd": str(home), "model": "good-model"})
    assert spawned, "a registered model must reach the coder subprocess"
    assert "--model" in spawned[0][0]


def test_pull_model_refuses_a_local_path_repo(home, evil_hf_dir, monkeypatch):
    """pull_model treats an existing path as a local add, which would REGISTER an
    arbitrary directory under a client-chosen name - and a registered name passes
    the membership check and resolves via the registry branch, around the gate."""
    called = []
    import localm.model_manager.pull as pull_mod
    monkeypatch.setattr(pull_mod, "pull_model",
                        lambda *a, **k: called.append((a, k)) or True)

    handler = _mcp_handler("pull_model", monkeypatch)
    out = handler({"repo": str(evil_hf_dir), "name": "x", "load": False})
    assert out.get("isError"), out
    assert called == [], "a local path must never reach pull_model/add_local"

    from localm.config import load_registry
    assert "x" not in load_registry(), "nothing may be registered"


# --------------------------------------------------------------------------- #
#  The eviction branch                                                         #
# --------------------------------------------------------------------------- #

def test_unregistered_name_cannot_evict_the_live_engine(home, evil_gguf, monkeypatch):
    """The registry check is the FIRST statement of _load_engine, so an
    unregistered name is refused before the eviction branch. A live engine is
    loaded here, which is the only shape in which that branch is reachable.
    """
    from localm.plugins.builtin.jobs import runner as runner_mod

    class Live:
        loaded = True
        display_name = "good-model"

        def unload(self):
            raise AssertionError("must not unload")

    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_engine", Live())

    evicted = []
    monkeypatch.setattr(runner_mod, "_evict_shared_engine_for_media",
                        lambda live: evicted.append(live))
    # Force the swap decision to say "yes, evict".
    from localm import vram
    monkeypatch.setattr(vram, "should_swap_for_media", lambda free, est: True)
    monkeypatch.setattr("localm.discover.vram_capacity",
                        lambda config=None: {"free": 1})

    with pytest.raises(RuntimeError):
        runner_mod._load_engine(str(evil_gguf))
    assert evicted == [], (
        "an unregistered name must be refused BEFORE the eviction branch, or it "
        "unloads the user's live chat model with no valid model and no load")


def test_pull_warns_when_repo_ships_python(tmp_path, capsys):
    """A downloaded repo's .py must be called out at the moment it lands."""
    from localm.model_manager.pull import _warn_if_repo_ships_code
    d = tmp_path / "snap"
    (d / "nested").mkdir(parents=True)
    (d / "modeling_custom.py").write_text("x = 1\n")
    (d / "nested" / "helper.py").write_text("y = 2\n")

    _warn_if_repo_ships_code(d, "someone/some-model")
    out = capsys.readouterr().out
    assert "modeling_custom.py" in out
    assert "hf_trust_remote_code" in out, "the note must name the flag"


def test_pull_is_quiet_for_an_ordinary_repo(tmp_path, capsys):
    from localm.model_manager.pull import _warn_if_repo_ships_code
    d = tmp_path / "snap2"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    _warn_if_repo_ships_code(d, "someone/plain")
    assert capsys.readouterr().out == ""


def test_pull_warning_filename_cannot_blank_the_notice(tmp_path, capsys):
    """A REMOTE repo names these files and they are interpolated into a Rich
    markup string, where an unescaped '[red]x.py' would parse as a STYLE TAG and
    vanish, leaving a notice that names nothing.

    A CLOSING tag like '[/b]x.py' is NOT reachable through a filename: '/' is the
    path separator on every platform. That vector goes through repo_id instead -
    see the next test.
    """
    from localm.model_manager.pull import _warn_if_repo_ships_code
    d = tmp_path / "hostile"
    d.mkdir()
    (d / "[red]hidden.py").write_bytes(b"x")
    (d / "[bold]loud.py").write_bytes(b"x")

    _warn_if_repo_ships_code(d, "someone/hostile")     # must not raise
    out = capsys.readouterr().out
    assert "hidden.py" in out and "loud.py" in out, (
        f"filenames must survive escaping, got: {out!r}")
    assert "2 Python file(s)" in out


def test_pull_warning_repo_id_cannot_crash_the_notice(tmp_path, capsys):
    """The repo id is remote-supplied too and, unlike a filename, is not
    constrained by the filesystem, so it CAN carry a closing tag. An unescaped
    '[/b]' raises MarkupError, and this call sits between the download and
    _register, so a crash there would leave the model on disk and
    UNREGISTERED."""
    from localm.model_manager.pull import _warn_if_repo_ships_code
    d = tmp_path / "repo"
    d.mkdir()
    (d / "mod.py").write_bytes(b"x")

    _warn_if_repo_ships_code(d, "someone/[/b]evil[red]")   # must not raise
    out = capsys.readouterr().out
    assert "mod.py" in out
    assert "1 Python file(s)" in out


def test_custom_code_detected_in_preprocessor_config(home, tmp_path):
    """preprocessor_config.json (with the 'pre') is the file AutoProcessor reads,
    so an auto_map there is detected too."""
    from localm.inference.backends.hf import _check_custom_code_allowed
    d = tmp_path / "vlm"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "preprocessor_config.json").write_text(json.dumps(
        {"auto_map": {"AutoProcessor": "proc.EvilProcessor"}}))
    with pytest.raises(RuntimeError, match="hf_trust_remote_code"):
        _check_custom_code_allowed(str(d))


def test_custom_code_detected_in_legacy_list_auto_map(home, tmp_path):
    """transformers still accepts the list/tuple auto_map form; an isinstance-dict
    test alone under-detects, and under-detecting is the unsafe direction."""
    from localm.inference.backends.hf import _check_custom_code_allowed
    d = tmp_path / "legacy"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"auto_map": ["modeling_x.Cls", "modeling_x.Other"]}))
    with pytest.raises(RuntimeError):
        _check_custom_code_allowed(str(d))


def test_refusal_names_a_command_that_actually_exists(home, evil_hf_dir):
    """The refusal names `localm config hf_trust_remote_code true`. The CLI takes
    two positionals with no 'set', so a `config set` form would not run."""
    from localm.inference.backends.hf import _check_custom_code_allowed
    with pytest.raises(RuntimeError) as exc:
        _check_custom_code_allowed(str(evil_hf_dir))
    assert "localm config hf_trust_remote_code true" in str(exc.value)
    assert "config set" not in str(exc.value)


def test_model_footprint_rglob_is_bounded(tmp_path, monkeypatch):
    """An attacker-named directory must not cost an unbounded walk."""
    from localm.inference import residency

    monkeypatch.setattr(residency, "_FOOTPRINT_MAX_FILES", 5)
    d = tmp_path / "many"
    d.mkdir()
    for i in range(50):
        (d / f"f{i}.bin").write_bytes(b"x" * 10)

    # Bounded: it stops early rather than summing all 50 files.
    assert residency.model_footprint_bytes(d) <= 5 * 10


def test_model_footprint_bound_counts_entries_not_just_files(tmp_path, monkeypatch):
    """The ceiling applies to ENTRIES WALKED, not to successfully measured files,
    so a tree of pure directories is bounded too."""
    from localm.inference import residency

    walked = []
    real_rglob = Path.rglob

    def counting_rglob(self, pat):
        for item in real_rglob(self, pat):
            walked.append(item)
            yield item

    monkeypatch.setattr(Path, "rglob", counting_rglob)
    monkeypatch.setattr(residency, "_FOOTPRINT_MAX_FILES", 5)

    d = tmp_path / "dirs_only"
    d.mkdir()
    for i in range(50):
        (d / f"sub{i}").mkdir()

    assert residency.model_footprint_bytes(d) == 0
    assert len(walked) <= 6, (
        f"a pure-directory tree must stop at the ceiling, walked {len(walked)}")


def test_jobs_cli_refuses_an_unregistered_model(home, evil_gguf):
    """The third write path into `model`: the CLI refuses at creation time rather
    than saving a row that fails on every unattended tick."""
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    res = CliRunner().invoke(main, ["add", "j", "--prompt", "hi",
                                    "--every", "60", "--model", str(evil_gguf)])
    assert res.exit_code != 0, res.output
    assert "not registered" in res.output
    assert JobStore().list() == [], "a refused job must not reach disk"
