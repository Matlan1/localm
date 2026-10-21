# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin-registered inference backends (Host.register_inference_backend).

A backend a plugin registers is asked only for a model path the built-in GGUF and
Hugging Face backends refuse; disabling the plugin brings the refusal back.
"""

from __future__ import annotations

import json
import textwrap
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI

from localm.inference import backend_registry
from localm.plugins.contract import InferenceBackendDescriptor

EXL2_QUANT = json.loads(
    '{"quant_method":"exl2","version":"0.1.7","bits":4.0,"head_bits":6,'
    '"calibration":{"rows":115,"length":2048,"dataset":"(default)"}}')  # turboderp/...-exl2 @4.0bpw
EXL2_SENTENCE = ("This model is quantized with EXL2, a format only ExLlamaV2 can run, "
                 "so localm cannot load it. Use a GGUF of the same model, or the "
                 "original Hugging Face weights.")


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch, tmp_path):
    import localm.config as cfg
    import localm.plugins.engine as plugins_engine
    monkeypatch.setattr(backend_registry, "_DESCRIPTORS", [])
    monkeypatch.setattr(plugins_engine, "_backend_manager", None)
    monkeypatch.setattr(plugins_engine, "_server_manager_attached", False)
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "home" / "config.json")
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "plugins").mkdir(parents=True)


def _hf_dir(d: Path, config: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        config if config is not None else {"architectures": ["LlamaForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"\0" * 64)
    (d / "tokenizer.json").write_text("{}")
    return d


def _exl2_dir(tmp_path: Path) -> Path:
    return _hf_dir(tmp_path / "exl2", {"architectures": ["LlamaForCausalLM"],
                                       "quantization_config": EXL2_QUANT})


class _Made:
    def __init__(self, path, opts):
        self.path, self.opts = path, opts


def _descriptor(calls: list, *, claim=True, name="fake", plugin="fakeplug"):
    def claims(p):
        calls.append(p)
        return claim

    return InferenceBackendDescriptor(name=name, claims=claims,
                                      create=lambda p, **o: _Made(p, o),
                                      plugin_name=plugin)


class TestCreateBackend:
    def test_a_claimed_refused_folder_gets_the_plugin_backend(self, tmp_path):
        from localm.inference.engine import create_backend
        d = _exl2_dir(tmp_path)
        backend_registry.register(_descriptor([]))
        made = create_backend(str(d), device="cuda:1", n_ctx=8192)
        assert isinstance(made, _Made)
        assert made.path == str(d)
        assert made.opts == {"device": "cuda:1", "n_ctx": 8192}

    def test_gguf_and_loadable_hf_paths_never_reach_a_descriptor(self, tmp_path):
        from localm.inference.backends.gguf import GgufBackend
        from localm.inference.backends.hf import HFBackend
        from localm.inference.engine import create_backend
        calls: list = []
        backend_registry.register(_descriptor(calls))
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"GGUF" + b"\3\0\0\0" + b"\0" * 16)
        plain = _hf_dir(tmp_path / "plain")
        assert isinstance(create_backend(str(gguf)), GgufBackend)
        assert isinstance(create_backend(str(plain)), HFBackend)
        assert calls == []

    def test_an_unclaimed_folder_keeps_the_unchanged_refusal(self, tmp_path):
        from localm.inference.backends.base import UnsupportedModelRoleError
        from localm.inference.backends.hf import HFBackend
        from localm.inference.engine import create_backend
        d = _exl2_dir(tmp_path)
        calls: list = []
        backend_registry.register(_descriptor(calls, claim=False))
        backend = create_backend(str(d))
        assert calls == [str(d)]
        assert isinstance(backend, HFBackend)
        with pytest.raises(UnsupportedModelRoleError) as exc:
            backend.load()
        assert str(exc.value) == EXL2_SENTENCE

    def test_a_claims_that_raises_is_logged_and_not_claiming(self, tmp_path, caplog):
        from localm.inference.backends.hf import HFBackend
        from localm.inference.engine import create_backend
        d = _exl2_dir(tmp_path)

        def boom(p):
            raise OSError("disk gone")
        backend_registry.register(InferenceBackendDescriptor(
            name="boom", claims=boom, create=lambda p, **o: None, plugin_name="x"))
        with caplog.at_level("WARNING", logger="localm"):
            assert isinstance(create_backend(str(d)), HFBackend)
        assert "claims() raised" in caplog.text and "disk gone" in caplog.text

    def test_first_registered_claim_wins(self, tmp_path):
        d = _exl2_dir(tmp_path)
        backend_registry.register(_descriptor([], name="first", plugin="a"))
        backend_registry.register(_descriptor([], name="second", plugin="b"))
        assert backend_registry.backend_for(str(d)).name == "first"

    def test_claims_runs_without_the_registry_lock(self, tmp_path):
        d = _exl2_dir(tmp_path)
        free = []

        def claims(p):
            got = backend_registry._LOCK.acquire(blocking=False)
            if got:
                backend_registry._LOCK.release()
            free.append(got)
            return True
        backend_registry.register(InferenceBackendDescriptor(
            name="lockcheck", claims=claims, create=lambda p, **o: None, plugin_name="x"))
        assert backend_registry.backend_for(str(d)) is not None
        assert free == [True]

    @pytest.mark.parametrize("bad", [
        dict(name="", claims=lambda p: True, create=lambda p: None),
        dict(name="x", claims=None, create=lambda p: None),
        dict(name="x", claims=lambda p: True, create="nope"),
    ])
    def test_register_rejects_a_malformed_descriptor(self, bad):
        with pytest.raises(ValueError):
            backend_registry.register(InferenceBackendDescriptor(**bad))


BACKEND_PLUGIN = '''
import os
from localm.plugins.contract import InferenceBackendDescriptor

_count = os.environ.get("FAKEEXL_IMPORTS")
if _count:
    with open(_count, "a", encoding="utf-8") as f:
        f.write("x")

class Made:
    def __init__(self, path, opts):
        self.path, self.opts = path, opts

def _claims(path):
    import json, pathlib
    try:
        cfg = json.loads((pathlib.Path(path) / "config.json").read_text())
    except Exception:
        return False
    return (cfg.get("quantization_config") or {}).get("quant_method") == "exl2"

def register(host):
    host.register_inference_backend(InferenceBackendDescriptor(
        name="fakeexl", claims=_claims, create=lambda p, **o: Made(p, o),
        plugin_name="someone-else"))

def unregister():
    pass
'''


def _make_backend_plugin(root: Path, name: str = "fakeexl") -> Path:
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n'
        'capabilities = ["inference-backend"]\n', encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(BACKEND_PLUGIN), encoding="utf-8")
    return pdir


def _enable(name: str) -> None:
    from localm.config import update_config

    def _mutate(c):
        c["plugins_enabled"] = sorted(set(c.get("plugins_enabled", [])) | {name})
    update_config(_mutate)


class TestPluginLifecycle:
    def test_disabling_the_plugin_brings_the_refusal_back(self, tmp_path):
        from localm.inference.backends.hf import HFBackend
        from localm.inference.engine import create_backend
        from localm.plugins.engine import PluginManager
        root = tmp_path / "installed"
        _make_backend_plugin(root)
        mgr = PluginManager(FastAPI(), installed_root=root, builtin_root=None)
        mgr.discover()
        mgr.install("fakeexl")
        d = _exl2_dir(tmp_path)
        made = create_backend(str(d))
        assert type(made).__name__ == "Made"
        assert [x.plugin_name for x in backend_registry.registered()] == ["fakeexl"]

        mgr.disable("fakeexl")
        assert backend_registry.registered() == []
        assert isinstance(create_backend(str(d)), HFBackend)

    def test_the_host_rejects_anything_but_a_descriptor(self):
        from localm.plugins.engine import PluginHost, PluginSpec
        host = PluginHost(FastAPI(), object(), PluginSpec(name="demo"))
        with pytest.raises(TypeError, match="InferenceBackendDescriptor"):
            host.register_inference_backend({"name": "x"})

    def test_an_in_process_load_finds_an_enabled_backend_plugin(self, tmp_path):
        from localm.inference.engine import create_backend
        _make_backend_plugin(tmp_path / "home" / "plugins")
        _enable("fakeexl")
        made = create_backend(str(_exl2_dir(tmp_path)))
        assert type(made).__name__ == "Made"

    def test_a_disabled_backend_plugin_is_not_loaded_in_process(self, tmp_path):
        from localm.inference.backends.hf import HFBackend
        from localm.inference.engine import create_backend
        _make_backend_plugin(tmp_path / "home" / "plugins")
        assert isinstance(create_backend(str(_exl2_dir(tmp_path))), HFBackend)
        assert backend_registry.registered() == []

    def test_lazy_plugin_load_takes_no_engine_load_lock(self, tmp_path):
        from localm.inference.engine import _LOAD_LOCK, create_backend
        _make_backend_plugin(tmp_path / "home" / "plugins")
        _enable("fakeexl")
        d = _exl2_dir(tmp_path)
        out: list = []
        t = threading.Thread(target=lambda: out.append(create_backend(str(d))), daemon=True)
        with _LOAD_LOCK:
            t.start()
            t.join(timeout=60)
            blocked = t.is_alive()
        t.join(timeout=60)
        assert not blocked, "the lazy plugin load waited for the engine load lock"
        assert type(out[0]).__name__ == "Made"


    def test_a_server_with_attached_plugins_never_loads_them_again(
            self, tmp_path, monkeypatch):
        from localm.inference.backends.hf import HFBackend
        from localm.inference.engine import create_backend
        from localm.plugins.engine import attach_engine
        imports = tmp_path / "imports.txt"
        monkeypatch.setenv("FAKEEXL_IMPORTS", str(imports))
        _make_backend_plugin(tmp_path / "home" / "plugins")
        _enable("fakeexl")
        attach_engine(FastAPI())
        assert imports.read_text() == "x"
        exl3 = _hf_dir(tmp_path / "exl3", {"architectures": ["LlamaForCausalLM"],
                                           "quantization_config": {"quant_method": "exl3"}})
        assert isinstance(create_backend(str(exl3)), HFBackend)
        assert imports.read_text() == "x"


class TestRegistryNote:
    def test_add_local_drops_the_cannot_load_note_when_a_backend_claims(
            self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        from localm.model_manager import _shared
        printed: list = []
        monkeypatch.setattr(_shared.console, "print",
                            lambda *a, **k: printed.append(" ".join(map(str, a))))
        monkeypatch.setattr(mm, "MODELS_DIR", tmp_path / "home" / "models")
        monkeypatch.setattr(mm, "REGISTRY_FILE", tmp_path / "home" / "registry.json")
        import localm.config as cfg
        monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "home" / "models")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "home" / "registry.json")
        (tmp_path / "home" / "models").mkdir()
        backend_registry.register(_descriptor([]))
        assert mm.add_local(str(_exl2_dir(tmp_path))) is True
        assert "cannot load it" not in "\n".join(printed)


class TestAdmission:
    def test_a_plugin_backend_model_is_sized_by_its_backend(self, tmp_path, monkeypatch):
        from localm.inference import http_server
        d = _exl2_dir(tmp_path)
        monkeypatch.setattr("localm.config.load_registry", lambda: {"exl": {"path": str(d)}})
        monkeypatch.setattr("localm.model_manager.get_model_info", lambda name: (d, None))
        assert http_server._switch_load_budget("exl").plugin_backend is False
        backend_registry.register(_descriptor([]))
        budget = http_server._switch_load_budget("exl")
        assert budget.plugin_backend is True and budget.check_split_fit is False


class TestCatalogGate:
    def test_exllama_is_not_listed_as_available(self, tmp_path):
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(FastAPI(), installed_root=tmp_path / "installed")
        names = {p["name"] for p in mgr.api_state()["plugins"]}
        assert "exllama" not in names
        assert "memory" in names

    def test_an_explicit_install_still_works(self, tmp_path):
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(FastAPI(), installed_root=tmp_path / "installed")
        mgr.discover()
        mgr.install("exllama")
        assert (tmp_path / "installed" / "exllama" / "plugin.toml").is_file()
        names = {p["name"] for p in mgr.api_state()["plugins"] if p["installed"]}
        assert "exllama" in names

    def test_setup_all_leaves_it_out(self):
        from localm.plugins import catalog
        assert catalog.offered("exllama") is False
        assert catalog.offered("memory") is True
        assert catalog.offered("not-in-the-catalog") is True


class TestRunnerTypedLoadRefusal:
    def test_a_model_refused_in_the_child_stays_a_refusal(self, tmp_path, monkeypatch):
        from localm.inference.backends._hf_runner import HFRunner
        from localm.inference.backends.base import ModelRefusedError
        worker = tmp_path / "refusing_worker.py"
        worker.write_text(textwrap.dedent('''
            from localm.inference.backends.base import ModelRefusedError

            class W:
                def __init__(self, **kw):
                    pass
                def load(self):
                    raise ModelRefusedError("no CUDA here")
        '''), encoding="utf-8")
        runner = HFRunner(worker_target=(str(worker), "W"), process_name="test-worker")
        with pytest.raises(ModelRefusedError, match="no CUDA here"):
            runner.spawn_and_load({}, timeout=120)
        assert not runner.is_alive()
