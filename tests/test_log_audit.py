# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/log_audit.py finds every warning, error, toast, model load and
download in a localm debug log and fails on any the manifest does not explain."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "log_audit.py"
_spec = importlib.util.spec_from_file_location("log_audit", _SCRIPT)
la = importlib.util.module_from_spec(_spec)
sys.modules["log_audit"] = la
_spec.loader.exec_module(la)

# Line shapes copied from a real localm 0.3.x debug log (paths shortened).
FRESH_START_LOG = textwrap.dedent("""\
    2026-11-03 11:21:45,750 DEBUG   localm: debug mode enabled (pid 1360)
    2026-11-03 11:21:46,936 INFO    localm.plugins: plugin browser loaded
    llama_model_loader: loaded meta data with 24 key-value pairs and 197 tensors from D:\\x\\home\\models\\embeddings\\bge-small-en-v1.5-q4_k_m.gguf (version GGUF V3 (latest))
    llama_model_loader: - kv   1:                               general.name str              = bge-small-en-v1.5
    load_tensors: loading model tensors, this can take a while... (load_mode = none)
    2026-11-03 11:21:56,217 INFO    localm: embedding model ready: D:\\x\\home\\models\\embeddings\\bge-small-en-v1.5-q4_k_m.gguf (dim=384, pooling=mean)
    2026-11-03 11:21:56,231 DEBUG   localm: GET /api/memory -> 200 (6848 ms, loop_lag=0.00s)
    2026-11-03 11:21:56,875 DEBUG   localm: GET /api/activity -> 200 (3 ms, loop_lag=0.00s)
    """)

MOE_SWITCH_LOG = textwrap.dedent("""\
    2026-11-03 12:44:55,794 DEBUG   localm: debug mode enabled (pid 5324)
    2026-11-03 12:54:39,770 WARNING localm: gpu layers auto: Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf: every layer on the GPU, with the routed experts of 24/48 layers in system RAM (Mixture-of-Experts)
    llama_model_loader: loaded meta data with 44 key-value pairs and 579 tensors from D:\\m\\Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf (version GGUF V3 (latest))
    2026-11-03 12:55:37,167 WARNING localm: switch_engine: loaded 'Qwen3-Coder-30B-A3B-Instruct-Q6_K': 48/48 layers on the GPU, routed experts of 24 layers in system RAM (slower)
    2026-11-03 12:55:37,254 WARNING localm: gui: [toast /] Model switched to Qwen3-Coder-30B-A3B-Instruct-Q6_K (experts of 24 layers in system RAM - slower)
    2026-11-03 12:55:37,281 DEBUG   localm: POST /api/gui-error -> 200 (103 ms, loop_lag=0.00s)
    """)

CLEAN_LOG = textwrap.dedent("""\
    2026-11-03 11:20:26,542 DEBUG   localm: debug mode enabled (pid 5300)
    2026-11-03 11:20:29,137 INFO    localm.plugins: plugin browser loaded
    2026-11-03 11:20:31,000 DEBUG   localm: GET /api/models -> 200 (5 ms, loop_lag=0.00s)
    """)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _manifest(tmp_path, data):
    return _write(tmp_path, "manifest.json", json.dumps(data))


def _events(tmp_path, text):
    events, _lines, _records = la.parse_log(_write(tmp_path, "localm_x.log", text))
    return events


def test_fresh_start_embedder_load_is_unexplained(tmp_path, capsys):
    log = _write(tmp_path, "localm_a.log", FRESH_START_LOG)
    code = la.main(["--manifest", str(_manifest(tmp_path, {})), str(log)])
    out = capsys.readouterr().out
    assert code == 1
    assert "UNEXPLAINED  model-load" in out
    assert "bge-small-en-v1.5-q4_k_m" in out


def test_loader_banner_and_ready_line_are_one_load(tmp_path):
    events = _events(tmp_path, FRESH_START_LOG)
    loads = [e for e in events if e.kind == "model-load"]
    assert [(e.subject, e.lines) for e in loads] == [("bge-small-en-v1.5-q4_k_m", [3, 6])]
    assert events == loads


def test_moe_switch_warning_toast_and_load(tmp_path):
    events = _events(tmp_path, MOE_SWITCH_LOG)
    kinds = [(e.kind, e.line) for e in events]
    assert kinds == [("warning", 2), ("model-load", 3), ("warning", 4), ("toast", 5)]
    load = events[1]
    assert load.subject == "Qwen3-Coder-30B-A3B-Instruct-Q6_K" and load.lines == [3, 4]
    assert events[3].subject.startswith("[toast /] Model switched to")


def test_manifest_explains_every_event(tmp_path, capsys):
    log = _write(tmp_path, "localm_b.log", MOE_SWITCH_LOG)
    manifest = _manifest(tmp_path, {
        "run": "MoE switch row",
        "models": [{"match": "qwen3-coder-30b-*", "why": "the step switches to it"}],
        "warnings": ["gpu layers auto: Qwen3-Coder-*", "switch_engine: loaded 'Qwen3-Coder*"],
        "toasts": [{"match": "[toast /] Model switched to *"}],
    })
    assert la.main(["--manifest", str(manifest), str(log)]) == 0
    out = capsys.readouterr().out
    assert "UNEXPLAINED" not in out
    assert "4 event(s): 4 expected, 0 unexplained" in out


def test_one_unexplained_toast_fails_the_audit(tmp_path, capsys):
    log = _write(tmp_path, "localm_c.log", MOE_SWITCH_LOG)
    manifest = _manifest(tmp_path, {
        "models": ["Qwen3-Coder-30B-A3B-Instruct-Q6_K"],
        "warnings": ["gpu layers auto: *", "switch_engine: loaded *"],
    })
    assert la.main(["--manifest", str(manifest), str(log)]) == 1
    out = capsys.readouterr().out
    assert "UNEXPLAINED  toast" in out
    assert "1 unexplained (1 toast)" in out


def test_clean_run_passes(tmp_path):
    log = _write(tmp_path, "localm_d.log", CLEAN_LOG)
    assert la.main(["--manifest", str(_manifest(tmp_path, {})), str(log)]) == 0


def test_downloads_errors_and_warnings_are_found(tmp_path):
    text = textwrap.dedent("""\
        2026-11-03 10:00:00,000 INFO    localm: downloading embedding model CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q4_k_m.gguf (one-time)...
        2026-11-03 10:00:01,000 INFO    localm: Model pull Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf started (requested from a connected client)
        model.safetensors:  10%|#         | 1.0M/10.0M [00:01<00:09, 1.0MB/s]
        model.safetensors:  90%|######### | 9.0M/10.0M [00:09<00:01, 1.0MB/s]
        2026-11-03 10:00:02,000 ERROR   localm: the model worker died: exit code 3
        llama_model_load: error loading model: unable to allocate ROCm0 buffer
        D:\\x\\site-packages\\torch\\cuda\\__init__.py:61: FutureWarning: pynvml is deprecated
        Exception in thread Thread-3:
        ggml.c:1234: GGML_ASSERT(ctx) failed
        2026-11-03 10:00:03,000 DEBUG   localm: raw model output:
        The build failed: error in step 2
        """)
    events = _events(tmp_path, text)
    got = [(e.kind, e.subject[:40]) for e in events]
    assert got == [
        ("download", "CompendiumLabs/bge-small-en-v1.5-gguf/bg"),
        ("download", "Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf"),
        ("download", "model.safetensors"),
        ("error", "the model worker died: exit code 3"),
        ("error", "llama_model_load: error loading model: u"),
        ("warning", "FutureWarning: pynvml is deprecated"),
        ("error", "Exception in thread Thread-3:"),
        ("error", "ggml.c:1234: GGML_ASSERT(ctx) failed"),
    ]
    assert events[2].lines == [3, 4]


def test_strict_fails_on_a_load_that_never_happened(tmp_path, capsys):
    log = _write(tmp_path, "localm_e.log", CLEAN_LOG)
    manifest = _manifest(tmp_path, {"models": ["Qwen3-0.6B-Q8_0"]})
    assert la.main(["--manifest", str(manifest), str(log)]) == 0
    assert "NOT SEEN" in capsys.readouterr().out
    assert la.main(["--manifest", str(manifest), "--strict", str(log)]) == 1


@pytest.mark.parametrize("manifest,needle", [
    ({"model": ["x"]}, "unknown key(s): model"),
    ({"models": "x"}, "must be a list"),
    ({"models": [{"pattern": "x"}]}, "is not a glob"),
    ({"models": [{"match": "x", "reason": "y"}]}, "unknown field(s): reason"),
    ([1, 2], "is not a JSON object"),
])
def test_malformed_manifest_cannot_pass(tmp_path, capsys, manifest, needle):
    log = _write(tmp_path, "localm_f.log", CLEAN_LOG)
    path = _write(tmp_path, "m.json", json.dumps(manifest))
    assert la.main(["--manifest", str(path), str(log)]) == 2
    assert needle in capsys.readouterr().out


def test_missing_or_foreign_log_cannot_pass(tmp_path, capsys):
    manifest = _manifest(tmp_path, {})
    assert la.main(["--manifest", str(manifest), str(tmp_path / "absent.log")]) == 2
    foreign = _write(tmp_path, "other.log", "hello\nworld\n")
    assert la.main(["--manifest", str(manifest), str(foreign)]) == 2
    assert "holds no localm log record" in capsys.readouterr().out


def test_json_report(tmp_path):
    log = _write(tmp_path, "localm_g.log", FRESH_START_LOG)
    out = tmp_path / "audit.json"
    assert la.main(["--manifest", str(_manifest(tmp_path, {})), "--json", str(out), str(log)]) == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [(e["verdict"], e["kind"]) for e in data["events"]] == [("UNEXPLAINED", "model-load")]


def _logger_formats(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr in ("info", "warning", "debug", "error") and node.args and \
                isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            out.append(node.args[0].value)
    return out


@pytest.mark.parametrize("rel,prefix,args,regex", [
    ("localm/inference/embedder.py", "embedding model ready:",
     ("D:/m/bge.gguf", 384, "mean"), la.EMBED_READY_RE),
    ("localm/inference/reranker.py", "embedding model ready:", ("rr.gguf", 1), la.EMBED_READY_RE),
    ("localm/inference/embedder.py", "downloading embedding model",
     ("owner/repo", "f.gguf"), la.DOWNLOADING_RE),
    ("localm/voice.py", "downloading Whisper", ("owner/whisper",), la.DOWNLOADING_RE),
])
def test_patterns_match_localm_log_messages(rel, prefix, args, regex):
    formats = [f for f in _logger_formats(_REPO / rel) if f.startswith(prefix)]
    assert formats, f"no logger call in {rel} starts with {prefix!r}"
    for fmt in formats:
        assert regex.match(fmt % args), (fmt, regex.pattern)


def test_switch_engine_loaded_message_matches():
    from types import SimpleNamespace

    from localm.inference.http_server import _describe_load_placement
    line = "switch_engine: " + _describe_load_placement("Qwen3-0.6B-Q8_0", SimpleNamespace())
    m = la.SWITCH_LOADED_RE.match(line)
    assert m and m.group("model") == "Qwen3-0.6B-Q8_0", line


def test_real_debug_log_written_by_localm(tmp_path):
    """A log written by localm's own debug logging, with a toast posted through
    the real /api/gui-error route and the embedder's ready line logged on the
    localm logger, audits as one toast and one model load."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"update_url": "", "bugreport_upload_url": ""}))
    child = textwrap.dedent("""\
        import logging
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from localm import debuglog
        path = debuglog.enable_debug()
        from localm.plugins.gui.routes import client_errors
        app = FastAPI()
        client_errors.register(app, None)
        with TestClient(app) as c:
            r = c.post("/api/gui-error", json={"message": "Model switched (slower)", "source": "toast", "page": "/"})
            assert r.status_code == 200, r.text
        logging.getLogger("localm").info("embedding model ready: %s (dim=%d, pooling=%s)",
                                         r"D:\\m\\bge-small-en-v1.5-q4_k_m.gguf", 384, "mean")
        debuglog.flush_log_handlers()
        print(path)
        """)
    env = dict(os.environ, LOCALM_HOME=str(home), PYTHONPATH=str(_REPO))
    env.pop("LOCALM_DEBUG", None)
    proc = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True,
                          timeout=180, cwd=str(tmp_path),
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert proc.returncode == 0, proc.stderr[-2000:]
    log = Path(proc.stdout.strip().splitlines()[-1])
    assert log.is_file() and log.parent.parent == home
    events, _lines, records = la.parse_log(log)
    assert records >= 2
    assert [(e.kind, e.subject) for e in events] == [
        ("toast", "[toast /] Model switched (slower)"),
        ("model-load", "bge-small-en-v1.5-q4_k_m"),
    ]
