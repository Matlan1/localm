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
        ("download", "Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.ggu"),
        ("download", "model.safetensors"),
        ("error", "the model worker died: exit code 3"),
        ("error", "llama_model_load: error loading model: u"),
        ("warning", "FutureWarning: pynvml is deprecated"),
        ("error", "Exception in thread Thread-3:"),
        ("error", "ggml.c:1234: GGML_ASSERT(ctx) failed"),
    ]
    assert events[2].lines == [3, 4]


def _real_format(level, msg, *args):
    import logging
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    rec = logging.LogRecord("localm", level, __file__, 1, msg, args, None)
    return fmt.format(rec)


@pytest.mark.parametrize("level,msg,args,subject", [
    (40, "%s", ("",), ""),
    (40, "%s", (RuntimeError(),), ""),
    (30, "\nDetail on the next line", (), "Detail on the next line"),
])
def test_record_with_an_empty_first_line_is_still_an_event(tmp_path, capsys, level, msg, args, subject):
    text = CLEAN_LOG + _real_format(level, msg, *args) + "\n"
    log = _write(tmp_path, "localm_h.log", text)
    events, _lines, records = la.parse_log(log)
    assert records == 4
    assert [(e.kind, e.subject) for e in events] == [
        ("error" if level == 40 else "warning", subject)]
    assert la.main(["--manifest", str(_manifest(tmp_path, {})), str(log)]) == 1


@pytest.mark.parametrize("line", [
    "ROCm error: out of memory",
    "CUDA error: an illegal memory access was encountered",
    "D:/a/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:88: ROCm error",
    "Fatal Python error: Segmentation fault",
    "Windows fatal exception: access violation",
    "Traceback (most recent call last):",
])
def test_fatal_native_and_python_lines_are_errors(tmp_path, line):
    events = _events(tmp_path, CLEAN_LOG + line + "\n")
    assert [(e.kind, e.subject) for e in events] == [("error", line)]


def test_traceback_of_a_flagged_record_is_part_of_it(tmp_path):
    text = CLEAN_LOG + textwrap.dedent("""\
        2026-11-03 11:20:32,000 ERROR   localm: worker crashed
        Traceback (most recent call last):
          File "x.py", line 1, in <module>
        ValueError: boom
        """)
    events = _events(tmp_path, text)
    assert [(e.kind, e.subject) for e in events] == [("error", "worker crashed")]


def test_crash_artefacts_in_the_home_are_errors(tmp_path, capsys):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True)
    (home / "logs").mkdir()
    (home / "run" / "server-crash-trace.abc.txt").write_text("")
    log = _write(tmp_path, "localm_i.log", CLEAN_LOG)
    manifest = _manifest(tmp_path, {})
    assert la.main(["--manifest", str(manifest), "--home", str(home), str(log)]) == 0
    (home / "run" / "server-crash-trace.abc.txt").write_text("Fatal Python error: Aborted\n")
    (home / "logs" / "hang_2026-11-03_1_2.log").write_text("stack\n")
    assert la.main(["--manifest", str(manifest), "--home", str(home), str(log)]) == 1
    out = capsys.readouterr().out
    assert "crash artefact run/server-crash-trace.abc.txt" in out
    assert "crash artefact logs/hang_2026-11-03_1_2.log" in out
    assert la.main(["--manifest", str(manifest), "--home", str(tmp_path / "nope"), str(log)]) == 2


def test_every_remote_pull_logs_a_download_line(monkeypatch, caplog):
    import contextlib
    import logging

    from localm.model_manager import pending_pulls, pull
    monkeypatch.setattr(pending_pulls, "tracking", lambda record: contextlib.nullcontext({}))
    monkeypatch.setattr(pull, "_pull_remote", lambda *a, **k: False)
    with caplog.at_level(logging.INFO, logger="localm"):
        pull.pull_model("Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf")
        pull.pull_model("https://user:secret@example.com/m.gguf?token=abc#x")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("pull: ")]
    assert lines == ["pull: downloading Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf",
                     "pull: downloading https://example.com/m.gguf"]
    for line in lines:
        assert la.PULL_LOG_RE.match(line)


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
    ("localm/inference/reranker.py", "reranker ready:", ("rr.gguf", "a,b"), la.READY_RE),
    ("localm/inference/speech.py", "speech model ready:", ("tts", 24000, "cpu"), la.READY_RE),
    ("localm/inference/backends/hf.py", "hf model ready:", ("my-hf-model",), la.READY_RE),
    ("localm/plugins/builtin/exllama/backend.py", "exllama model ready:", ("my-exl2",), la.READY_RE),
    ("localm/media/sdcpp/runner.py", "sd.cpp model ready:", ("flux.gguf",), la.READY_RE),
    ("localm/media/sdcpp/runner.py", "sd.cpp upscaler ready:", ("esrgan.pth",), la.READY_RE),
    ("localm/voice.py", "whisper model ready:", ("base",), la.READY_RE),
    ("localm/media/comfy_client.py", "comfy: job uses model", ("wan.safetensors",), la.JOB_MODEL_RE),
    ("localm/media/koboldcpp/music.py", "koboldcpp music: job uses model",
     ("ace.gguf",), la.JOB_MODEL_RE),
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


def _marker_home(tmp_path, *names):
    home = tmp_path / "home"
    (home / "run").mkdir(parents=True)
    for name, text in names:
        (home / "run" / name).write_text(text)
    return home


@pytest.mark.parametrize("files,left", [
    ([("server-crash.abc.marker", "{}"), ("server-crash-trace.abc.txt", "")],
     "run/server-crash.abc.marker"),
    ([("server-crash.abc.marker", "")], "run/server-crash.abc.marker"),
    ([("server-crash.marker", "{}")], "run/server-crash.marker"),
    ([("server-crash.abc.stopping", "{}"), ("server-crash-trace.abc.txt", "")],
     "run/server-crash.abc.stopping"),
])
def test_a_crash_marker_fails_the_audit_with_an_empty_or_absent_trace(tmp_path, capsys, files, left):
    home = _marker_home(tmp_path, *files)
    log = _write(tmp_path, "localm_j.log", CLEAN_LOG)
    assert la.main(["--manifest", str(_manifest(tmp_path, {})), "--home", str(home), str(log)]) == 1
    out = capsys.readouterr().out
    assert f"crash marker {left}" in out
    assert "UNEXPLAINED  error" in out


def test_the_marker_the_real_guard_writes_is_an_event_until_a_clean_stop(tmp_path, monkeypatch, capsys):
    from localm import bugreport
    monkeypatch.setattr(bugreport, "_diagnostics_allowed", lambda: False)
    home = tmp_path / "home"
    home.mkdir()
    log = _write(tmp_path, "localm_k.log", CLEAN_LOG)
    argv = ["--manifest", str(_manifest(tmp_path, {})), "--home", str(home), str(log)]
    assert bugreport.arm_crash_guard(home=home, instance_id="real") is True
    try:
        assert la.main(argv) == 1
        assert "crash marker run/server-crash.real.marker" in capsys.readouterr().out
        bugreport.clear_crash_marker(home=home, instance_id="real")
        assert la.main(argv) == 1
        assert "crash marker run/server-crash.real.stopping" in capsys.readouterr().out
    finally:
        bugreport.disarm_crash_guard(home=home, instance_id="real")
    assert la.main(argv) == 0


def test_a_manifest_can_name_a_crash_marker(tmp_path):
    home = _marker_home(tmp_path, ("server-crash.abc.marker", "{}"))
    log = _write(tmp_path, "localm_l.log", CLEAN_LOG)
    manifest = _manifest(tmp_path, {"errors": [{"match": "crash marker *", "why": "step kills it"}]})
    assert la.main(["--manifest", str(manifest), "--home", str(home), str(log)]) == 0


FLAGGED = "2026-11-03 11:20:32,000 WARNING localm: expected warning\n"
RAW_TRACEBACK = textwrap.dedent("""\
    Traceback (most recent call last):
      File "x.py", line 1, in <module>
    ValueError: boom
    """)


@pytest.mark.parametrize("between", [
    "native chatter from another thread\n",
    "\n",
    "2026-11-03 11:20:33,000 DEBUG   localm: GET /api/models -> 200 (5 ms)\n",
])
def test_a_traceback_not_directly_after_the_warning_is_an_error(tmp_path, capsys, between):
    log = _write(tmp_path, "localm_m.log", CLEAN_LOG + FLAGGED + between + RAW_TRACEBACK)
    manifest = _manifest(tmp_path, {"warnings": ["expected warning"]})
    assert la.main(["--manifest", str(manifest), str(log)]) == 1
    assert "UNEXPLAINED  error" in capsys.readouterr().out


def test_a_second_traceback_after_the_warnings_own_is_an_error(tmp_path):
    text = CLEAN_LOG + FLAGGED + RAW_TRACEBACK + RAW_TRACEBACK
    events = _events(tmp_path, text)
    assert [(e.kind, e.subject) for e in events] == [
        ("warning", "expected warning"),
        ("error", "Traceback (most recent call last):")]


def _format_with_exc(exc):
    import logging
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    rec = logging.LogRecord("localm", logging.WARNING, __file__, 1, "handled it", (),
                            (type(exc), exc, exc.__traceback__))
    return fmt.format(rec) + "\n"


def _raises(fn):
    try:
        fn()
    except Exception as e:
        return e
    raise AssertionError("did not raise")


def _chained_from():
    try:
        raise ValueError("first")
    except ValueError as e:
        raise RuntimeError("second") from e


def _raise_second():
    raise RuntimeError("second")


def _chained_during():
    try:
        raise ValueError("first")
    except ValueError:
        _raise_second()


@pytest.mark.parametrize("fn", [_chained_from, _chained_during])
def test_a_real_chained_exception_of_a_warning_stays_one_event(tmp_path, fn):
    text = CLEAN_LOG + _format_with_exc(_raises(fn))
    assert text.count("Traceback (most recent call last):") == 2
    events = _events(tmp_path, text)
    assert [(e.kind, e.subject) for e in events] == [("warning", "handled it")]


def test_a_raw_chain_marker_without_an_owned_traceback_does_not_excuse_the_next(tmp_path):
    text = CLEAN_LOG + FLAGGED + textwrap.dedent("""\
        native chatter
        During handling of the above exception, another exception occurred:

        Traceback (most recent call last):
        """)
    events = _events(tmp_path, text)
    assert [e.kind for e in events] == ["warning", "error"]


GLUED = "....2026-11-03 11:00:00,000 WARNING localm: real problem\n"


def test_a_record_glued_behind_a_partial_native_line_is_an_event(tmp_path, capsys):
    log = _write(tmp_path, "localm_n.log", CLEAN_LOG + GLUED)
    events, _lines, records = la.parse_log(log)
    assert records == 4
    assert [(e.kind, e.subject, e.time) for e in events] == [
        ("warning", "real problem", "2026-11-03 11:00:00,000")]
    assert la.main(["--manifest", str(_manifest(tmp_path, {})), str(log)]) == 1
    assert "UNEXPLAINED  warning" in capsys.readouterr().out


def test_the_native_text_before_a_glued_record_is_still_read(tmp_path):
    text = CLEAN_LOG + "llama_decode: failed to decode 2026-11-03 11:00:00,000 DEBUG   localm: fine\n"
    events = _events(tmp_path, text)
    assert [(e.kind, e.subject) for e in events] == [
        ("error", "llama_decode: failed to decode")]


def test_a_record_quoting_a_record_is_one_record(tmp_path):
    text = CLEAN_LOG + ("2026-11-03 11:00:00,000 DEBUG   localm: saw "
                        "2026-11-03 10:00:00,000 WARNING localm: old line\n")
    log = _write(tmp_path, "localm_o.log", text)
    events, _lines, records = la.parse_log(log)
    assert records == 4
    assert events == []


@pytest.mark.parametrize("msg,model", [
    ("hf model ready: my-hf-model", "my-hf-model"),
    ("exllama model ready: my-exl2", "my-exl2"),
    ("sd.cpp model ready: flux1-dev-Q4_0.gguf", "flux1-dev-Q4_0"),
    ("sd.cpp upscaler ready: esrgan.pth", "esrgan.pth"),
    ("whisper model ready: base", "base"),
    ("reranker ready: bge-reranker.gguf (labels=a,b)", "bge-reranker"),
    ("speech model ready: tts-model (24000 Hz, projector on cpu)", "tts-model"),
    ("speech model ready: tts-model (v3, 24000 Hz, mmproj on cpu)", "tts-model"),
    ("comfy: job uses model wan/wan2.2_ti2v_5B_fp16.safetensors", "wan2.2_ti2v_5B_fp16.safetensors"),
    ("koboldcpp music: job uses model ace-step-dit.gguf", "ace-step-dit"),
])
def test_every_model_ready_line_is_a_model_load(tmp_path, msg, model):
    text = CLEAN_LOG + f"2026-11-03 11:00:00,000 INFO    localm: {msg}\n"
    assert [(e.kind, e.subject) for e in _events(tmp_path, text)] == [("model-load", model)]


def _stub_sd_runner(monkeypatch, runner):
    monkeypatch.setattr(runner, "_spawn", lambda: None)
    monkeypatch.setattr(runner, "shutdown", lambda grace=5.0: None)
    monkeypatch.setattr(runner, "is_alive", lambda: True)
    monkeypatch.setattr(runner, "_request", lambda name, payload, timeout, **kw: {"name": name})


@pytest.mark.parametrize("ctx,name", [
    ({"model_path": "D:/m/flux1-dev-Q4_0.gguf"}, "flux1-dev-Q4_0.gguf"),
    ({"model_path": "", "diffusion_model_path": "D:/m/unet.safetensors",
      "vae_path": "D:/m/vae.safetensors"}, "unet.safetensors"),
])
def test_a_clean_sdcpp_load_logs_one_event_and_a_reuse_none(monkeypatch, caplog, ctx, name):
    import logging

    from localm.media.sdcpp.runner import SdRunner
    runner = SdRunner()
    _stub_sd_runner(monkeypatch, runner)
    with caplog.at_level(logging.INFO, logger="localm"):
        runner.ensure_loaded("k", "rt", ctx)
        runner.ensure_loaded("k", "rt", ctx)
    lines = [r.getMessage() for r in caplog.records if "ready" in r.getMessage()]
    assert lines == [f"sd.cpp model ready: {name}"]
    assert la.READY_RE.match(lines[0])


def test_a_failed_sdcpp_load_logs_no_ready_line(monkeypatch, caplog):
    import logging

    from localm.media.sdcpp.runner import SdRunner, SdWorkerError
    runner = SdRunner()
    _stub_sd_runner(monkeypatch, runner)

    def boom(name, payload, timeout, **kw):
        raise SdWorkerError("no")
    monkeypatch.setattr(runner, "_request", boom)
    with caplog.at_level(logging.INFO, logger="localm"), pytest.raises(SdWorkerError):
        runner.ensure_loaded("k", "rt", {"model_path": "D:/m/x.gguf"})
    assert not [r for r in caplog.records if "ready" in r.getMessage()]


def test_an_sdcpp_upscale_logs_its_model(monkeypatch, caplog):
    import logging

    from localm.media.sdcpp.runner import SdRunner
    runner = SdRunner()
    _stub_sd_runner(monkeypatch, runner)
    with caplog.at_level(logging.INFO, logger="localm"):
        runner.upscale_image({"model_path": "D:/m/esrgan.pth"}, "rt")
    assert [r.getMessage() for r in caplog.records if "ready" in r.getMessage()] == [
        "sd.cpp upscaler ready: esrgan.pth"]


class _ComfyResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def test_a_comfy_job_logs_each_model_of_the_shipped_workflow(monkeypatch, caplog):
    import logging

    from localm.media import comfy_client
    workflow = json.loads((_REPO / "localm/video_gen/wan_workflow.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(comfy_client, "_comfy_urlopen",
                        lambda req, timeout=None: _ComfyResponse(b'{"prompt_id": "p1"}'))
    with caplog.at_level(logging.INFO, logger="localm"):
        kind, value = comfy_client.comfy_submit_prompt("http://127.0.0.1:1", workflow)
    assert (kind, value) == (comfy_client.SUBMIT_OK, "p1")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("comfy: job")]
    assert sorted(lines) == ["comfy: job uses model umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                             "comfy: job uses model wan2.2_ti2v_5B_fp16.safetensors",
                             "comfy: job uses model wan2.2_vae.safetensors"]
    for line in lines:
        assert la.JOB_MODEL_RE.match(line)


def test_a_refused_comfy_job_logs_no_model(monkeypatch, caplog):
    import logging
    import urllib.error

    from localm.media import comfy_client

    def refused(req, timeout=None):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(comfy_client, "_comfy_urlopen", refused)
    workflow = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}}}
    with caplog.at_level(logging.INFO, logger="localm"):
        kind, _value = comfy_client.comfy_submit_prompt("http://127.0.0.1:1", workflow)
    assert kind == comfy_client.SUBMIT_URL_ERROR
    assert not [r for r in caplog.records if r.getMessage().startswith("comfy: job")]


def test_workflow_model_names_skip_what_is_not_a_model_file():
    from localm.media.comfy_client import workflow_model_names
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "sub/u.GGUF"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "a.safetensors",
                                                         "clip_name2": "b.safetensors"}},
        "3": {"class_type": "KSampler", "inputs": {"sampler_name": "euler", "model": ["1", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x.safetensors"}},
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "sub/u.GGUF"}},
        "6": {"class_type": "Custom", "inputs": {"model_name": ["1", 0]}},
        "7": "not a node",
        "8": {"class_type": "NoInputs"},
    }
    assert workflow_model_names(workflow) == ["sub/u.GGUF", "a.safetensors", "b.safetensors"]
    assert workflow_model_names("nonsense") == []


def test_a_clean_hf_load_logs_a_ready_line(tmp_path, monkeypatch, caplog):
    import logging

    from localm.inference.backends import hf
    model = tmp_path / "my-hf-model"
    model.mkdir()
    (model / "config.json").write_text("{}")

    class FakeRunner:
        def spawn_and_load(self, params, timeout):
            return {"device": "cpu"}
    monkeypatch.setattr(hf, "HFRunner", FakeRunner)
    backend = hf.HFBackend(str(model), device="cpu")
    with caplog.at_level(logging.INFO, logger="localm"):
        backend.load()
    lines = [r.getMessage() for r in caplog.records if "ready" in r.getMessage()]
    assert lines == ["hf model ready: my-hf-model"]


def test_a_native_music_job_logs_its_models_by_their_real_names(monkeypatch, caplog):
    import logging

    from localm.media.koboldcpp import music, server
    models = server.ModelSet(text_encoder="D:/m/te.gguf", dit="D:/m/ace-dit.gguf",
                             vae="D:/m/vae.gguf", lm="D:/m/ace-lm.gguf")
    monkeypatch.setattr(music, "resolve_models", lambda *a, **k: models)
    monkeypatch.setattr(music, "runtime_paths", lambda m: server.ModelSet(
        text_encoder="X~1", dit="DIT~1", vae="V~1", lm="LM~1"))
    monkeypatch.setattr(music, "_ensure_runtime", lambda *a, **k: object())
    monkeypatch.setattr(music.server, "run", lambda *a, **k: b"wav")
    monkeypatch.setattr(music, "broken_reason", lambda data: None)
    monkeypatch.setattr(music.runtime, "backend_worked", lambda b: True)
    with caplog.at_level(logging.INFO, logger="localm"):
        assert music.generate_wav({}, "cpu", {})[0] == b"wav"
    assert [r.getMessage() for r in caplog.records if r.getMessage().startswith("koboldcpp music")] == [
        "koboldcpp music: job uses model ace-dit.gguf",
        "koboldcpp music: job uses model ace-lm.gguf"]


def test_the_whisper_worker_writes_to_the_debug_log():
    source = (_REPO / "localm/voice.py").read_text(encoding="utf-8")
    worker = next(n for n in ast.parse(source).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_worker_main")
    calls = {n.func.id for n in ast.walk(worker)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "attach_child_logging" in calls
