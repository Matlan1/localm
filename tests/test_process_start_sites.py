# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm knows all its processes (localm.procs): every place in a shipped file that
starts a program goes through ``procs.start``, starts something that imports localm
(and so records itself), or is named below with the kind of child it starts.

A start site is any mention of a program-starting function, called or passed on as a
value, whatever name it was imported under. ``subprocess.run`` and its siblings wait
for their child, which is found as a running child of a recorded process, so they are
start sites only when they detach it. A program of the user (a web browser, a viewer)
is opened through ``procs.open_for_user``."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = [REPO / "launcher.pyw", REPO / "hatch_build.py", REPO / "localm.py",
      *sorted((REPO / "localm").rglob("*.py")), *sorted((REPO / "installer").rglob("*.py")),
      *sorted((REPO / "runtime").rglob("*.py")),
      *(REPO / "scripts" / n for n in (
          "crash_recovery_watchdog.py", "report_issue.py", "rollback_update.py",
          "update_gate.pyw", "update_watchdog.py", "verify_release.py"))]

#: The child imports localm: ``python -m localm...`` or a multiprocessing worker, whose
#: target is a function of localm. The test looks for that in the function.
LOCALM = frozenset({
    "localm/cli/chat.py:run:Popen",
    "localm/diagnostics.py:_run_hf_backend_probe:Process",
    "localm/diagnostics.py:check_worker_spawn:Process",
    "localm/discover.py:_ask_torch_probe_child:Popen",
    "localm/inference/_embedder_runner.py:EmbedderRunner._spawn:Process",
    "localm/inference/_speech_runner.py:SpeechRunner._spawn:Process",
    "localm/inference/backends/_hf_runner.py:HFRunner._spawn:Process",
    "localm/inference/backends/llamacpp/_loader.py:_spawn_probe_daemon:Popen",
    "localm/inference/backends/llamacpp/_runner.py:ModelRunner._spawn:Process",
    "localm/inference/gbnf.py:_spawn_trigger_probe_daemon:Popen",
    "localm/inference/hf_tokenizer_safety.py:_run_probe_subprocess:Popen",
    "localm/jobmanager.py:JobManager.start_cli._run:Popen",
    "localm/media/koboldcpp/models.py:_pull:Popen",
    "localm/media/sdcpp/runner.py:SdRunner._spawn:Process",
    "localm/plugins/coder/cli/_main.py:_build_backend_gated:Popen",
    "localm/plugins/coder/server.py:ManagedServer.start:Popen",
    "localm/update_mechanism/steps_probe.py:probe:Popen",
    "localm/voice.py:_spawn_worker:Process",
})
#: The child imports localm, by a command line that is built elsewhere.
LOCALM_ELSEWHERE = {
    "installer/gui.py:Wizard._launch:Popen": "LocaLM.exe gui",
    "localm/diagnostics.py:run_report_isolated:Popen": "python -c, imports localm.diagnostics",
    "localm/inference/http_server.py:_execv_restart:execv": "python -m localm gui, same process",
    "localm/update_mechanism/steps_update.py:run_child:Popen":
        "python -m localm, waited for; or python -c import torch, waited for",
    "localm/update_mechanism/applier.py:Applier._relaunch:Popen":
        "the command line of the server, python -m localm",
    "scripts/crash_recovery_watchdog.py:relaunch:Popen": "python -m localm gui",
}
#: A short tool the starter waits for: a running child of a recorded process.
TOOLS = {
    "installer/gui.py:_run:Popen": "one step of setup",
    "scripts/update_gate.pyw:run:Popen": "the gate (localm/__init__.py as a program)",
    "localm/browser/provision.py:_stream_install:Popen": "playwright install",
    "localm/discover.py:_list_gpus_probe:Popen": "nvidia-smi",
    "localm/discover.py:nvidia_smi_used_by_bus:Popen": "nvidia-smi",
    "localm/media/managed_comfy_provision.py:_run:Popen": "git, uv, pip",
    "localm/model_manager/model_packages.py:_stream:Popen": "uv, pip",
    "localm/plugins/coder/hooks.py:run_hook:Popen": "a hook command",
    "localm/plugins/coder/tools/base.py:_run_cancellable:Popen": "a shell command",
    "localm/plugins/deps.py:_run_attempts:Popen": "uv, pip",
    "localm/updater_handoff.py:_run_check:Popen": "the check the update mechanism of a build runs",
    "localm/vc_runtime.py:run_installer:ShellExecuteExW": "Microsoft's runtime installer",
}
#: A program of the user: each of these goes through procs.open_for_user.
USERS = {
    "localm/appface.py:_StatusWindow._open:open": "web browser",
    "localm/appface.py:_WinTray._open:open": "web browser",
    "localm/appface.py:open_logs:open": "web browser",
    "localm/appface.py:open_logs:startfile": "text editor",
    "localm/cli/media.py:_open_file:Popen": "image, music or video viewer (open, xdg-open)",
    "localm/cli/media.py:_open_file:startfile": "image, music or video viewer",
    "localm/plugins/gui/cli.py:_open_attached:open": "web browser",
    "localm/plugins/gui/cli.py:_open_when_ready:open": "web browser",
    "localm/plugins/gui/cli.py:_serve_beside_native_window:open": "web browser",
}
#: The default of a parameter that is called through procs.open_for_user.
USERS_DEFAULT = {
    "localm/bugreport/consent.py:offer_to_send:open": "mail program",
    "localm/bugreport/consent.py:report_failure:open": "mail program",
}
#: Started without the record only where localm cannot be imported at all.
NO_LOCALM = {"launcher.pyw:_spawn_detached:Popen": "a window of the launcher"}
#: A program of the user that may hold other windows and stays open: not recorded, and in
#: a session of its own. The localm in it records itself.
TERMINAL = {"launcher.pyw:_open_terminal:Popen": "terminal program"}
RECORDED = "localm/procs.py:start:Popen"
#: Every place that starts a long-lived or detached program that does not import localm.
STARTED = {
    "launcher.pyw": 1,
    "localm/media/comfy_client.py": 1,
    "localm/media/koboldcpp/_proc.py": 3,
    "localm/media/koboldcpp/runtime.py": 1,
    "localm/plugins/coder/background.py": 1,
    "localm/plugins/coder/mcp.py": 1,
    "localm/portmux.py": 1,
    "localm/updater.py": 1,
    "localm/updater_finish.py": 1,
    "localm/updater_handoff.py": 1,
}
DETACHED = ("start_new_session", "process_group", "preexec_fn", "CREATE_NEW_PROCESS_GROUP",
            "DETACHED_PROCESS", "CREATE_NEW_CONSOLE", "CREATE_BREAKAWAY_FROM_JOB")


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    parts.append(node.id if isinstance(node, ast.Name) else "?")
    return ".".join(reversed(parts))


STARTS = frozenset({
    "Popen", "Process", "SpawnProcess", "ForkProcess", "ForkServerProcess", "Pool", "Manager",
    "SyncManager", "ProcessPoolExecutor", "openpty", "ShellExecuteW", "ShellExecuteExW",
    "ShellExecuteA", "WinExec", "CreateProcess", "CreateProcessW", "CreateProcessAsUserW",
    "CreateProcessWithTokenW", "CreateProcessWithLogonW"})
#: Of these, only a call or a base class starts something: other things have the name too.
CALLED = {"Process", "Pool", "Manager"}
OS_STARTS = {"startfile", "popen", "system", "fork", "forkpty", "posix_spawn", "posix_spawnp",
             *(a + b for a in ("exec", "spawn") for b in (
                 "l", "le", "lp", "lpe", "v", "ve", "vp", "vpe"))}
WAITS = {"run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
STAR = ("subprocess", "os", "multiprocessing", "concurrent.futures", "asyncio", "pty")


def _is_start(full: str) -> bool:
    module, _, last = full.rpartition(".")
    if module == "os":
        return last in OS_STARTS
    if full == "psutil.Process":
        return False
    return (last in STARTS
            or last.startswith(("create_subprocess", "subprocess_exec", "subprocess_shell"))
            or full in ("webbrowser.open", "webbrowser.open_new", "webbrowser.open_new_tab",
                        "webbrowser.get", "pty.spawn", "pty.fork"))


def _waits_detached(call, function) -> bool:
    """Whether a subprocess.run(...) detaches its child: by a keyword of the call, or by
    a dict of keywords of a function that names a detaching flag."""
    if not isinstance(call, ast.Call):
        return False
    if any(d in ast.unparse(call) for d in DETACHED):
        return True
    return (any(k.arg is None for k in call.keywords) and function is not None
            and any(d in ast.unparse(function) for d in DETACHED))


def sites_of(source: str) -> list:
    """[(enclosing function, what)] for each start site, in order."""
    tree = ast.parse(source)
    parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    names: dict = {}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                names[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                names[a.asname or a.name] = f"{n.module}.{a.name}"
                if a.name == "*" and n.module.split(".")[0] in {m.split(".")[0] for m in STAR}:
                    out.append((n.lineno, "", f"{n.module}.*"))
    notes = {m for n in ast.walk(tree) for f in ("annotation", "returns")
             if isinstance(getattr(n, f, None), ast.AST) for m in ast.walk(getattr(n, f))}
    for n in ast.walk(tree):
        if n in notes:
            continue
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            full = _dotted(n)
            head, _, rest = full.partition(".")
            full = f"{names.get(head, head)}.{rest}"
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and "." in names.get(n.id, ""):
            full = names[n.id]
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str)
              and isinstance(parent.get(n), ast.Call)
              and getattr(parent[n].func, "id", getattr(parent[n].func, "attr", ""))
              in ("getattr", "__import__", "import_module")):
            if n.value in ("subprocess", "multiprocessing"):
                full = "by name.Popen"
            else:
                full = "os." + n.value
                full = full if _is_start(full) else "by name." + n.value
        else:
            continue
        up, where, function = n, [], None
        while up in parent:
            up = parent[up]
            if isinstance(up, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                where.insert(0, up.name)
                if function is None and not isinstance(up, ast.ClassDef):
                    function = up
        call = parent.get(n)
        last = full.rsplit(".", 1)[-1]
        if isinstance(call, ast.Attribute):
            continue
        if full.startswith("subprocess.") and last in WAITS:
            if _waits_detached(call, function):
                out.append((n.lineno, ".".join(where), last + " detached"))
            continue
        if not _is_start(full):
            continue
        base = isinstance(call, ast.ClassDef)
        called = isinstance(call, ast.Call) and call.func is n
        if last in CALLED and not (called or base):
            continue
        out.append((n.lineno, ".".join(where), last))
    return [s[1:] for s in sorted(out)]


def _function(path: Path, where: str):
    node: ast.AST = ast.parse(path.read_text(encoding="utf-8"))
    for name in where.split("."):
        node = next(n for n in ast.walk(node) if getattr(n, "name", None) == name
                    and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    return node


def _split(key: str) -> tuple:
    rel, where, what = key.rsplit(":", 2)
    return REPO / rel, where, what


def found() -> dict:
    out: dict = {}
    for path in PY:
        rel = path.relative_to(REPO).as_posix()
        for where, what in sites_of(path.read_text(encoding="utf-8")):
            key = f"{rel}:{where}:{what}"
            out[key] = out.get(key, 0) + 1
    return out


def test_every_start_site_is_recorded_or_named():
    known = {RECORDED, *LOCALM, *LOCALM_ELSEWHERE, *TOOLS, *USERS, *USERS_DEFAULT, *NO_LOCALM,
             *TERMINAL}
    now = found()
    assert set(now) - known == set(), (
        "These places start a program that localm would not know of. Start a long-lived or "
        "detached one with localm.procs.start; name one that imports localm, a tool that is "
        "waited for or a program of the user in this file.")
    assert known - set(now) == set(), "These places start nothing any more: remove them here."


def test_one_function_starts_one_program():
    assert {k: n for k, n in found().items() if n > 1} == {
        "localm/cli/media.py:_open_file:Popen": 2}


def test_a_localm_child_is_started_as_localm():
    for key in LOCALM:
        path, where, what = _split(key)
        fn = _function(path, where)
        if what == "Process":
            assert any(isinstance(n, ast.keyword) and n.arg == "target" for n in ast.walk(fn)), key
            continue
        text = ast.unparse(fn if what != "Popen" or "." not in where
                           else _function(path, where.rsplit(".", 1)[0]))
        assert "'-m', 'localm" in text, key


def test_a_program_of_the_user_is_opened_as_one():
    for key in USERS:
        path, where, what = _split(key)
        fn = _function(path, where)
        given = [_dotted(c.args[0]) for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and _dotted(c.func) == "procs.open_for_user" and c.args]
        bare = [_dotted(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)
                and _dotted(c.func).rsplit(".", 1)[-1] == what]
        assert [g for g in given if g.rsplit(".", 1)[-1] == what] and not bare, key
    text = (REPO / "localm/bugreport/consent.py").read_text(encoding="utf-8")
    calls = [c for c in ast.walk(ast.parse(text)) if isinstance(c, ast.Call)]
    assert not [c for c in calls if _dotted(c.func) == "open_browser"]
    assert [c for c in calls if _dotted(c.func) == "procs.open_for_user"
            and _dotted(c.args[0]) == "open_browser"]


def test_a_start_without_the_record_needs_a_missing_localm():
    for key in NO_LOCALM:
        path, where, _ = _split(key)
        fn = _function(path, where)
        guarded = [h for t in ast.walk(fn) if isinstance(t, ast.Try) for h in t.handlers
                   if _dotted(h.type) == "ImportError" and "subprocess.Popen" in ast.unparse(h)]
        assert len(guarded) == 1 and ast.unparse(fn).count("subprocess.Popen") == 1, key


def test_the_terminal_is_started_in_a_session_of_its_own():
    for key in TERMINAL:
        path, where, _ = _split(key)
        assert "start_new_session=True" in ast.unparse(_function(path, where)), key


def test_a_tool_is_not_detached():
    for key in TOOLS:
        path, where, _ = _split(key)
        text = ast.unparse(_function(path, where))
        assert not [d for d in DETACHED if d in text], key


def test_the_long_lived_programs_go_through_start():
    now: dict = {}
    for path in PY:
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and _dotted(n.func) == "procs.start":
                rel = path.relative_to(REPO).as_posix()
                now[rel] = now.get(rel, 0) + 1
    assert now == STARTED


def test_the_scan_sees_every_spelling():
    assert sites_of("""
import subprocess as _sp, os, asyncio, multiprocessing as mp, webbrowser, subprocess
from subprocess import Popen as P, run
from os import execv
def a():
    _sp.Popen([])
    P([])
    execv("x", [])
def b(popen=subprocess.Popen):
    os.spawnv(0, "x", [])
    getattr(subprocess, "Popen")([])
class C:
    async def c(self):
        await asyncio.create_subprocess_exec("x")
        mp.get_context("spawn").Process(target=a).start()
        ctypes.windll.shell32.ShellExecuteW(None)
        webbrowser.open("x")
def d(x: subprocess.Popen) -> subprocess.Popen:
    subprocess.run([]); run([]); psutil.Process(1); subprocess.call([], cwd="x")
""") == [("a", "Popen"), ("a", "Popen"), ("a", "execv"), ("b", "Popen"), ("b", "spawnv"),
         ("b", "Popen"), ("C.c", "create_subprocess_exec"), ("C.c", "Process"),
         ("C.c", "ShellExecuteW"), ("C.c", "open")]


def test_the_scan_sees_the_rarer_spellings():
    assert sites_of("""
import os, psutil, subprocess, importlib, multiprocessing as mp
from concurrent.futures.process import ProcessPoolExecutor as E
from multiprocessing.context import SpawnProcess
from subprocess import *
def a():
    os.system("x")
    psutil.Popen([])
    mp.Manager()
    E()
    SpawnProcess(target=a).start()
    getattr(os, "spawnv")(0, "x", [])
    getattr(subprocess, "Popen")([])
    importlib.import_module("subprocess").Popen([])
    __import__("subprocess")
class W(mp.Process): pass
class P(psutil.Popen): pass
def b(**kw):
    subprocess.run([], start_new_session=True)
    subprocess.call([], creationflags=subprocess.CREATE_BREAKAWAY_FROM_JOB)
    subprocess.run([], capture_output=True)
def c():
    kw = {}
    kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.check_call([], **kw)
""") == [("", "subprocess.*"), ("a", "system"), ("a", "Popen"), ("a", "Manager"),
         ("a", "ProcessPoolExecutor"), ("a", "SpawnProcess"), ("a", "spawnv"), ("a", "Popen"),
         ("a", "Popen"), ("a", "Popen"), ("a", "Popen"), ("W", "Process"), ("P", "Popen"),
         ("b", "run detached"), ("b", "call detached"), ("c", "check_call detached")]


if __name__ == "__main__":
    for k, v in found().items():
        print(f"{k} {v}")
