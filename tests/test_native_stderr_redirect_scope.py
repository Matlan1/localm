# SPDX-License-Identifier: AGPL-3.0-or-later
"""``dedup_native_stderr`` redirects the process-wide fd 2 and appends every
line it reads to the activity ring a bug report carries. It is entered only by
code that runs in an isolated worker process (the GGUF model runner and the
embedder runner), so a stderr write elsewhere in the server process never
reaches the server's ring. These tests pin where it is entered and where the
classes that enter it are constructed."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "localm"

_REDIRECT = "dedup_native_stderr"
_ALLOWED_REDIRECT_USERS = {
    "inference/backends/llamacpp/llama.py",
    "inference/embedder.py",
    "inference/_embedder_runner.py",
}
# Class -> the only module that may construct it. Each module is a worker
# process entry: _runner.py's child loop, _worker.py inside it, and the
# embedder runner's child loop.
_ALLOWED_CONSTRUCTORS = {
    "LlamaCpp": {"inference/backends/llamacpp/_worker.py"},
    "GgufWorker": {"inference/backends/llamacpp/_runner.py"},
    "GGUFEmbedder": {"inference/_embedder_runner.py"},
}


def _name_of(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan():
    redirect_users, constructors = {}, {}
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)) and \
                    isinstance(getattr(node, "ctx", None), ast.Load) and \
                    _name_of(node) == _REDIRECT:
                redirect_users.setdefault(rel, 0)
                redirect_users[rel] += 1
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == _REDIRECT:
                        redirect_users.setdefault(rel, 0)
                        redirect_users[rel] += 1
            if isinstance(node, ast.Call) and _name_of(node.func) in _ALLOWED_CONSTRUCTORS:
                constructors.setdefault(_name_of(node.func), set()).add(rel)
    return redirect_users, constructors


def test_fd2_redirect_is_entered_only_from_worker_process_code():
    users, _ = _scan()
    users.pop("debuglog.py", None)
    assert set(users) == _ALLOWED_REDIRECT_USERS


def test_redirecting_classes_are_constructed_only_in_worker_processes():
    _, constructors = _scan()
    for cls, allowed in _ALLOWED_CONSTRUCTORS.items():
        assert constructors.get(cls) == allowed, cls
