# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/affected_tests.py: a test that reaches a changed module through one
other module is selected even when a hub module makes the full --depth
selection too wide.

Two shapes, each pinned on a throwaway checkout and on the real tree:
  - a test that calls a route by URL through the app factory, where the
    route's module imports the changed module (the request model of the
    route lives there): selected at every depth by the route it names;
  - a test that imports a narrow module which imports the changed module,
    beside a hub (the config module) that most tests import: selected at
    --depth 1 because the hub, not the changed module, is left out.

A route prefix that keeps only the namespace (`/api/` of an f-string route)
names no route.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "affected_tests.py"
_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _load():
    spec = importlib.util.spec_from_file_location("affected_tests_indirect", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True, env={**os.environ, **_GIT_ENV},
                          creationflags=_NO_WINDOW).stdout


_HUB_TESTS = 8
_FILES = {
    "localm/__init__.py": "",
    "localm/proto.py": "class ChatRequest:\n    max_tokens = 1\n",
    "localm/routes_chat.py": (
        "from fastapi import FastAPI\n\nfrom localm.proto import ChatRequest\n\n"
        "app = FastAPI()\n\n\n"
        "@app.post('/v1/chat/completions')\n"
        "async def chat(req: ChatRequest):\n    return {}\n\n\n"
        "@app.get(f'/api/{MEDIA}/workflows')\n"
        "async def workflows():\n    return []\n"),
    "localm/app.py": "from localm import proto, routes_chat\n",
    "localm/store.py": "def lock():\n    return 1\n",
    "localm/config.py": "from localm import store\n",
    "localm/auth.py": "from localm import store\n",
    "localm/maint.py": "def run():\n    from localm.store import lock\n    return lock()\n",
    "tests/conftest.py": "",
    "tests/test_chat_url.py": (
        "from localm.app import routes_chat\n\n\n"
        "def test_chat(client):\n"
        "    assert client.post('/v1/chat/completions', json={'max_tokens': 0})\n"),
    "tests/test_app_only.py": "import localm.app\n\n\ndef test_app():\n    assert localm.app\n",
    "tests/test_maint.py": "from localm.maint import run\n\n\ndef test_maint():\n    assert run()\n",
    "tests/test_auth.py": "import localm.auth\n\n\ndef test_auth():\n    assert localm.auth\n",
    "tests/test_api_namespace.py": (
        "def test_other(client):\n    assert client.get('/api/other/thing')\n"),
    "tests/test_none.py": "def test_none():\n    assert True\n",
    **{f"tests/test_hub_{i}.py": "import localm.config\n\n\ndef test_hub():\n    assert True\n"
       for i in range(_HUB_TESTS)},
}
_TOTAL = sum(1 for p in _FILES if Path(p).name.startswith("test_"))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A committed throwaway checkout with the files above; the module's REPO
    points at it."""
    for rel, src in _FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    mod = _load()
    monkeypatch.setattr(mod, "REPO", tmp_path)
    return mod, tmp_path


def _run_cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    """The script's main() in a fresh interpreter with REPO pointed at *root*."""
    driver = ("import importlib.util, sys\n"
              f"spec = importlib.util.spec_from_file_location('m', {str(_SCRIPT)!r})\n"
              "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
              "from pathlib import Path\n"
              f"m.REPO = Path({str(root)!r})\n"
              "raise SystemExit(m.main(sys.argv[1:]))")
    return subprocess.run([sys.executable, "-c", driver, *args], capture_output=True,
                          text=True, env={**os.environ, **_GIT_ENV}, creationflags=_NO_WINDOW)


# --------------------------------------------------------------------------- #
#  A route whose module imports the changed module                            #
# --------------------------------------------------------------------------- #

def test_a_route_of_an_importer_selects_the_test_calling_it_at_depth_zero(repo):
    mod, _ = repo
    selected = mod.select(["localm/proto.py"], mod.Graph())
    assert selected["tests/test_chat_url.py"] == [
        "names route /v1/chat/completions of localm.routes_chat, which imports localm.proto"]
    assert "tests/test_app_only.py" not in selected
    assert "tests/test_none.py" not in selected


def test_a_namespace_only_route_prefix_names_no_route(repo):
    mod, root = repo
    assert mod.route_prefixes((root / "localm/routes_chat.py").read_text(encoding="utf-8")) \
        == {"/v1/chat/completions"}
    selected = mod.select(["localm/routes_chat.py"], mod.Graph())
    assert "tests/test_api_namespace.py" not in selected
    assert selected["tests/test_chat_url.py"] == ["names route /v1/chat/completions"]


# --------------------------------------------------------------------------- #
#  A hub beside a narrow importer                                             #
# --------------------------------------------------------------------------- #

def test_the_hub_is_left_out_and_the_narrow_importers_stay_at_depth_one(repo):
    mod, _ = repo
    graph = mod.Graph()
    full = mod.select(["localm/store.py"], graph, depth=1)
    assert len(full) / _TOTAL > 0.5
    held: dict[str, float] = {}
    selected = mod.select(["localm/store.py"], graph, depth=1, max_share=0.5, held=held)
    assert held == {"localm.config": _HUB_TESTS / _TOTAL}
    assert selected == {"tests/test_auth.py": ["imports localm.auth (1 hop(s) from a change)"],
                        "tests/test_maint.py": ["imports localm.maint (1 hop(s) from a change)"]}


def test_the_command_line_keeps_depth_one_by_leaving_the_hub_out(repo):
    _, root = repo
    out = _run_cli(root, "--depth", "1", "--max-share", "0.5", "--files", "localm/store.py")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ["tests/test_auth.py", "tests/test_maint.py"]
    assert (f"not followed: localm.config, which {_HUB_TESTS / _TOTAL:.0%} of the test files "
            "import, so following it would exceed --max-share 50%") in out.stderr


def test_nothing_is_left_out_when_depth_zero_is_already_too_wide(repo):
    mod, _ = repo
    held: dict[str, float] = {}
    selected = mod.select(["localm/config.py"], mod.Graph(), depth=1, max_share=0.25, held=held)
    assert held == {}
    assert {f"tests/test_hub_{i}.py" for i in range(_HUB_TESTS)} <= set(selected)


def test_a_changed_module_is_never_left_out(repo):
    mod, _ = repo
    held: dict[str, float] = {}
    selected = mod.select(["localm/store.py", "localm/config.py"], mod.Graph(), depth=1,
                          max_share=0.65, held=held)
    assert held == {"localm.auth": 1 / _TOTAL}
    assert {f"tests/test_hub_{i}.py" for i in range(_HUB_TESTS)} <= set(selected)
    assert "tests/test_maint.py" in selected
    assert len(selected) / _TOTAL <= 0.65


# --------------------------------------------------------------------------- #
#  The real tree                                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real():
    mod = _load()
    return mod, mod.Graph()


def test_real_tree_a_protocol_change_selects_the_test_posting_to_its_route(real):
    mod, graph = real
    selected = mod.select(["localm/inference/protocol.py"], graph)
    assert ("names route /v1/chat/completions of localm.inference.routes.chat, which imports "
            "localm.inference.protocol") in selected["tests/test_rag_reembed.py"]


@pytest.mark.parametrize(("changed", "test", "via"), [
    (["localm/memory/store.py", "localm/storekit.py"], "tests/test_cli_setup_embeddings_msg.py",
     "imports localm.cli.maintenance (1 hop(s) from a change)"),
    (["localm/storekit.py"], "tests/test_auth_keystore_cross_process.py",
     "imports localm.auth (1 hop(s) from a change)"),
])
def test_real_tree_a_storekit_change_leaves_config_out_and_keeps_the_narrow_paths(real, changed, test, via):
    mod, graph = real
    held: dict[str, float] = {}
    selected = mod.select(changed, graph, depth=1, max_share=0.25, held=held)
    assert "localm.config" in held
    assert via in selected[test]
    assert len(selected) / len(graph.test_files) <= 0.25
