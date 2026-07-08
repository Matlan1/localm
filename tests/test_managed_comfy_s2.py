# SPDX-License-Identifier: AGPL-3.0-or-later
"""STAGE S2 (COPY path) for the localm-managed ComfyUI feature.

S2 provisions localm's OWN ComfyUI by REPLICATING a user's existing stack
(design decisions 2 + 3): when the user HAS a working ComfyUI (comfy_workdir set
and a venv discoverable under it), read its repo commit + a pip-freeze of their
venv, clone that same commit into <LOCALM_HOME>/comfyui, create a FRESH localm
venv, and pip-install the SAME package versions into it (NOT a byte-copy of the
user's venv). Custom nodes are copied only when asked. Models are shared via S1's
extra_model_paths.yaml. When NO usable user ComfyUI is present, setup HONESTLY
reports that the fresh hardware-matched install is stage S3 (not yet built) and
changes nothing (AGENTS.md rule 5: no facade).

The heavy end-to-end test exercises the copy path FOR REAL against a MINIMAL fake:
a real throwaway git repo (stub main.py committed, so rev-parse works) with a real
minimal venv holding ONE tiny hand-built wheel (no torch - the copy logic is
version-agnostic, so a tiny freeze exercises it fully and fast). The wheel is
installed via a ``name @ file://`` reference so pip freeze round-trips OFFLINE and
the copy path can replay it with PIP_NO_INDEX. The multi-GB torch replication is a
documented manual/integration check; this proves the mechanism end to end.

Design + locked decisions: dev-notes/DESIGN-localm-managed-comfyui-2026-07-08.md
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

import localm.config as cfg
from localm.media import managed_comfy as mc


# --------------------------------------------------------------------------- #
#  Isolation: a throwaway LOCALM_HOME wired through BOTH the lazy home_dir()   #
#  and the import-frozen config attrs, exactly like the S1 test's `home`.      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    return h


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _build_tiny_wheel(dest_dir: Path, name: str, version: str) -> Path:
    """A minimal, dependency-free, pure-python py3-none-any wheel. Real enough for
    pip to install offline (no build step); tiny enough to be fast. A wheel is a
    zip of the package plus a ``.dist-info`` with METADATA/WHEEL/RECORD."""
    dist_info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f'__version__ = "{version}"\n',
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            "Summary: localm S2 test fixture package\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: localm-s2-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = [f"{p},," for p in files] + [f"{dist_info}/RECORD,,"]
    files[f"{dist_info}/RECORD"] = "\n".join(record) + "\n"
    whl = dest_dir / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, content in files.items():
            z.writestr(arcname, content)
    return whl


# --------------------------------------------------------------------------- #
#  Session-scoped fake "user ComfyUI": a real git checkout + a real venv with  #
#  a tiny replicable package + custom nodes + a models dir. Built ONCE (the     #
#  venv+wheel is the only slow part) and reused read-only by the copy tests.    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def fake_user_comfy(tmp_path_factory):
    if not _git_available():
        pytest.skip("git not on PATH (S2 copy path clones the user's ComfyUI)")

    root = tmp_path_factory.mktemp("user_comfy")
    workdir = root / "ComfyUI"
    workdir.mkdir()

    # ComfyUI-shaped source: a committed main.py + a .gitignore that (like real
    # ComfyUI) keeps the venv, models, custom_nodes and outputs OUT of git, so a
    # clone brings ONLY the source - never the venv or custom nodes.
    (workdir / "main.py").write_text("# fake ComfyUI entry point\nprint('comfy')\n",
                                      encoding="utf-8")
    (workdir / ".gitignore").write_text(
        "venv/\n.venv/\nmodels/\ncustom_nodes/\noutput/\ninput/\ntemp/\n",
        encoding="utf-8")

    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    ident = ["-c", "user.email=s2@localhost", "-c", "user.name=s2-test",
             "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]

    def _git(*args):
        subprocess.run(["git", *ident, *args], cwd=str(workdir), env=env,
                       check=True, capture_output=True, text=True)

    _git("init")
    _git("add", "main.py", ".gitignore")
    _git("commit", "-m", "init fake ComfyUI")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(workdir),
                            capture_output=True, text=True, check=True).stdout.strip()

    # Untracked custom nodes (two of them) - only copied when the user opts in.
    for node in ("NodeAlpha", "NodeBeta"):
        nd = workdir / "custom_nodes" / node
        nd.mkdir(parents=True)
        (nd / "__init__.py").write_text(f"# {node}\n", encoding="utf-8")

    # A models dir so S1's extra_model_paths generator has <workdir>/models to point at.
    (workdir / "models" / "checkpoints").mkdir(parents=True)

    # A REAL venv holding one tiny replicable package, installed via `name @ file://`
    # so pip freeze emits the file URL and the copy path can replay it OFFLINE.
    venv_dir = workdir / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True,
                   capture_output=True, text=True)
    user_py = _venv_python(venv_dir)
    pkg_name, pkg_version = "localms2pkg", "0.0.1"
    whl = _build_tiny_wheel(root, pkg_name, pkg_version)
    req = f"{pkg_name} @ {whl.as_uri()}"
    subprocess.run([str(user_py), "-m", "pip", "install", "--no-index",
                    "--disable-pip-version-check", req],
                   check=True, capture_output=True, text=True)

    return types.SimpleNamespace(
        workdir=workdir, commit=commit, venv_python=user_py,
        pkg_name=pkg_name, pkg_version=pkg_version, wheel=whl)


def _freeze(venv_python: Path) -> list:
    out = subprocess.run([str(venv_python), "-m", "pip", "freeze"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
#  Discovery of the user's ComfyUI stack (repo state + venv).                  #
# --------------------------------------------------------------------------- #

def test_discover_none_without_workdir(home):
    from localm.media import managed_comfy_provision as prov
    assert prov.discover_user_comfy() is None


def test_discover_none_when_no_venv(home, tmp_path):
    """comfy_workdir set with a main.py but NO venv is not copyable - discover
    returns None (the dispatcher then reports the fresh install is S3)."""
    from localm.media import managed_comfy_provision as prov
    ud = tmp_path / "no-venv-comfy"
    ud.mkdir()
    (ud / "main.py").write_text("# comfy\n", encoding="utf-8")
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(ud)})
    assert prov.discover_user_comfy() is None


def test_discover_finds_git_stack(home, fake_user_comfy):
    from localm.media import managed_comfy_provision as prov
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})
    stack = prov.discover_user_comfy()
    assert stack is not None
    assert stack.commit == fake_user_comfy.commit
    assert Path(stack.venv_python).is_file()
    assert stack.version_marker.startswith("git:")


def test_read_user_comfy_commit_git_and_nongit(tmp_path, fake_user_comfy):
    from localm.media import managed_comfy_provision as prov
    assert prov.read_user_comfy_commit(fake_user_comfy.workdir) == fake_user_comfy.commit
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    assert prov.read_user_comfy_commit(plain) is None


# --------------------------------------------------------------------------- #
#  torch index-url derivation (pip does NOT record it per-package; we derive    #
#  it from the local version label).                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("line,expected", [
    ("torch==2.4.1+rocm6.2", "https://download.pytorch.org/whl/rocm6.2"),
    ("torch==2.5.1+cu124", "https://download.pytorch.org/whl/cu124"),
    ("torch==2.5.1+cpu", "https://download.pytorch.org/whl/cpu"),
    ("torchvision==0.19.1+rocm6.2", "https://download.pytorch.org/whl/rocm6.2"),
])
def test_torch_index_url_derived(line, expected):
    from localm.media import managed_comfy_provision as prov
    assert prov.torch_index_url_from_freeze([line]) == expected


def test_torch_index_url_none_for_plain_pypi():
    from localm.media import managed_comfy_provision as prov
    assert prov.torch_index_url_from_freeze(["torch==2.5.1", "numpy==1.26.4"]) is None
    assert prov.torch_index_url_from_freeze([]) is None


# --------------------------------------------------------------------------- #
#  The copy path, end to end and FOR REAL (the core oracle).                   #
# --------------------------------------------------------------------------- #

def _yaml_base_paths(yaml_path: Path) -> set:
    import yaml
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return {e["base_path"] for e in data.values()
            if isinstance(e, dict) and "base_path" in e}


def test_provision_copy_end_to_end(home, fake_user_comfy, monkeypatch):
    """Replicate the whole stack: same commit, a FRESH venv with the same tiny
    package, custom nodes copied, extra_model_paths at both model dirs, and the
    S1 routing now targets the managed instance."""
    from localm.media import managed_comfy_provision as prov

    # Deterministic + offline: the replayed `name @ file://` requirement resolves
    # from the on-disk wheel without ever touching an index.
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})

    stack = prov.discover_user_comfy()
    assert stack is not None
    result = prov.provision_by_copy(stack, copy_custom_nodes=True)
    assert result.ok, result.message

    paths = mc.managed_comfy_paths()
    # 1) ComfyUI checkout at the SAME commit.
    assert paths.main_py.is_file()
    managed_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(paths.root),
                                  capture_output=True, text=True, check=True).stdout.strip()
    assert managed_head == fake_user_comfy.commit
    # 2) A FRESH localm venv (its own interpreter, NOT the user's).
    assert paths.venv_python.is_file()
    assert Path(paths.venv_python).resolve() != Path(fake_user_comfy.venv_python).resolve()
    # 3) The replicated venv has the SAME tiny package version (real pip install ran).
    managed_freeze = _freeze(paths.venv_python)
    assert any(fake_user_comfy.pkg_name in ln for ln in managed_freeze), managed_freeze
    # 4) Custom nodes copied (opted in).
    assert (paths.root / "custom_nodes" / "NodeAlpha").is_dir()
    assert (paths.root / "custom_nodes" / "NodeBeta").is_dir()
    # 5) extra_model_paths.yaml points at BOTH the user's and localm's models dirs.
    assert paths.extra_model_paths.is_file()
    bases = _yaml_base_paths(paths.extra_model_paths)
    assert str(paths.models_dir) in bases
    assert str(fake_user_comfy.workdir / "models") in bases
    # 6) Installed -> S1 routing targets the managed instance when comfy_target=own.
    assert mc.is_managed_comfy_installed() is True
    cfg.save_config({**cfg.load_config(),
                     "managed_comfy_enabled": True, "comfy_target": "own"})
    target = mc.resolve_comfy_target()
    assert target.managed is True
    assert target.api_url == mc.MANAGED_COMFY_API_URL
    assert target.workdir == str(paths.root)


def test_provision_copy_leaves_out_custom_nodes_when_declined(home, fake_user_comfy,
                                                              monkeypatch):
    """--no-custom-nodes: everything provisions EXCEPT the user's custom nodes."""
    from localm.media import managed_comfy_provision as prov
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})

    stack = prov.discover_user_comfy()
    result = prov.provision_by_copy(stack, copy_custom_nodes=False)
    assert result.ok, result.message

    paths = mc.managed_comfy_paths()
    assert paths.main_py.is_file()            # source still cloned
    assert paths.venv_python.is_file()        # fresh venv still made
    assert mc.is_managed_comfy_installed() is True
    # The user's custom nodes were NOT brought over.
    assert not (paths.root / "custom_nodes" / "NodeAlpha").exists()
    assert not (paths.root / "custom_nodes" / "NodeBeta").exists()

    # Reversibility on a REAL git-cloned managed dir: git marks its object store
    # read-only, so plain rmtree fails on Windows - rmtree_robust must still remove it
    # (or `localm comfy remove` and rollback break after S2).
    mc.rmtree_robust(paths.root)
    assert not paths.root.exists()
    assert mc.is_managed_comfy_installed() is False


def test_provision_fails_and_rolls_back_when_freeze_unreadable(home, fake_user_comfy,
                                                               monkeypatch):
    """A FAILED provision must leave NOTHING that reads as installed. Simulate a pip
    freeze failure AFTER the clone + venv exist (when is_managed_comfy_installed()
    would otherwise be true) and assert it rolls back: ok=False, not installed, and
    the partial dir removed - never a broken-but-reads-as-ready facade (rule 5)."""
    from localm.media import managed_comfy_provision as prov
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})
    # None == pip freeze FAILED (distinct from an empty venv).
    monkeypatch.setattr(prov, "read_user_freeze", lambda venv_python: None)

    stack = prov.discover_user_comfy()
    result = prov.provision_by_copy(stack, copy_custom_nodes=False)
    assert result.ok is False
    assert "freeze" in result.message.lower() or "package set" in result.message.lower()
    assert mc.is_managed_comfy_installed() is False
    assert not mc.managed_comfy_paths().root.exists()   # rolled back, nothing lingers


def test_read_user_freeze_none_on_failure(tmp_path):
    """pip freeze failure returns None (distinct from [] for a genuinely empty venv),
    so the caller can tell a swallowed failure from an empty venv."""
    from localm.media import managed_comfy_provision as prov
    bogus = tmp_path / "nope" / ("python.exe" if os.name == "nt" else "python")
    assert prov.read_user_freeze(bogus) is None


def test_provision_refuses_when_managed_dir_exists(home, fake_user_comfy, monkeypatch):
    """A pre-existing managed dir is not silently overwritten - honest refuse."""
    from localm.media import managed_comfy_provision as prov
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})
    paths = mc.managed_comfy_paths()
    paths.root.mkdir(parents=True)
    (paths.root / "sentinel").write_text("keep me\n", encoding="utf-8")

    stack = prov.discover_user_comfy()
    result = prov.provision_by_copy(stack, copy_custom_nodes=False)
    assert result.ok is False
    assert "exists" in result.message.lower() or "remove" in result.message.lower()
    assert (paths.root / "sentinel").is_file()   # untouched


def test_count_user_custom_nodes(fake_user_comfy):
    from localm.media import managed_comfy_provision as prov
    assert prov.count_user_custom_nodes(fake_user_comfy.workdir) == 2


# --------------------------------------------------------------------------- #
#  The dispatcher + CLI: copy path when present, honest S3 report when not.    #
# --------------------------------------------------------------------------- #

def test_cli_setup_no_user_comfy_reports_s3_and_changes_nothing(cli_runner):
    """No usable user ComfyUI -> setup reports the fresh install is S3 (not yet
    implemented) and creates nothing (AGENTS.md rule 5)."""
    from localm.cli import main
    res = cli_runner.invoke(main, ["comfy", "setup"])
    assert res.exit_code == 0, res.output
    low = res.output.lower()
    assert "not yet implemented" in low or "not built" in low
    assert "s3" in low or "fresh" in low
    # Nothing provisioned.
    assert not mc.is_managed_comfy_installed()
    assert not mc.managed_comfy_paths().root.exists()


def test_cli_setup_selects_copy_path_and_passes_flag(cli_runner, fake_user_comfy,
                                                     monkeypatch):
    """With a usable user ComfyUI, setup selects the COPY path and forwards the
    custom-nodes choice from the flags. The real copy behaviour is proven by the
    core tests above; here we assert the CLI wiring (flag -> the right bool)."""
    import localm.config as cfg2
    from localm.cli import main
    from localm.media import managed_comfy_provision as prov

    cfg2.save_config({**cfg2.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})

    seen = {}

    def _spy(stack, cfg=None, *, copy_custom_nodes, on_progress=None):
        seen["copy"] = copy_custom_nodes
        seen["workdir"] = str(stack.workdir)
        return prov.ProvisionResult(ok=True, status="copied",
                                    message="ok", managed_root=stack.workdir)

    monkeypatch.setattr(prov, "provision_by_copy", _spy)

    res = cli_runner.invoke(main, ["comfy", "setup", "--copy-custom-nodes"])
    assert res.exit_code == 0, res.output
    assert seen["copy"] is True
    assert seen["workdir"] == str(fake_user_comfy.workdir)

    seen.clear()
    res = cli_runner.invoke(main, ["comfy", "setup", "--no-custom-nodes"])
    assert res.exit_code == 0, res.output
    assert seen["copy"] is False


def test_resolve_copy_custom_nodes_helper(monkeypatch):
    """The ASK resolution: an explicit flag wins; with no flag and no nodes it is
    False; with no flag, nodes present, and no TTY it defaults to False (never
    hangs on a prompt in a non-interactive run)."""
    from localm.media import managed_comfy_provision as prov
    assert prov.resolve_copy_custom_nodes(True, n_nodes=2, interactive=False) is True
    assert prov.resolve_copy_custom_nodes(False, n_nodes=2, interactive=False) is False
    assert prov.resolve_copy_custom_nodes(None, n_nodes=0, interactive=False) is False
    assert prov.resolve_copy_custom_nodes(None, n_nodes=3, interactive=False) is False


def test_resolve_copy_custom_nodes_interactive_confirm():
    """The interactive ASK (spec point 3): with no flag and nodes present, the user's
    confirm() answer is honored both ways."""
    from localm.media import managed_comfy_provision as prov
    assert prov.resolve_copy_custom_nodes(
        None, n_nodes=2, interactive=True, confirm=lambda: True) is True
    assert prov.resolve_copy_custom_nodes(
        None, n_nodes=2, interactive=True, confirm=lambda: False) is False
