# SPDX-License-Identifier: AGPL-3.0-or-later
"""STAGE S2 (COPY path) for the localm-managed ComfyUI feature.

S2 provisions localm's OWN ComfyUI by REPLICATING a user's existing stack: when
the user HAS a working ComfyUI (comfy_workdir set and a venv discoverable under
it), read its repo commit + a pip-freeze of their venv, clone that same commit
into <LOCALM_HOME>/comfyui, create a FRESH localm venv, and pip-install the SAME
package versions into it (NOT a byte-copy of the user's venv). Custom nodes are
copied only when asked. Models are shared via S1's extra_model_paths.yaml. When
NO usable user ComfyUI is present, setup runs the fresh hardware-matched install
instead (stage S3).

The heavy end-to-end test exercises the copy path FOR REAL against a MINIMAL
fake: a real throwaway git repo (stub main.py committed, so rev-parse works) with
a real minimal venv holding ONE tiny hand-built wheel (no torch - the copy logic
is version-agnostic, so a tiny freeze exercises it fully and fast). The wheel is
installed via a ``name @ file://`` reference so pip freeze round-trips OFFLINE
and the copy path can replay it with PIP_NO_INDEX. Multi-GB torch replication is
a manual check; this proves the mechanism end to end.
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
#  and the import-frozen config attrs.                                         #
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


def test_discover_finds_stack_via_plugin_only_workdir(home, fake_user_comfy):
    """A folder set ONLY via a plugin's own comfy.workdir field (no global
    comfy_workdir) - the shape the modern Settings UI actually produces - must
    still be discovered, so `localm comfy setup` copies the user's existing
    ComfyUI (S2) instead of silently running a redundant multi-GB fresh install
    (S3) because a global-only check found nothing."""
    from localm.media import managed_comfy_provision as prov
    cfg.save_config({**cfg.load_config(), "plugins": {
        "video": {"comfy": {"workdir": str(fake_user_comfy.workdir)}}}})
    stack = prov.discover_user_comfy()
    assert stack is not None
    assert stack.commit == fake_user_comfy.commit


def test_discover_prefers_legacy_global_over_plugin_workdir(home, fake_user_comfy, tmp_path):
    """The legacy global comfy_workdir, when ALSO set, still wins - matching
    discover_user_comfy()'s own documented order ("checks the legacy GLOBAL
    comfy_workdir first, then falls back to any plugin's own comfy.workdir").
    A plugin-only value pointed at a folder with no usable ComfyUI (no venv)
    must not silently override a genuinely usable global one."""
    from localm.media import managed_comfy_provision as prov
    unusable = tmp_path / "plugin-pointed-elsewhere"
    unusable.mkdir()
    (unusable / "main.py").write_text("# comfy, no venv\n", encoding="utf-8")
    cfg.save_config({**cfg.load_config(),
                     "comfy_workdir": str(fake_user_comfy.workdir),
                     "plugins": {"image": {"comfy": {"workdir": str(unusable)}}}})
    stack = prov.discover_user_comfy()
    assert stack is not None
    assert stack.commit == fake_user_comfy.commit


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
    # 2) A FRESH localm venv (its own directory tree, NOT the user's). Compare the
    # venv-relative paths, not resolve() through them: on POSIX, `python -m venv`
    # symlinks the venv's python back to the shared BASE interpreter, so two
    # distinct venvs built from the same base legitimately resolve() to the
    # identical real binary - that collapse is normal venv behavior, not a sign
    # provisioning reused the user's venv.
    assert paths.venv_python.is_file()
    assert Path(paths.venv_python) != Path(fake_user_comfy.venv_python)
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
    cfg.save_config({**cfg.load_config(), "comfy_target": "own"})
    target = mc.resolve_comfy_target()
    assert target.managed is True
    assert target.api_url == mc.MANAGED_COMFY_API_URL
    assert target.workdir == str(paths.root)


def test_isolated_env_strips_pythonpath(monkeypatch):
    """_isolated_env() must never leak localm's own PYTHONPATH into a subprocess that
    drives the user's or the managed venv - see its docstring for why."""
    from localm.media import managed_comfy_provision as prov
    monkeypatch.setenv("PYTHONPATH", "some/leaked/path")
    monkeypatch.setenv("SOME_OTHER_VAR", "kept")
    env = prov._isolated_env()
    assert "PYTHONPATH" not in env
    assert env.get("SOME_OTHER_VAR") == "kept"


def test_isolated_env_pins_the_pip_cache_into_the_data_dir(home, monkeypatch):
    """Provisioning's pip subprocesses must cache INSIDE the data dir.

    Unset, pip caches to a per-user location outside the data dir, without asking
    and without telling. An ambient PIP_CACHE_DIR must NOT win: containment any
    stray environment variable can silently switch off is not a guarantee.
    LOCALM_HOME is the knob (see config.cache_dir())."""
    from localm.media import managed_comfy_provision as prov
    monkeypatch.setenv("PIP_CACHE_DIR", str(home.parent / "ambient-cache"))

    env = prov._isolated_env()
    assert env["PIP_CACHE_DIR"] == str(prov.pip_cache_dir())
    assert prov.pip_cache_dir() == home / "cache" / "pip"
    assert home in prov.pip_cache_dir().parents            # inside the data dir
    if Path.home() not in home.parents:
        assert Path.home() not in prov.pip_cache_dir().parents  # NOT the home profile


def test_pip_subprocess_really_honours_the_contained_cache_dir(home, fake_user_comfy):
    """The EFFECT, not the setting: a REAL pip child, launched through the real
    _run()/_isolated_env() path, must resolve its cache inside the data dir.

    Setting an env var and asserting the env var proves nothing about the child - and
    provisioning's pip runs in the MANAGED ComfyUI's own venv, a different interpreter
    from localm's, so nothing about localm's own process env reaches it implicitly.
    ``pip cache dir`` makes the child report the location it would actually write to,
    so this asserts real resolved behaviour rather than our intent."""
    from localm.media import managed_comfy_provision as prov
    ok, out = prov._run([str(fake_user_comfy.venv_python), "-m", "pip",
                         "cache", "dir", "--disable-pip-version-check"], timeout=120)
    assert ok, out
    reported = Path(out.strip().splitlines()[-1].strip())
    assert reported == prov.pip_cache_dir(), out   # Windows Path eq is case-insensitive
    assert home in reported.parents, out


def test_read_user_freeze_ignores_leaked_pythonpath(home, fake_user_comfy, monkeypatch):
    """A PYTHONPATH set on the CALLING localm process must not contaminate pip
    freeze of the user's venv. With a dev PYTHONPATH pointing at localm's own
    venv, `pip freeze` on a 1-package venv reports the whole leaked environment -
    one of which needs a source build this feature never intended to trigger,
    surfacing downstream as a confusing "offline pip cache is missing a build
    backend" failure with no hint that the cause was environment leakage (see
    managed_comfy_provision._isolated_env()).

    ``pip freeze`` enumerates installed DISTRIBUTIONS via their ``.dist-info``
    metadata (not just importable modules), so the "leaked" directory here must
    be dist-info-shaped - a bare importable package would not be picked up either
    way and this test would pass even on the unfixed code."""
    from localm.media import managed_comfy_provision as prov
    leaked = home.parent / "leaked-site-packages"
    dist_info = leaked / "unrelatedpkg-1.0.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unrelatedpkg\nVersion: 1.0.0\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(leaked))

    freeze = prov.read_user_freeze(fake_user_comfy.venv_python)
    assert freeze is not None
    assert len(freeze) == 1, freeze
    assert fake_user_comfy.pkg_name in freeze[0]


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
    # read-only, so plain rmtree fails on Windows - rmtree_robust must still
    # remove it.
    mc.rmtree_robust(paths.root)
    assert not paths.root.exists()
    assert mc.is_managed_comfy_installed() is False


def test_provision_fails_and_rolls_back_when_freeze_unreadable(home, fake_user_comfy,
                                                               monkeypatch):
    """A FAILED provision must leave NOTHING that reads as installed. Simulates a
    pip freeze failure AFTER the clone + venv exist (when
    is_managed_comfy_installed() would otherwise be true) and asserts it rolls
    back: ok=False, not installed, and the partial dir removed."""
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


def test_unresolvable_packages_parses_pip_no_match_lines():
    """_unresolvable_packages parses pip's own 'No matching distribution found
    for <req>' wording, stripping the version pin so the message names the bare
    package, deduped and in first-seen order."""
    from localm.media import managed_comfy_provision as prov
    out = (
        "Collecting foo==1.0\n"
        "ERROR: Could not find a version that satisfies the requirement "
        "amd-torch-device-gfx1030==2.12.0+rocm7.14.0 (from versions: none)\n"
        "ERROR: No matching distribution found for "
        "amd-torch-device-gfx1030==2.12.0+rocm7.14.0\n"
    )
    assert prov._unresolvable_packages(out) == ["amd-torch-device-gfx1030"]
    assert prov._unresolvable_packages("Collecting foo==1.0\nSuccessfully installed foo-1.0\n") == []


def test_provision_copy_names_the_unresolvable_package_not_just_a_pip_tail(
        home, fake_user_comfy, monkeypatch):
    """When the user's freeze names a package pip can find NO version for at all
    (the shape produced by a vendor-bundled driver package installed from a
    non-public index or a local wheel on the ORIGINAL machine), the failure must
    name the package and say why, not just echo a generic pip-transcript tail.

    Injects one synthetic unresolvable line alongside the REAL freeze (so the
    real, tiny replicable package still installs and only the synthetic one
    fails) and runs the real pip subprocess under PIP_NO_INDEX=1, same as
    test_provision_copy_end_to_end - no mock of pip itself.

    Asserts wording that is NOT a substring of pip's own raw output ("pip found
    no matching version", "non-public index", "local wheel") rather than merely
    checking the package name appears somewhere: the package name alone would
    also appear in a generic tail-of-pip-output message.

    Also pins the recovery hint: the message must point at the actual next step
    (clear the ComfyUI folder setting, re-run setup for the fresh
    hardware-matched path) rather than leaving the user stuck on a dead-end copy
    path."""
    from localm.media import managed_comfy_provision as prov

    monkeypatch.setenv("PIP_NO_INDEX", "1")
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})

    real_freeze = prov.read_user_freeze
    bogus_pkg = "vendor-only-package"

    def _freeze_with_unresolvable(venv_python):
        lines = real_freeze(venv_python)
        return None if lines is None else [*lines, f"{bogus_pkg}==9.9.9"]

    monkeypatch.setattr(prov, "read_user_freeze", _freeze_with_unresolvable)

    stack = prov.discover_user_comfy()
    assert stack is not None
    result = prov.provision_by_copy(stack, copy_custom_nodes=False)

    assert result.ok is False
    assert bogus_pkg in result.message
    assert "pip found no matching version" in result.message
    assert "non-public index" in result.message or "local wheel" in result.message
    # The actionable next step: this path is a dead end, and the message must say
    # what to do instead (clear the workdir setting, re-run setup for S3 fresh).
    assert "cannot be replicated" in result.message
    assert "comfy setup" in result.message
    assert mc.is_managed_comfy_installed() is False
    assert not mc.managed_comfy_paths().root.exists()   # rolled back, nothing lingers


def test_provision_copy_fails_loudly_when_venv_has_no_pip(home, fake_user_comfy,
                                                           monkeypatch):
    """S2 copy path: the same pip probe as S3, so a pip-less managed venv is
    diagnosed immediately after creation rather than surfacing as an opaque
    failure two steps later inside the replicated-package install. Mocks only
    `_run` (git clone/checkout are trivially satisfied, no real network needed for
    this probe-only check) - without the probe, the fake `_run` below would
    receive the subsequent `pip install` replay call and raise, which is what
    makes this a real fires-control."""
    from localm.media import managed_comfy_provision as prov

    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})
    stack = prov.discover_user_comfy()
    assert stack is not None

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "clone" in cmd or "checkout" in cmd:
            return True, ""
        if "venv" in cmd:
            # Venv creation genuinely "succeeds" (dest exists) - only pip is
            # missing, matching the real broken instance this entry describes.
            dest = Path(cmd[-1])
            venv_py = _venv_python(dest)
            venv_py.parent.mkdir(parents=True, exist_ok=True)
            venv_py.write_bytes(b"")
            return True, ""
        if "pip" in cmd and "--version" in cmd:
            return False, "No module named pip"
        raise AssertionError(f"unexpected call reached past the pip probe: {cmd}")

    monkeypatch.setattr(prov, "_run", _fake_run)

    result = prov.provision_by_copy(stack, copy_custom_nodes=False)

    assert result.ok is False
    assert "no working pip" in result.message
    assert "localm doctor" in result.message
    install_calls = [c for c in calls if "install" in c]
    assert install_calls == [], f"reached an install call past the probe: {install_calls}"
    assert mc.is_managed_comfy_installed() is False
    assert not mc.managed_comfy_paths().root.exists()   # rolled back, nothing lingers


def test_copy_custom_nodes_returns_warnings_not_silent(tmp_path, monkeypatch):
    """A node that fails to copy is RETURNED as a warning (so the caller can route
    it into the run log), not only streamed and forgotten. Both the dir-copy and
    the .py-file-copy failure paths are covered."""
    from localm.media import managed_comfy_provision as prov
    user = tmp_path / "user"
    (user / "custom_nodes" / "NodeDir").mkdir(parents=True)
    (user / "custom_nodes" / "NodeDir" / "__init__.py").write_text("", encoding="utf-8")
    (user / "custom_nodes" / "node_file.py").write_text("# a node\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()

    def _boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(prov.shutil, "copytree", _boom)
    monkeypatch.setattr(prov.shutil, "copy2", _boom)

    count, warnings = prov._copy_custom_nodes(user, managed)
    assert count == 0
    assert len(warnings) == 2
    joined = " ".join(warnings)
    assert "NodeDir" in joined and "node_file.py" in joined
    assert "disk full" in joined


# --------------------------------------------------------------------------- #
#  ProgressCb carries a structured ProgressEvent, not just text, and           #
#  _copy_custom_nodes threads on_progress through with a per-item emit.        #
# --------------------------------------------------------------------------- #

def test_progress_event_is_a_string_and_carries_structure():
    """The widened type IS the human-readable line (every existing sink just prints
    it, unaffected) and, for a countable step, carries the facts that line was built
    from directly - so a consumer never has to regex them back out of the text."""
    from localm.media import managed_comfy_provision as prov
    ev = prov.ProgressEvent("Cloning your ComfyUI (abc123) into /x ...", phase="clone")
    assert isinstance(ev, str)
    assert ev == "Cloning your ComfyUI (abc123) into /x ..."
    assert ev.phase == "clone"
    assert ev.done is None and ev.total is None and ev.unit is None

    counted = prov.ProgressEvent("3/9", phase="custom_nodes", done=3, total=9,
                                 unit="nodes")
    assert (counted.done, counted.total, counted.unit) == (3, 9, "nodes")


def test_emit_wraps_a_plain_string_into_a_bare_progress_event():
    """_run()'s raw subprocess-line relay (and any other caller with nothing
    structured to add) still just hands a str to _emit - it must arrive at the sink
    as a ProgressEvent with every structured field absent, never crash or silently
    stay a plain str."""
    from localm.media import managed_comfy_provision as prov
    seen = []
    prov._emit(seen.append, "a raw subprocess line")
    assert len(seen) == 1
    ev = seen[0]
    assert isinstance(ev, prov.ProgressEvent)
    assert ev == "a raw subprocess line"
    assert ev.phase is None and ev.done is None and ev.total is None


def test_emit_raising_sink_is_swallowed_not_raised():
    """A raising progress sink is best-effort and must never propagate out of
    _emit."""
    from localm.media import managed_comfy_provision as prov

    def _boom(event):
        raise RuntimeError("a broken progress sink")

    prov._emit(_boom, "line")   # must not raise


def test_copy_custom_nodes_reports_structured_progress_per_node(tmp_path):
    """Each node gets its own progress event with phase/done/total/unit set
    directly - a listener never has to parse "Copying custom node X (i/n) ..." to
    learn i, n or the unit."""
    from localm.media import managed_comfy_provision as prov
    user = tmp_path / "user"
    for node in ("Alpha", "Beta", "Gamma"):
        nd = user / "custom_nodes" / node
        nd.mkdir(parents=True)
        (nd / "__init__.py").write_text("", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()

    events = []
    count, warnings = prov._copy_custom_nodes(user, managed, on_progress=events.append)

    assert count == 3 and not warnings
    assert all(isinstance(e, prov.ProgressEvent) for e in events)
    assert [e.phase for e in events] == ["custom_nodes"] * 3
    assert [e.done for e in events] == [1, 2, 3]
    assert all(e.total == 3 for e in events)
    assert all(e.unit == "nodes" for e in events)


def test_copy_custom_nodes_raising_sink_does_not_abort_the_copy(tmp_path):
    """A raising sink must not abort the copy it is merely reporting on."""
    from localm.media import managed_comfy_provision as prov
    user = tmp_path / "user"
    (user / "custom_nodes" / "NodeA").mkdir(parents=True)
    (user / "custom_nodes" / "NodeA" / "__init__.py").write_text("", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()

    def _raising_sink(event):
        raise RuntimeError("a broken progress sink")

    count, warnings = prov._copy_custom_nodes(user, managed, on_progress=_raising_sink)
    assert count == 1 and not warnings
    assert (managed / "custom_nodes" / "NodeA").is_dir()


def test_provision_survives_a_raising_progress_sink(home, fake_user_comfy, monkeypatch):
    """End to end: a raising on_progress sink must not abort a real provision that
    would otherwise succeed - covers both the per-step narration (_say) and the
    per-node emit (_copy_custom_nodes) a real copy exercises."""
    from localm.media import managed_comfy_provision as prov
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})

    def _raising_sink(event):
        raise RuntimeError("a broken progress sink")

    stack = prov.discover_user_comfy()
    result = prov.provision_by_copy(stack, copy_custom_nodes=True,
                                    on_progress=_raising_sink)
    assert result.ok, result.message
    assert mc.is_managed_comfy_installed() is True


def test_provision_copy_node_failures_land_in_result(home, fake_user_comfy, monkeypatch):
    """End to end: a non-fatal custom-node copy failure must SURVIVE into the
    result, not only the live progress stream. It lands in ProvisionResult.log and
    its count is folded into the success message. Provisioning still succeeds - a
    failed node only breaks the workflow needing it, not the whole install."""
    from localm.media import managed_comfy_provision as prov

    # Skip the real pip replicate (empty freeze) to keep this fast; the fresh venv is
    # still really created so the result reads as installed. The copy path is what we
    # exercise. Only the user's ComfyUI git clone (not copytree) provides the source.
    monkeypatch.setattr(prov, "read_user_freeze", lambda venv_python: [])
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(fake_user_comfy.workdir)})

    def _boom(*a, **k):
        raise OSError("simulated copy failure")
    monkeypatch.setattr(prov.shutil, "copytree", _boom)  # NodeAlpha + NodeBeta are dirs

    stack = prov.discover_user_comfy()
    result = prov.provision_by_copy(stack, copy_custom_nodes=True)

    assert result.ok, result.message                     # non-fatal
    assert mc.is_managed_comfy_installed() is True
    assert "could not copy custom node" in result.log     # survived into the log ...
    assert "NodeAlpha" in result.log and "NodeBeta" in result.log
    assert "2 custom node(s) could not be copied" in result.message  # ... and the message


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
#  The dispatcher + CLI: copy path when a user ComfyUI is present (fresh when not). #
# --------------------------------------------------------------------------- #

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
    """The interactive ASK: with no flag and nodes present, the user's confirm()
    answer is honored both ways."""
    from localm.media import managed_comfy_provision as prov
    assert prov.resolve_copy_custom_nodes(
        None, n_nodes=2, interactive=True, confirm=lambda: True) is True
    assert prov.resolve_copy_custom_nodes(
        None, n_nodes=2, interactive=True, confirm=lambda: False) is False
